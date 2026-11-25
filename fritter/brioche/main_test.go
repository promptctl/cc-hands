package main

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/bmf/cc-hands/fritter/testrun"
)

// program names what the test binary is when a test starts it as a child, as Go's own
// os/exec tests start theirs: brioche in its process, for how it stops, which only its
// process shows. Unset, the binary runs the tests.
const program = "BRIOCHE_TEST_PROGRAM"

func TestMain(m *testing.M) {
	switch named := os.Getenv(program); named {
	case "":
		os.Exit(testrun.Main(m))
	case "brioche":
		os.Exit(serve(os.Args[1:]))
	default:
		fmt.Fprintf(os.Stderr, "%s=%q names no program\n", program, named)
		os.Exit(2)
	}
}

// Each test package reads the hold again, as testrun.Hold says why.
func TestAnEditToTheHoldIsNeverACachedPass(t *testing.T) {
	if _, err := testrun.Hold(); err != nil {
		t.Error(err)
	}
}

func unproxied(*http.Request) (*url.URL, error) { return nil, nil }

// said is what brioche says, read while it may still be saying more.
type said struct {
	mu      sync.Mutex
	more    *sync.Cond // broadcast at each write, and once the test is out of time
	lines   bytes.Buffer
	expired bool
}

func saying() *said {
	s := &said{}
	s.more = sync.NewCond(&s.mu)
	return s
}

func (s *said) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	defer s.more.Broadcast()
	return s.lines.Write(p)
}

func (s *said) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lines.String()
}

// until returns once brioche has said `line`: what the test waits for is the saying itself.
// A test that runs out of time first fails saying what brioche did say, short of go test's
// own timeout, whose dump would not.
func (s *said) until(t *testing.T, line string) {
	t.Helper()
	if deadline, ok := t.Deadline(); ok {
		expire := time.AfterFunc(time.Until(deadline)-5*time.Second, func() {
			s.mu.Lock()
			defer s.mu.Unlock()
			s.expired = true
			s.more.Broadcast()
		})
		defer expire.Stop()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for !strings.Contains(s.lines.String(), line) && !s.expired {
		s.more.Wait()
	}
	if s.expired {
		t.Fatalf("brioche never said %q; it said %q", line, s.lines.String())
	}
}

// arguments is a whole brioche command line for `to`, with what `changed` gives each flag
// in place of its own: a value, or "" for a flag left out.
func arguments(to, file string, changed map[string]string) []string {
	given := map[string]string{"--intercept": "https://api.anthropic.com", "--to": to, "--trust-variable": "NODE_EXTRA_CA_CERTS", "--trust-file": file}
	var args []string
	for _, flag := range flags {
		value, ok := changed[flag]
		if !ok {
			value = given[flag]
		}
		if value != "" {
			args = append(args, flag, value)
		}
	}
	return args
}

// serving is a brioche started for a test: the environment it printed, and what it says.
type serving struct {
	environment map[string]string
	said        *said
}

// started runs brioche intercepting https://api.anthropic.com for `to`, in a client
// environment of `client`, reaching the world through `outer`, and stops it as the test
// ends, which it must stop for.
func started(t *testing.T, to string, client map[string]string, outer func(*http.Request) (*url.URL, error)) serving {
	t.Helper()
	read, printed, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer read.Close()
	says := saying()
	ended := make(chan stop, 1)
	exited := make(chan int, 1)
	args := arguments(to, filepath.Join(t.TempDir(), "trusted.pem"), nil)
	go func() {
		exited <- run(args, func(name string) string { return client[name] }, outer, printed, says, ended)
	}()
	environment := map[string]string{}
	lines := bufio.NewScanner(read)
	for lines.Scan() {
		name, value, found := strings.Cut(lines.Text(), "=")
		if !found {
			t.Fatalf("brioche printed %q, which is no NAME=VALUE", lines.Text())
		}
		environment[name] = value
	}
	if len(environment) == 0 {
		t.Fatalf("brioche printed no environment, ending %d and saying %q", <-exited, says)
	}
	t.Cleanup(func() {
		ended <- stop{"the test ended", nil}
		if code := <-exited; code != 0 {
			t.Errorf("brioche ended %d once told to stop, saying %q", code, says)
		}
	})
	return serving{environment: environment, said: says}
}

// trusting is a client of brioche's that trusts what roots holds, and nothing else.
func trusting(t *testing.T, brioche serving, roots *x509.CertPool) *http.Client {
	t.Helper()
	proxy, err := url.Parse(brioche.environment["HTTPS_PROXY"])
	if err != nil {
		t.Fatal(err)
	}
	transport := &http.Transport{Proxy: http.ProxyURL(proxy), TLSClientConfig: &tls.Config{RootCAs: roots}}
	t.Cleanup(transport.CloseIdleConnections)
	return &http.Client{Transport: transport}
}

// trusted is what the file brioche printed for the client to trust holds.
func trusted(t *testing.T, brioche serving) *x509.CertPool {
	t.Helper()
	held, err := os.ReadFile(brioche.environment["NODE_EXTRA_CA_CERTS"])
	if err != nil {
		t.Fatal(err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(held) {
		t.Fatalf("the file brioche gave the client to trust holds no certificate: %q", held)
	}
	return roots
}

// refusing is a loopback origin that takes each connection and hangs up on it: one that
// refuses, held for the test, so no other listener can come to answer on its port.
func refusing(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { listener.Close() })
	go func() {
		for {
			connection, err := listener.Accept()
			if err != nil {
				return
			}
			connection.Close()
		}
	}()
	return "http://" + listener.Addr().String()
}

func TestAClientTrustingBriochesFileReachesTheOriginWithTheRequestAsSentAndGetsItsAnswerAsItComes(t *testing.T) {
	type asked struct{ method, uri, host, header, forwarded, body string }
	got := make(chan asked, 1)
	// The origin holds its answer's second part until the client has read the first: the
	// client reads it only if brioche passed it on as it came.
	second := make(chan struct{})
	var letGo sync.Once
	release := func() { letGo.Do(func() { close(second) }) }
	origin := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(request.Body)
		got <- asked{request.Method, request.RequestURI, request.Host, request.Header.Get("X-Asked"), request.Header.Get("X-Forwarded-For"), string(body)}
		writer.WriteHeader(http.StatusOK)
		io.WriteString(writer, "first,")
		writer.(http.Flusher).Flush()
		<-second
		io.WriteString(writer, "second")
	}))
	t.Cleanup(origin.Close)
	// Last in, so run first: a test that fails before letting go never leaves the origin's
	// handler holding its Close.
	t.Cleanup(release)
	brioche := started(t, origin.URL, nil, unproxied)
	// A query ReverseProxy cannot parse, which it would rewrite, and a forwarding header,
	// which it would drop: as sent, both reach the origin.
	request, err := http.NewRequest(http.MethodPost, "https://api.anthropic.com/v1/messages?beta=true&a=1;b=2", strings.NewReader(`{"asked":true}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("X-Asked", "as sent")
	request.Header.Set("X-Forwarded-For", "203.0.113.9")
	response, err := trusting(t, brioche, trusted(t, brioche)).Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	first := make([]byte, len("first,"))
	if _, err := io.ReadFull(response.Body, first); err != nil || string(first) != "first," {
		t.Fatalf("the answer began %q (%v), want first,", first, err)
	}
	release()
	rest, err := io.ReadAll(response.Body)
	if err != nil || string(rest) != "second" {
		t.Errorf("the answer went on %q (%v), want second", rest, err)
	}
	want := asked{http.MethodPost, "/v1/messages?beta=true&a=1;b=2", "api.anthropic.com", "as sent", "203.0.113.9", `{"asked":true}`}
	if reached := <-got; reached != want {
		t.Errorf("the origin was asked %+v, want %+v", reached, want)
	}
}

func TestAConnectToAnyOtherHostIsTunnelled(t *testing.T) {
	elsewhere := httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		io.WriteString(writer, "elsewhere")
	}))
	t.Cleanup(elsewhere.Close)
	brioche := started(t, refusing(t), nil, unproxied)
	roots := x509.NewCertPool()
	roots.AddCert(elsewhere.Certificate())
	// Trusting elsewhere's own certificate and not brioche's: the TLS is elsewhere's.
	response, err := trusting(t, brioche, roots).Get(elsewhere.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if body, err := io.ReadAll(response.Body); err != nil || string(body) != "elsewhere" {
		t.Errorf("the tunnelled request was answered %q (%v), want elsewhere's answer", body, err)
	}
}

func TestAPlainRequestForAnyOtherOriginGoesOnAsItWasAsked(t *testing.T) {
	got := make(chan string, 1)
	elsewhere := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		got <- request.RequestURI + " " + request.Header.Get("X-Forwarded-For")
		io.WriteString(writer, "elsewhere")
	}))
	t.Cleanup(elsewhere.Close)
	brioche := started(t, refusing(t), nil, unproxied)
	request, err := http.NewRequest(http.MethodGet, elsewhere.URL+"/page?a=1;b=2", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("X-Forwarded-For", "203.0.113.9")
	response, err := trusting(t, brioche, nil).Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if body, err := io.ReadAll(response.Body); err != nil || string(body) != "elsewhere" {
		t.Errorf("the request was answered %q (%v), want elsewhere's answer", body, err)
	}
	if asked := <-got; asked != "/page?a=1;b=2 203.0.113.9" {
		t.Errorf("elsewhere was asked %q, want the request as it was asked", asked)
	}
}

func TestWhatGoesAnywhereElseGoesThroughTheOuterProxyAndWhatGoesToTheOriginDoesNot(t *testing.T) {
	// The outer proxy notes each CONNECT it is asked for, and refuses it.
	connects := make(chan string, 1)
	outer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		connects <- request.Method + " " + request.Host
		http.Error(writer, "the outer proxy refuses", http.StatusForbidden)
	}))
	t.Cleanup(outer.Close)
	through, err := url.Parse(outer.URL)
	if err != nil {
		t.Fatal(err)
	}
	origin := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		io.WriteString(writer, "the origin")
	}))
	t.Cleanup(origin.Close)
	brioche := started(t, origin.URL, nil, http.ProxyURL(through))
	if _, err := trusting(t, brioche, nil).Get("https://elsewhere.example:8443/"); err == nil {
		t.Error("a tunnel the outer proxy refused was joined")
	}
	if asked := <-connects; asked != "CONNECT elsewhere.example:8443" {
		t.Errorf("the outer proxy was asked %q, want the tunnel to elsewhere", asked)
	}
	response, err := trusting(t, brioche, trusted(t, brioche)).Get("https://api.anthropic.com/v1/messages")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if body, err := io.ReadAll(response.Body); err != nil || string(body) != "the origin" {
		t.Errorf("a request for the origin was answered %q (%v), want the origin's answer, reached directly", body, err)
	}
}

func TestAnOriginThatRefusesIsAnsweredAsA502NamingBriocheAndIsSaid(t *testing.T) {
	down := refusing(t)
	brioche := started(t, down, nil, unproxied)
	response, err := trusting(t, brioche, trusted(t, brioche)).Get("https://api.anthropic.com/v1/messages")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(response.Body)
	if response.StatusCode != http.StatusBadGateway || !strings.HasPrefix(string(body), "brioche could not reach "+down+"/v1/messages") {
		t.Errorf("the client was answered %d %q, want a 502 naming brioche and where it could not reach", response.StatusCode, body)
	}
	if !strings.Contains(brioche.said.String(), "brioche: could not reach "+down+" for /v1/messages: ") {
		t.Errorf("brioche said %q, want the request it could not take to the origin", brioche.said)
	}
}

func TestAnAnswerTheOriginCutShortIsSaid(t *testing.T) {
	origin := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		io.WriteString(writer, "part")
		writer.(http.Flusher).Flush()
		// The origin's server hangs up mid-answer.
		panic(http.ErrAbortHandler)
	}))
	t.Cleanup(origin.Close)
	brioche := started(t, origin.URL, nil, unproxied)
	response, err := trusting(t, brioche, trusted(t, brioche)).Get("https://api.anthropic.com/v1/messages")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadAll(response.Body); err == nil {
		t.Error("an answer the origin cut short reached the client whole")
	}
	response.Body.Close()
	brioche.said.until(t, "brioche: httputil: ReverseProxy read error during body copy: ")
}

func TestTheClientIsGivenTheCertificatesItsVariableNamedBesideBriochesAuthority(t *testing.T) {
	theirs := httptest.NewTLSServer(http.NotFoundHandler())
	t.Cleanup(theirs.Close)
	own := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: theirs.Certificate().Raw})
	named := filepath.Join(t.TempDir(), "theirs.pem")
	if err := os.WriteFile(named, own, 0o600); err != nil {
		t.Fatal(err)
	}
	brioche := started(t, refusing(t), map[string]string{"NODE_EXTRA_CA_CERTS": named}, unproxied)
	held, err := os.ReadFile(brioche.environment["NODE_EXTRA_CA_CERTS"])
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(held, own) || len(held) == len(own) {
		t.Errorf("the client is given %q to trust, want what it named, then brioche's authority", held)
	}
	for _, name := range []string{"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"} {
		if !strings.HasPrefix(brioche.environment[name], "http://127.0.0.1:") {
			t.Errorf("%s=%q, want brioche's loopback address", name, brioche.environment[name])
		}
	}
	for _, name := range []string{"NO_PROXY", "no_proxy"} {
		if value, given := brioche.environment[name]; !given || value != "" {
			t.Errorf("%s is %q (given: %v), want it given empty, so no host goes around brioche", name, value, given)
		}
	}
}

func TestARelativeTrustFileIsGivenTheClientAsTheFileBriocheWrote(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	read, printed, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	ended := make(chan stop, 1)
	ended <- stop{"the test ended", nil}
	says := saying()
	if code := run(arguments(refusing(t), "trusted.pem", nil), func(string) string { return "" }, unproxied, printed, says, ended); code != 0 {
		t.Fatalf("brioche ended %d, saying %q", code, says)
	}
	environment, _ := io.ReadAll(read)
	if want := "NODE_EXTRA_CA_CERTS=" + filepath.Join(dir, "trusted.pem") + "\n"; !strings.HasPrefix(string(environment), want) {
		t.Errorf("brioche printed %q, want it to begin %q, found from any directory", environment, want)
	}
}

func TestCertificatesTheClientNamesRelativeToItsOwnDirectoryAreSaidAndNotGiven(t *testing.T) {
	brioche := started(t, refusing(t), map[string]string{"NODE_EXTRA_CA_CERTS": "corp.pem"}, unproxied)
	if want := `brioche: cannot read the certificates NODE_EXTRA_CA_CERTS names, "corp.pem": it is relative to the client's directory`; !strings.Contains(brioche.said.String(), want) {
		t.Errorf("brioche said %q, want %q", brioche.said, want)
	}
	if held, err := os.ReadFile(brioche.environment["NODE_EXTRA_CA_CERTS"]); err != nil || bytes.Count(held, []byte("BEGIN CERTIFICATE")) != 1 {
		t.Errorf("the client is given %q (%v) to trust, want brioche's authority alone", held, err)
	}
}

func TestAClientWhoseVariableNamesBriochesOwnFileIsRefused(t *testing.T) {
	file := filepath.Join(t.TempDir(), "trusted.pem")
	_, printed, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	says := saying()
	// Told to stop already: a brioche that took the client's file would serve, and end 0.
	ended := make(chan stop, 1)
	ended <- stop{"the test ended", nil}
	code := run(arguments(refusing(t), file, nil), func(string) string { return file }, unproxied, printed, says, ended)
	if code != failed || !strings.Contains(says.String(), "NODE_EXTRA_CA_CERTS names "+file+", the file brioche writes") {
		t.Errorf("brioche ended %d saying %q, want %d saying the client's variable names brioche's own file", code, says, failed)
	}
}

func TestAClientThatRefusesBriochesCertificateIsSaidToHaveRefusedIt(t *testing.T) {
	brioche := started(t, refusing(t), nil, unproxied)
	// Trusting nothing of brioche's: the client refuses the TLS brioche answers it with, as Go's
	// client does a certificate of an authority it does not know.
	if _, err := trusting(t, brioche, x509.NewCertPool()).Get("https://api.anthropic.com/v1/messages"); err == nil {
		t.Fatal("a client that trusts nothing of brioche's was answered")
	}
	brioche.said.until(t, "brioche: the client refused brioche's answer for api.anthropic.com:443: tls: bad certificate\n")
}

func TestArgumentsBriocheCannotTakeAreRefusedSayingWhich(t *testing.T) {
	file := filepath.Join(t.TempDir(), "trusted.pem")
	for _, refused := range []struct {
		args []string
		says string
	}{
		{arguments("http://127.0.0.1:1", file, map[string]string{"--intercept": ""}), "--intercept is needed"},
		{append(arguments("http://127.0.0.1:1", file, nil), "--to", "http://127.0.0.1:2"), "--to is given twice"},
		{append(arguments("http://127.0.0.1:1", file, map[string]string{"--trust-file": ""}), "--trust-file", ""), "--trust-file needs a value"},
		{append(arguments("http://127.0.0.1:1", file, nil), "--tap"), `unknown argument "--tap"`},
		{arguments("http://127.0.0.1:1", file, map[string]string{"--intercept": "api.anthropic.com"}), "--intercept: an origin is an http or https URL"},
		{arguments("http://127.0.0.1:1/proxy", file, nil), `--to: "http://127.0.0.1:1/proxy" names more than an origin`},
		{arguments("http://127.0.0.1:1", file, map[string]string{"--trust-variable": "HTTPS_PROXY"}), "--trust-variable:"},
		{arguments("http://127.0.0.1:1", file+"\nHTTPS_PROXY=elsewhere", nil), "--trust-file: a path printed on a line of its own holds no line break"},
	} {
		says := saying()
		read, printed, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		code := run(refused.args, func(string) string { return "" }, unproxied, printed, says, nil)
		// run closes what it prints to however it ends: a read of it ends.
		if environment, err := io.ReadAll(read); err != nil || len(environment) != 0 {
			t.Errorf("brioche %q printed %q (%v), want nothing", refused.args, environment, err)
		}
		read.Close()
		if code != misuse || !strings.Contains(says.String(), refused.says) {
			t.Errorf("brioche %q ended %d saying %q, want %d saying %q", refused.args, code, says, misuse, refused.says)
		}
	}
}

// process is brioche started as its own process, in the test's environment, once it has
// printed its environment: its standard input, and what it has said and will say.
type process struct {
	child *exec.Cmd
	input io.WriteCloser
	said  *said
}

func processed(t *testing.T) process {
	t.Helper()
	child := exec.Command(os.Args[0], arguments(refusing(t), filepath.Join(t.TempDir(), "trusted.pem"), nil)...)
	child.Env = append(os.Environ(), program+"=brioche")
	input, err := child.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	printed, err := child.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	says := saying()
	child.Stderr = says
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { child.Process.Kill() })
	if environment, err := io.ReadAll(printed); err != nil || !bytes.Contains(environment, []byte("HTTPS_PROXY=http://127.0.0.1:")) {
		t.Fatalf("brioche printed %q (%v), saying %q", environment, err, says)
	}
	return process{child: child, input: input, said: says}
}

func TestABriocheWhoseInputEndsStopsSayingSo(t *testing.T) {
	brioche := processed(t)
	brioche.input.Close()
	if err := brioche.child.Wait(); err != nil {
		t.Fatalf("brioche ended %v once its input ended, saying %q", err, brioche.said)
	}
	if !strings.Contains(brioche.said.String(), "brioche: stopping: its input ended\n") {
		t.Errorf("brioche said %q, want why it stopped", brioche.said)
	}
}

func TestABriocheToldToStopStopsSayingSo(t *testing.T) {
	brioche := processed(t)
	if err := brioche.child.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	if err := brioche.child.Wait(); err != nil {
		t.Fatalf("brioche ended %v once told to stop, saying %q", err, brioche.said)
	}
	if !strings.Contains(brioche.said.String(), "brioche: stopping: told to by terminated\n") {
		t.Errorf("brioche said %q, want why it stopped", brioche.said)
	}
}
