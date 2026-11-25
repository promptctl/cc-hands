package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
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

// listening is where brioche listens in a test that has no listener of its own to hand it.
func listening() (net.Listener, error) { return net.Listen("tcp", "127.0.0.1:0") }

// said is what brioche says, read while it may still be saying more: each report, and the
// reports in words, a "brioche: " line each, as a reader of them sees it. A line that is no
// report is said as one of kind "unframed", which no test waits for.
type said struct {
	mu      sync.Mutex
	more    *sync.Cond // broadcast at each write, and once the test is out of time
	pending []byte     // a line not yet ended
	reports []report
	lines   bytes.Buffer
	expired bool
}

func saying() *said {
	s := &said{}
	s.more = sync.NewCond(&s.mu)
	return s
}

// Write takes what brioche says, in whatever pieces it comes: from its sayer a report a
// write, and from its process's standard error as a pipe gives it.
func (s *said) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	defer s.more.Broadcast()
	s.pending = append(s.pending, p...)
	for {
		line, rest, whole := bytes.Cut(s.pending, []byte("\n"))
		if !whole {
			return len(p), nil
		}
		s.pending = rest
		var told report
		if err := json.Unmarshal(line, &told); err != nil {
			told = report{Kind: "unframed", Level: failure, Said: string(line)}
		}
		s.reports = append(s.reports, told)
		fmt.Fprintf(&s.lines, "brioche: %s\n", told.Said)
	}
}

func (s *said) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lines.String()
}

// of is each report of `kind` brioche has made.
func (s *said) of(kind string) []report {
	s.mu.Lock()
	defer s.mu.Unlock()
	var found []report
	for _, told := range s.reports {
		if told.Kind == kind {
			found = append(found, told)
		}
	}
	return found
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
		value, ok := changed[flag.name]
		if !ok {
			value = given[flag.name]
		}
		if value != "" {
			args = append(args, flag.name, value)
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
	ended := stops()
	exited := make(chan int, 1)
	args := arguments(to, filepath.Join(t.TempDir(), "trusted.pem"), nil)
	go func() {
		exited <- run(args, func(name string) string { return client[name] }, outer, listening, printed, says, ended)
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
		ended.send(stop{"the test ended", nil})
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
	brioche.said.until(t, "brioche: intercepted https://api.anthropic.com/v1/messages?beta=true&a=1;b=2: answered 200\n")
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
	brioche.said.until(t, "brioche: could not tunnel to elsewhere.example:8443: ")
	// A plain request, which brioche forwards itself, goes through the outer proxy too: one
	// that went around it would reach elsewhere, and be answered.
	elsewhere := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		io.WriteString(writer, "reached around the outer proxy")
	}))
	t.Cleanup(elsewhere.Close)
	response, err := trusting(t, brioche, nil).Get(elsewhere.URL + "/page")
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusForbidden {
		t.Fatalf("a plain request for elsewhere was answered %s, want the outer proxy's refusal", response.Status)
	}
	if asked, want := <-connects, "GET "+strings.TrimPrefix(elsewhere.URL, "http://"); asked != want {
		t.Errorf("the outer proxy was asked %q, want %q", asked, want)
	}
	response, err = trusting(t, brioche, trusted(t, brioche)).Get("https://api.anthropic.com/v1/messages")
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
	if !strings.Contains(brioche.said.String(), "brioche: could not reach "+down+"/v1/messages: ") {
		t.Errorf("brioche said %q, want the request it could not take to the origin", brioche.said)
	}
	// Hands' proxy unreached leaves every request of the brain's failing: an error, never traffic.
	if told := brioche.said.of("bad_gateway"); len(told) != 1 || told[0].Level != failure || told[0].URL != down+"/v1/messages" {
		t.Errorf("brioche reported %+v, want one error naming the request it could not take", told)
	}
}

func TestAClientThatGivesUpBeforeTheOriginAnswersIsSaidToHaveGivenUp(t *testing.T) {
	asked := make(chan struct{})
	held := make(chan struct{})
	origin := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		close(asked)
		<-held
	}))
	t.Cleanup(origin.Close)
	t.Cleanup(func() { close(held) })
	brioche := started(t, origin.URL, nil, unproxied)
	giving, giveUp := context.WithCancel(context.Background())
	request, err := http.NewRequestWithContext(giving, http.MethodPost, "https://api.anthropic.com/v1/messages", strings.NewReader("{}"))
	if err != nil {
		t.Fatal(err)
	}
	client := trusting(t, brioche, trusted(t, brioche))
	answered := make(chan error, 1)
	go func() {
		_, err := client.Do(request)
		answered <- err
	}()
	<-asked
	giveUp()
	if err := <-answered; err == nil {
		t.Fatal("a request the client gave up on was answered")
	}
	brioche.said.until(t, "brioche: the client gave up on "+origin.URL+"/v1/messages before it was answered: ")
	if strings.Contains(brioche.said.String(), "could not reach") {
		t.Errorf("brioche said %q, want no word of an origin it reached", brioche.said)
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
	ended := stops()
	ended.send(stop{"the test ended", nil})
	says := saying()
	if code := run(arguments(refusing(t), "trusted.pem", nil), func(string) string { return "" }, unproxied, listening, printed, says, ended); code != 0 {
		t.Fatalf("brioche ended %d, saying %q", code, says)
	}
	environment, _ := io.ReadAll(read)
	if want := "NODE_EXTRA_CA_CERTS=" + filepath.Join(dir, "trusted.pem") + "\n"; !strings.HasPrefix(string(environment), want) {
		t.Errorf("brioche printed %q, want it to begin %q, found from any directory", environment, want)
	}
}

// refused is how brioche ends for a client whose variable names `named`, told to stop
// already: a brioche that took it would serve, and end 0.
func refused(t *testing.T, file, named string) (int, string) {
	t.Helper()
	_, printed, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	says := saying()
	ended := stops()
	ended.send(stop{"the test ended", nil})
	code := run(arguments(refusing(t), file, nil), func(string) string { return named }, unproxied, listening, printed, says, ended)
	return code, says.String()
}

func TestAClientWhoseVariableNamesCertificatesRelativeToItsOwnDirectoryIsRefused(t *testing.T) {
	code, says := refused(t, filepath.Join(t.TempDir(), "trusted.pem"), "corp.pem")
	if want := `brioche: NODE_EXTRA_CA_CERTS names "corp.pem", relative to the client's directory, which brioche does not know`; code != failed || !strings.Contains(says, want) {
		t.Errorf("brioche ended %d saying %q, want %d saying %q", code, says, failed, want)
	}
}

func TestAClientWhoseVariableNamesBriochesOwnFileByAnotherNameIsRefused(t *testing.T) {
	// A brioche before this one wrote the file, and the client was given it.
	file := filepath.Join(t.TempDir(), "trusted.pem")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(t.TempDir(), "linked.pem")
	if err := os.Symlink(file, link); err != nil {
		t.Fatal(err)
	}
	code, says := refused(t, file, link)
	if want := "NODE_EXTRA_CA_CERTS names " + link + ", the file the client is given in its place"; code != failed || !strings.Contains(says, want) {
		t.Errorf("brioche ended %d saying %q, want %d saying %q", code, says, failed, want)
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
	// A client that distrusts brioche has every request of its own fail: an error, never traffic.
	if told := brioche.said.of("handshake_refused"); len(told) != 1 || told[0].Level != failure || told[0].Target != "api.anthropic.com:443" {
		t.Errorf("brioche reported %+v, want one error naming the target it was refused for", told)
	}
}

func TestAClientThatHangsUpBeforeTLSIsSaidToHaveDoneSo(t *testing.T) {
	brioche := started(t, refusing(t), nil, unproxied)
	proxy, err := url.Parse(brioche.environment["HTTPS_PROXY"])
	if err != nil {
		t.Fatal(err)
	}
	connection, err := net.Dial("tcp", proxy.Host)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprint(connection, "CONNECT api.anthropic.com:443 HTTP/1.1\r\nHost: api.anthropic.com:443\r\n\r\n")
	if answer, err := http.ReadResponse(bufio.NewReader(connection), nil); err != nil || answer.StatusCode != http.StatusOK {
		t.Fatalf("the CONNECT was answered %v (%v), want 200", answer, err)
	}
	connection.Close()
	brioche.said.until(t, "brioche: the client's TLS for api.anthropic.com:443 ended before it began: ")
}

func TestABriocheWhoseDoorStopsServingEndsAsFailedSayingWhy(t *testing.T) {
	listener, err := listening()
	if err != nil {
		t.Fatal(err)
	}
	read, printed, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer read.Close()
	says := saying()
	exited := make(chan int, 1)
	args := arguments(refusing(t), filepath.Join(t.TempDir(), "trusted.pem"), nil)
	go func() {
		exited <- run(args, func(string) string { return "" }, unproxied, func() (net.Listener, error) { return listener, nil }, printed, says, stops())
	}()
	if environment, err := io.ReadAll(read); err != nil || len(environment) == 0 {
		t.Fatalf("brioche printed %q (%v), saying %q", environment, err, says)
	}
	listener.Close()
	if code := <-exited; code != failed || !strings.Contains(says.String(), "brioche: stopping: it stopped taking the client's requests: ") {
		t.Errorf("brioche ended %d saying %q, want %d saying its door stopped", code, says, failed)
	}
}

func TestToIsTakenForAnyNameOfLoopback(t *testing.T) {
	for _, named := range []string{"http://LocalHost:1", "http://127.0.0.2:1", "http://[::1]:1", "https://localhost"} {
		if _, err := loopback(named); err != nil {
			t.Errorf("--to %s was refused: %v", named, err)
		}
	}
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
		{arguments("http://127.0.0.1:1?", file, nil), `--to: "http://127.0.0.1:1?" names more than an origin`},
		{arguments("http://hands:secret@127.0.0.1:1", file, nil), `names more than an origin`},
		{arguments("https://hands.lan:8443", file, nil), `--to: "https://hands.lan:8443" is not on loopback`},
		{arguments("http://127.0.0.1:1", file, map[string]string{"--trust-variable": "HTTPS_PROXY"}), "--trust-variable:"},
		{arguments("http://127.0.0.1:1", file+"\nHTTPS_PROXY=elsewhere", nil), "--trust-file: a path printed on a line of its own holds no line break"},
	} {
		says := saying()
		read, printed, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		code := run(refused.args, func(string) string { return "" }, unproxied, listening, printed, says, stops())
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
// printed its environment: what it has said and will say.
type process struct {
	child *exec.Cmd
	said  *said
}

// processed starts brioche reading `input` as its standard input.
func processed(t *testing.T, input *os.File) process {
	t.Helper()
	child := exec.Command(os.Args[0], arguments(refusing(t), filepath.Join(t.TempDir(), "trusted.pem"), nil)...)
	child.Env = append(os.Environ(), program+"=brioche")
	child.Stdin = input
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
	return process{child: child, said: says}
}

// piped is a standard input for brioche, and the end the test writes or closes.
func piped(t *testing.T) (*os.File, *os.File) {
	t.Helper()
	read, write, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		read.Close()
		write.Close()
	})
	return read, write
}

func TestABriocheWhoseInputEndsStopsSayingSo(t *testing.T) {
	input, written := piped(t)
	brioche := processed(t, input)
	written.Close()
	if err := brioche.child.Wait(); err != nil {
		t.Fatalf("brioche ended %v once its input ended, saying %q", err, brioche.said)
	}
	if !strings.Contains(brioche.said.String(), "brioche: stopping: its input ended\n") {
		t.Errorf("brioche said %q, want why it stopped", brioche.said)
	}
}

func TestABriocheWhoseInputFailsEndsAsFailedSayingWhy(t *testing.T) {
	// A directory, which a read of fails.
	input, err := os.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { input.Close() })
	brioche := processed(t, input)
	var exit *exec.ExitError
	if err := brioche.child.Wait(); !errors.As(err, &exit) || exit.ExitCode() != failed {
		t.Fatalf("brioche ended %v once its input failed, want %d, saying %q", err, failed, brioche.said)
	}
	if !strings.Contains(brioche.said.String(), "brioche: stopping: its input failed: ") {
		t.Errorf("brioche said %q, want why it stopped", brioche.said)
	}
}

func TestOnlyTheFirstStopIsTakenAndOneAfterItIsLeftToBeSaid(t *testing.T) {
	ended := stops()
	if !ended.send(stop{"its input ended", nil}) {
		t.Fatal("the first stop was not taken")
	}
	// Taken while brioche is stopping, it would be lost unsaid: it is not taken.
	if ended.send(stop{"it stopped taking the client's requests", errors.New("closed")}) {
		t.Error("a stop after the first was taken")
	}
	if why := <-ended.why; why.why != "its input ended" {
		t.Errorf("brioche stopped for %q, want the first stop", why.why)
	}
	if ended.send(stop{"told to by terminated", nil}) {
		t.Error("a stop after the first was taken once the first was read")
	}
}

func TestABriocheReadsTheClientsTrustVariableFromItsOwnEnvironment(t *testing.T) {
	child := exec.Command(os.Args[0], arguments(refusing(t), filepath.Join(t.TempDir(), "trusted.pem"), nil)...)
	child.Env = append(os.Environ(), program+"=brioche", "NODE_EXTRA_CA_CERTS=corp.pem")
	says := saying()
	child.Stderr = says
	var exit *exec.ExitError
	if err := child.Run(); !errors.As(err, &exit) || exit.ExitCode() != failed {
		t.Fatalf("brioche ended %v, want %d, saying %q", err, failed, says)
	}
	if !strings.Contains(says.String(), `NODE_EXTRA_CA_CERTS names "corp.pem"`) {
		t.Errorf("brioche said %q, want the variable its environment gave it", says)
	}
}

func TestABriocheToldToStopStopsSayingSo(t *testing.T) {
	input, _ := piped(t)
	brioche := processed(t, input)
	if err := brioche.child.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	if err := brioche.child.Wait(); err != nil {
		t.Fatalf("brioche ended %v once told to stop, saying %q", err, brioche.said)
	}
	if !strings.Contains(brioche.said.String(), "brioche: stopping: told to by terminated\n") {
		t.Errorf("brioche said %q, want why it stopped", brioche.said)
	}
	// Every line its process wrote is a report: none is left for a reader to take as words.
	if told := brioche.said.of("unframed"); len(told) != 0 {
		t.Errorf("brioche wrote %+v, which are no reports", told)
	}
	if told := brioche.said.of("stopping"); len(told) != 1 || told[0].Level != info {
		t.Errorf("brioche reported %+v, want one stop it was asked for", told)
	}
}
