package main

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/bmf/cc-hands/fritter/testrun"
)

func TestMain(m *testing.M) {
	os.Exit(testrun.Main(m))
}

// Each test package reads the hold again, as testrun.Hold says why.
func TestAnEditToTheHoldIsNeverACachedPass(t *testing.T) {
	if _, err := testrun.Hold(); err != nil {
		t.Error(err)
	}
}

// said is what brioche says, read while it may still be saying more.
type said struct {
	mu    sync.Mutex
	more  *sync.Cond // broadcast at each write
	lines bytes.Buffer
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

// until returns once brioche has said `line`, however long that takes: what the test waits
// for is the saying itself, never a time it is hoped to come by.
func (s *said) until(line string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for !strings.Contains(s.lines.String(), line) {
		s.more.Wait()
	}
}

func (s *said) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lines.String()
}

// serving is a brioche started for a test: the environment it printed, and what it says.
type serving struct {
	environment map[string]string
	said        *said
}

// started runs brioche intercepting https://api.anthropic.com for `to`, in a client
// environment of `client`, and stops it as the test ends, which it must stop for, saying
// nothing it was not asked to.
func started(t *testing.T, to string, client map[string]string) serving {
	t.Helper()
	read, printed, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer read.Close()
	says := saying()
	ended := make(chan struct{})
	exited := make(chan int, 1)
	args := []string{"--intercept", "https://api.anthropic.com", "--to", to, "--trust-variable", "NODE_EXTRA_CA_CERTS", "--trust-file", filepath.Join(t.TempDir(), "trusted.pem")}
	go func() { exited <- run(args, func(name string) string { return client[name] }, printed, says, ended) }()
	environment := map[string]string{}
	lines := bufio.NewScanner(read)
	for lines.Scan() {
		name, value, _ := strings.Cut(lines.Text(), "=")
		environment[name] = value
	}
	if len(environment) == 0 {
		t.Fatalf("brioche printed no environment, ending %d and saying %q", <-exited, says)
	}
	t.Cleanup(func() {
		close(ended)
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

func TestAClientTrustingBriochesFileReachesTheOriginWithTheRequestAsSentAndGetsItsAnswerAsItComes(t *testing.T) {
	type asked struct{ method, uri, host, header, body string }
	got := make(chan asked, 1)
	// The origin holds its answer's second part until the client has read the first: the
	// client reads it only if brioche passed it on as it came.
	second := make(chan struct{})
	var letGo sync.Once
	release := func() { letGo.Do(func() { close(second) }) }
	t.Cleanup(release)
	origin := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(request.Body)
		got <- asked{request.Method, request.RequestURI, request.Host, request.Header.Get("X-Asked"), string(body)}
		writer.WriteHeader(http.StatusOK)
		io.WriteString(writer, "first,")
		writer.(http.Flusher).Flush()
		<-second
		io.WriteString(writer, "second")
	}))
	t.Cleanup(origin.Close)
	brioche := started(t, origin.URL, nil)
	request, err := http.NewRequest(http.MethodPost, "https://api.anthropic.com/v1/messages?beta=true", strings.NewReader(`{"asked":true}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("X-Asked", "as sent")
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
	want := asked{http.MethodPost, "/v1/messages?beta=true", "api.anthropic.com", "as sent", `{"asked":true}`}
	if reached := <-got; reached != want {
		t.Errorf("the origin was asked %+v, want %+v", reached, want)
	}
}

func TestAConnectToAnyOtherHostIsTunnelled(t *testing.T) {
	elsewhere := httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		io.WriteString(writer, "elsewhere")
	}))
	t.Cleanup(elsewhere.Close)
	brioche := started(t, "http://127.0.0.1:1", nil)
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

func TestAnOriginThatIsDownIsAnsweredAsA502NamingBrioche(t *testing.T) {
	// A port nothing listens on any more: the origin refuses.
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	down := "http://" + listener.Addr().String()
	listener.Close()
	brioche := started(t, down, nil)
	response, err := trusting(t, brioche, trusted(t, brioche)).Get("https://api.anthropic.com/v1/messages")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(response.Body)
	if response.StatusCode != http.StatusBadGateway || !strings.HasPrefix(string(body), "brioche could not reach "+down+"/v1/messages") {
		t.Errorf("the client was answered %d %q, want a 502 naming brioche and where it could not reach", response.StatusCode, body)
	}
}

func TestTheClientIsGivenTheCertificatesItsVariableNamedBesideBriochesAuthority(t *testing.T) {
	theirs := httptest.NewTLSServer(http.NotFoundHandler())
	t.Cleanup(theirs.Close)
	own := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: theirs.Certificate().Raw})
	named := filepath.Join(t.TempDir(), "theirs.pem")
	if err := os.WriteFile(named, own, 0o600); err != nil {
		t.Fatal(err)
	}
	brioche := started(t, "http://127.0.0.1:1", map[string]string{"NODE_EXTRA_CA_CERTS": named})
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

func TestAClientThatRefusesBriochesCertificateIsSaidToHaveRefusedIt(t *testing.T) {
	brioche := started(t, "http://127.0.0.1:1", nil)
	// Trusting nothing of brioche's: the client refuses the TLS brioche answers it with, as Go's
	// client does a certificate of an authority it does not know.
	if _, err := trusting(t, brioche, x509.NewCertPool()).Get("https://api.anthropic.com/v1/messages"); err == nil {
		t.Fatal("a client that trusts nothing of brioche's was answered")
	}
	// Said once the door reads the client's alert: a test that fails here hangs until go
	// test's own timeout, which names it.
	brioche.said.until("brioche: the client refused brioche's answer for api.anthropic.com:443: tls: bad certificate\n")
}

func TestArgumentsBriocheCannotTakeAreRefusedSayingWhich(t *testing.T) {
	whole := []string{"--intercept", "https://api.anthropic.com", "--to", "http://127.0.0.1:1", "--trust-variable", "NODE_EXTRA_CA_CERTS", "--trust-file", "trusted.pem"}
	for _, refused := range []struct {
		args []string
		says string
	}{
		{whole[2:], "--intercept is needed"},
		{append(whole, "--to", "http://127.0.0.1:2"), "--to is given twice"},
		{append(whole[:7:7], ""), "--trust-file needs a value"},
		{append(whole, "--tap"), `unknown argument "--tap"`},
		{[]string{"--intercept", "api.anthropic.com", "--to", "http://127.0.0.1:1", "--trust-variable", "NODE_EXTRA_CA_CERTS", "--trust-file", "trusted.pem"}, "--intercept: an origin is an http or https URL"},
		{[]string{"--intercept", "https://api.anthropic.com", "--to", "http://127.0.0.1:1", "--trust-variable", "HTTPS_PROXY", "--trust-file", "trusted.pem"}, "--trust-variable:"},
	} {
		says := saying()
		_, printed, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		code := run(refused.args, func(string) string { return "" }, printed, says, nil)
		printed.Close()
		if code != misuse || !strings.Contains(says.String(), refused.says) {
			t.Errorf("brioche %q ended %d saying %q, want %d saying %q", refused.args, code, says, misuse, refused.says)
		}
	}
}
