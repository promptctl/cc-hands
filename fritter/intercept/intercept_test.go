package intercept

import (
	"bufio"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

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

func unproxied(*http.Request) (*url.URL, error) { return nil, nil }

func origin(t *testing.T, raw string) Origin {
	t.Helper()
	parsed, err := ParseOrigin(raw)
	if err != nil {
		t.Fatal(err)
	}
	return parsed
}

func interception(t *testing.T, raw string) *Interception {
	t.Helper()
	made, err := Intercept("door", origin(t, raw), time.Now())
	if err != nil {
		t.Fatal(err)
	}
	return made
}

// opened is a door intercepting `raw`, with proxy as how it reaches what it tunnels to: its
// address, the file a client is given to trust, and each event it saw.
type opened struct {
	door    *Door
	address *url.URL
	trusted string
	seen    chan Event
}

func open(t *testing.T, raw string, proxy Proxy) opened {
	t.Helper()
	made := interception(t, raw)
	trusted := filepath.Join(t.TempDir(), "trusted.pem")
	if err := made.WriteTrusted(trusted, nil); err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	seen := make(chan Event, 16)
	door, err := Open(listener, Opening{
		Interception: made,
		Intercepted: http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			io.WriteString(writer, "intercepted "+request.Method+" "+request.URL.RequestURI())
		}),
		Forwarded: http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			io.WriteString(writer, "forwarded "+request.URL.String())
		}),
		Proxy: proxy,
		Seen:  func(event Event) { seen <- event },
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(door.Close)
	address, err := url.Parse(door.Address())
	if err != nil {
		t.Fatal(err)
	}
	return opened{door: door, address: address, trusted: trusted, seen: seen}
}

// client is an HTTP client with the door as its proxy, trusting the file it was given.
func (o opened) client(t *testing.T) *http.Client {
	t.Helper()
	held, err := os.ReadFile(o.trusted)
	if err != nil {
		t.Fatal(err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(held) {
		t.Fatalf("the trust file holds no certificate: %q", held)
	}
	return &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(o.address), TLSClientConfig: &tls.Config{RootCAs: roots}}}
}

// read is a response's status and body, for a test that fails where it cannot be read.
func read(t *testing.T) func(*http.Response, error) (int, string) {
	return func(response *http.Response, err error) (int, string) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		body, err := io.ReadAll(response.Body)
		if err != nil {
			t.Fatal(err)
		}
		return response.StatusCode, string(body)
	}
}

// connect sends a CONNECT for target to the door, and is what came back with the
// connection it was asked on.
func (o opened) connect(t *testing.T, target string) (*http.Response, net.Conn, *bufio.Reader) {
	t.Helper()
	connection, err := net.Dial("tcp", o.address.Host)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { connection.Close() })
	if _, err := connection.Write([]byte("CONNECT " + target + " HTTP/1.1\r\nHost: " + target + "\r\n\r\n")); err != nil {
		t.Fatal(err)
	}
	reader := bufio.NewReader(connection)
	response, err := http.ReadResponse(reader, &http.Request{Method: http.MethodConnect})
	if err != nil {
		t.Fatal(err)
	}
	return response, connection, reader
}

func TestAClientTrustingTheFileItIsGivenIsAnsweredForTheOriginByWhoeverTakesWhatWasIntercepted(t *testing.T) {
	door := open(t, "https://api.example.com", unproxied)
	response, err := door.client(t).Post("https://api.example.com/v1/messages?beta=true", "application/json", strings.NewReader("{}"))
	if err != nil {
		t.Fatalf("the origin was not reached through the door: %v", err)
	}
	if issuer := response.TLS.PeerCertificates[0].Issuer.CommonName; !strings.HasPrefix(issuer, "door ") {
		t.Errorf("api.example.com was answered by a certificate from %q, want the door's own authority", issuer)
	}
	if status, body := read(t)(response, err); status != http.StatusOK || body != "intercepted POST /v1/messages?beta=true" {
		t.Errorf("the door answered %d %q, want the request handed to whoever takes what was intercepted", status, body)
	}
	for _, want := range []Event{Opened{Target: "api.example.com:443"}, Answered{Route: Intercepted, URL: "https://api.example.com/v1/messages?beta=true", Status: http.StatusOK}} {
		if event := <-door.seen; event != want {
			t.Errorf("the door saw %+v, want %+v", event, want)
		}
	}
}

func TestAnOriginNamedByAddressIsAnsweredWithACertificateForThatAddress(t *testing.T) {
	door := open(t, "https://127.0.0.1:8443", unproxied)
	if status, body := read(t)(door.client(t).Get("https://127.0.0.1:8443/v1/models")); status != http.StatusOK || body != "intercepted GET /v1/models" {
		t.Errorf("the door answered %d %q, want the request intercepted", status, body)
	}
}

func TestTheOriginsHostIsInterceptedWhateverCaseItIsNamedIn(t *testing.T) {
	door := open(t, "https://API.Example.com", unproxied)
	response, connection, _ := door.connect(t, "api.EXAMPLE.com:443")
	if response.StatusCode != http.StatusOK {
		t.Fatalf("the CONNECT came back %s", response.Status)
	}
	// Nothing follows the CONNECT's answer until the client speaks: the reader holds nothing more.
	secured := tls.Client(connection, &tls.Config{ServerName: "api.example.com", InsecureSkipVerify: true})
	if err := secured.Handshake(); err != nil {
		t.Errorf("the connection was not answered over TLS: %v", err)
	}
}

func TestAPlainRequestForTheOriginIsInterceptedAndOneForAnyOtherIsForwarded(t *testing.T) {
	door := open(t, "http://127.0.0.1:8080", unproxied)
	client := &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(door.address)}}
	if status, body := read(t)(client.Get("http://127.0.0.1:8080/v1/messages")); status != http.StatusOK || body != "intercepted GET /v1/messages" {
		t.Errorf("a request for the origin came back %d %q, want it intercepted", status, body)
	}
	if status, body := read(t)(client.Get("http://elsewhere.example.com/page")); status != http.StatusOK || body != "forwarded http://elsewhere.example.com/page" {
		t.Errorf("a request for another origin came back %d %q, want it forwarded", status, body)
	}
	for _, want := range []Event{Answered{Route: Intercepted, URL: "http://127.0.0.1:8080/v1/messages", Status: http.StatusOK}, Answered{Route: Forwarded, URL: "http://elsewhere.example.com/page", Status: http.StatusOK}} {
		if event := <-door.seen; event != want {
			t.Errorf("the door saw %+v, want %+v", event, want)
		}
	}
}

func TestACertificateTheClientRefusesIsSeen(t *testing.T) {
	door := open(t, "https://api.example.com", unproxied)
	stranger := &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(door.address)}}
	if _, err := stranger.Get("https://api.example.com/v1/models"); err == nil {
		t.Fatal("a client that does not trust the door's authority took its certificate")
	}
	// crypto/tls's own client refuses a certificate from an authority it does not know so:
	// this holds the door's reading of a received alert to crypto/tls.
	if event := <-door.seen; event != (HandshakeRefused{Target: "api.example.com:443", Alert: tls.AlertError(42)}) {
		t.Errorf("the door saw %+v, want the client's refusal of its certificate", event)
	}
}

func TestAClientThatHangsUpBeforeTLSIsAHandshakeThatFailed(t *testing.T) {
	door := open(t, "https://api.example.com", unproxied)
	response, connection, _ := door.connect(t, "api.example.com:443")
	if response.StatusCode != http.StatusOK {
		t.Fatalf("the CONNECT came back %s", response.Status)
	}
	connection.Close()
	if event, ok := (<-door.seen).(HandshakeFailed); !ok || event.Target != "api.example.com:443" || !errors.Is(event.Err, io.EOF) {
		t.Errorf("the door saw %+v, want the handshake ended by the client's hanging up", event)
	}
}

func TestWhatTheClientTrustedAlreadyIsKeptBesideTheAuthorityEndedByALineOrNot(t *testing.T) {
	other := interception(t, "https://elsewhere.example.com")
	already := filepath.Join(t.TempDir(), "already.pem")
	if err := os.WriteFile(already, []byte(strings.TrimSuffix(string(other.pem), "\n")), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("EXTRA_CERTS", already)
	held, err := AlsoTrusted(trust(t, "EXTRA_CERTS"))
	if err != nil {
		t.Fatal(err)
	}
	trusted := filepath.Join(t.TempDir(), "trusted.pem")
	if err := interception(t, "https://api.example.com").WriteTrusted(trusted, held); err != nil {
		t.Fatal(err)
	}
	written, err := os.ReadFile(trusted)
	if err != nil {
		t.Fatal(err)
	}
	roots := 0
	for block, rest := pem.Decode(written); block != nil; block, rest = pem.Decode(rest) {
		roots++
	}
	if roots != 2 {
		t.Errorf("the trust file holds %d certificates a reader finds, want what was trusted already and the authority: %q", roots, written)
	}
}

func TestAVariableNamingNoFileIsNothingTrustedAndOneNamingAnUnreadableFileIsAnError(t *testing.T) {
	if held, err := AlsoTrusted(trust(t, "EXTRA_CERTS")); held != nil || err != nil {
		t.Errorf("an unset variable gave %q, %v, want nothing", held, err)
	}
	t.Setenv("EXTRA_CERTS", filepath.Join(t.TempDir(), "missing.pem"))
	if _, err := AlsoTrusted(trust(t, "EXTRA_CERTS")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a file the variable names that cannot be read gave %v, want the error reading it", err)
	}
}

// outer is a proxy that answers each CONNECT with `status`, and echoes what crosses one it
// answered 200, telling each CONNECT it was asked.
func outer(t *testing.T, status string) (*url.URL, <-chan *http.Request) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { listener.Close() })
	asked := make(chan *http.Request, 1)
	go func() {
		connection, err := listener.Accept()
		if err != nil {
			t.Errorf("the outer proxy took no connection: %v", err)
			return
		}
		defer connection.Close()
		reader := bufio.NewReader(connection)
		request, err := http.ReadRequest(reader)
		if err != nil {
			t.Errorf("the outer proxy was asked no request it can read: %v", err)
			return
		}
		asked <- request
		if _, err := connection.Write([]byte("HTTP/1.1 " + status + "\r\n\r\n")); err != nil {
			t.Errorf("the outer proxy could not answer: %v", err)
			return
		}
		io.Copy(connection, reader)
	}()
	return &url.URL{Scheme: "http", User: url.UserPassword("user", "secret"), Host: listener.Addr().String()}, asked
}

func TestAConnectToAnyOtherHostIsTunnelledThroughTheProxyItIsGiven(t *testing.T) {
	via, asked := outer(t, "200 Connection Established")
	door := open(t, "https://api.example.com", func(*http.Request) (*url.URL, error) { return via, nil })
	response, connection, reader := door.connect(t, "elsewhere.example.com:443")
	if response.StatusCode != http.StatusOK {
		t.Fatalf("the CONNECT came back %s", response.Status)
	}
	request := <-asked
	if request.Method != http.MethodConnect || request.Host != "elsewhere.example.com:443" || request.Header.Get("Proxy-Authorization") != "Basic dXNlcjpzZWNyZXQ=" {
		t.Errorf("the outer proxy was asked %s %q with %q, want a CONNECT for the host the client named, with the proxy's credential", request.Method, request.Host, request.Header)
	}
	if _, err := connection.Write([]byte("ping")); err != nil {
		t.Fatal(err)
	}
	echoed := make([]byte, 4)
	if _, err := io.ReadFull(reader, echoed); err != nil || string(echoed) != "ping" {
		t.Errorf("the tunnel carried back %q, %v, want what was sent through it", echoed, err)
	}
	connection.Close()
	if event := <-door.seen; event != (Tunnelled{Target: "elsewhere.example.com:443"}) {
		t.Errorf("the door saw %+v, want the tunnel ended as its ends ended it", event)
	}
}

func TestAnOuterProxyThatRefusesTheTunnelIsAnsweredAsABadGatewayNamingTheDoor(t *testing.T) {
	via, _ := outer(t, "407 Proxy Authentication Required")
	door := open(t, "https://api.example.com", func(*http.Request) (*url.URL, error) { return via, nil })
	response, _, _ := door.connect(t, "elsewhere.example.com:443")
	if status, body := read(t)(response, nil); status != http.StatusBadGateway || !strings.HasPrefix(body, "door could not reach elsewhere.example.com:443") || !strings.Contains(body, "407") {
		t.Errorf("the CONNECT came back %d %q, want a 502 naming the door and what the proxy answered", status, body)
	}
	if event, ok := (<-door.seen).(Unreached); !ok || event.Target != "elsewhere.example.com:443" || !strings.Contains(event.Err.Error(), "407") {
		t.Errorf("the door saw %+v, want the host unreached through the proxy", event)
	}
}

func TestAnOuterProxyThatIsNotHTTPIsAnsweredAsABadGateway(t *testing.T) {
	door := open(t, "https://api.example.com", func(*http.Request) (*url.URL, error) { return url.Parse("socks5://127.0.0.1:1080") })
	response, _, _ := door.connect(t, "elsewhere.example.com:443")
	if status, body := read(t)(response, nil); status != http.StatusBadGateway || !strings.Contains(body, "only through an http proxy") {
		t.Errorf("the CONNECT came back %d %q, want a 502 saying a tunnel goes only through an http proxy", status, body)
	}
}

func TestATunnelThatCannotTakeTheClientsConnectionEndsTheOneItDialled(t *testing.T) {
	far, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer far.Close()
	door := open(t, "https://api.example.com", unproxied)
	// A writer whose connection cannot be taken, as the client's cannot once it is gone.
	door.door.tunnel(httptest.NewRecorder(), far.Addr().String())
	dialled, err := far.Accept()
	if err != nil {
		t.Fatal(err)
	}
	defer dialled.Close()
	dialled.SetReadDeadline(time.Now().Add(5 * time.Second))
	if _, err := dialled.Read(make([]byte, 1)); err != io.EOF {
		t.Errorf("the far end read %v, want its connection ended", err)
	}
	if event, ok := (<-door.seen).(Tunnelled); !ok || event.Err == nil {
		t.Errorf("the door saw %+v, want a tunnel cut short", event)
	}
}

func TestAClientThatEndsWhatItSendsStillGetsAllTheFarEndAnswers(t *testing.T) {
	far, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer far.Close()
	go func() {
		dialled, err := far.Accept()
		if err != nil {
			return
		}
		defer dialled.Close()
		// The far end answers once the client has sent all it will.
		asked, _ := io.ReadAll(dialled)
		dialled.Write([]byte("answered " + string(asked)))
	}()
	door := open(t, "https://api.example.com", unproxied)
	response, connection, reader := door.connect(t, far.Addr().String())
	if response.StatusCode != http.StatusOK {
		t.Fatalf("the CONNECT came back %s", response.Status)
	}
	connection.Write([]byte("ping"))
	connection.(*net.TCPConn).CloseWrite()
	connection.SetReadDeadline(time.Now().Add(5 * time.Second))
	if answered, err := io.ReadAll(reader); err != nil || string(answered) != "answered ping" {
		t.Errorf("the client read %q, %v, want the far end's whole answer", answered, err)
	}
}

func TestClosingTheDoorEndsItsTunnels(t *testing.T) {
	far, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer far.Close()
	door := open(t, "https://api.example.com", unproxied)
	response, connection, reader := door.connect(t, far.Addr().String())
	if response.StatusCode != http.StatusOK {
		t.Fatalf("the CONNECT came back %s", response.Status)
	}
	dialled, err := far.Accept()
	if err != nil {
		t.Fatal(err)
	}
	defer dialled.Close()
	door.door.Close()
	connection.SetReadDeadline(time.Now().Add(5 * time.Second))
	if _, err := reader.ReadByte(); err != io.EOF {
		t.Errorf("the client's tunnel read %v once the door closed, want it ended", err)
	}
}

func TestAConnectThatNamesNoPortAndARequestThatIsNoProxyRequestAreRefusedInTheDoorsName(t *testing.T) {
	door := open(t, "https://api.example.com", unproxied)
	for _, target := range []string{"api.example.com", "api.example.com:"} {
		response, _, _ := door.connect(t, target)
		if status, body := read(t)(response, nil); status != http.StatusBadRequest || !strings.Contains(body, "a CONNECT names a host and port") {
			t.Errorf("a CONNECT for %q came back %d %q, want it refused", target, status, body)
		}
	}
	if status, body := read(t)(http.Get(door.address.String() + "/v1/messages")); status != http.StatusBadRequest || !strings.HasPrefix(body, "door is a proxy") {
		t.Errorf("a request for a path alone came back %d %q, want a refusal naming the door", status, body)
	}
	for _, asked := range []string{"api.example.com", "api.example.com:", "/v1/messages"} {
		if event, ok := (<-door.seen).(Refused); !ok || event.Asked != asked || event.Err == nil {
			t.Errorf("the door saw %+v, want %q refused", event, asked)
		}
	}
}

func TestADoorMissingAPartIsRefusedBeforeItServes(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	served := http.NotFoundHandler()
	for name, opening := range map[string]Opening{
		"nobody to take what it intercepts": {Interception: interception(t, "https://api.example.com"), Forwarded: served, Proxy: unproxied, Seen: func(Event) {}},
		"nothing to intercept":              {Intercepted: served, Forwarded: served, Proxy: unproxied, Seen: func(Event) {}},
	} {
		if _, err := Open(listener, opening); err == nil {
			t.Errorf("a door with %s was opened", name)
		}
	}
	if _, err := Intercept("door", Origin{}, time.Now()); err == nil {
		t.Error("an origin ParseOrigin never made was intercepted")
	}
}

func TestEachLineTheDoorsServersLogIsSeenAsALineOfItsOwn(t *testing.T) {
	var seen []string
	lines(func(line string) { seen = append(seen, line) }).Write([]byte("http: panic serving 1.2.3.4: boom\ngoroutine 1 [running]:\n"))
	if len(seen) != 2 || seen[0] != "http: panic serving 1.2.3.4: boom" || seen[1] != "goroutine 1 [running]:" {
		t.Errorf("the log was seen as %q, want each of its lines", seen)
	}
}

func TestATrustVariableIsAVariablesNameAndNoneThatPointsTheClientAtItsProxy(t *testing.T) {
	for _, name := range []string{"", "A=B", "1ST", "HTTPS_PROXY", "no_proxy"} {
		if _, err := ParseTrustVariable(name); err == nil {
			t.Errorf("%q was taken for a trust variable", name)
		}
	}
}

func trust(t *testing.T, name string) TrustVariable {
	t.Helper()
	variable, err := ParseTrustVariable(name)
	if err != nil {
		t.Fatal(err)
	}
	return variable
}

func TestAnOriginIsAnHTTPOrHTTPSURLWithAHost(t *testing.T) {
	for _, raw := range []string{"api.example.com", "ftp://api.example.com", "https://:8443", "://"} {
		if _, err := ParseOrigin(raw); err == nil {
			t.Errorf("%q was taken for an origin", raw)
		}
	}
	if parsed := origin(t, "https://api.example.com/v1?x=1"); parsed.String() != "https://api.example.com" {
		t.Errorf("an origin with a path is %q, want its scheme and host alone", parsed)
	}
}

func TestTheAuthorityIsValidFromAnHourBeforeItWasMadeForAYear(t *testing.T) {
	made := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	minted, err := Intercept("door", origin(t, "https://api.example.com"), made)
	if err != nil {
		t.Fatal(err)
	}
	leaf, err := x509.ParseCertificate(minted.leaf.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	if !leaf.NotBefore.Equal(made.Add(-time.Hour)) || !leaf.NotAfter.Equal(made.AddDate(1, 0, 0)) {
		t.Errorf("the certificate is valid from %s until %s", leaf.NotBefore, leaf.NotAfter)
	}
}

func TestTheEnvironmentPointsAClientAtTheDoorForEveryHostTrustingItsFile(t *testing.T) {
	door := open(t, "https://api.example.com", unproxied)
	at := door.door.Address()
	want := []Variable{{"EXTRA_CERTS", "/tmp/trusted.pem"}, {"HTTPS_PROXY", at}, {"https_proxy", at}, {"HTTP_PROXY", at}, {"http_proxy", at}, {"NO_PROXY", ""}, {"no_proxy", ""}}
	got := door.door.Environment(trust(t, "EXTRA_CERTS"), "/tmp/trusted.pem")
	if len(got) != len(want) {
		t.Fatalf("the environment is %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("the environment is %v, want %v", got, want)
		}
	}
}
