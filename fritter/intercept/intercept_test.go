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
	"sync"
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

// environment is a client's environment holding `variables` and nothing else, as Trust
// reads it.
func environment(variables map[string]string) func(string) string {
	return func(name string) string { return variables[name] }
}

// trusted is `raw` intercepted, trusted by a client of EXTRA_CERTS whose environment names
// nothing more to trust.
func trusted(t *testing.T, raw string) Trusted {
	t.Helper()
	made, err := interception(t, raw).Trust(trust(t, "EXTRA_CERTS"), environment(nil), filepath.Join(t.TempDir(), "trusted.pem"), func(err error) { t.Error(err) })
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
	trusted Trusted
	seen    chan Event
}

func open(t *testing.T, raw string, proxy Proxy) opened {
	t.Helper()
	given := trusted(t, raw)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	seen := make(chan Event, 16)
	door, err := Open(listener, Opening{
		Trusted: given,
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
	return opened{door: door, address: address, trusted: given, seen: seen}
}

// client is an HTTP client with the door as its proxy, trusting the file it was given.
func (o opened) client(t *testing.T) *http.Client {
	t.Helper()
	held, err := os.ReadFile(o.trusted.Path())
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
	if roots := trusting(t, environment(map[string]string{"EXTRA_CERTS": already}), func(err error) { t.Error(err) }); roots != 2 {
		t.Errorf("the trust file holds %d certificates a reader finds, want what was trusted already and the authority", roots)
	}
}

func TestAVariableNamingNoFileIsNothingTrustedAndOneNamingAnUnreadableFileIsSaidAndLeftOut(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing.pem")
	// The door's own environment is not the client's: what it names is not read.
	t.Setenv("EXTRA_CERTS", missing)
	if roots := trusting(t, environment(nil), func(err error) { t.Error(err) }); roots != 1 {
		t.Errorf("an unset variable left %d certificates to trust, want the authority alone", roots)
	}
	var unread error
	if roots := trusting(t, environment(map[string]string{"EXTRA_CERTS": missing}), func(err error) { unread = err }); roots != 1 || !errors.Is(unread, os.ErrNotExist) {
		t.Errorf("a file the variable names that cannot be read left %d certificates and said %v, want the authority alone and the error reading it", roots, unread)
	}
}

// trusting is how many certificates a reader finds in the file the door's interception
// writes for a client of EXTRA_CERTS, with the environment `client`, to trust.
func trusting(t *testing.T, client func(string) string, unread func(error)) int {
	t.Helper()
	trusted, err := interception(t, "https://api.example.com").Trust(trust(t, "EXTRA_CERTS"), client, filepath.Join(t.TempDir(), "trusted.pem"), unread)
	if err != nil {
		t.Fatal(err)
	}
	written, err := os.ReadFile(trusted.Path())
	if err != nil {
		t.Fatal(err)
	}
	roots := 0
	for block, rest := pem.Decode(written); block != nil; block, rest = pem.Decode(rest) {
		roots++
	}
	return roots
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

func TestAnOuterProxyThatIsNeitherHTTPNorHTTPSIsAnsweredAsABadGateway(t *testing.T) {
	door := open(t, "https://api.example.com", func(*http.Request) (*url.URL, error) { return url.Parse("socks5://127.0.0.1:1080") })
	response, _, _ := door.connect(t, "elsewhere.example.com:443")
	if status, body := read(t)(response, nil); status != http.StatusBadGateway || !strings.Contains(body, "only through an http or https proxy") {
		t.Errorf("the CONNECT came back %d %q, want a 502 saying a tunnel goes only through an http or https proxy", status, body)
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
	// The far end says nothing, so only the door's closing ends its side.
	dialled.SetReadDeadline(time.Now().Add(5 * time.Second))
	if _, err := dialled.Read(make([]byte, 1)); err != io.EOF {
		t.Errorf("the far end of the tunnel read %v once the door closed, want it ended", err)
	}
	select {
	case event := <-door.seen:
		if ended, ok := event.(Tunnelled); !ok || ended.Target != far.Addr().String() {
			t.Errorf("the door saw %+v, want the tunnel said to have ended", event)
		}
	case <-time.After(5 * time.Second):
		t.Error("the tunnel was never said to have ended once the door closed")
	}
}

func TestATunnelThroughAnHTTPSProxySpeaksTLSToIt(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	first := make(chan byte, 1)
	go func() {
		asked, err := listener.Accept()
		if err != nil {
			return
		}
		defer asked.Close()
		read := make([]byte, 1)
		asked.Read(read)
		first <- read[0]
	}()
	proxy := &url.URL{Scheme: "https", Host: listener.Addr().String()}
	door := open(t, "https://api.example.com", func(*http.Request) (*url.URL, error) { return proxy, nil })
	door.connect(t, "elsewhere.example.com:443")
	// 22 opens every TLS record of a handshake.
	if opened := <-first; opened != 22 {
		t.Errorf("the door opened its tunnel through the https proxy with byte %d, want a TLS handshake's", opened)
	}
}

// failing is a listener whose every Accept fails, as one whose socket is lost.
type failing struct{ net.Listener }

func (failing) Accept() (net.Conn, error) { return nil, errors.New("the socket is lost") }

func TestADoorWhoseListenerFailsSaysItStopped(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	seen := make(chan Event, 4)
	served := http.NotFoundHandler()
	door, err := Open(failing{listener}, Opening{Trusted: trusted(t, "https://api.example.com"), Intercepted: served, Forwarded: served, Proxy: unproxied, Seen: func(event Event) { seen <- event }})
	if err != nil {
		t.Fatal(err)
	}
	defer door.Close()
	if event, ok := (<-seen).(Stopped); !ok || event.Err == nil || event.Err.Error() != "the socket is lost" {
		t.Errorf("the door saw %+v, want it said to have stopped with its listener's failure", event)
	}
}

func TestAnAnswerItsTakerGaveUpPartWayIsSaidToHaveBeenCut(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	seen := make(chan Event, 4)
	door, err := Open(listener, Opening{
		Trusted:     trusted(t, "https://api.example.com"),
		Intercepted: http.NotFoundHandler(),
		Forwarded: http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.WriteHeader(http.StatusOK)
			// As a reverse proxy gives up an answer whose body ended short.
			panic(http.ErrAbortHandler)
		}),
		Proxy: unproxied,
		Seen:  func(event Event) { seen <- event },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer door.Close()
	address, err := url.Parse(door.Address())
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(address)}}
	if response, err := client.Get("http://elsewhere.example.com/page"); err == nil {
		io.ReadAll(response.Body)
		response.Body.Close()
	}
	if event := <-seen; event != (Cut{Route: Forwarded, URL: "http://elsewhere.example.com/page", Status: http.StatusOK}) {
		t.Errorf("the door saw %+v, want the answer said to have been cut", event)
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
		"nobody to take what it intercepts": {Trusted: trusted(t, "https://api.example.com"), Forwarded: served, Proxy: unproxied, Seen: func(Event) {}},
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
	want := []Variable{{"EXTRA_CERTS", door.trusted.Path()}, {"HTTPS_PROXY", at}, {"https_proxy", at}, {"HTTP_PROXY", at}, {"http_proxy", at}, {"NO_PROXY", ""}, {"no_proxy", ""}}
	got := door.door.Environment()
	if len(got) != len(want) {
		t.Fatalf("the environment is %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("the environment is %v, want %v", got, want)
		}
	}
}

func TestClosingTheDoorReturnsOnceTheRequestItWasServingIsSeenAnswered(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	entered, cancelled, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
	seen := make(chan Event, 4)
	door, err := Open(listener, Opening{
		Trusted:     trusted(t, "https://api.example.com"),
		Intercepted: http.NotFoundHandler(),
		// A handler that ends only when let: it sees its request ended by the closing, then waits.
		Forwarded: http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			close(entered)
			<-request.Context().Done()
			close(cancelled)
			<-release
		}),
		Proxy: unproxied,
		Seen:  func(event Event) { seen <- event },
	})
	if err != nil {
		t.Fatal(err)
	}
	address, err := url.Parse(door.Address())
	if err != nil {
		t.Fatal(err)
	}
	go (&http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(address)}}).Get("http://elsewhere.example.com/page")
	<-entered
	closed := make(chan struct{})
	go func() {
		door.Close()
		close(closed)
	}()
	<-cancelled
	select {
	case <-closed:
		t.Fatal("the door closed while the request it was serving was still being answered")
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	<-closed
	select {
	case event := <-seen:
		if answered, ok := event.(Answered); !ok || answered.URL != "http://elsewhere.example.com/page" {
			t.Errorf("the door saw %+v, want the request it was serving answered", event)
		}
	default:
		t.Error("the door closed before it said how the request it was serving was answered")
	}
}

func TestAnAnswerWhoseConnectionCouldNotBeTakenIsTheStatusItWentWith(t *testing.T) {
	// A recorder cannot give up its connection, as no writer of HTTP/2 can.
	answering := &answering{ResponseWriter: httptest.NewRecorder()}
	if _, _, err := answering.Hijack(); err == nil {
		t.Fatal("a recorder's connection was taken")
	}
	http.Error(answering, "cannot take the connection", http.StatusInternalServerError)
	if status := answering.answered(); status != http.StatusInternalServerError {
		t.Errorf("an answer whose connection was not taken went with %d, want the status it was answered with", status)
	}
}

// closes is whether Close returns within five seconds: a door's Close returns once what it
// ends has ended, and a dial's own timeout is ten.
func closes(door *Door) bool {
	closed := make(chan struct{})
	go func() {
		door.Close()
		close(closed)
	}()
	select {
	case <-closed:
		return true
	case <-time.After(5 * time.Second):
		return false
	}
}

func TestClosingTheDoorEndsAConnectionAHandlerTookAndIsWaitingOn(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	taken := make(chan struct{})
	door, err := Open(listener, Opening{
		Trusted:     trusted(t, "https://api.example.com"),
		Intercepted: http.NotFoundHandler(),
		// As a ReverseProxy joins an upgraded connection to the upstream's: until it ends.
		Forwarded: http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			connection, _, err := http.NewResponseController(writer).Hijack()
			if err != nil {
				t.Error(err)
				return
			}
			close(taken)
			io.Copy(io.Discard, connection)
			connection.Close()
		}),
		Proxy: unproxied,
		Seen:  func(Event) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	client, err := net.Dial("tcp", listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if _, err := client.Write([]byte("GET http://elsewhere.example.com/socket HTTP/1.1\r\nHost: elsewhere.example.com\r\n\r\n")); err != nil {
		t.Fatal(err)
	}
	<-taken
	if !closes(door) {
		t.Fatal("the door did not close while a handler waited on the connection it took")
	}
}

func TestClosingTheDoorEndsATunnelStillWaitingOnTheOuterProxyAndSaysTheDoorClosed(t *testing.T) {
	// An http proxy that never answers the CONNECT, and an https one that never answers the
	// handshake before it.
	for _, scheme := range []string{"http", "https"} {
		t.Run(scheme, func(t *testing.T) {
			silent, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			defer silent.Close()
			asked := make(chan net.Conn, 1)
			go func() {
				if connection, err := silent.Accept(); err == nil {
					asked <- connection
				}
			}()
			outer := &url.URL{Scheme: scheme, Host: silent.Addr().String()}
			door := open(t, "https://api.example.com", func(*http.Request) (*url.URL, error) { return outer, nil })
			client, err := net.Dial("tcp", door.address.Host)
			if err != nil {
				t.Fatal(err)
			}
			defer client.Close()
			if _, err := client.Write([]byte("CONNECT elsewhere.example.com:443 HTTP/1.1\r\nHost: elsewhere.example.com:443\r\n\r\n")); err != nil {
				t.Fatal(err)
			}
			defer (<-asked).Close()
			if !closes(door.door) {
				t.Fatal("the door did not close while a tunnel waited on the outer proxy")
			}
			if event, ok := (<-door.seen).(Unreached); !ok || !errors.Is(event.Err, net.ErrClosed) {
				t.Errorf("the door saw %+v, want the tunnel unreached because the door closed", event)
			}
		})
	}
}

func TestAHandlerThatTookItsConnectionAndPanickedHasItEnded(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	door, err := Open(listener, Opening{
		Trusted:     trusted(t, "https://api.example.com"),
		Intercepted: http.NotFoundHandler(),
		Forwarded: http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			http.NewResponseController(writer).Hijack()
			panic("boom")
		}),
		Proxy: unproxied,
		Seen:  func(Event) {},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer door.Close()
	client, err := net.Dial("tcp", listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if _, err := client.Write([]byte("GET http://elsewhere.example.com/socket HTTP/1.1\r\nHost: elsewhere.example.com\r\n\r\n")); err != nil {
		t.Fatal(err)
	}
	client.SetReadDeadline(time.Now().Add(5 * time.Second))
	if _, err := client.Read(make([]byte, 1)); !errors.Is(err, io.EOF) {
		t.Errorf("the client read %v, want its connection ended", err)
	}
}

func TestAHandlersPanicIsSeenLoggedBeforeTheDoorCloses(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	logging, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	door, err := Open(listener, Opening{
		Trusted:     trusted(t, "https://api.example.com"),
		Intercepted: http.NotFoundHandler(),
		Forwarded:   http.HandlerFunc(func(http.ResponseWriter, *http.Request) { panic("boom") }),
		Proxy:       unproxied,
		// The first line of the panic's log is held until let go.
		Seen: func(event Event) {
			if logged, ok := event.(Logged); ok && strings.HasPrefix(logged.Line, "http: panic serving") && strings.HasSuffix(logged.Line, ": boom") {
				once.Do(func() {
					close(logging)
					<-release
				})
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	address, err := url.Parse(door.Address())
	if err != nil {
		t.Fatal(err)
	}
	go (&http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(address)}}).Get("http://elsewhere.example.com/page")
	<-logging
	closed := make(chan struct{})
	go func() {
		door.Close()
		close(closed)
	}()
	select {
	case <-closed:
		t.Error("the door closed while the panic it met was still being said")
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	<-closed
}
