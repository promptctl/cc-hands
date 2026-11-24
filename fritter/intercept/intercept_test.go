package intercept

import (
	"bufio"
	"crypto/tls"
	"crypto/x509"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// door is a Door answering for https://api.example.com itself, its trust file written to
// a test directory, with `proxied` as how it reaches what it joins.
func door(t *testing.T, proxied Proxied) (*url.URL, string) {
	t.Helper()
	authority, err := NewAuthority("door 1", "api.example.com")
	if err != nil {
		t.Fatal(err)
	}
	trusted := filepath.Join(t.TempDir(), "trusted.pem")
	if err := WriteTrusted(trusted, nil, authority); err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	opened := http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		io.WriteString(writer, "opened "+request.Method+" "+request.URL.RequestURI())
	})
	passing := http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		io.WriteString(writer, "passing "+request.URL.String())
	})
	tapped, _ := url.Parse("https://api.example.com")
	served := Open(listener, Opening{Name: "door", Tapped: tapped, Leaf: authority.Leaf, Opened: opened, Passing: passing, Proxied: proxied, ErrorLog: log.New(io.Discard, "", 0)})
	t.Cleanup(served.Close)
	return &url.URL{Scheme: "http", Host: listener.Addr().String()}, trusted
}

func unproxied(*http.Request) (*url.URL, error) { return nil, nil }

func TestAClientTrustingTheFileItIsGivenIsAnsweredForTheTappedHostByWhoeverTakesWhatItOpened(t *testing.T) {
	address, trusted := door(t, unproxied)
	held, err := os.ReadFile(trusted)
	if err != nil {
		t.Fatal(err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(held) {
		t.Fatalf("the trust file holds no certificate: %q", held)
	}
	client := &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(address), TLSClientConfig: &tls.Config{RootCAs: roots}}}
	response, err := client.Post("https://api.example.com/v1/messages?beta=true", "application/json", strings.NewReader("{}"))
	if err != nil {
		t.Fatalf("the tapped host was not reached through the door: %v", err)
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(response.Body)
	if string(body) != "opened POST /v1/messages?beta=true" {
		t.Errorf("the door answered %q, want the request handed to whoever takes what it opened", body)
	}
	if issuer := response.TLS.PeerCertificates[0].Issuer.CommonName; issuer != "door 1" {
		t.Errorf("api.example.com was answered by a certificate from %q, want the door's own authority", issuer)
	}
}

func TestWhatTheClientTrustedAlreadyIsKeptBesideTheAuthority(t *testing.T) {
	already := filepath.Join(t.TempDir(), "already.pem")
	other, err := NewAuthority("other", "elsewhere.example.com")
	if err != nil {
		t.Fatal(err)
	}
	os.WriteFile(already, other.PEM, 0o600)
	t.Setenv("EXTRA_CERTS", already)
	held, err := AlsoTrusted("EXTRA_CERTS")
	if err != nil {
		t.Fatal(err)
	}
	authority, _ := NewAuthority("door", "api.example.com")
	trusted := filepath.Join(t.TempDir(), "trusted.pem")
	if err := WriteTrusted(trusted, held, authority); err != nil {
		t.Fatal(err)
	}
	written, _ := os.ReadFile(trusted)
	if got := strings.Count(string(written), "BEGIN CERTIFICATE"); got != 2 {
		t.Errorf("the trust file holds %d certificates, want what was trusted already and the authority", got)
	}
	t.Setenv("EXTRA_CERTS", filepath.Join(t.TempDir(), "missing.pem"))
	if _, err := AlsoTrusted("EXTRA_CERTS"); err == nil {
		t.Error("a file the variable names that cannot be read was taken for none")
	}
}

// outer is a proxy that answers each CONNECT and echoes what crosses it, telling the target
// each CONNECT named.
func outer(t *testing.T) (*url.URL, <-chan string) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { listener.Close() })
	named := make(chan string, 1)
	go func() {
		connection, err := listener.Accept()
		if err != nil {
			return
		}
		defer connection.Close()
		reader := bufio.NewReader(connection)
		request, err := http.ReadRequest(reader)
		if err != nil {
			return
		}
		named <- request.Host
		connection.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
		io.Copy(connection, reader)
	}()
	return &url.URL{Scheme: "http", Host: listener.Addr().String()}, named
}

func TestAConnectToAnyOtherHostIsJoinedThroughTheProxyItIsGiven(t *testing.T) {
	via, named := outer(t)
	address, _ := door(t, func(*http.Request) (*url.URL, error) { return via, nil })
	connection, err := net.Dial("tcp", address.Host)
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	connection.Write([]byte("CONNECT elsewhere.example.com:443 HTTP/1.1\r\nHost: elsewhere.example.com:443\r\n\r\n"))
	reader := bufio.NewReader(connection)
	if response, err := http.ReadResponse(reader, &http.Request{Method: http.MethodConnect}); err != nil || response.StatusCode != http.StatusOK {
		t.Fatalf("the CONNECT came back %v, %v", response, err)
	}
	if target := <-named; target != "elsewhere.example.com:443" {
		t.Errorf("the outer proxy was asked for %q, want the host the client named", target)
	}
	connection.Write([]byte("ping"))
	echoed := make([]byte, 4)
	if _, err := io.ReadFull(reader, echoed); err != nil || string(echoed) != "ping" {
		t.Errorf("the tunnel carried back %q, %v, want what was sent through it", echoed, err)
	}
}

func TestARequestThatIsNoProxyRequestIsRefusedInTheDoorsName(t *testing.T) {
	address, _ := door(t, unproxied)
	response, err := http.Get(address.String() + "/v1/messages")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(response.Body)
	if response.StatusCode != http.StatusBadRequest || !strings.HasPrefix(string(body), "door is a proxy") {
		t.Errorf("a request for a path alone came back %d %q, want a refusal naming the door", response.StatusCode, body)
	}
}
