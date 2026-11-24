// Package intercept is how a proxy of hands' answers a client's connections to one origin
// itself, over TLS under a certificate authority of its own, and tunnels every other
// connection to wherever it was going: what fritter's tap and brioche share. A change here
// is a change to fritter, which every session runs under, so it holds only what the two
// must change at the same moment: how a client is pointed at the proxy and given a
// certificate to trust, and how a connection is told apart, intercepted or tunnelled.
// What either does with a request it intercepted is its own.
package intercept

import (
	"bufio"
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"log"
	"math/big"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// Origin is the one origin a door intercepts: an http or https scheme and a host, nothing
// more. [LAW:parse-dont-validate] ParseOrigin is the only way to make one.
type Origin struct {
	url url.URL
	// The origin's host and port as a CONNECT names it, lowered: a host is named in any
	// case, and a client may send it lowered.
	hostPort string
}

// ParseOrigin reads raw as the origin it names: a path on it is the client's to send, as it
// sends every other part of a request.
func ParseOrigin(raw string) (Origin, error) {
	parsed, err := url.Parse(raw)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" {
		return Origin{}, fmt.Errorf("an origin is an http or https URL with a host, and %q is not", raw)
	}
	origin := url.URL{Scheme: parsed.Scheme, Host: parsed.Host}
	return Origin{url: origin, hostPort: strings.ToLower(hostPort(&origin))}, nil
}

// URL is the origin as a URL of its own, for the caller to forward to.
func (o Origin) URL() *url.URL {
	reached := o.url
	return &reached
}

func (o Origin) String() string {
	return o.url.String()
}

// is whether a request in scheme for hostPort is one for the origin. [LAW:single-enforcer]
// the one test of it, for a CONNECT and a plain request alike.
func (o Origin) is(scheme, hostPort string) bool {
	return scheme == o.url.Scheme && strings.ToLower(hostPort) == o.hostPort
}

// hostPort is the host and port a URL reaches, as a CONNECT names it.
func hostPort(target *url.URL) string {
	if target.Port() != "" {
		return target.Host
	}
	port := map[string]string{"http": "80", "https": "443"}[target.Scheme]
	return net.JoinHostPort(target.Hostname(), port)
}

// Interception is an origin, and the certificate authority a door answers it with. The
// authority's key is never written anywhere: it lives and dies with the process.
type Interception struct {
	// Who intercepts, which the client reads in every error it is answered with.
	name   string
	origin Origin
	pem    []byte // the authority's certificate, which the client is given to trust
	leaf   tls.Certificate
}

// Intercept makes a certificate authority for `name`, valid from an hour before now for a
// year, and with it a certificate for origin's host.
func Intercept(name string, origin Origin, now time.Time) (Interception, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return Interception{}, err
	}
	rootSerial, err := serial()
	if err != nil {
		return Interception{}, err
	}
	from, until := now.Add(-time.Hour), now.AddDate(1, 0, 0)
	root := &x509.Certificate{
		SerialNumber:          rootSerial,
		Subject:               pkix.Name{CommonName: fmt.Sprintf("%s %d", name, os.Getpid())},
		NotBefore:             from,
		NotAfter:              until,
		IsCA:                  true,
		BasicConstraintsValid: true,
		MaxPathLenZero:        true,
		KeyUsage:              x509.KeyUsageCertSign,
	}
	signed, err := x509.CreateCertificate(rand.Reader, root, root, &key.PublicKey, key)
	if err != nil {
		return Interception{}, err
	}
	root, err = x509.ParseCertificate(signed)
	if err != nil {
		return Interception{}, err
	}
	leafKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return Interception{}, err
	}
	leafSerial, err := serial()
	if err != nil {
		return Interception{}, err
	}
	host := origin.url.Hostname()
	leaf := &x509.Certificate{
		SerialNumber: leafSerial,
		Subject:      pkix.Name{CommonName: host},
		NotBefore:    from,
		NotAfter:     until,
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	if ip := net.ParseIP(host); ip != nil {
		leaf.IPAddresses = []net.IP{ip}
	} else {
		leaf.DNSNames = []string{host}
	}
	leafSigned, err := x509.CreateCertificate(rand.Reader, leaf, root, &leafKey.PublicKey, key)
	if err != nil {
		return Interception{}, err
	}
	return Interception{
		name:   name,
		origin: origin,
		pem:    pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: signed}),
		leaf:   tls.Certificate{Certificate: [][]byte{leafSigned}, PrivateKey: leafKey},
	}, nil
}

func serial() (*big.Int, error) {
	n, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return nil, fmt.Errorf("cannot read randomness for a certificate's serial: %w", err)
	}
	return n, nil
}

// AlsoTrusted is what the file `variable` names holds: the certificates a client trusts
// besides its own before it is given another file in its place. Nothing where `variable`
// names none; an error where the file it names cannot be read.
func AlsoTrusted(variable string) ([]byte, error) {
	named := os.Getenv(variable)
	if named == "" {
		return nil, nil
	}
	return os.ReadFile(named)
}

// WriteTrusted writes the file the client is given to trust: what it trusted already, and
// the authority, each on lines of its own.
func (i Interception) WriteTrusted(path string, already []byte) error {
	return os.WriteFile(path, bytes.Join([][]byte{already, i.pem}, []byte("\n")), 0o600)
}

// The variables HTTP clients read their proxy from, each spelling.
var proxyVariables = []string{"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"}

// The variables HTTP clients read the hosts they reach without a proxy from, which the
// client is given empty: every connection it makes reaches the door, and the door reaches
// the hosts they named as directly as the client did, from its own environment.
var exemptVariables = []string{"NO_PROXY", "no_proxy"}

// Variable is one variable of a client's environment.
type Variable struct {
	Name, Value string
}

func (v Variable) String() string {
	return v.Name + "=" + v.Value
}

// Proxy is how the process finds the proxy it reaches a URL through, nil for none, as
// http.Transport's Proxy is: http.ProxyFromEnvironment, for one that reaches the world as
// its own environment says.
type Proxy func(*http.Request) (*url.URL, error)

// Route is what a door did with a connection or request.
type Route string

const (
	// Intercepted: a CONNECT to the origin answered over TLS, or a plain request for it.
	Intercepted Route = "intercepted"
	// Tunnelled: a CONNECT to any other host, joined to that host.
	Tunnelled Route = "tunnelled"
	// Forwarded: a plain request for any other origin.
	Forwarded Route = "forwarded"
	// Refused: a request that is no proxy request, answered 400.
	Refused Route = "refused"
	// Served: what the door's own servers met, a handler's panic among it.
	Served Route = "served"
)

// Event is one connection or request a door took, as it was routed: Err is nil where the
// route was taken, and what stopped it where it was not.
type Event struct {
	Route  Route
	Target string // host:port for a CONNECT, the URL for a plain request
	Err    error
}

// Opening is what a Door is given: what it intercepts; who takes each request it
// intercepted, `Intercepted`, and each plain request for any other origin, `Forwarded`;
// how it reaches a host it tunnels to, `Proxy`; and where each Event goes, `Seen`.
type Opening struct {
	Interception Interception
	Intercepted  http.Handler
	Forwarded    http.Handler
	Proxy        Proxy
	Seen         func(Event)
}

// Door is a client's proxy, serving until closed.
type Door struct {
	opening Opening
	address string
	outer   *http.Server
	inner   *http.Server // serves the client's intercepted connections, once TLS is ended
	// The connections taken from the servers to tunnel or intercept, which neither server
	// tracks: Close ends them. [LAW:no-ambient-temporal-coupling] once closed is set, no
	// connection is taken.
	mu     sync.Mutex
	closed bool
	taken  map[net.Conn]struct{}
}

// Open serves `o` on listener. A CONNECT to the origin's host, where it is https, is
// answered over TLS with the interception's certificate, and each request on it goes to
// Intercepted; a plain request for the origin goes there too. Every other CONNECT is
// tunnelled to the host it names, and every other plain request goes to Forwarded.
// [LAW:single-enforcer] the one place a connection is told apart, for fritter and brioche
// alike.
func Open(listener net.Listener, o Opening) (*Door, error) {
	// [LAW:parse-dont-validate] a door missing any of its parts is refused here, before it
	// serves, and never fails at its first request.
	if o.Intercepted == nil || o.Forwarded == nil || o.Proxy == nil || o.Seen == nil {
		return nil, errors.New("a door needs Intercepted, Forwarded, Proxy and Seen")
	}
	serving := log.New(lines(func(line string) { o.Seen(Event{Route: Served, Err: errors.New(line)}) }), "", 0)
	door := &Door{
		opening: o,
		address: "http://" + listener.Addr().String(),
		inner:   &http.Server{Handler: o.Intercepted, ErrorLog: serving},
		taken:   map[net.Conn]struct{}{},
	}
	ended := handing(listener.Addr())
	door.outer = &http.Server{Handler: door.route(ended), ErrorLog: serving}
	// Serve returns only once its server is closed: every connection it met before then
	// went to its handler or its ErrorLog.
	go door.inner.Serve(ended)
	go door.outer.Serve(listener)
	return door, nil
}

// lines is a log writer that hands on each line it is written.
type lines func(string)

func (l lines) Write(p []byte) (int, error) {
	l(strings.TrimSuffix(string(p), "\n"))
	return len(p), nil
}

func (door *Door) route(ended *handed) http.Handler {
	o := door.opening
	origin := o.Interception.origin
	// The door ends TLS itself and hands the connection to a server that serves it as it
	// comes, which speaks HTTP/1.1 alone: only a server that ends TLS itself speaks h2.
	answer := &tls.Config{Certificates: []tls.Certificate{o.Interception.leaf}, NextProtos: []string{"http/1.1"}}
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch {
		case request.Method == http.MethodConnect:
			target, err := connectTarget(request.Host)
			switch {
			case err != nil:
				door.refuse(writer, request.Host, err)
			// A CONNECT is a tunnel for TLS, so it is the origin's only where the origin is https.
			case origin.is("https", target):
				door.intercept(writer, target, answer, ended)
			default:
				door.tunnel(writer, target)
			}
		case !request.URL.IsAbs():
			door.refuse(writer, request.URL.String(), fmt.Errorf("%s is a proxy: ask it for an absolute URL, or CONNECT", o.Interception.name))
		case origin.is(request.URL.Scheme, hostPort(request.URL)):
			o.Seen(Event{Route: Intercepted, Target: request.URL.String()})
			o.Intercepted.ServeHTTP(writer, request)
		default:
			o.Seen(Event{Route: Forwarded, Target: request.URL.String()})
			o.Forwarded.ServeHTTP(writer, request)
		}
	})
}

// connectTarget is the host and port a CONNECT names, which it must name both of.
func connectTarget(named string) (string, error) {
	host, _, err := net.SplitHostPort(named)
	if err == nil && host == "" {
		err = errors.New("no host")
	}
	if err != nil {
		return "", fmt.Errorf("a CONNECT names a host and port, and %q is not one: %w", named, err)
	}
	return named, nil
}

func (door *Door) refuse(writer http.ResponseWriter, target string, err error) {
	door.opening.Seen(Event{Route: Refused, Target: target, Err: err})
	http.Error(writer, err.Error(), http.StatusBadRequest)
}

// intercept answers a CONNECT to the origin, ends TLS on it, and hands what it opened to
// the server of intercepted connections.
func (door *Door) intercept(writer http.ResponseWriter, target string, answer *tls.Config, ended *handed) {
	near, err := door.take(writer)
	if err != nil {
		door.opening.Seen(Event{Route: Intercepted, Target: target, Err: err})
		return
	}
	opened := tls.Server(near, answer)
	opened.SetDeadline(time.Now().Add(10 * time.Second))
	err = opened.Handshake()
	door.opening.Seen(Event{Route: Intercepted, Target: target, Err: err})
	if err != nil {
		door.release(near)
		return
	}
	opened.SetDeadline(time.Time{})
	// The server of intercepted connections ends it from here, closing included.
	door.forget(near)
	ended.hand(opened)
}

// tunnel joins the client to target, through the outer proxy the door was given, if any.
func (door *Door) tunnel(writer http.ResponseWriter, target string) {
	far, err := dial(door.opening.Proxy, target)
	if err != nil {
		door.opening.Seen(Event{Route: Tunnelled, Target: target, Err: err})
		http.Error(writer, fmt.Sprintf("%s could not reach %s: %v", door.opening.Interception.name, target, err), http.StatusBadGateway)
		return
	}
	near, err := door.take(writer)
	door.opening.Seen(Event{Route: Tunnelled, Target: target, Err: err})
	if err != nil {
		far.Close()
		return
	}
	go func() {
		io.Copy(far, near)
		far.Close()
	}()
	io.Copy(near, far)
	door.release(near)
}

// take answers a CONNECT and takes the connection it opened from the server, the door's to
// end from then on.
func (door *Door) take(writer http.ResponseWriter) (net.Conn, error) {
	connection, read, err := http.NewResponseController(writer).Hijack()
	if err != nil {
		http.Error(writer, fmt.Sprintf("%s cannot take the connection: %v", door.opening.Interception.name, err), http.StatusInternalServerError)
		return nil, err
	}
	door.mu.Lock()
	closed := door.closed
	if !closed {
		door.taken[connection] = struct{}{}
	}
	door.mu.Unlock()
	if closed {
		connection.Close()
		return nil, net.ErrClosed
	}
	if _, err := connection.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n")); err != nil {
		door.release(connection)
		return nil, err
	}
	return &buffered{Conn: connection, from: read.Reader}, nil
}

// release ends a connection the door took.
func (door *Door) release(connection net.Conn) {
	door.forget(connection)
	connection.Close()
}

// forget gives up a connection the door took, without ending it.
func (door *Door) forget(connection net.Conn) {
	if held, ok := connection.(*buffered); ok {
		connection = held.Conn
	}
	door.mu.Lock()
	delete(door.taken, connection)
	door.mu.Unlock()
}

// Address is the door as a client's proxy variables name it.
func (door *Door) Address() string {
	return door.address
}

// Environment is what points a client at the door, each variable it sets: the door as its
// proxy for every host, no host exempted, and the file of certificates to trust, `trusted`,
// in the variable the client reads them from, `trust`.
func (door *Door) Environment(trust, trusted string) []Variable {
	given := []Variable{{trust, trusted}}
	for _, name := range proxyVariables {
		given = append(given, Variable{name, door.address})
	}
	for _, name := range exemptVariables {
		given = append(given, Variable{name, ""})
	}
	return given
}

// Close stops serving, and ends every connection the door took.
func (door *Door) Close() {
	door.outer.Close()
	door.inner.Close()
	door.mu.Lock()
	door.closed = true
	taken := door.taken
	door.taken = map[net.Conn]struct{}{}
	door.mu.Unlock()
	for connection := range taken {
		connection.Close()
	}
}

// buffered is a connection whose first bytes may already have been read into a buffer.
type buffered struct {
	net.Conn
	from *bufio.Reader
}

func (b *buffered) Read(p []byte) (int, error) {
	return b.from.Read(p)
}

// dial is a connection to target, through the proxy `proxy` names for it, if any.
func dial(proxy Proxy, target string) (net.Conn, error) {
	// A CONNECT is a tunnel for TLS, and is routed as https is, whatever its port.
	via, err := proxy(&http.Request{URL: &url.URL{Scheme: "https", Host: target}})
	if err != nil {
		return nil, err
	}
	if via == nil {
		return net.DialTimeout("tcp", target, 10*time.Second)
	}
	if via.Scheme != "http" {
		return nil, fmt.Errorf("a tunnel goes only through an http proxy, and the proxy given is %s", via.Redacted())
	}
	connection, err := net.DialTimeout("tcp", hostPort(via), 10*time.Second)
	if err != nil {
		return nil, err
	}
	// As net/http's own Transport asks a proxy for a tunnel.
	asked := &http.Request{Method: http.MethodConnect, URL: &url.URL{Opaque: target}, Host: target, Header: http.Header{}}
	if via.User != nil {
		password, _ := via.User.Password()
		asked.Header.Set("Proxy-Authorization", "Basic "+base64.StdEncoding.EncodeToString([]byte(via.User.Username()+":"+password)))
	}
	connection.SetDeadline(time.Now().Add(10 * time.Second))
	reader := bufio.NewReader(connection)
	err = asked.Write(connection)
	var answer *http.Response
	if err == nil {
		answer, err = http.ReadResponse(reader, asked)
	}
	if err == nil && answer.StatusCode != http.StatusOK {
		err = fmt.Errorf("the proxy %s answered %s", via.Redacted(), answer.Status)
	}
	if err != nil {
		connection.Close()
		return nil, err
	}
	connection.SetDeadline(time.Time{})
	return &buffered{Conn: connection, from: reader}, nil
}

// handed is a listener whose connections are given to it, one at a time, rather than
// accepted from the network: the client's connections to the origin, once TLS is ended.
type handed struct {
	given  chan net.Conn
	closed chan struct{}
	once   sync.Once
	addr   net.Addr
}

func handing(addr net.Addr) *handed {
	return &handed{given: make(chan net.Conn), closed: make(chan struct{}), addr: addr}
}

func (h *handed) hand(connection net.Conn) {
	select {
	case h.given <- connection:
	case <-h.closed:
		connection.Close()
	}
}

func (h *handed) Accept() (net.Conn, error) {
	select {
	case connection := <-h.given:
		return connection, nil
	case <-h.closed:
		return nil, net.ErrClosed
	}
}

func (h *handed) Close() error {
	h.once.Do(func() { close(h.closed) })
	return nil
}

func (h *handed) Addr() net.Addr {
	return h.addr
}
