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
	"context"
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
	"reflect"
	"regexp"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"time"
)

// Origin is the one origin a door intercepts: an http or https scheme and a host, nothing
// more, as ParseOrigin reads it: Intercept refuses any other.
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
func Intercept(name string, origin Origin, now time.Time) (*Interception, error) {
	// [LAW:parse-dont-validate] the one place an Origin is taken that ParseOrigin did not make.
	if origin.hostPort == "" {
		return nil, errors.New("an origin to intercept is one ParseOrigin made")
	}
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, err
	}
	rootSerial, err := serial()
	if err != nil {
		return nil, err
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
		return nil, err
	}
	root, err = x509.ParseCertificate(signed)
	if err != nil {
		return nil, err
	}
	leafKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, err
	}
	leafSerial, err := serial()
	if err != nil {
		return nil, err
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
		return nil, err
	}
	return &Interception{
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

// Trusted is an interception, and the file of certificates written for its client to trust
// it, with the variable that is to name that file: what a door is opened with, so the door
// and what its client is pointed at to trust cannot disagree.
type Trusted struct {
	interception *Interception
	variable     TrustVariable
	path         string
}

// Path is where the file is.
func (t Trusted) Path() string {
	return t.path
}

// Trust writes, at path, the file a client is given to trust in the place of the one
// `variable` names in its environment, read by `client`: the certificates that one holds,
// if it names one, and the authority, each on lines of its own. [LAW:single-enforcer] the
// one way a client comes to trust a door, for fritter and brioche alike.
// [LAW:effects-at-boundaries] the client's environment is given, as its proxy is: a door
// need not run in the environment its client does.
func (i *Interception) Trust(variable TrustVariable, client func(name string) string, path string, unread func(error)) (Trusted, error) {
	already, err := alsoTrusted(client(variable.name))
	if err != nil {
		// As Node does with a NODE_EXTRA_CA_CERTS it cannot read: says so, and goes on
		// without them, so the client runs as it would have without the door.
		unread(fmt.Errorf("cannot read the certificates %s names, so the client is not given them: %w", variable, err))
	}
	if err := os.WriteFile(path, bytes.Join([][]byte{already, i.pem}, []byte("\n")), 0o600); err != nil {
		return Trusted{}, fmt.Errorf("cannot write the certificates the client is to trust: %w", err)
	}
	return Trusted{interception: i, variable: variable, path: path}, nil
}

// alsoTrusted is what the file `named` holds: the certificates a client trusts besides its
// own before it is given another file in its place. Nothing where it is named none; an
// error, and nothing, where the file cannot be read: a read that fails part way has read a
// certificate cut short, which the client is not given.
func alsoTrusted(named string) ([]byte, error) {
	if named == "" {
		return nil, nil
	}
	held, err := os.ReadFile(named)
	if err != nil {
		return nil, err
	}
	return held, nil
}

// The variables HTTP clients read their proxy from, each spelling.
var proxyVariables = []string{"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"}

// The variables HTTP clients read the hosts they reach without a proxy from, which the
// client is given empty: every connection it makes reaches the door, and the door reaches
// the hosts they named as directly as the client did, from its own environment.
var exemptVariables = []string{"NO_PROXY", "no_proxy"}

var variableName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// TrustVariable names the variable a client reads the certificates it trusts besides its
// own from: an environment variable's name, and none of the ones Environment sets besides.
type TrustVariable struct {
	name string
}

// ParseTrustVariable reads name as a TrustVariable. [LAW:single-enforcer] the one test of
// it, for fritter's flag and brioche alike.
func ParseTrustVariable(name string) (TrustVariable, error) {
	if !variableName.MatchString(name) || slices.Contains(proxyVariables, name) || slices.Contains(exemptVariables, name) {
		return TrustVariable{}, fmt.Errorf("%q is no variable a client reads certificates from: it names no variable, or one that points the client at its proxy", name)
	}
	return TrustVariable{name}, nil
}

func (v TrustVariable) String() string {
	return v.name
}

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

// Event is what a door did with one connection or request, said once it is known: one of
// Opened, HandshakeRefused, HandshakeFailed, Answered, Cut, Tunnelled, Unreached, Refused,
// Logged and Stopped. [LAW:types-are-the-program] each outcome its own type, holding what that outcome
// has and nothing it does not.
type Event interface {
	event()
}

// Opened: TLS ended on a CONNECT to the origin, its requests served from here on.
type Opened struct {
	Target string // host:port, as the CONNECT named it
}

// HandshakeRefused: the client refused the door's TLS, with Alert; a certificate it does
// not trust is "unknown certificate authority" or "bad certificate".
type HandshakeRefused struct {
	Target string
	Alert  tls.AlertError
}

// HandshakeFailed: a CONNECT to the origin ended short of TLS, with no alert from the
// client: it hung up, ran out of time, or spoke something else.
type HandshakeFailed struct {
	Target string
	Err    error
}

// Route is who took a request: Intercepted or Forwarded.
type Route string

const (
	// Intercepted: a request for the origin, over TLS the door ended or in plain HTTP.
	Intercepted Route = "intercepted"
	// Forwarded: a plain request for any other origin.
	Forwarded Route = "forwarded"
)

// Answered: a request was taken by Route and answered with Status; 0 where the one who
// took it took its connection instead, as a switch of protocols does.
type Answered struct {
	Route  Route
	URL    string
	Status int
}

// Cut: a request whose taker gave up its answer part way, by a panic, as a reverse proxy
// does when the answer it is copying ends short; Status is what had been answered by then.
type Cut struct {
	Route  Route
	URL    string
	Status int
}

// Tunnelled: a CONNECT to any other host, joined to it until both ways ended. Err is what
// cut either short, nil where both ended as their senders ended them.
type Tunnelled struct {
	Target string
	Err    error
}

// Unreached: a CONNECT to a host the door could not reach, answered 502.
type Unreached struct {
	Target string
	Err    error
}

// Refused: a request that is no proxy request, answered 400.
type Refused struct {
	Asked string
	Err   error
}

// Logged: a line the door's servers logged, a handler's panic among them: a fault of the
// door's own, or of who it gave a request to.
type Logged struct {
	Line string
}

// Stopped: one of the door's servers stopped serving before the door was closed, with Err.
type Stopped struct {
	Err error
}

func (Opened) event()           {}
func (HandshakeRefused) event() {}
func (HandshakeFailed) event()  {}
func (Answered) event()         {}
func (Cut) event()              {}
func (Stopped) event()          {}
func (Tunnelled) event()        {}
func (Unreached) event()        {}
func (Refused) event()          {}
func (Logged) event()           {}

// Opening is what a Door is given: what it intercepts, and the file its client trusts it
// by; who takes each request it intercepted, `Intercepted`, and each plain request for any other origin, `Forwarded`;
// how it reaches a host it tunnels to, `Proxy`; and where each Event goes, `Seen`: called
// from the door's goroutines, many at once, as each thing happens, so what it waits on the
// door's handlers and its Close wait on too.
type Opening struct {
	Trusted     Trusted
	Intercepted http.Handler
	Forwarded   http.Handler
	Proxy       Proxy
	Seen        func(Event)
}

// Door is a client's proxy, serving until closed.
type Door struct {
	opening Opening
	address string
	outer   *http.Server
	inner   *http.Server // serves the client's intercepted connections, once TLS is ended
	log     *log.Logger  // the servers' log, each line of it Seen
	// The connections taken from the servers, by the door or by a handler, and those dialled
	// to tunnel to, which neither server tracks: Close ends them. And the door's own work,
	// its servers and each request they handed it: Close waits for it, and every connection
	// that work can wait on is one Close ends, or one still being dialled under `closing`.
	// [LAW:no-ambient-temporal-coupling] once `closing` is done, set under mu, no connection
	// is taken and no work begins.
	mu      sync.Mutex
	taken   map[net.Conn]struct{}
	work    sync.WaitGroup
	closing context.Context
	close   context.CancelFunc
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
	if o.Trusted.interception == nil || o.Intercepted == nil || o.Forwarded == nil || o.Proxy == nil || o.Seen == nil {
		return nil, errors.New("a door needs a Trusted, Intercepted, Forwarded, Proxy and Seen")
	}
	serving := log.New(lines(func(line string) { o.Seen(Logged{line}) }), "", 0)
	door := &Door{opening: o, address: "http://" + listener.Addr().String(), log: serving, taken: map[net.Conn]struct{}{}}
	door.closing, door.close = context.WithCancel(context.Background())
	origin := o.Trusted.interception.origin
	door.inner = &http.Server{
		Handler: door.handles(func(writer http.ResponseWriter, request *http.Request) {
			// Asked in origin-form over the TLS the door ended: the origin is the URL's.
			door.answer(Intercepted, o.Intercepted, origin.String()+request.URL.RequestURI(), writer, request)
		}),
		ErrorLog: serving,
	}
	ended := handing(listener.Addr())
	door.outer = &http.Server{Handler: door.route(ended), ErrorLog: serving}
	// Serve returns once its server is closed, or its listener fails: every connection it
	// met before then went to its handler or its ErrorLog.
	for _, serving := range []func() error{func() error { return door.inner.Serve(ended) }, func() error { return door.outer.Serve(listener) }} {
		door.work.Add(1)
		go func() {
			defer door.work.Done()
			if err := serving(); !errors.Is(err, http.ErrServerClosed) {
				o.Seen(Stopped{err})
			}
		}()
	}
	return door, nil
}

// lines is a log writer that hands on each line it is written, one at a time: a panic's
// stack is many.
type lines func(string)

func (l lines) Write(p []byte) (int, error) {
	for _, line := range strings.Split(strings.TrimSuffix(string(p), "\n"), "\n") {
		l(line)
	}
	return len(p), nil
}

// handles has handler take each request the door is handed, as work Close waits for: a
// request handed it once it is closed is answered that it is. A handler's panic is logged
// here, within that work, as net/http would log it, and net/http is then told only that the
// handler gave up, which it ends the connection for and does not log.
func (door *Door) handles(handler http.HandlerFunc) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		door.mu.Lock()
		closed := door.closing.Err() != nil
		if !closed {
			door.work.Add(1)
		}
		door.mu.Unlock()
		if closed {
			http.Error(writer, door.opening.Trusted.interception.name+" is closed", http.StatusServiceUnavailable)
			return
		}
		defer door.work.Done()
		defer func() {
			if cut := recover(); cut != nil {
				if cut != http.ErrAbortHandler {
					door.log.Printf("http: panic serving %v: %v\n%s", request.RemoteAddr, cut, debug.Stack())
				}
				panic(http.ErrAbortHandler)
			}
		}()
		handler(writer, request)
	})
}

func (door *Door) route(ended *handed) http.Handler {
	o := door.opening
	interception := o.Trusted.interception
	origin := interception.origin
	// The door ends TLS itself and hands the connection to a server that serves it as it
	// comes, which speaks HTTP/1.1 alone: only a server that ends TLS itself speaks h2.
	answer := &tls.Config{Certificates: []tls.Certificate{interception.leaf}, NextProtos: []string{"http/1.1"}}
	return door.handles(func(writer http.ResponseWriter, request *http.Request) {
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
			door.refuse(writer, request.URL.String(), fmt.Errorf("%s is a proxy: ask it for an absolute URL, or CONNECT", interception.name))
		case origin.is(request.URL.Scheme, hostPort(request.URL)):
			door.answer(Intercepted, o.Intercepted, request.URL.String(), writer, request)
		default:
			door.answer(Forwarded, o.Forwarded, request.URL.String(), writer, request)
		}
	})
}

// answer has request taken by handler, and says how it was answered once it was.
func (door *Door) answer(route Route, handler http.Handler, asked string, writer http.ResponseWriter, request *http.Request) {
	answering := &answering{ResponseWriter: writer, door: door}
	defer func() {
		cut := recover()
		// A handler that returned ended what it took, or is ending it: the door gives it up.
		// One that panicked ends nothing, and net/http ends no connection it gave away.
		switch {
		case answering.taken != nil && cut != nil:
			door.release(answering.taken)
		case answering.taken != nil:
			door.forget(answering.taken)
		}
		if cut != nil {
			// The server ends the connection, and the door logs what is no http.ErrAbortHandler.
			door.opening.Seen(Cut{Route: route, URL: asked, Status: answering.status})
			panic(cut)
		}
	}()
	handler.ServeHTTP(answering, request)
	door.opening.Seen(Answered{Route: route, URL: asked, Status: answering.answered()})
}

// answering is a ResponseWriter that keeps the status it answered with, and the
// connection its handler took, which the door holds.
type answering struct {
	http.ResponseWriter
	door   *Door
	status int
	taken  net.Conn
}

func (a *answering) WriteHeader(code int) {
	// A 1xx goes before the answer, and is not it.
	if a.status == 0 && code >= 200 {
		a.status = code
	}
	a.ResponseWriter.WriteHeader(code)
}

func (a *answering) Write(p []byte) (int, error) {
	if a.status == 0 {
		a.status = http.StatusOK
	}
	return a.ResponseWriter.Write(p)
}

// Unwrap lets an http.ResponseController reach the writer beneath, to flush it.
func (a *answering) Unwrap() http.ResponseWriter {
	return a.ResponseWriter
}

// Hijack takes the connection for the handler, and holds it for the door to end until the
// handler returns: a handler joining it to another, as a ReverseProxy does an upgrade, ends
// once it is.
func (a *answering) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	connection, read, err := http.NewResponseController(a.ResponseWriter).Hijack()
	if err == nil {
		err = a.door.hold(connection)
	}
	if err != nil {
		return nil, nil, err
	}
	a.taken = connection
	return connection, read, nil
}

// answered is the status the answer went with: net/http's 200 where nothing was written,
// and 0 where the connection was taken.
func (a *answering) answered() int {
	switch {
	case a.taken != nil:
		return 0
	case a.status == 0:
		return http.StatusOK
	}
	return a.status
}

// connectTarget is the host and port a CONNECT names, which it must name both of, lowered:
// a host is named in any case.
func connectTarget(named string) (string, error) {
	host, port, err := net.SplitHostPort(named)
	if err == nil && (host == "" || port == "") {
		err = errors.New("no host, or no port")
	}
	if err != nil {
		return "", fmt.Errorf("a CONNECT names a host and port, and %q is not one: %w", named, err)
	}
	return strings.ToLower(named), nil
}

func (door *Door) refuse(writer http.ResponseWriter, asked string, err error) {
	door.opening.Seen(Refused{Asked: asked, Err: err})
	http.Error(writer, err.Error(), http.StatusBadRequest)
}

// intercept answers a CONNECT to the origin, ends TLS on it, and hands what it opened to
// the server of intercepted connections.
func (door *Door) intercept(writer http.ResponseWriter, target string, answer *tls.Config, ended *handed) {
	near, err := door.take(writer)
	if err != nil {
		door.opening.Seen(HandshakeFailed{Target: target, Err: err})
		return
	}
	opened := tls.Server(near, answer)
	opened.SetDeadline(time.Now().Add(10 * time.Second))
	if err := opened.Handshake(); err != nil {
		door.release(near)
		door.opening.Seen(handshakeEnded(target, err))
		return
	}
	opened.SetDeadline(time.Time{})
	door.opening.Seen(Opened{Target: target})
	// The server of intercepted connections ends it from here, closing included.
	door.forget(near)
	ended.hand(opened)
}

// handshakeEnded is what ended a handshake short. crypto/tls gives an alert the client
// sent as a net.OpError "remote error" around a value of its own unexported alert type, a
// byte: read here and nowhere else, and held to crypto/tls by the package's tests.
func handshakeEnded(target string, err error) Event {
	var received *net.OpError
	if errors.As(err, &received) && received.Op == "remote error" {
		if alert := reflect.ValueOf(received.Err); alert.Kind() == reflect.Uint8 {
			return HandshakeRefused{Target: target, Alert: tls.AlertError(alert.Uint())}
		}
	}
	return HandshakeFailed{Target: target, Err: err}
}

// tunnel joins the client to target, through the outer proxy the door was given, if any,
// each way until its sender ends it.
func (door *Door) tunnel(writer http.ResponseWriter, target string) {
	far, err := door.dial(target)
	if err == nil {
		err = door.hold(far)
	}
	if err != nil {
		door.opening.Seen(Unreached{Target: target, Err: err})
		http.Error(writer, fmt.Sprintf("%s could not reach %s: %v", door.opening.Trusted.interception.name, target, err), http.StatusBadGateway)
		return
	}
	near, err := door.take(writer)
	if err != nil {
		door.release(far)
		door.opening.Seen(Tunnelled{Target: target, Err: err})
		return
	}
	sent := make(chan error, 1)
	go func() {
		_, err := io.Copy(far, near)
		closeWrite(far)
		sent <- err
	}()
	_, err = io.Copy(near, far)
	closeWrite(near)
	err = errors.Join(<-sent, err)
	door.release(near)
	door.release(far)
	door.opening.Seen(Tunnelled{Target: target, Err: err})
}

// closeWrite ends what is sent on connection, leaving what comes back to come: a
// connection that cannot end one way alone is ended both.
func closeWrite(connection net.Conn) {
	if halved, ok := connection.(interface{ CloseWrite() error }); ok {
		halved.CloseWrite()
		return
	}
	connection.Close()
}

// take answers a CONNECT and takes the connection it opened from the server, the door's to
// end from then on.
func (door *Door) take(writer http.ResponseWriter) (*buffered, error) {
	connection, read, err := http.NewResponseController(writer).Hijack()
	if err != nil {
		http.Error(writer, fmt.Sprintf("%s cannot take the connection: %v", door.opening.Trusted.interception.name, err), http.StatusInternalServerError)
		return nil, err
	}
	taken := &buffered{Conn: connection, from: read.Reader}
	if err := door.hold(taken); err != nil {
		return nil, err
	}
	if _, err := connection.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n")); err != nil {
		door.release(taken)
		return nil, err
	}
	return taken, nil
}

// hold makes connection the door's to end, or ends it where the door is closed.
func (door *Door) hold(connection net.Conn) error {
	door.mu.Lock()
	closed := door.closing.Err() != nil
	if !closed {
		door.taken[connection] = struct{}{}
	}
	door.mu.Unlock()
	if closed {
		connection.Close()
		return net.ErrClosed
	}
	return nil
}

// release ends a connection the door took.
func (door *Door) release(connection net.Conn) {
	door.forget(connection)
	connection.Close()
}

// forget gives up a connection the door took, without ending it.
func (door *Door) forget(connection net.Conn) {
	door.mu.Lock()
	delete(door.taken, connection)
	door.mu.Unlock()
}

// Address is the door as a client's proxy variables name it.
func (door *Door) Address() string {
	return door.address
}

// Environment is what points a client at the door, each variable it sets: the file of
// certificates the door was opened with, in the variable it is to be named in, and the door
// as its proxy for every host, no host exempted.
func (door *Door) Environment() []Variable {
	trusted := door.opening.Trusted
	given := []Variable{{trusted.variable.name, trusted.path}}
	for _, name := range proxyVariables {
		given = append(given, Variable{name, door.address})
	}
	for _, name := range exemptVariables {
		given = append(given, Variable{name, ""})
	}
	return given
}

// Close stops serving, ends every connection the door took, and returns once the door's
// work has: each request and tunnel it was serving is seen ended before Close returns.
func (door *Door) Close() {
	door.mu.Lock()
	door.close()
	taken := door.taken
	door.taken = map[net.Conn]struct{}{}
	door.mu.Unlock()
	door.outer.Close()
	door.inner.Close()
	for connection := range taken {
		connection.Close()
	}
	door.work.Wait()
}

// buffered is a connection whose first bytes may already have been read into a buffer.
type buffered struct {
	net.Conn
	from *bufio.Reader
}

func (b *buffered) Read(p []byte) (int, error) {
	return b.from.Read(p)
}

func (b *buffered) CloseWrite() error {
	closeWrite(b.Conn)
	return nil
}

// dial is a connection to target, through the proxy the door's Proxy names for it, if any:
// net.ErrClosed where the door closed on it.
func (door *Door) dial(target string) (net.Conn, error) {
	connection, err := door.dialing(target)
	if err != nil && door.closing.Err() != nil {
		return nil, net.ErrClosed
	}
	return connection, err
}

func (door *Door) dialing(target string) (net.Conn, error) {
	dialer := &net.Dialer{Timeout: 10 * time.Second}
	// A CONNECT is a tunnel for TLS, and is routed as https is, whatever its port.
	via, err := door.opening.Proxy(&http.Request{URL: &url.URL{Scheme: "https", Host: target}})
	if err != nil {
		return nil, err
	}
	if via == nil {
		return dialer.DialContext(door.closing, "tcp", target)
	}
	if via.Scheme != "http" && via.Scheme != "https" {
		return nil, fmt.Errorf("a tunnel goes only through an http or https proxy, and the proxy given is %s", via.Redacted())
	}
	reached, err := dialer.DialContext(door.closing, "tcp", hostPort(via))
	if err == nil {
		// The door's to end while the proxy is asked, so it is waited on only until the door
		// closes; the tunnel holds what the proxy answers with.
		err = door.hold(reached)
	}
	if err != nil {
		return nil, err
	}
	defer door.forget(reached)
	connection := reached
	if via.Scheme == "https" {
		// As net/http's own Transport reaches an https proxy: over TLS, trusting what the
		// machine trusts. The handshake is made under the deadline set below.
		connection = tls.Client(connection, &tls.Config{ServerName: via.Hostname()})
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
