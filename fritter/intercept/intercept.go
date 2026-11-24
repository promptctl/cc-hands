// Package intercept is how a proxy of hands' answers a client's connections to one host
// itself, over TLS under an authority of its own, and joins every other connection to
// wherever it was going: what fritter's tap and brioche share. A change here is a change
// to fritter, which every session runs under, so it holds only what the two must change at
// the same moment: how a client is given a certificate to trust, and how a connection is
// told apart, opened or joined. What either does with a request it opened is its own.
package intercept

import (
	"bufio"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/pem"
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

// Authority is a certificate authority that lives only in memory, and the certificate it
// signed for the one host it answers for.
type Authority struct {
	PEM  []byte // the authority's certificate, which the client is given to trust
	Leaf tls.Certificate
}

// NewAuthority makes an authority named `name`, and with it a certificate for host. Its
// key is never written anywhere: it lives and dies with the process.
func NewAuthority(name, host string) (Authority, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return Authority{}, err
	}
	from, until := time.Now().Add(-time.Hour), time.Now().AddDate(1, 0, 0)
	root := &x509.Certificate{
		SerialNumber:          serial(),
		Subject:               pkix.Name{CommonName: name},
		NotBefore:             from,
		NotAfter:              until,
		IsCA:                  true,
		BasicConstraintsValid: true,
		MaxPathLenZero:        true,
		KeyUsage:              x509.KeyUsageCertSign,
	}
	signed, err := x509.CreateCertificate(rand.Reader, root, root, &key.PublicKey, key)
	if err != nil {
		return Authority{}, err
	}
	root, err = x509.ParseCertificate(signed)
	if err != nil {
		return Authority{}, err
	}
	leafKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return Authority{}, err
	}
	leaf := &x509.Certificate{
		SerialNumber: serial(),
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
		return Authority{}, err
	}
	return Authority{
		PEM:  pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: signed}),
		Leaf: tls.Certificate{Certificate: [][]byte{leafSigned}, PrivateKey: leafKey},
	}, nil
}

func serial() *big.Int {
	n, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		// crypto/rand does not fail on any system hands runs on.
		panic(fmt.Sprintf("cannot read randomness: %v", err))
	}
	return n
}

// AlsoTrusted is what the file `variable` names holds, ended by a line: the certificates
// the client trusted besides its own before it is given another file in its place. Nothing
// where `variable` names none; an error where the file it names cannot be read, which the
// client would have been told too, and gone on without them.
func AlsoTrusted(variable string) ([]byte, error) {
	named := os.Getenv(variable)
	if named == "" {
		return nil, nil
	}
	held, err := os.ReadFile(named)
	if err != nil {
		return nil, err
	}
	return append(held, '\n'), nil
}

// WriteTrusted writes the file the client is given to trust: what it trusted already, and
// the authority.
func WriteTrusted(path string, already []byte, authority Authority) error {
	return os.WriteFile(path, append(already, authority.PEM...), 0o600)
}

// AuthorityOf is the host and port a URL reaches, as a CONNECT names it.
func AuthorityOf(target *url.URL) string {
	if target.Port() != "" {
		return target.Host
	}
	port := map[string]string{"http": "80", "https": "443"}[target.Scheme]
	return net.JoinHostPort(target.Hostname(), port)
}

// Proxied is how the process finds the proxy it reaches a URL through, nil for none:
// http.ProxyFromEnvironment, for one that reaches the world as its own environment says.
type Proxied func(*http.Request) (*url.URL, error)

// Door is a client's proxy, serving until closed.
type Door struct {
	outer  *http.Server
	opened *http.Server
}

// Opening is what a Door is given: who it is, `name`, which the client reads in every
// error it answers; the one origin it answers for itself, `tapped`, with the certificate
// it answers that host with; who takes each request it opened or that was asked of that
// origin in plain HTTP, `opened`; who takes every other request asked in plain HTTP,
// `passing`; and how it reaches a host it joins the client to, `proxied`.
type Opening struct {
	Name    string
	Tapped  *url.URL
	Leaf    tls.Certificate
	Opened  http.Handler
	Passing http.Handler
	Proxied Proxied
	// Where the servers' own complaints go: every error reaches the client already.
	ErrorLog *log.Logger
}

// Open serves `o` on listener. A CONNECT to the tapped origin's host, where it is https,
// is answered over TLS with the leaf, and each request on it goes to Opened; a request
// asked of that origin in plain HTTP goes there too. Every other CONNECT is joined to the
// host it names, and every other request goes to Passing. [LAW:single-enforcer] the one
// place a connection is told apart, for fritter and brioche alike.
func Open(listener net.Listener, o Opening) *Door {
	opening := handing(listener.Addr())
	door := &Door{opened: &http.Server{Handler: o.Opened, ErrorLog: o.ErrorLog}}
	go door.opened.Serve(opening)
	// A host is named in any case, and a client may send it lowered: one host either way.
	tapped := AuthorityOf(o.Tapped)
	answer := &tls.Config{Certificates: []tls.Certificate{o.Leaf}, NextProtos: []string{"http/1.1"}}
	door.outer = &http.Server{
		Handler: http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			switch {
			case request.Method == http.MethodConnect && strings.EqualFold(request.Host, tapped) && o.Tapped.Scheme == "https":
				joined(o.Name, writer, func(connection net.Conn) { opening.hand(tls.Server(connection, answer)) })
			case request.Method == http.MethodConnect:
				tunnel(o.Name, o.Proxied, writer, request.Host)
			case !request.URL.IsAbs():
				http.Error(writer, o.Name+" is a proxy: ask it for an absolute URL, or CONNECT", http.StatusBadRequest)
			case request.URL.Scheme == o.Tapped.Scheme && strings.EqualFold(AuthorityOf(request.URL), tapped):
				o.Opened.ServeHTTP(writer, request)
			default:
				o.Passing.ServeHTTP(writer, request)
			}
		}),
		ErrorLog: o.ErrorLog,
	}
	go door.outer.Serve(listener)
	return door
}

// Close stops serving.
func (door *Door) Close() {
	door.outer.Close()
	door.opened.Close()
}

// joined answers a CONNECT and gives the connection it opened to then.
func joined(name string, writer http.ResponseWriter, then func(net.Conn)) {
	connection, buffered, err := http.NewResponseController(writer).Hijack()
	if err != nil {
		http.Error(writer, fmt.Sprintf("%s cannot take the connection: %v", name, err), http.StatusInternalServerError)
		return
	}
	if _, err := connection.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n")); err != nil {
		connection.Close()
		return
	}
	then(&Read{Conn: connection, From: buffered.Reader})
}

// Read is a connection whose first bytes may already have been read into a buffer.
type Read struct {
	net.Conn
	From *bufio.Reader
}

func (r *Read) Read(p []byte) (int, error) {
	return r.From.Read(p)
}

// tunnel joins the client to target, through the proxy `proxied` names, if any.
func tunnel(name string, proxied Proxied, writer http.ResponseWriter, target string) {
	far, err := Dial(name, proxied, target)
	if err != nil {
		http.Error(writer, fmt.Sprintf("%s could not reach %s: %v", name, target, err), http.StatusBadGateway)
		return
	}
	joined(name, writer, func(near net.Conn) {
		go func() {
			io.Copy(far, near)
			far.Close()
		}()
		io.Copy(near, far)
		near.Close()
	})
}

// Dial is a connection to target, through the proxy `proxied` names for it, if any.
func Dial(name string, proxied Proxied, target string) (net.Conn, error) {
	via, err := proxied(&http.Request{URL: &url.URL{Scheme: "https", Host: target}})
	if err != nil {
		return nil, err
	}
	if via == nil {
		return net.DialTimeout("tcp", target, 10*time.Second)
	}
	if via.Scheme != "http" {
		return nil, fmt.Errorf("%s tunnels only through an http proxy, and was given %s", name, via.Redacted())
	}
	connection, err := net.DialTimeout("tcp", AuthorityOf(via), 10*time.Second)
	if err != nil {
		return nil, err
	}
	asked := "CONNECT " + target + " HTTP/1.1\r\nHost: " + target + "\r\n"
	if via.User != nil {
		password, _ := via.User.Password()
		asked += "Proxy-Authorization: Basic " + base64.StdEncoding.EncodeToString([]byte(via.User.Username()+":"+password)) + "\r\n"
	}
	connection.SetDeadline(time.Now().Add(10 * time.Second))
	reader := bufio.NewReader(connection)
	_, err = connection.Write([]byte(asked + "\r\n"))
	var answer *http.Response
	if err == nil {
		answer, err = http.ReadResponse(reader, &http.Request{Method: http.MethodConnect})
	}
	if err == nil && answer.StatusCode != http.StatusOK {
		err = fmt.Errorf("the proxy %s answered %s", via.Redacted(), answer.Status)
	}
	if err != nil {
		connection.Close()
		return nil, err
	}
	connection.SetDeadline(time.Time{})
	return &Read{Conn: connection, From: reader}, nil
}

// handed is a listener whose connections are given to it, one at a time, rather than
// accepted from the network: the client's connections to the tapped origin, once opened.
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
