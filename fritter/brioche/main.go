// Command brioche is the brain's HTTPS proxy: the brain's Claude Code reaches Anthropic
// through it, takes it for Anthropic's own API, and every exchange it has goes to hands'
// proxy instead.
//
//	brioche --intercept ORIGIN --to ORIGIN --trust-variable VARIABLE --trust-file FILE
//
// A CONNECT to the intercepted origin's host is answered over TLS under brioche's own
// certificate authority, and each request on it goes to the --to origin, hands' proxy on
// loopback, in plain HTTP as it was sent, its answer streamed back as it comes. Every other
// CONNECT is tunnelled to the host it names, and every other plain request goes on as it was
// asked, through the proxy brioche's own environment names, if any.
//
// brioche writes, at FILE, the certificates the client is to trust: those its VARIABLE
// names in brioche's own environment, which is the client's, and brioche's authority. It
// then prints the environment that points a client at it, a NAME=VALUE line for each
// variable, and closes its standard output: whoever started it reads that to its end. It
// serves until its standard input ends or it is told to stop, so a brioche whose starter has
// gone goes too, and it says why it stopped.
//
// brioche records nothing: hands' proxy, at the --to origin, is the one recorder of what
// passes through it. brioche says, on its standard error, what it did with each connection
// and request, and what it could not do: one JSON object a line, its kind, how bad it is,
// and what it is in words (`said`), so whoever started it reads each line as what it is.
//
// It is built from fritter's module, for the interception it shares with fritter's tap, and
// it is not fritter: fritter's binary links none of this, so a change here restarts no
// session. [LAW:one-source-of-truth] exception: what brioche does with a request it took,
// the transports and proxies and how a failure is answered, is fritter's tap's twin
// (fritter/tap.go), kept apart on purpose: shared, every change to it would be a fritter
// release and a restart of every session, for what only the brain needs. A fix to one is
// weighed for the other.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/bmf/cc-hands/fritter/intercept"
)

const (
	failed = 1 // brioche could not start, or could not go on serving
	misuse = 64
)

func main() {
	os.Exit(serve(os.Args[1:]))
}

// serve is brioche in its process: its arguments, its environment, its standard streams,
// the signals it is sent, and a port of its own on loopback.
func serve(args []string) int {
	// Armed first, so a stop that comes while brioche starts waits here, and is never the
	// default disposition that kills it with its trust file half written.
	killed := make(chan os.Signal, 1)
	signal.Notify(killed, syscall.SIGTERM, syscall.SIGINT, syscall.SIGHUP)
	ended := stops()
	go func() {
		if _, err := io.Copy(io.Discard, os.Stdin); err != nil {
			ended.send(stop{"its input failed", err})
			return
		}
		ended.send(stop{"its input ended", nil})
	}()
	go func() {
		told := <-killed
		// A second signal takes its default course: a stop that stalls is never one that
		// nothing short of SIGKILL ends.
		signal.Stop(killed)
		ended.send(stop{"told to by " + told.String(), nil})
	}()
	listen := func() (net.Listener, error) { return net.Listen("tcp", "127.0.0.1:0") }
	return run(args, os.Getenv, http.ProxyFromEnvironment, listen, os.Stdout, os.Stderr, ended)
}

// stop is why brioche is to stop: Err is what failed, which makes the stop a failure; nil
// for one it was asked for.
type stop struct {
	why string
	err error
}

// stopping is why brioche stops: the first stop sent it, and only that one. [LAW:types-are-the-program]
// one stop is taken however many come, and when: one that comes once brioche is stopping
// already is never taken, so whoever sent it says it.
type stopping struct {
	first sync.Once
	why   chan stop // holds the first
}

func stops() *stopping {
	return &stopping{why: make(chan stop, 1)}
}

// send is whether `why` is the stop brioche stops for.
func (s *stopping) send(why stop) (taken bool) {
	s.first.Do(func() {
		s.why <- why
		taken = true
	})
	return taken
}

// flag is one of brioche's flags, every one of them needed: what its value is called, and
// how it is read into what brioche is asked to do.
type flag struct {
	name, value string
	read        func(*options, string) error
}

// [LAW:one-source-of-truth] the one table of brioche's flags: what is taken, what is
// needed, how each is read, and the usage that names them.
var flags = []flag{
	{"--intercept", "ORIGIN", func(asked *options, value string) (err error) {
		asked.intercepted, err = intercept.ParseOrigin(value)
		return err
	}},
	{"--to", "ORIGIN", func(asked *options, value string) (err error) {
		asked.to, err = loopback(value)
		return err
	}},
	{"--trust-variable", "VARIABLE", func(asked *options, value string) (err error) {
		asked.trust, err = intercept.ParseTrustVariable(value)
		return err
	}},
	{"--trust-file", "FILE", func(asked *options, value string) error {
		file, err := filepath.Abs(value)
		if err == nil && strings.ContainsAny(file, "\n\r") {
			err = errors.New("a path printed on a line of its own holds no line break")
		}
		asked.file = file
		return err
	}},
}

var usage = func() string {
	line := "usage: brioche"
	for _, flag := range flags {
		line += " " + flag.name + " " + flag.value
	}
	return line
}()

// options is what brioche is asked to do, each part of it parsed.
type options struct {
	intercepted intercept.Origin
	to          intercept.Origin // on loopback
	trust       intercept.TrustVariable
	file        string // absolute, so a client in any directory finds it
}

// parse reads brioche's arguments. [LAW:parse-dont-validate] each flag given once, with a
// value, and all of them: a flag left empty or given twice is refused, never taken for one
// not given or for its last spelling.
func parse(args []string) (options, error) {
	given := map[string]string{}
	for i := 0; i < len(args); i++ {
		name := args[i]
		if !slices.ContainsFunc(flags, func(flag flag) bool { return flag.name == name }) {
			return options{}, fmt.Errorf("unknown argument %q", name)
		}
		if i+1 >= len(args) || args[i+1] == "" {
			return options{}, fmt.Errorf("%s needs a value", name)
		}
		if _, twice := given[name]; twice {
			return options{}, fmt.Errorf("%s is given twice", name)
		}
		given[name] = args[i+1]
		i++
	}
	var asked options
	for _, flag := range flags {
		value, ok := given[flag.name]
		if !ok {
			return options{}, fmt.Errorf("%s is needed", flag.name)
		}
		if err := flag.read(&asked, value); err != nil {
			return options{}, fmt.Errorf("%s: %w", flag.name, err)
		}
	}
	return asked, nil
}

// loopback is what --to names: hands' proxy, an origin on loopback and no more. A path,
// query or credentials would be dropped, since each request goes to the path it was sent
// to, and an origin anywhere else would be reached around the proxy brioche was given.
func loopback(raw string) (intercept.Origin, error) {
	named, err := url.Parse(raw)
	if err != nil {
		return intercept.Origin{}, err
	}
	if named.User != nil || strings.TrimSuffix(named.Path, "/") != "" || named.RawQuery != "" || named.ForceQuery || named.Fragment != "" {
		return intercept.Origin{}, fmt.Errorf("%q names more than an origin: each request goes to the path it was sent to", raw)
	}
	if host := named.Hostname(); !strings.EqualFold(host, "localhost") && !net.ParseIP(host).IsLoopback() {
		return intercept.Origin{}, fmt.Errorf("%q is not on loopback, where hands' proxy listens", raw)
	}
	return intercept.ParseOrigin(raw)
}

// run is brioche, with what it reaches of its process passed in: the environment it reads
// the client's trust from; how it reaches the world, `outer`; where it listens; where it
// prints the client's environment, which it closes however it returns; where it says what it
// must; and `ended`, which says when it is to stop and why, and which its door says on too.
func run(args []string, client func(string) string, outer intercept.Proxy, listen func() (net.Listener, error), printed io.WriteCloser, said io.Writer, ended *stopping) int {
	closePrinted := sync.OnceValue(printed.Close)
	defer closePrinted()
	sayer := &sayer{to: said}
	asked, err := parse(args)
	if err != nil {
		sayer.sayf("misused", failure, "%v\n%s", err, usage)
		return misuse
	}
	door, err := open(asked, client(asked.trust.String()), outer, listen, sayer, ended)
	if err != nil {
		sayer.sayf("unopened", failure, "%v", err)
		return failed
	}
	defer door.Close()
	for _, variable := range door.Environment() {
		if _, err := fmt.Fprintln(printed, variable); err != nil {
			sayer.sayf("unprinted", failure, "cannot print the environment that points a client at brioche: %v", err)
			return failed
		}
	}
	if err := closePrinted(); err != nil {
		sayer.sayf("unprinted", failure, "cannot end the environment that points a client at brioche: %v", err)
		return failed
	}
	why := <-ended.why
	if why.err != nil {
		// [LAW:no-silent-failure] a stop for a failure, a door that stopped serving among them,
		// ends brioche as failed: its starter sees it gone, never a brioche alive and deaf.
		sayer.sayf("stopping", failure, "stopping: %s: %v", why.why, why.err)
		return failed
	}
	sayer.sayf("stopping", info, "stopping: %s", why.why)
	return 0
}

// level is how bad a thing brioche says is, in the words of hands' audit, which judges each
// line it records by it.
type level string

const (
	info    level = "info"
	warning level = "warning"
	failure level = "error"
)

// report is one thing brioche says: what kind of thing it is, how bad, and it in words; and
// what the door's event held of the connection or request it is of.
type report struct {
	Kind   string `json:"kind"`
	Level  level  `json:"level"`
	Said   string `json:"said"`
	Target string `json:"target,omitempty"` // host:port, as a CONNECT named it
	URL    string `json:"url,omitempty"`
	Status *int   `json:"status,omitempty"` // what a request was answered with, 0 for a connection taken
}

// sayer says each thing whole on its own line, so things said from the door's goroutines
// at once are never interleaved.
type sayer struct {
	mu sync.Mutex
	to io.Writer
}

func (s *sayer) say(said report) {
	// A struct of strings and an int, which Marshal always encodes, any line break in a
	// string escaped: one thing said is one line.
	line, _ := json.Marshal(said)
	s.mu.Lock()
	defer s.mu.Unlock()
	// A standard error that cannot be written to leaves brioche nowhere to say so.
	_, _ = s.to.Write(append(line, '\n'))
}

func (s *sayer) sayf(kind string, how level, format string, args ...any) {
	s.say(report{Kind: kind, Level: how, Said: fmt.Sprintf(format, args...)})
}

// logger is what a server or proxy of the standard library logs, each line said as `kind`.
func (s *sayer) logger(kind string, how level) *log.Logger {
	return log.New(writing(func(p []byte) {
		s.say(report{Kind: kind, Level: how, Said: strings.TrimSuffix(string(p), "\n")})
	}), "", 0)
}

// writing is an io.Writer that hands each write to `to`, whole.
type writing func([]byte)

func (to writing) Write(p []byte) (int, error) {
	to(p)
	return len(p), nil
}

// open is brioche's door, for a client whose trust variable names `named`. A server of its
// that stops serving is a stop, sent on `ended`.
func open(asked options, named string, outer intercept.Proxy, listen func() (net.Listener, error), sayer *sayer, ended *stopping) (*intercept.Door, error) {
	interception, err := intercept.Intercept("brioche", asked.intercepted, time.Now())
	if err != nil {
		return nil, fmt.Errorf("cannot make the certificate the client is answered with for %s: %w", asked.intercepted, err)
	}
	if err := theirs(named, asked); err != nil {
		return nil, err
	}
	trusted, err := interception.Trust(asked.trust, named, asked.file, func(unread error) { sayer.sayf("untrusted", warning, "%v", unread) })
	if err != nil {
		return nil, err
	}
	listener, err := listen()
	if err != nil {
		return nil, fmt.Errorf("cannot listen for the client's requests: %w", err)
	}
	// Hands' proxy, on loopback, is reached directly; anywhere else, as brioche's own
	// environment says, so the client reaches the world as it would without brioche.
	// Hands' proxy unreached leaves every request of the brain's failing; another origin
	// unreached, only the request for it.
	forwarding := asSent(func(out *httputil.ProxyRequest) { out.SetURL(asked.to.URL()) }, transport(nil), sayer, failure)
	passing := asSent(func(*httputil.ProxyRequest) {}, transport(outer), sayer, warning)
	told := func(event intercept.Event) {
		if stopped, ok := event.(intercept.Stopped); ok {
			if ended.send(stop{"it stopped taking the client's requests", stopped.Err}) {
				return // run says it, as why brioche stops
			}
		}
		sayer.say(account(event))
	}
	door, err := intercept.Open(listener, intercept.Opening{Trusted: trusted, Intercepted: forwarding, Forwarded: passing, Proxy: outer, Seen: told})
	if err != nil {
		listener.Close()
		return nil, err
	}
	return door, nil
}

// theirs refuses a file the client's trust variable names relative to the client's
// directory, which brioche does not run in or know: the client's starter names it whole.
func theirs(named string, asked options) error {
	if named != "" && !filepath.IsAbs(named) {
		return fmt.Errorf("%s names %q, relative to the client's directory, which brioche does not know: the client's starter names it whole", asked.trust, named)
	}
	return nil
}

// account is what brioche says of what its door did. [LAW:nothing-unseen] every event,
// whatever the client was answered with besides: this is the one account of the run
// brioche keeps, and an event of a type this does not name is said as it is. How bad each
// is, is what it leaves the brain: a client that refuses brioche's TLS, or a door that
// stops, leaves every request of the brain's failing.
func account(event intercept.Event) report {
	switch event := event.(type) {
	case intercept.Opened:
		return report{Kind: "opened", Level: info, Target: event.Target, Said: fmt.Sprintf("answering the client's TLS for %s", event.Target)}
	case intercept.HandshakeRefused:
		return report{Kind: "handshake_refused", Level: failure, Target: event.Target, Said: fmt.Sprintf("the client refused brioche's answer for %s: %v", event.Target, event.Alert)}
	case intercept.HandshakeFailed:
		return report{Kind: "handshake_failed", Level: warning, Target: event.Target, Said: fmt.Sprintf("the client's TLS for %s ended before it began: %v", event.Target, event.Err)}
	case intercept.Answered:
		return report{Kind: "answered", Level: info, URL: event.URL, Status: &event.Status, Said: fmt.Sprintf("%s %s: answered %d", event.Route, event.URL, event.Status)}
	case intercept.Cut:
		return report{Kind: "cut", Level: warning, URL: event.URL, Status: &event.Status, Said: fmt.Sprintf("%s %s: cut short, answered %d", event.Route, event.URL, event.Status)}
	case intercept.Tunnelled:
		if event.Err != nil {
			return report{Kind: "tunnelled", Level: warning, Target: event.Target, Said: fmt.Sprintf("tunnelled to %s, cut short: %v", event.Target, event.Err)}
		}
		return report{Kind: "tunnelled", Level: info, Target: event.Target, Said: fmt.Sprintf("tunnelled to %s", event.Target)}
	case intercept.Unreached:
		return report{Kind: "unreached", Level: warning, Target: event.Target, Said: fmt.Sprintf("could not tunnel to %s: %v", event.Target, event.Err)}
	case intercept.Refused:
		return report{Kind: "refused", Level: warning, Said: fmt.Sprintf("refused %s, which is no proxy request: %v", event.Asked, event.Err)}
	case intercept.Logged:
		return report{Kind: "logged", Level: failure, Said: event.Line}
	case intercept.Stopped:
		return report{Kind: "stopped", Level: failure, Said: fmt.Sprintf("stopped taking the client's requests: %v", event.Err)}
	default:
		return report{Kind: "unnamed", Level: failure, Said: fmt.Sprintf("%T %+v", event, event)}
	}
}

func transport(proxy intercept.Proxy) *http.Transport {
	return &http.Transport{
		Proxy: proxy,
		// The answer's bytes as they were written, compressed or not: the client asked for
		// whatever encoding it asked for, and nothing is added to that.
		DisableCompression:  true,
		ForceAttemptHTTP2:   true,
		TLSHandshakeTimeout: 10 * time.Second,
		IdleConnTimeout:     90 * time.Second,
	}
}

// asSent is a proxy that sends each request on where `route` says, as the client sent it,
// the Host the client named among it, which a route to another origin would change. What it
// could not do, and an answer cut short as it gave up copying it, it says: an origin it
// could not reach as `unreached`.
func asSent(route func(*httputil.ProxyRequest), over *http.Transport, sayer *sayer, unreached level) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Rewrite: func(out *httputil.ProxyRequest) {
			route(out)
			intercept.AsSent(out)
			out.Out.Host = out.In.Host
		},
		Transport: over,
		// Each chunk reaches the client as it arrives: a streamed answer is the point.
		FlushInterval: -1,
		ErrorLog:      sayer.logger("proxy_logged", warning),
		ErrorHandler: func(writer http.ResponseWriter, request *http.Request, err error) {
			// The request as it was sent on, whose context is the client's.
			if request.Context().Err() != nil {
				sayer.sayf("abandoned", info, "the client gave up on %s before it was answered: %v", request.URL.Redacted(), err)
				return
			}
			sayer.say(report{Kind: "bad_gateway", Level: unreached, URL: request.URL.Redacted(), Said: fmt.Sprintf("could not reach %s: %v", request.URL.Redacted(), err)})
			http.Error(writer, fmt.Sprintf("brioche could not reach %s: %v", request.URL.Redacted(), err), http.StatusBadGateway)
		},
	}
}
