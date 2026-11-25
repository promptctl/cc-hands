// Command brioche is the brain's HTTPS proxy: the brain's Claude Code reaches Anthropic
// through it, takes it for Anthropic's own API, and every exchange it has goes to hands'
// proxy instead.
//
//	brioche --intercept ORIGIN --to ORIGIN --trust-variable VARIABLE --trust-file FILE
//
// A CONNECT to the intercepted origin's host is answered over TLS under brioche's own
// certificate authority, and each request on it goes to the --to origin in plain HTTP as
// it was sent, its answer streamed back as it comes. Every other CONNECT is tunnelled to
// the host it names, and every other plain request goes on as it was asked, through the
// proxy brioche's own environment names, if any.
//
// brioche writes, at FILE, the certificates the client is to trust: those its VARIABLE
// names in brioche's own environment, which is the client's, and brioche's authority. It
// then prints the environment that points a client at it, a NAME=VALUE line for each
// variable, and closes its standard output: whoever started it reads that to its end. It
// serves until its standard input ends or it is told to stop, so a brioche whose starter has
// gone goes too, and it says why it stopped.
//
// brioche records nothing: hands' proxy, at the --to origin, is the one recorder of what
// passes through it. What never reaches that recorder, a request brioche could not take
// there or an answer from there cut short, brioche says.
//
// It is built from fritter's module, for the interception it shares with fritter's tap, and
// it is not fritter: fritter's binary links none of this, so a change here restarts no
// session. What brioche does with a request it took is its own for that reason, though
// fritter's tap does much the same: shared, it would be a change to fritter each time.
package main

import (
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

const usage = "usage: brioche --intercept ORIGIN --to ORIGIN --trust-variable VARIABLE --trust-file FILE"

const (
	failed = 1 // brioche could not start, or could not go on serving
	misuse = 64
)

func main() {
	os.Exit(serve(os.Args[1:]))
}

// serve is brioche in its process: its arguments, its environment, its standard streams,
// and the signals it is sent.
func serve(args []string) int {
	// Armed first, so a stop that comes while brioche starts waits here, and is never the
	// default disposition that kills it with its trust file half written.
	killed := make(chan os.Signal, 1)
	signal.Notify(killed, syscall.SIGTERM, syscall.SIGINT, syscall.SIGHUP)
	// Room for both: whichever comes second is never waited on.
	ended := make(chan stop, 2)
	go func() {
		_, err := io.Copy(io.Discard, os.Stdin)
		if err != nil {
			ended <- stop{"its input failed", err}
			return
		}
		ended <- stop{"its input ended", nil}
	}()
	go func() {
		told := <-killed
		// A second signal takes its default course: a stop that stalls is never one that
		// nothing short of SIGKILL ends.
		signal.Stop(killed)
		ended <- stop{"told to by " + told.String(), nil}
	}()
	return run(args, os.Getenv, http.ProxyFromEnvironment, os.Stdout, os.Stderr, ended)
}

// stop is why brioche is to stop: its input ended or failed, or it was told to. Err is
// what failed, nil for a stop asked for.
type stop struct {
	why string
	err error
}

// The flags brioche takes, every one of them needed. [LAW:one-source-of-truth] the one list,
// for what is taken and what is needed.
var flags = []string{"--intercept", "--to", "--trust-variable", "--trust-file"}

// options is what brioche is asked to do, each part of it parsed.
type options struct {
	intercepted intercept.Origin
	to          intercept.Origin
	trust       intercept.TrustVariable
	file        string // absolute, so a client in any directory finds it
}

// parse reads brioche's arguments. [LAW:parse-dont-validate] each flag given once, with a
// value, and all of them: a flag left empty or given twice is refused, never taken for one
// not given or for its last spelling.
func parse(args []string) (options, error) {
	given := map[string]string{}
	for i := 0; i < len(args); i++ {
		flag := args[i]
		if !slices.Contains(flags, flag) {
			return options{}, fmt.Errorf("unknown argument %q", flag)
		}
		if i+1 >= len(args) || args[i+1] == "" {
			return options{}, fmt.Errorf("%s needs a value", flag)
		}
		if _, twice := given[flag]; twice {
			return options{}, fmt.Errorf("%s is given twice", flag)
		}
		given[flag] = args[i+1]
		i++
	}
	for _, flag := range flags {
		if _, ok := given[flag]; !ok {
			return options{}, fmt.Errorf("%s is needed", flag)
		}
	}
	intercepted, err := intercept.ParseOrigin(given["--intercept"])
	if err != nil {
		return options{}, fmt.Errorf("--intercept: %w", err)
	}
	to, err := origin(given["--to"])
	if err != nil {
		return options{}, fmt.Errorf("--to: %w", err)
	}
	trust, err := intercept.ParseTrustVariable(given["--trust-variable"])
	if err != nil {
		return options{}, fmt.Errorf("--trust-variable: %w", err)
	}
	file, err := filepath.Abs(given["--trust-file"])
	if err == nil && strings.ContainsAny(file, "\n\r") {
		err = errors.New("a path printed on a line of its own holds no line break")
	}
	if err != nil {
		return options{}, fmt.Errorf("--trust-file: %w", err)
	}
	return options{intercepted: intercepted, to: to, trust: trust, file: file}, nil
}

// origin is what --to names: an origin and no more, since each request goes to the path it
// was sent to, and a path or query here would be dropped.
func origin(raw string) (intercept.Origin, error) {
	if named, err := url.Parse(raw); err == nil && (strings.TrimSuffix(named.Path, "/") != "" || named.RawQuery != "" || named.Fragment != "") {
		return intercept.Origin{}, fmt.Errorf("%q names more than an origin: each request goes to the path it was sent to", raw)
	}
	return intercept.ParseOrigin(raw)
}

// run is brioche, with what it reaches of its process passed in: the environment it reads
// the client's trust from; how it reaches the world, `outer`; where it prints the client's
// environment, which it closes however it returns; where it says what it must; and `ended`,
// which says when it is to stop and why.
func run(args []string, client func(string) string, outer intercept.Proxy, printed io.WriteCloser, said io.Writer, ended <-chan stop) int {
	closePrinted := sync.OnceValue(printed.Close)
	defer closePrinted()
	// A logger, so lines said from the door's goroutines at once are never interleaved.
	sayer := log.New(said, "brioche: ", 0)
	asked, err := parse(args)
	if err != nil {
		sayer.Printf("%v\n%s", err, usage)
		return misuse
	}
	// Room for both of the door's servers: a second that stops is never waited on.
	stopped := make(chan error, 2)
	door, err := open(asked, client, outer, sayer, stopped)
	if err != nil {
		sayer.Print(err)
		return failed
	}
	defer door.Close()
	for _, variable := range door.Environment() {
		if _, err := fmt.Fprintln(printed, variable); err != nil {
			sayer.Printf("cannot print the environment that points a client at brioche: %v", err)
			return failed
		}
	}
	if err := closePrinted(); err != nil {
		sayer.Printf("cannot end the environment that points a client at brioche: %v", err)
		return failed
	}
	select {
	case why := <-ended:
		if why.err != nil {
			sayer.Printf("stopping: %s: %v", why.why, why.err)
			return failed
		}
		sayer.Printf("stopping: %s", why.why)
		return 0
	case err := <-stopped:
		// [LAW:no-silent-failure] a door that stopped serving leaves every request refused:
		// brioche ends, so its starter sees it gone, never a brioche alive and deaf.
		sayer.Printf("stopping: it stopped taking the client's requests: %v", err)
		return failed
	}
}

// open is brioche's door, serving on a loopback port of its own. A server of its that stops
// serving is sent on `stopped`.
func open(asked options, client func(string) string, outer intercept.Proxy, sayer *log.Logger, stopped chan<- error) (*intercept.Door, error) {
	interception, err := intercept.Intercept("brioche", asked.intercepted, time.Now())
	if err != nil {
		return nil, fmt.Errorf("cannot make the certificate the client is answered with for %s: %w", asked.intercepted, err)
	}
	theirs, err := clientTrusts(client(asked.trust.String()), asked, sayer)
	if err != nil {
		return nil, err
	}
	trusted, err := interception.Trust(asked.trust, func(string) string { return theirs }, asked.file, func(unread error) { sayer.Print(unread) })
	if err != nil {
		return nil, err
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("cannot listen for the client's requests: %w", err)
	}
	// Hands' proxy, on loopback, is reached directly; anywhere else, as brioche's own
	// environment says, so the client reaches the world as it would without brioche.
	direct := transport(nil)
	onward := transport(outer)
	forwarding := asSent(func(out *httputil.ProxyRequest) { out.SetURL(asked.to.URL()) }, direct)
	// [LAW:nothing-unseen] what never reaches the recorder is said: a request brioche could
	// not take to it, and an answer from it cut short, which ReverseProxy logs as it gives up.
	forwarding.ErrorLog = sayer
	forwarding.ErrorHandler = func(writer http.ResponseWriter, request *http.Request, err error) {
		sayer.Printf("could not reach %s for %s: %v", asked.to, request.URL.RequestURI(), err)
		unreached(writer, request, err)
	}
	// What the client asks of anywhere else, in plain HTTP, goes on as it was asked; what
	// fails there the client is answered with, and it is no exchange of the recorder's.
	passing := asSent(func(*httputil.ProxyRequest) {}, onward)
	passing.ErrorHandler = unreached
	told := func(event intercept.Event) {
		// The client refusing brioche's TLS leaves its every request to the origin failing
		// with nothing to answer it that, so brioche says it; and so it does of what its
		// servers logged, and of a server that stopped, faults of brioche's own. Everything
		// else the client was answered with, or did itself. [LAW:nothing-unseen]
		switch event := event.(type) {
		case intercept.HandshakeRefused:
			sayer.Printf("the client refused brioche's answer for %s: %v", event.Target, event.Alert)
		case intercept.Logged:
			sayer.Print(event.Line)
		case intercept.Stopped:
			stopped <- event.Err
		}
	}
	door, err := intercept.Open(listener, intercept.Opening{Trusted: trusted, Intercepted: forwarding, Forwarded: passing, Proxy: outer, Seen: told})
	if err != nil {
		listener.Close()
		return nil, err
	}
	return door, nil
}

// clientTrusts is the file of certificates the client's trust variable names, as brioche
// can read it: none where it names none.
func clientTrusts(named string, asked options, sayer *log.Logger) (string, error) {
	switch {
	case named == "":
		return "", nil
	case !filepath.IsAbs(named):
		// The client reads a relative path from its own directory, which brioche does not
		// run in or know: as Node does with a file it cannot read, it goes on without them.
		sayer.Printf("cannot read the certificates %s names, %q: it is relative to the client's directory, which brioche does not know, so the client is not given them", asked.trust, named)
		return "", nil
	case filepath.Clean(named) == asked.file:
		// The file a brioche gave the client before: what it holds besides that brioche's
		// authority is not known, and that authority's key is gone.
		return "", fmt.Errorf("%s names %s, the file brioche writes: the client is to be given the certificates it trusts besides brioche's", asked.trust, named)
	}
	return named, nil
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

// asSent is a proxy that sends each request on where `route` says, as the client sent it:
// ReverseProxy drops the client's forwarding headers and rewrites a query it cannot parse
// before `route` sees it, which this puts back, and keeps the Host the client named, which
// a route to another origin would change.
func asSent(route func(*httputil.ProxyRequest), over *http.Transport) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Rewrite: func(out *httputil.ProxyRequest) {
			route(out)
			out.Out.Host = out.In.Host
			out.Out.URL.RawQuery = out.In.URL.RawQuery
			for _, name := range []string{"Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto"} {
				if values, sent := out.In.Header[name]; sent {
					out.Out.Header[name] = values
				}
			}
		},
		Transport: over,
		// Each chunk reaches the client as it arrives: a streamed answer is the point.
		FlushInterval: -1,
		ErrorLog:      log.New(io.Discard, "", 0),
	}
}

// unreached answers a request brioche could not take where it was going with a 502 that
// names brioche, and why.
func unreached(writer http.ResponseWriter, request *http.Request, err error) {
	http.Error(writer, fmt.Sprintf("brioche could not reach %s: %v", request.URL.Redacted(), err), http.StatusBadGateway)
}
