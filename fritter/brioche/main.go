// Command brioche is the brain's HTTPS proxy: the brain's Claude Code reaches Anthropic
// through it, takes it for Anthropic's own API, and every exchange it has goes to hands'
// proxy instead.
//
//	brioche --intercept ORIGIN --to ORIGIN --trust-variable VARIABLE --trust-file FILE
//
// A CONNECT to the intercepted origin's host is answered over TLS under brioche's own
// certificate authority, and each request on it goes to the --to origin in plain HTTP as
// it was sent, its answer streamed back as it comes. Every other CONNECT is tunnelled to
// the host it names, through the proxy brioche's own environment names, if any.
//
// brioche writes, at FILE, the certificates the client is to trust: those its VARIABLE
// names in brioche's own environment, which is the client's, and brioche's authority. It
// then prints the environment that points a client at it, a NAME=VALUE line for each
// variable, and closes its standard output: whoever started it reads that to its end. It
// serves until its standard input ends or it is told to stop, so a brioche whose starter has
// gone goes too.
//
// brioche records nothing: hands' proxy, at the --to origin, is the one recorder of what
// passes through it.
//
// It is built from fritter's module, for the interception it shares with fritter's tap, and
// it is not fritter: fritter's binary links none of this, so a change here restarts no
// session.
package main

import (
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/bmf/cc-hands/fritter/intercept"
)

const usage = "usage: brioche --intercept ORIGIN --to ORIGIN --trust-variable VARIABLE --trust-file FILE"

const (
	failed = 1 // brioche could not start, or could not serve
	misuse = 64
)

func main() {
	// Armed first, so a stop that comes while brioche starts waits here, and is never the
	// default disposition that kills it with its trust file half written.
	killed := make(chan os.Signal, 1)
	signal.Notify(killed, syscall.SIGTERM, syscall.SIGINT, syscall.SIGHUP)
	read := make(chan struct{})
	go func() {
		// [LAW:no-silent-failure] exception: an error reading stdin ends it as its end does:
		// either way nobody is left to send brioche anything, and it stops as asked.
		io.Copy(io.Discard, os.Stdin)
		close(read)
	}()
	ended := make(chan struct{})
	go func() {
		select {
		case <-killed:
		case <-read:
		}
		close(ended)
	}()
	os.Exit(run(os.Args[1:], os.Getenv, os.Stdout, os.Stderr, ended))
}

// options is what brioche is asked to do, each part of it parsed.
type options struct {
	intercepted intercept.Origin
	to          intercept.Origin
	trust       intercept.TrustVariable
	file        string
}

// parse reads brioche's arguments. [LAW:parse-dont-validate] each flag given once, with a
// value, and all of them: a flag left empty or given twice is refused, never taken for one
// not given or for its last spelling.
func parse(args []string) (options, error) {
	given := map[string]string{}
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--intercept", "--to", "--trust-variable", "--trust-file":
			flag := args[i]
			if i+1 >= len(args) || args[i+1] == "" {
				return options{}, fmt.Errorf("%s needs a value", flag)
			}
			if _, twice := given[flag]; twice {
				return options{}, fmt.Errorf("%s is given twice", flag)
			}
			given[flag] = args[i+1]
			i++
		default:
			return options{}, fmt.Errorf("unknown argument %q", args[i])
		}
	}
	for _, flag := range []string{"--intercept", "--to", "--trust-variable", "--trust-file"} {
		if _, ok := given[flag]; !ok {
			return options{}, fmt.Errorf("%s is needed", flag)
		}
	}
	intercepted, err := intercept.ParseOrigin(given["--intercept"])
	if err != nil {
		return options{}, fmt.Errorf("--intercept: %w", err)
	}
	to, err := intercept.ParseOrigin(given["--to"])
	if err != nil {
		return options{}, fmt.Errorf("--to: %w", err)
	}
	trust, err := intercept.ParseTrustVariable(given["--trust-variable"])
	if err != nil {
		return options{}, fmt.Errorf("--trust-variable: %w", err)
	}
	return options{intercepted: intercepted, to: to, trust: trust, file: given["--trust-file"]}, nil
}

// run is brioche, with what it reaches of its process passed in: the environment it reads
// the client's trust from, where it prints the client's environment, which it closes once
// printed, where it says what it must, and `ended`, closed when it is to stop.
func run(args []string, client func(string) string, printed io.WriteCloser, said io.Writer, ended <-chan struct{}) int {
	// A logger, so lines said from the door's goroutines at once are never interleaved.
	say := log.New(said, "brioche: ", 0).Printf
	asked, err := parse(args)
	if err != nil {
		say("%v\n%s", err, usage)
		return misuse
	}
	door, err := open(asked, client, say)
	if err != nil {
		say("%v", err)
		return failed
	}
	defer door.Close()
	for _, variable := range door.Environment() {
		if _, err := fmt.Fprintln(printed, variable); err != nil {
			say("cannot print the environment that points a client at brioche: %v", err)
			return failed
		}
	}
	if err := printed.Close(); err != nil {
		say("cannot end the environment that points a client at brioche: %v", err)
		return failed
	}
	<-ended
	return 0
}

// open is brioche's door, serving on a loopback port of its own.
func open(asked options, client func(string) string, say func(string, ...any)) (*intercept.Door, error) {
	interception, err := intercept.Intercept("brioche", asked.intercepted, time.Now())
	if err != nil {
		return nil, fmt.Errorf("cannot make the certificate the client is answered with for %s: %w", asked.intercepted, err)
	}
	trusted, err := interception.Trust(asked.trust, client, asked.file, func(unread error) { say("%v", unread) })
	if err != nil {
		return nil, err
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("cannot listen for the client's requests: %w", err)
	}
	// [LAW:no-silent-failure] every error the proxies meet reaches the client, as a 502 that
	// names brioche or an answer cut short; what is brioche's own fault, `told` says.
	quiet := log.New(io.Discard, "", 0)
	// [LAW:one-source-of-truth] the proxy brioche was itself given, if any, for what it
	// forwards and what it tunnels alike: the client reaches the world as it would without
	// brioche. Loopback, where hands' proxy listens, it never names.
	outer := http.ProxyFromEnvironment
	onward := &http.Transport{
		Proxy: outer,
		// The answer's bytes as they were written, compressed or not: the client asked for
		// whatever encoding it asked for, and nothing is added to that.
		DisableCompression:  true,
		ForceAttemptHTTP2:   true,
		TLSHandshakeTimeout: 10 * time.Second,
		IdleConnTimeout:     90 * time.Second,
	}
	unreached := func(writer http.ResponseWriter, request *http.Request, err error) {
		http.Error(writer, fmt.Sprintf("brioche could not reach %s: %v", request.URL.Redacted(), err), http.StatusBadGateway)
	}
	forwarding := &httputil.ReverseProxy{
		Rewrite: func(out *httputil.ProxyRequest) {
			out.SetURL(asked.to.URL())
			// As sent: the Host the client named, which SetURL would make the --to origin's.
			out.Out.Host = out.In.Host
		},
		Transport: onward,
		// Each chunk reaches the client as it arrives: a streamed answer is the point.
		FlushInterval: -1,
		ErrorHandler:  unreached,
		ErrorLog:      quiet,
	}
	// What the client asks of anywhere else, in plain HTTP, goes on as it was asked.
	passing := &httputil.ReverseProxy{Rewrite: func(*httputil.ProxyRequest) {}, Transport: onward, FlushInterval: -1, ErrorHandler: unreached, ErrorLog: quiet}
	told := func(event intercept.Event) {
		// The client refusing brioche's TLS leaves its every request to the origin failing
		// with nothing to answer it that, so brioche says it; and so it does of what its
		// servers logged, and of a server that stopped, faults of brioche's own. Everything
		// else the client was answered with, or did itself. [LAW:nothing-unseen]
		switch event := event.(type) {
		case intercept.HandshakeRefused:
			say("the client refused brioche's answer for %s: %v", event.Target, event.Alert)
		case intercept.Logged:
			say("%s", event.Line)
		case intercept.Stopped:
			say("stopped taking the client's requests: %v", event.Err)
		}
	}
	door, err := intercept.Open(listener, intercept.Opening{Trusted: trusted, Intercepted: forwarding, Forwarded: passing, Proxy: outer, Seen: told})
	if err != nil {
		listener.Close()
		return nil, err
	}
	return door, nil
}
