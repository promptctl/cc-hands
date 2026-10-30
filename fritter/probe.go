package main

// PROBE for promptctl-testing-px1f.6i5, never merged: every byte typed into the child and every
// chunk it prints, with the time and whether its terminal was in canonical mode at that moment.

import (
	"fmt"
	"os"
	"sync"
	"time"

	"golang.org/x/sys/unix"
)

type tracer struct {
	mu     sync.Mutex
	out    *os.File
	master *os.File
}

func newTracer(master *os.File) *tracer {
	_ = os.MkdirAll("/tmp/fritter-trace", 0o755)
	out, err := os.Create(fmt.Sprintf("/tmp/fritter-trace/%d-%d.log", os.Getpid(), time.Now().UnixNano()))
	if err != nil {
		warn("probe: %v", err)
	}
	return &tracer{out: out, master: master}
}

func (t *tracer) line(kind string, b []byte) {
	if t == nil || t.out == nil {
		return
	}
	mode := "?"
	if conn, err := t.master.SyscallConn(); err == nil {
		_ = conn.Control(func(fd uintptr) {
			if tio, err := unix.IoctlGetTermios(int(fd), unix.TIOCGETA); err == nil {
				mode = map[bool]string{true: "cooked", false: "raw"}[tio.Lflag&unix.ICANON != 0]
			}
		})
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	fmt.Fprintf(t.out, "%s %s %s %q\n", time.Now().Format("15:04:05.000000"), kind, mode, b)
}

func (t *tracer) Write(b []byte) (int, error) {
	t.line("out", b)
	return len(b), nil
}
