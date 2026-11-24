package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/bmf/cc-hands/fritter/testrun"
)

// program names what the test binary is when a test starts it as a child, as Go's own
// os/exec tests start theirs: fritter's own main, for what holds only once the program
// has ended, which nothing that keeps running can show, and the recorder, a child of
// Claude Code's shape as far as input goes. Unset, the binary runs the tests. A program
// takes its arguments as its own: TestMain runs before the testing flags are parsed.
const program = "FRITTER_TEST_PROGRAM"

func TestMain(m *testing.M) {
	switch named := os.Getenv(program); named {
	case "":
		os.Exit(testrun.Main(m))
	case "fritter":
		os.Exit(run(os.Args[1:], os.Stdin, os.Stdout))
	case "recorder":
		os.Exit(recorder(os.Args[1:]))
	default:
		fmt.Fprintf(os.Stderr, "%s=%q names no program\n", program, named)
		os.Exit(2)
	}
}

// as starts the test binary as the program named, with args, in the test's environment.
func as(named string, args ...string) *exec.Cmd {
	child := exec.Command(os.Args[0], args...)
	child.Env = append(os.Environ(), program+"="+named)
	return child
}

// Each test package reads the hold again, as testrun.Hold says why.
func TestAnEditToTheHoldIsNeverACachedPass(t *testing.T) {
	if _, err := testrun.Hold(); err != nil {
		t.Error(err)
	}
}

func TestAProgramNobodyNamedIsRefused(t *testing.T) {
	said, err := as("nonesuch").CombinedOutput()
	if exit, ended := err.(*exec.ExitError); !ended || exit.ExitCode() != 2 || !strings.Contains(string(said), `"nonesuch" names no program`) {
		t.Errorf("a child told to be nonesuch ended %v saying %q, want exit 2 naming it", err, said)
	}
}
