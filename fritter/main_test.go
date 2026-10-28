package main

import (
	"fmt"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
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
		os.Exit(tested(m))
	case "fritter":
		os.Exit(run(os.Args[1:], os.Stdin, os.Stdout))
	case "recorder":
		os.Exit(recorder(os.Args[1:]))
	default:
		fmt.Fprintf(os.Stderr, "%s=%q names no program\n", program, named)
		os.Exit(2)
	}
}

// hermetic is the environment the tests run in, which every child they start inherits:
// the machine's TMPDIR, where os.TempDir puts what fritter makes, as it does under hands,
// the system's tools, all a child here needs (sh, curl, head and the test binary), and a
// HOME of their own. Nothing else of the machine's reaches them: no credential, no proxy,
// no certificates the tap would hand on, no Claude Code session's names, no ~/.curlrc.
func hermetic(tmpdir, home string) map[string]string {
	return map[string]string{"TMPDIR": tmpdir, "PATH": "/usr/bin:/bin", "HOME": home}
}

// as starts the test binary as the program named, with args, in the test's environment.
func as(named string, args ...string) *exec.Cmd {
	child := exec.Command(os.Args[0], args...)
	child.Env = append(os.Environ(), program+"="+named)
	return child
}

// tested runs the tests in the hermetic environment, with a HOME that holds nothing and
// that nothing may write to: every test shares it, so a test that writes there fails at
// the write, by its own name, rather than leaving something the next test finds.
//
// [LAW:single-enforcer] TestMain is the one owner of the environment a test runs in: a
// test that needs a name sets it on top, with t.Setenv or its child's Env. It is set
// before the first test, so http.ProxyFromEnvironment, which reads the environment once
// and keeps what it read, reads this one.
func tested(m *testing.M) int {
	home, err := os.MkdirTemp("/tmp", "fritter-home-")
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot make the tests a HOME: %v\n", err)
		return 1
	}
	defer os.Remove(home)
	if err := os.Chmod(home, 0o500); err != nil {
		fmt.Fprintf(os.Stderr, "cannot make the tests' HOME read-only: %v\n", err)
		return 1
	}
	names := hermetic(os.Getenv("TMPDIR"), home)
	os.Clearenv()
	for name, value := range names {
		os.Setenv(name, value)
	}
	return m.Run()
}

func TestATestRunsInTheHermeticEnvironment(t *testing.T) {
	home := os.Getenv("HOME")
	var want []string
	for name, value := range hermetic(os.Getenv("TMPDIR"), home) {
		want = append(want, name+"="+value)
	}
	environment := os.Environ()
	slices.Sort(environment)
	slices.Sort(want)
	if !slices.Equal(environment, want) || !strings.HasPrefix(home, "/tmp/fritter-home-") {
		t.Errorf("a test runs with %q, want %q with a HOME of the tests' own", environment, want)
	}
	if err := os.WriteFile(home+"/.curlrc", nil, 0o600); err == nil {
		t.Errorf("a test wrote to the tests' HOME, which the next test would find")
	}
}

func TestAProgramNobodyNamedIsRefused(t *testing.T) {
	said, err := as("nonesuch").CombinedOutput()
	if exit, ended := err.(*exec.ExitError); !ended || exit.ExitCode() != 2 || !strings.Contains(string(said), `"nonesuch" names no program`) {
		t.Errorf("a child told to be nonesuch ended %v saying %q, want exit 2 naming it", err, said)
	}
}
