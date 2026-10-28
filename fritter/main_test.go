package main

import (
	"fmt"
	"os"
	"os/exec"
	"os/user"
	"slices"
	"strings"
	"testing"
)

// program names what the test binary is when a test starts it as a child, as Go's own
// os/exec tests start theirs: fritter's own main, for what holds only once the program
// has ended, which nothing that keeps running can show, and the recorder, a child of
// Claude Code's shape as far as input goes. Unset, the binary runs the tests.
const program = "FRITTER_TEST_PROGRAM"

func TestMain(m *testing.M) {
	switch named := os.Getenv(program); named {
	case "":
		os.Exit(tested(m))
	case "fritter":
		os.Exit(run(strings.Split(os.Getenv("FRITTER_TEST_ARGS"), "\x1f"), os.Stdin, os.Stdout))
	case "recorder":
		os.Exit(recorder())
	default:
		fmt.Fprintf(os.Stderr, "%s=%q names no program\n", program, named)
		os.Exit(2)
	}
}

// hermetic is the environment the tests run in, which every child they start inherits:
// the system's tools, all a child here needs (sh, curl, head and the test binary), and a
// HOME of their own. Nothing else of the machine's reaches them: no credential, no proxy,
// no certificates the tap would hand on, no Claude Code session's names, no ~/.curlrc.
func hermetic(home string) map[string]string {
	return map[string]string{"PATH": "/usr/bin:/bin", "HOME": home}
}

// tested runs the tests in the hermetic environment, with a HOME that holds nothing and
// must hold nothing once they have run, since every test shares it.
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
	defer os.RemoveAll(home)
	os.Clearenv()
	for name, value := range hermetic(home) {
		os.Setenv(name, value)
	}
	code := m.Run()
	left, err := os.ReadDir(home)
	if err != nil || len(left) > 0 {
		fmt.Fprintf(os.Stderr, "the tests' HOME holds %v (%v), which the next test to read it finds\n", left, err)
		return 1
	}
	return code
}

func TestATestRunsInTheHermeticEnvironment(t *testing.T) {
	home := os.Getenv("HOME")
	environment := os.Environ()
	slices.Sort(environment)
	if want := []string{"HOME=" + home, "PATH=/usr/bin:/bin"}; !slices.Equal(environment, want) {
		t.Errorf("a test runs with %q, want %q", environment, want)
	}
	// The directory service, not HOME, says where the developer's home is.
	developer, err := user.Current()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(home, "/tmp/fritter-home-") || home == developer.HomeDir {
		t.Errorf("a test's HOME is %q, want one of the tests' own, not %q", home, developer.HomeDir)
	}
}

func TestAProgramNobodyNamedIsRefused(t *testing.T) {
	child := exec.Command(os.Args[0])
	child.Env = append(os.Environ(), program+"=nonesuch")
	said, err := child.CombinedOutput()
	if exit, ended := err.(*exec.ExitError); !ended || exit.ExitCode() != 2 || !strings.Contains(string(said), `"nonesuch" names no program`) {
		t.Errorf("a child told to be nonesuch ended %v saying %q, want exit 2 naming it", err, said)
	}
}
