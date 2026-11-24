package testrun

import (
	"context"
	"errors"
	"io/fs"
	"net"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	os.Exit(Main(m))
}

func TestAnEditToTheHoldIsNeverACachedPass(t *testing.T) {
	if _, err := Hold(); err != nil {
		t.Error(err)
	}
}

// [LAW:single-enforcer] the hold is Main's, so a package of the module whose tests do not
// run under it runs them unheld: every one does, and reads the hold again.
func TestEveryPackageOfTheModuleRunsItsTestsUnderMainAndReadsTheHold(t *testing.T) {
	root, err := Root()
	if err != nil {
		t.Fatal(err)
	}
	tested := map[string]string{}
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() || !strings.HasSuffix(path, "_test.go") {
			return err
		}
		said, err := os.ReadFile(path)
		tested[filepath.Dir(path)] += string(said)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	for dir, said := range tested {
		if !strings.Contains(said, "Main(m)") || !strings.Contains(said, "Hold()") {
			t.Errorf("the tests in %s do not run under testrun.Main, or do not read testrun.Hold", dir)
		}
	}
}

// 192.0.2.1 is TEST-NET-1 (RFC 5737), routed nowhere: outside the sandbox a connection to
// it times out, inside it the OS refuses it at once.
func TestAConnectionATestMakesPastLoopbackIsRefusedByTheOS(t *testing.T) {
	if _, err := net.DialTimeout("tcp", "192.0.2.1:443", 5*time.Second); !errors.Is(err, syscall.EPERM) {
		t.Errorf("a connection past loopback ended %v, want the OS's EPERM", err)
	}
}

func TestANameOffThisMachineIsNotResolved(t *testing.T) {
	if addresses, err := net.DefaultResolver.LookupHost(context.Background(), "example.com"); err == nil {
		t.Errorf("example.com resolved to %v", addresses)
	}
}

func TestTheFloorIsReadFromTheOneOwnerOfTheHold(t *testing.T) {
	if lowest, err := Floor("OFFLINE = 1\nSANDBOXED_MACOS = 15\n"); err != nil || lowest != 15 {
		t.Errorf("the floor read %d, %v, want 15", lowest, err)
	}
	if _, err := Floor("SANDBOXED_MACOS = fifteen\n"); err == nil || !strings.Contains(err.Error(), "says no `SANDBOXED_MACOS = <n>`") {
		t.Errorf("a floor nobody can read ended %v, want an error naming the line", err)
	}
}

func TestAMacOSAtTheFloorOrLaterIsHeld(t *testing.T) {
	for _, release := range []string{"15.0", "15.7.9", "26.5.2"} {
		if err := Sandboxable(release, 15); err != nil {
			t.Errorf("macOS %s was refused: %v", release, err)
		}
	}
}

func TestAMacOSBeforeTheFloorStopsTheRunSayingSo(t *testing.T) {
	err := Sandboxable("14.8.9", 15)
	if err == nil || !strings.HasSuffix(err.Error(), "refuses loopback at random before macOS 15: this is macOS 14.8.9") {
		t.Errorf("macOS 14.8.9 ended %v, want the refusal naming the floor", err)
	}
}

func TestATestRunsInTheHermeticEnvironment(t *testing.T) {
	home := os.Getenv("HOME")
	var want []string
	for name, value := range Hermetic(os.LookupEnv, home) {
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
