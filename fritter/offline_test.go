package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/bmf/cc-hands/fritter/internal/offline"
)

// The repository's one owner of the hold a test run is kept off the network by, and the
// profile it holds runs to. go test runs a package's tests in its own directory.
const (
	holds   = "../tests/offline.py"
	profile = "../tests/offline.sb"
)

// floor is the first macOS whose sandbox holds a run to the profile without refusing
// loopback, read from the `SANDBOXED_MACOS = <n>` line of what holds, as test.yml reads
// the pins it caches by. [LAW:one-source-of-truth] never a copy of it here.
func floor(said string) (int, error) {
	found := regexp.MustCompile(`(?m)^SANDBOXED_MACOS = (\d+)$`).FindStringSubmatch(said)
	if found == nil {
		return 0, fmt.Errorf("%s says no `SANDBOXED_MACOS = <n>` this can read", holds)
	}
	return strconv.Atoi(found[1])
}

// sandboxable refuses a macOS release before floor, saying why, never a run whose
// loopback connections fail at random.
func sandboxable(release string, floor int) error {
	major, err := strconv.Atoi(strings.Split(release, ".")[0])
	if err != nil {
		return fmt.Errorf("macOS's release %q names no major version: %v", release, err)
	}
	if major < floor {
		return fmt.Errorf("the run holds itself offline with macOS's sandbox, which refuses loopback at random before macOS %d: this is macOS %s", floor, release)
	}
	return nil
}

// held puts the test binary in the profile before the first test: fritter's tests fetch
// nothing, so the run is held from its start, and so is every child a test starts.
func held() error {
	said, err := os.ReadFile(holds)
	if err != nil {
		return err
	}
	lowest, err := floor(string(said))
	if err != nil {
		return err
	}
	release, err := syscall.Sysctl("kern.osproductversion")
	if err != nil {
		return fmt.Errorf("cannot read macOS's release: %v", err)
	}
	if err := sandboxable(release, lowest); err != nil {
		return err
	}
	held, err := os.ReadFile(profile)
	if err != nil {
		return err
	}
	if err := offline.Hold(held); err != nil {
		return fmt.Errorf("the run cannot hold itself to %s: %v", profile, err)
	}
	return nil
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
	if lowest, err := floor("OFFLINE = 1\nSANDBOXED_MACOS = 15\n"); err != nil || lowest != 15 {
		t.Errorf("the floor read %d, %v, want 15", lowest, err)
	}
	if _, err := floor("SANDBOXED_MACOS = fifteen\n"); err == nil || !strings.Contains(err.Error(), "says no `SANDBOXED_MACOS = <n>`") {
		t.Errorf("a floor nobody can read ended %v, want an error naming the line", err)
	}
}

func TestAMacOSAtTheFloorOrLaterIsHeld(t *testing.T) {
	for _, release := range []string{"15.0", "15.7.9", "26.5.2"} {
		if err := sandboxable(release, 15); err != nil {
			t.Errorf("macOS %s was refused: %v", release, err)
		}
	}
}

func TestAMacOSBeforeTheFloorStopsTheRunSayingSo(t *testing.T) {
	err := sandboxable("14.8.9", 15)
	if err == nil || !strings.HasSuffix(err.Error(), "refuses loopback at random before macOS 15: this is macOS 14.8.9") {
		t.Errorf("macOS 14.8.9 ended %v, want the refusal naming the floor", err)
	}
}
