// Package testrun is the one owner of what every test binary of fritter's module runs
// under: held offline and away from the machine's Claude Code policy, in an environment of
// its own. Each package's TestMain is Main, so no package's tests run unheld, and each has
// a test that reads Hold, so an edit to the hold runs it again.
package testrun

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"testing"

	"github.com/bmf/cc-hands/fritter/internal/offline"
)

// The repository's one owner of the hold a test run is kept to, and the profile it holds
// runs to, each a link from the module's testdata/ to the repository's tests/: go test
// runs a package's tests in its own directory, and rechecks no file outside the module
// before it reports a pass it cached.
const (
	holds   = "testdata/offline.py"
	profile = "testdata/offline.sb"
)

// Root is the module's own directory, found from the package directory go test runs a
// package's tests in.
func Root() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", errors.New("no go.mod above the directory the tests run in")
		}
		dir = parent
	}
}

// Floor is the first macOS whose sandbox holds a run to the profile without refusing
// loopback, read from the `SANDBOXED_MACOS = <n>` line of what holds, as test.yml reads
// the pins it caches by. [LAW:one-source-of-truth] never a copy of it here.
func Floor(said string) (int, error) {
	found := regexp.MustCompile(`(?m)^SANDBOXED_MACOS = (\d+)$`).FindStringSubmatch(said)
	if found == nil {
		return 0, fmt.Errorf("%s says no `SANDBOXED_MACOS = <n>` this can read", holds)
	}
	return strconv.Atoi(found[1])
}

// Sandboxable refuses a macOS release before floor, saying why, never a run whose
// loopback connections fail at random.
func Sandboxable(release string, floor int) error {
	major, err := strconv.Atoi(strings.Split(release, ".")[0])
	if err != nil {
		return fmt.Errorf("macOS's release %q names no major version: %v", release, err)
	}
	if major < floor {
		return fmt.Errorf("the run holds itself offline with macOS's sandbox, which refuses loopback at random before macOS %d: this is macOS %s", floor, release)
	}
	return nil
}

// Hold is the profile the run is held to, once this macOS is known to apply it faithfully.
// go test keeps a passing result until a file its tests opened changes, and Main reads
// the hold before go test records what is opened: each package's tests read it again, so
// an edit to the profile or the floor runs them again, never a pass it cached.
func Hold() ([]byte, error) {
	root, err := Root()
	if err != nil {
		return nil, err
	}
	said, err := os.ReadFile(filepath.Join(root, holds))
	if err != nil {
		return nil, err
	}
	lowest, err := Floor(string(said))
	if err != nil {
		return nil, err
	}
	release, err := syscall.Sysctl("kern.osproductversion")
	if err != nil {
		return nil, fmt.Errorf("cannot read macOS's release: %v", err)
	}
	if err := Sandboxable(release, lowest); err != nil {
		return nil, err
	}
	return os.ReadFile(filepath.Join(root, profile))
}

// Hermetic is the environment the tests run in, which every child they start inherits:
// the TMPDIR `machine` has, where os.TempDir puts what fritter makes, as it does under
// hands (none where the machine has none, so os.TempDir falls back to /tmp as it does
// there, never to an empty name), the system's tools, all a child here needs (sh, curl, head and the test binary), and a
// HOME of their own. Nothing else of the machine's reaches them: no credential, no proxy,
// no certificates the tap would hand on, no Claude Code session's names, no ~/.curlrc.
func Hermetic(machine func(string) (string, bool), home string) map[string]string {
	names := map[string]string{"PATH": "/usr/bin:/bin", "HOME": home}
	if tmpdir, ok := machine("TMPDIR"); ok {
		names["TMPDIR"] = tmpdir
	}
	return names
}

// Main runs the tests held offline and in the hermetic environment, with a HOME that holds
// nothing and that nothing may write to: every test shares it, so a test that writes there
// fails at the write, by its own name, rather than leaving something the next test finds.
//
// [LAW:single-enforcer] Main is the one owner of the environment a test runs in: a test
// that needs a name sets it on top, with t.Setenv or its child's Env. It is set before the
// first test, so http.ProxyFromEnvironment, which reads the environment once and keeps
// what it read, reads this one.
func Main(m *testing.M) int {
	// Held first, so nothing the run does, making its HOME included, is unheld: fritter's
	// tests fetch nothing, so the run is held from its start, and so is every child a test
	// starts.
	rules, err := Hold()
	if err == nil {
		err = offline.Hold(rules)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "the run cannot hold itself to %s: %v\n", profile, err)
		return 1
	}
	home, err := os.MkdirTemp("/tmp", "fritter-home-")
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot make the tests a HOME: %v\n", err)
		return 1
	}
	if err := os.Chmod(home, 0o500); err != nil {
		fmt.Fprintf(os.Stderr, "cannot make the tests' HOME read-only: %v\n", err)
		return 1
	}
	names := Hermetic(os.LookupEnv, home)
	os.Clearenv()
	for name, value := range names {
		os.Setenv(name, value)
	}
	code := m.Run()
	// [LAW:no-silent-failure] a HOME that will not go held something a test wrote, which is a
	// failure of the run, however its tests went.
	if err := os.Remove(home); err != nil {
		fmt.Fprintf(os.Stderr, "the tests' HOME was left holding something: %v\n", err)
		return 1
	}
	return code
}
