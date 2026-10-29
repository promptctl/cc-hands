// Package offline holds a process in a macOS sandbox profile, as the repository's
// tests/offline.py holds hands' and wake-training's test runs: sandbox_init, the call
// sandbox-exec makes, on the process itself. It is cgo, which Go refuses in a _test.go
// file, so it is a package only fritter's tests import; fritter's binary is built
// without cgo and never links it. It has no tests of its own: a test binary of its own
// would run unheld, and fritter's held proof tests show what it does.
package offline

/*
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
int sandbox_init(const char *profile, uint64_t flags, char **errorbuf);
void sandbox_free_error(char *errorbuf);
int sandbox_check(pid_t pid, const char *operation, int type, ...);
// cgo calls no variadic function: whether this process is in a sandbox, of any profile.
static int sandboxed(void) { return sandbox_check(getpid(), NULL, 0); }
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Hold puts this process in profile, which every process it starts inherits and none can
// leave. [LAW:no-silent-failure] a profile the OS does not apply is an error with the OS's
// reason, and whether the process was in a sandbox already, inside which macOS applies no
// other.
func Hold(profile []byte) error {
	text := C.CString(string(profile))
	defer C.free(unsafe.Pointer(text))
	var reason *C.char
	if C.sandbox_init(text, 0, &reason) != 0 {
		defer C.sandbox_free_error(reason)
		return fmt.Errorf("%s; in a sandbox already: %v", C.GoString(reason), C.sandboxed() != 0)
	}
	return nil
}
