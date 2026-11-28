# pyright: basic, reportAttributeAccessIssue=false
"""LaunchServices held to the real lsappinfo (lsappinfo of macOS 26): the app it reads in front is the app AppKit knows."""

import AppKit
import pytest

from conftest import runner_on
from hands.clock import SYSTEM
from hands.sessions.launchservices import Launched, LaunchServices

pytestmark = pytest.mark.outside


async def test_the_app_lsappinfo_names_in_front_is_the_one_appkit_knows_by_its_pid() -> None:
    # AppKit is the oracle the answer is held to: a second reader of LaunchServices, asked about the pid read, so the
    # front changing between the two reads changes nothing.
    match await LaunchServices(runner_on(SYSTEM)).launched():
        case Launched(pid=pid, bundle=bundle, name=name):
            app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            assert app is not None, f"no app runs as pid {pid}"
            assert (bundle, name) == (app.bundleIdentifier(), app.localizedName())
        case unread:
            pytest.fail(f"nothing read in front: {unread}")
