"""What lsappinfo's answers say is in front, read as lsappinfo writes them (lsappinfo of macOS 26). The real lsappinfo's
test is test_launchservices_lsappinfo.py."""

from hands.core.front import FrontUnread
from hands.sessions.launchservices import Launched, launched, serial_of

SERIAL = "ASN:0x0-0x3003:"


def test_the_app_in_front_is_the_serial_lsappinfo_names_and_none_is_when_it_names_none() -> None:
    assert serial_of(f"{SERIAL}\n") == SERIAL
    assert serial_of("\n") == FrontUnread("no app is in front: lsappinfo said ''")


def test_an_app_is_its_pid_bundle_and_name_and_one_with_no_bundle_is_still_named() -> None:
    info = '"pid"=449\n"CFBundleIdentifier"="com.apple.loginwindow"\n"LSDisplayName"="loginwindow"\n'
    assert launched(SERIAL, info) == Launched(449, "com.apple.loginwindow", "loginwindow")
    assert launched(SERIAL, '"pid"=97\n"CFBundleIdentifier"=[ NULL ] \n"LSDisplayName"="tool"\n') == Launched(97, None, "tool")


def test_an_app_gone_since_it_was_named_in_front_is_left_unread() -> None:
    # lsappinfo answers a serial it no longer knows with every field NULL, and exits 0.
    gone = '"pid"=[ NULL ] \n"CFBundleIdentifier"=[ NULL ] \n"LSDisplayName"=[ NULL ] \n'
    assert launched(SERIAL, gone) == FrontUnread(f"lsappinfo did not say which app {SERIAL} is: {gone!r}")
