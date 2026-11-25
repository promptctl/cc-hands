"""The authority hands' proxy ends the API's TLS with, the file a client trusts it by, and how the proxy reaches every
other host: each refusal of the start said, naming what refused it."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography import x509

from hands.core.wire import API_HOST
from hands.sessions.authority import Uninterceptable, mint, trusted
from hands.sessions.connect import Dialed, Outer, Outward, exempted, outward
from hands.sessions.proxy import serve_proxy
from hands.sessions.untap import TRUST
from hands.virtualclock import VirtualClock

NOW = datetime(2026, 11, 24, 12, tzinfo=UTC)
AUTHORITY = mint(API_HOST, VirtualClock(epoch=NOW))
ELSEWHERE = b"-----BEGIN CERTIFICATE-----\nthe machine's own\n-----END CERTIFICATE-----\n"


def test_the_authority_signs_certificates_alone_from_an_hour_before_it_was_made_for_a_year() -> None:
    # What it signed for the API's host is read back from a handshake in tests/test_proxy_connect.py.
    root = x509.load_pem_x509_certificate(AUTHORITY.certificate)
    assert root.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    usage = root.extensions.get_extension_for_class(x509.KeyUsage).value
    assert usage.key_cert_sign and not usage.digital_signature
    assert (root.not_valid_before_utc, root.not_valid_after_utc) == (datetime(2026, 11, 24, 11, tzinfo=UTC), datetime(2027, 11, 24, 12, tzinfo=UTC))


def test_a_client_trusts_the_authority_and_what_hands_environment_already_trusts(tmp_path: Path) -> None:
    named = tmp_path / "corp.pem"
    named.write_bytes(ELSEWHERE)
    written = trusted(AUTHORITY, {TRUST: str(named)}, tmp_path / "proxy" / "trusted.pem")
    assert written.read_bytes() == ELSEWHERE + b"\n" + AUTHORITY.certificate
    assert written.stat().st_mode & 0o777 == 0o600
    # With nothing else named, the authority alone.
    assert x509.load_pem_x509_certificates(trusted(AUTHORITY, {}, tmp_path / "alone.pem").read_bytes()) == [x509.load_pem_x509_certificate(AUTHORITY.certificate)]


@pytest.mark.parametrize(
    ("named", "said"),
    [
        ("corp.pem", f"{TRUST} names corp.pem, a path relative to a directory hands does not know"),
        ("/nowhere/corp.pem", f"cannot read the certificates {TRUST} names"),
    ],
    ids=["relative", "unreadable"],
)
def test_certificates_hands_cannot_read_refuse_the_proxy_naming_them(named: str, said: str, tmp_path: Path) -> None:
    with pytest.raises(Uninterceptable, match=f"^{said}"):
        trusted(AUTHORITY, {TRUST: named}, tmp_path / "trusted.pem")
    assert not (tmp_path / "trusted.pem").exists()


def test_the_proxys_own_file_named_as_what_to_trust_besides_it_is_refused(tmp_path: Path) -> None:
    written = trusted(AUTHORITY, {}, tmp_path / "trusted.pem")
    with pytest.raises(Uninterceptable, match="the file hands' proxy gives its clients in its place"):
        trusted(AUTHORITY, {TRUST: str(written)}, written)


async def test_a_proxy_that_cannot_intercept_refuses_to_start_saying_why(tmp_path: Path) -> None:
    # Before it listens: no connection made to it is ever answered without its TLS.
    with pytest.raises(Uninterceptable, match="a path relative to a directory hands does not know"):
        await serve_proxy("http://127.0.0.1:1", lambda _: None, lambda _: pytest.fail("routed"), VirtualClock(epoch=NOW), {TRUST: "corp.pem"}, AUTHORITY, tmp_path / "trusted.pem")
    with pytest.raises(Uninterceptable, match="^HTTPS_PROXY names socks5://corp:1080: hands' proxy reaches an outer proxy at an http:// URL"):
        await serve_proxy("http://127.0.0.1:1", lambda _: None, lambda _: pytest.fail("routed"), VirtualClock(epoch=NOW), {"HTTPS_PROXY": "socks5://corp:1080"}, AUTHORITY, tmp_path / "trusted.pem")


def test_the_world_is_reached_as_hands_environment_says() -> None:
    reached = outward({"https_proxy": "http://me:p%40ss@corp:3128", "HTTP_PROXY": "corp:8080", "NO_PROXY": ".internal, localhost:9000"})
    assert reached == Outward(Outer("corp", 3128, (("Proxy-Authorization", "Basic bWU6cEBzcw=="),)), Outer("corp", 8080, ()), ".internal, localhost:9000")
    assert reached.onward("api.example.com", secure=True) == Outer("corp", 3128, (("Proxy-Authorization", "Basic bWU6cEBzcw=="),))
    assert reached.onward("db.internal", secure=True) == Dialed()
    assert reached.onward("localhost", secure=False) == Dialed()
    assert outward({}).onward("api.example.com", secure=True) == Dialed()


@pytest.mark.parametrize(
    ("host", "exempt", "named"),
    [
        ("api.anthropic.com", ".anthropic.com", True),
        ("api.anthropic.com", "*.anthropic.com", True),
        ("anthropic.com", "anthropic.com", True),
        ("evilanthropic.com", "anthropic.com", False),
        ("example.com", "*", True),
        ("127.0.0.1", "127.0.0.1:8080", True),
        ("::1", "[::1]:443", True),
        ("::1", "::1", True),
        ("example.com", "", False),
    ],
)
def test_no_proxy_names_a_host_and_the_hosts_under_it(host: str, exempt: str, named: bool) -> None:
    assert exempted(host, exempt) is named

