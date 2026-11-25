"""The authority hands' proxy ends the API's TLS with, the file a client trusts it by, and how the proxy reaches every
other host: each refusal of the start said, naming what refused it."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from yarl import URL

from hands.core.wire import API_HOST
from hands.sessions.authority import Uninterceptable, also_trusted, mint, onward_trust, trusted
from hands.sessions.connect import outward
from hands.sessions.proxy import serve_proxy
from hands.sessions.untap import TRUST
from hands.virtualclock import VirtualClock

NOW = datetime(2026, 11, 24, 12, tzinfo=UTC)
AUTHORITY = mint(API_HOST, VirtualClock(epoch=NOW))
# A certificate hands' environment trusts besides the system's, as a corporate proxy's would be.
ELSEWHERE = mint("corp.example", VirtualClock(epoch=NOW)).certificate


def pem(*certificates: x509.Certificate) -> bytes:
    return b"".join(certificate.public_bytes(serialization.Encoding.PEM) for certificate in certificates)


def test_the_authority_signs_certificates_alone_from_an_hour_before_it_was_made_for_a_year() -> None:
    # What it signed for the API's host is read back from a handshake in tests/test_proxy_connect.py.
    root = AUTHORITY.certificate
    assert root.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    usage = root.extensions.get_extension_for_class(x509.KeyUsage).value
    assert usage.key_cert_sign and not usage.digital_signature
    assert (root.not_valid_before_utc, root.not_valid_after_utc) == (datetime(2026, 11, 24, 11, tzinfo=UTC), datetime(2027, 11, 24, 12, tzinfo=UTC))


@pytest.mark.parametrize(
    ("host", "port", "intercepted"),
    [(API_HOST, 443, True), (API_HOST, 8080, False), (API_HOST, 80, False), ("example.com", 443, False), (None, None, False)],
    ids=["the-api", "another-port", "http", "another-host", "no-host"],
)
def test_what_a_client_asks_for_is_the_apis_at_the_apis_host_and_https_port_alone(host: str | None, port: int | None, intercepted: bool) -> None:
    # The one reading of what is the API's, which a CONNECT and a request in absolute form are both judged by: the
    # origin https://api.anthropic.com, and no other port of its host.
    assert AUTHORITY.intercepts(host, port) is intercepted


def test_a_client_trusts_the_authority_and_what_hands_environment_already_trusts(tmp_path: Path) -> None:
    named = tmp_path / "corp.pem"
    named.write_bytes(pem(ELSEWHERE))
    written = trusted(AUTHORITY, also_trusted({TRUST: str(named)}, tmp_path / "proxy" / "trusted.pem"), tmp_path / "proxy" / "trusted.pem")
    assert written.read_bytes() == pem(ELSEWHERE, AUTHORITY.certificate)
    assert written.stat().st_mode & 0o777 == 0o600
    # With nothing else named, the authority alone.
    assert trusted(AUTHORITY, also_trusted({}, tmp_path / "alone.pem"), tmp_path / "alone.pem").read_bytes() == pem(AUTHORITY.certificate)


# What NODE_EXTRA_CA_CERTS names, in the directory a test writes in, and what its refusal begins with.
NAMED: list[tuple[Callable[[Path], str], str]] = [
    (lambda _root: "corp.pem", f"{TRUST} names corp.pem, a path relative to a directory hands does not know"),
    (lambda root: str(root / "nowhere.pem"), f"cannot read the certificates {TRUST} names"),
    (lambda root: str(root / "garbled.pem"), f"cannot read the certificates {TRUST} names"),
]


@pytest.mark.parametrize(("named", "said"), NAMED, ids=["relative", "unreadable", "not-certificates"])
def test_certificates_hands_cannot_read_refuse_the_proxy_naming_them(named: Callable[[Path], str], said: str, tmp_path: Path) -> None:
    (tmp_path / "garbled.pem").write_text("not a certificate")
    with pytest.raises(Uninterceptable, match=f"^{said}"):
        also_trusted({TRUST: named(tmp_path)}, tmp_path / "trusted.pem")


def test_the_proxys_own_file_named_as_what_to_trust_besides_it_is_refused(tmp_path: Path) -> None:
    written = trusted(AUTHORITY, (), tmp_path / "trusted.pem")
    with pytest.raises(Uninterceptable, match="the file hands' proxy gives its clients in its place"):
        also_trusted({TRUST: str(written)}, written)


@pytest.mark.parametrize(
    ("environ", "said"),
    [
        ({TRUST: "corp.pem"}, "a path relative to a directory hands does not know"),
        ({"HTTPS_PROXY": "socks5://corp:1080"}, "^https_proxy or HTTPS_PROXY names an outer proxy of its scheme 'socks5'"),
        ({"http_proxy": "http://corp:0"}, "^http_proxy or HTTP_PROXY names an outer proxy of .* its port 0"),
        ({"HTTPS_PROXY": "http://me:secret@corp:abc"}, r"^https_proxy or HTTPS_PROXY names an outer proxy by no URL hands' proxy can read \(Invalid URL: port can't be converted to integer\)"),
        ({"http_proxy": "http://corp:99999"}, r"^http_proxy or HTTP_PROXY names an outer proxy by no URL hands' proxy can read \(Port out of range"),
    ],
    ids=["trust", "scheme", "port", "unread-port", "port-out-of-range"],
)
async def test_a_proxy_that_cannot_intercept_refuses_to_start_saying_why(environ: dict[str, str], said: str, tmp_path: Path) -> None:
    # Before it listens: no connection made to it is ever answered without its TLS, or reaches the world around it.
    with pytest.raises(Uninterceptable, match=said) as refused:
        await serve_proxy("http://127.0.0.1:1", lambda _: None, lambda _: pytest.fail("routed"), VirtualClock(epoch=NOW), environ, tmp_path / "trusted.pem", lambda _: None)
    assert not (tmp_path / "trusted.pem").exists()
    # An outer proxy's url may carry its credentials: never said.
    assert "secret" not in str(refused.value)


@pytest.mark.parametrize(
    ("upstream", "said"),
    [
        ("http://api:abc", r"^the API the proxy forwards to is no URL, 'http://api:abc': Invalid URL: port can't be converted to integer"),
        ("/v1", "^the API the proxy forwards to is reached at a host and a port, not /v1"),
    ],
    ids=["unread-port", "no-host"],
)
async def test_an_api_the_proxy_cannot_forward_to_refuses_it_before_it_writes_its_trust(upstream: str, said: str, tmp_path: Path) -> None:
    with pytest.raises(Uninterceptable, match=said):
        await serve_proxy(upstream, lambda _: None, lambda _: pytest.fail("routed"), VirtualClock(epoch=NOW), {}, tmp_path / "trusted.pem", lambda _: None)
    assert not (tmp_path / "trusted.pem").exists()


class Untrustable:
    """A stand-in for a certificate cryptography read and OpenSSL refuses to trust, its bytes none OpenSSL reads as one."""

    def public_bytes(self, encoding: serialization.Encoding) -> bytes:
        return b"\x30\x03\x02\x01\x00"


def test_a_certificate_openssl_will_not_trust_refuses_the_proxy_naming_the_variable() -> None:
    with pytest.raises(Uninterceptable, match=f"^cannot trust the certificates {TRUST} names, which hands trusts besides the system's: "):
        onward_trust((ELSEWHERE, cast(x509.Certificate, Untrustable())))


def test_the_world_is_reached_as_hands_environment_says_read_as_urllib_reads_it() -> None:
    reached = outward({"https_proxy": "http://me:p%40ss@corp:3128", "HTTP_PROXY": "http://corp:8080", "NO_PROXY": ".internal, localhost:9000"})
    assert reached.onward("https", "api.example.com", 443) == URL("http://me:p%40ss@corp:3128")
    assert reached.onward("http", "localhost", 8000) == URL("http://corp:8080")
    # no_proxy names a host and every host under it, and a host at one port alone.
    assert reached.onward("https", "db.internal", 443) is None
    assert reached.onward("http", "localhost", 9000) is None
    assert outward({}).onward("https", "api.example.com", 443) is None


def test_a_lower_case_proxy_variable_outranks_its_upper_case_and_set_empty_names_none() -> None:
    assert outward({"HTTPS_PROXY": "http://upper:1", "https_proxy": "http://lower:2"}).onward("https", "example.com", 443) == URL("http://lower:2")
    assert outward({"HTTPS_PROXY": "http://upper:1", "https_proxy": ""}).onward("https", "example.com", 443) is None
