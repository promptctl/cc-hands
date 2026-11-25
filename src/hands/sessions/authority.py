"""The certificate authority hands' proxy answers the API's host with, and the file a client is given to trust it by.

A client that reaches the API with the proxy as its HTTPS proxy opens TLS to the API's host through it, and the proxy
ends that TLS itself, with a certificate for the host signed by an authority it made as it started. The authority's key
is never written anywhere: it lives and dies with the proxy, as fritter's tap's does (fritter/intercept).
"""

import datetime
import os
import ssl
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from hands.clock import Clock
from hands.sessions.untap import TRUST

# The one HTTP the proxy serves, so the only one a client may speak inside the TLS it ends.
ALPN = "http/1.1"


class Uninterceptable(Exception):
    """The proxy cannot end a client's TLS to the API: the certificates it is to trust besides the proxy's cannot be
    read, or the file that holds them all cannot be written. Nothing reaches the API around the proxy in its place."""


@dataclass(frozen=True)
class Authority:
    """An authority made for one run of the proxy, and what it signed: `certificate` is what a client trusts, and
    `answer` is the server side of the TLS the proxy ends for `host`, offering ALPN's http/1.1 alone."""

    host: str
    certificate: bytes
    answer: ssl.SSLContext


def mint(host: str, clock: Clock) -> Authority:
    """An authority valid from an hour before now for a year, as fritter's tap's is, and a certificate for `host`
    under it."""
    now = clock.wall()
    key = ec.generate_private_key(ec.SECP256R1())
    named = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"hands' proxy {os.getpid()}")])
    begins, ends = now - datetime.timedelta(hours=1), now + datetime.timedelta(days=365)
    root = (
        x509.CertificateBuilder()
        .subject_name(named)
        .issuer_name(named)
        .public_key(key.public_key())
        .serial_number(_serial(clock))
        .not_valid_before(begins)
        .not_valid_after(ends)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)]))
        .issuer_name(named)
        .public_key(leaf_key.public_key())
        .serial_number(_serial(clock))
        .not_valid_before(begins)
        .not_valid_after(ends)
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(host)]), critical=False)
        .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    answer = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    answer.set_alpn_protocols([ALPN])
    # [LAW:effects-at-boundaries] exception: Python's ssl loads a key from a file alone. The leaf's is written for the
    # load encrypted, under a password held here only, to a file read back at once and removed.
    password = clock.token(32).encode()
    with tempfile.NamedTemporaryFile(prefix="hands-proxy-", suffix=".pem") as loaded:
        loaded.write(leaf.public_bytes(serialization.Encoding.PEM))
        loaded.write(leaf_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.BestAvailableEncryption(password)))
        loaded.flush()
        answer.load_cert_chain(loaded.name, password=password)
    return Authority(host, root.public_bytes(serialization.Encoding.PEM), answer)


def _serial(clock: Clock) -> int:
    """A serial number as RFC 5280 has it: positive, and of 20 octets at most."""
    return int(clock.token(19), 16) | 1


def trusted(authority: Authority, environ: Mapping[str, str], path: Path) -> Path:
    """Writes, at `path`, the file a client is given to trust in place of the one `environ`'s NODE_EXTRA_CA_CERTS names:
    the certificates that one holds, and the authority, each on lines of its own. Raises Uninterceptable when that one
    cannot be read, or is named relative, which the client would read from a directory hands does not know."""
    # [LAW:one-source-of-truth] the client trusts what it would have without the proxy, and the proxy's authority besides.
    already = _also_trusted(environ.get(TRUST, ""), path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as written:
            written.write(b"\n".join((already, authority.certificate)))
    except OSError as error:
        raise Uninterceptable(f"cannot write the certificates a client of hands' proxy is to trust: {error}") from error
    return path


def _also_trusted(named: str, path: Path) -> bytes:
    """What the file `named` holds: nothing where it names none."""
    if not named:
        return b""
    file = Path(named)
    if not file.is_absolute():
        raise Uninterceptable(f"{TRUST} names {named}, a path relative to a directory hands does not know: name the file of certificates by its whole path")
    try:
        # [LAW:no-silent-failure] the proxy's own file from before names an authority whose key is gone, and what it
        # holds besides is not known.
        if path.exists() and file.samefile(path):
            raise Uninterceptable(f"{TRUST} names {named}, the file hands' proxy gives its clients in its place: it is to name the certificates they trust besides the proxy's")
        return file.read_bytes()
    except OSError as error:
        raise Uninterceptable(f"cannot read the certificates {TRUST} names, which a client of hands' proxy is to trust besides its authority: {error}") from error
