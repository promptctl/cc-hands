"""The certificate authority hands' proxy answers the API's host with, the file a client is given to trust it by, and what
the proxy trusts itself as it reaches the world.

A client that reaches the API with the proxy as its HTTPS proxy opens TLS to the API's host through it, and the proxy
ends that TLS itself, with a certificate for the host signed by an authority it made as it started. The authority's key
is never written anywhere: it lives and dies with the proxy, as fritter's tap's does (fritter/intercept). What hands'
own environment trusts beside the system's authorities, its NODE_EXTRA_CA_CERTS, the client trusts too, and so does the
proxy, on every connection it makes onward.
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
from hands.sessions.files import replace_whole
from hands.sessions.untap import TRUST

# The one HTTP the proxy serves, so the only one a client may speak inside the TLS it ends.
ALPN = "http/1.1"


class Uninterceptable(Exception):
    """The proxy cannot end a client's TLS to the API, or reach the world past it: its authority cannot be made, the
    certificates it is to trust besides the system's cannot be read, the file that holds them all cannot be written, or
    the outer proxy hands' environment names is one it cannot speak to. Nothing reaches the API around the proxy in its
    place."""


@dataclass(frozen=True)
class Authority:
    """An authority made for one run of the proxy, and what it signed: `certificate` is what a client trusts, and
    `answer` is the server side of the TLS the proxy ends for `host`, offering ALPN's http/1.1 alone."""

    host: str
    certificate: x509.Certificate
    answer: ssl.SSLContext


def mint(host: str, clock: Clock) -> Authority:
    """An authority valid from an hour before now for a year, as fritter's tap's is, and a certificate for `host`
    under it; raises Uninterceptable when the leaf cannot be loaded for the TLS it answers.

    [LAW:effects-at-boundaries] exception: the keys are the operating system's randomness, drawn by cryptography and
    never by `clock`, whose tokens a test makes predictable, and the authority is named by this process's id, so two
    runs' authorities are told apart in a keychain. The serial numbers and the load's password are `clock`'s."""
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
    try:
        with tempfile.NamedTemporaryFile(prefix="hands-proxy-", suffix=".pem") as loaded:
            loaded.write(leaf.public_bytes(serialization.Encoding.PEM))
            loaded.write(leaf_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.BestAvailableEncryption(password)))
            loaded.flush()
            answer.load_cert_chain(loaded.name, password=password)
    except OSError as error:
        raise Uninterceptable(f"cannot load the certificate hands' proxy answers {host} with, through a file of its own: {error}") from error
    return Authority(host, root, answer)


def _serial(clock: Clock) -> int:
    """A serial number as RFC 5280 has it: positive, and of 20 octets at most."""
    return int(clock.token(19), 16) | 1


def also_trusted(environ: Mapping[str, str], path: Path) -> tuple[x509.Certificate, ...]:
    """The certificates `environ`'s NODE_EXTRA_CA_CERTS names, which hands trusts besides the system's: none where it
    names none. Raises Uninterceptable when its file cannot be read as certificates, is named relative, which a client
    would read from a directory hands does not know, or is `path`, the proxy's own file."""
    named = environ.get(TRUST, "")
    if not named:
        return ()
    file = Path(named)
    if not file.is_absolute():
        raise Uninterceptable(f"{TRUST} names {named}, a path relative to a directory hands does not know: name the file of certificates by its whole path")
    try:
        # [LAW:no-silent-failure] the proxy's own file from before names an authority whose key is gone, and what it
        # holds besides is not known.
        if path.exists() and file.samefile(path):
            raise Uninterceptable(f"{TRUST} names {named}, the file hands' proxy gives its clients in its place: it is to name the certificates they trust besides the proxy's")
        return tuple(x509.load_pem_x509_certificates(file.read_bytes()))
    except (OSError, ValueError) as error:
        raise Uninterceptable(f"cannot read the certificates {TRUST} names, which hands trusts besides the system's: {error}") from error


def trusted(authority: Authority, also: tuple[x509.Certificate, ...], path: Path) -> Path:
    """Writes, whole, at `path`, the file a client is given to trust in place of the one hands' NODE_EXTRA_CA_CERTS
    names: a snapshot of `also`, the certificates that one held as the proxy started, and the authority, readable by the
    user alone. Raises Uninterceptable when it cannot be written."""
    # [LAW:one-source-of-truth] the client trusts what it would have without the proxy, and the proxy's authority besides.
    written = "".join(certificate.public_bytes(serialization.Encoding.PEM).decode() for certificate in (*also, authority.certificate))
    try:
        replace_whole(path, written, 0o600)
    except OSError as error:
        raise Uninterceptable(f"cannot write the certificates a client of hands' proxy is to trust: {error}") from error
    return path


def onward_trust(also: tuple[x509.Certificate, ...]) -> ssl.SSLContext:
    """What the proxy trusts on each connection it makes onward, to the API among them: the system's authorities, and
    `also`, as Node trusts them with NODE_EXTRA_CA_CERTS."""
    trust = ssl.create_default_context()
    for certificate in also:
        trust.load_verify_locations(cadata=certificate.public_bytes(serialization.Encoding.DER))
    return trust
