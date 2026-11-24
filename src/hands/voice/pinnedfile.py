"""A file hands fetches whole from a URL and holds to the sha256 digest it pins it to: DTLN-aec's models
(`hands.voice.dtln`) and ONNX Runtime Web's files (`hands.voice.ortweb`) are fetched this way.
"""

import hashlib
import os
import threading
import urllib.request
from pathlib import Path

# How long a read of the file's server may wait for bytes, in seconds, before the fetch fails.
FETCH_TIMEOUT = 30


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetched(directory: Path, name: str, url: str, digest: str, at: str) -> Path:
    """`name` in `directory`, fetched from `url` unless a file there already has `digest`; `at` says which release or
    revision `url` names.

    [LAW:no-silent-failure] a file that is not the one pinned raises naming it and both digests: a model or a runtime
    with other bytes would fail, or worse run, with nothing to say why. A file is fetched whole and held to its digest
    before it is written beside its place, by a name of this process's and thread's own, and moved to it: a fetch cut short leaves none
    to be taken for the file, and two fetches at once do not write the same one.
    """
    path = directory / name
    if not (path.is_file() and digest_of(path) == digest):
        with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT) as response:  # noqa: S310
            body = response.read()
        if (found := hashlib.sha256(body).hexdigest()) != digest:
            raise ValueError(f"{name} at {at} is {found}, not the {digest} pinned")
        beside = directory / f"{name}.{os.getpid()}.{threading.get_ident()}.part"
        beside.write_bytes(body)
        beside.replace(path)
    return path
