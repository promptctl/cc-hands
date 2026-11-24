# pyright: basic, reportAttributeAccessIssue=false
# [LAW:types-are-the-program] exception: PyObjC ships no type stubs, as in coreaudio.py.
"""CoreAudio's adapter: how it tells the Mac's own microphone and speakers, on devices of the test's own and against
the system's."""

import struct
import threading
from collections.abc import Iterator
from typing import NamedTuple

import CoreAudio
import pytest
from loguru import logger

from hands.voice import coreaudio

_TRANSPORT = coreaudio._address(CoreAudio.kAudioDevicePropertyTransportType)
_TORN_DOWN = 9


def _source(code: bytes) -> tuple[int, bytes]:
    return 0, struct.pack("I", int.from_bytes(code))


_NONE = (CoreAudio.kAudioHardwareUnknownPropertyError, b"")

# Devices of the test's own, by the data source each side is on: IOAudio's port subtypes. A side with none answers
# CoreAudio's unknown property, as Bluetooth's and a virtual device's do.
_SOURCES = {
    (1, "input"): _source(b"imic"),  # the Mac's own microphone
    (1, "output"): _NONE,
    (2, "input"): _NONE,
    (2, "output"): _source(b"ispk"),  # the Mac's own speakers
    (3, "input"): _NONE,
    (3, "output"): _source(b"hdpn"),  # the headphone jack, built in too
    (4, "input"): _source(b"emic"),  # a microphone plugged into the jack
    (4, "output"): _NONE,
    (5, "input"): _NONE,  # a Bluetooth headset's, on no source
    (5, "output"): _NONE,
    (6, "input"): (CoreAudio.kAudioHardwareUnspecifiedError, b""),
    (_TORN_DOWN, "input"): (CoreAudio.kAudioHardwareBadObjectError, b""),  # gone since it was listed
    (_TORN_DOWN, "output"): (CoreAudio.kAudioHardwareBadObjectError, b""),
}

# The side each address the adapter reads a data source by faces: the adapter's own address objects, told by identity.
_SIDES = ((coreaudio._INPUT_SOURCE, "input"), (coreaudio._OUTPUT_SOURCE, "output"))


def _read(device: int, address: object) -> tuple[int, bytes]:
    # A read of anything the table does not hold fails: the fake answers only what it was written to.
    return _SOURCES[device, next(side for known, side in _SIDES if known is address)]


def test_a_device_is_the_macs_own_microphone_or_speakers_only_on_their_data_source() -> None:
    built_ins = coreaudio._built_ins(_read)
    read = {device: (built_ins.microphone(device), built_ins.speakers(device)) for device in (1, 2, 3, 4, 5, _TORN_DOWN)}
    assert read == {1: (True, False), 2: (False, True), 3: (False, False), 4: (False, False), 5: (False, False), _TORN_DOWN: (False, False)}


def test_a_data_source_coreaudio_will_not_say_is_an_error() -> None:
    with pytest.raises(OSError, match="say which data source an audio device is on"):
        coreaudio._built_ins(_read).microphone(6)


class _Bound(NamedTuple):
    """Whether a device could be the Mac's own microphone, and its own speakers."""

    microphone: bool
    speakers: bool


def _bound(device: int) -> _Bound:
    # [LAW:verifiable-goals] a bound by facts the adapter does not decide by: the Mac's own pair has the built-in
    # transport type, and streams in the scope it faces. It is no more than a bound, since the headphone jack has the
    # built-in transport type too, and a name is no better: a virtual Mac's own pair, CI's runner's, is "Apple Virtual
    # Sound Device". The streams are read by the adapter's own `_has_streams`; the transport, which discriminates, is
    # read by its raw read and compared here, where the adapter compares none, so no fault in its comparisons widens
    # the bound with the verdicts.
    try:
        status, data = coreaudio._read(device, _TRANSPORT)
        coreaudio._of_device(status, "say how an audio device is connected")
        built_in = struct.unpack("I", data)[0] == CoreAudio.kAudioDeviceTransportTypeBuiltIn
        has_input = coreaudio._has_streams(device, coreaudio._INPUT_STREAMS)
        has_output = coreaudio._has_streams(device, coreaudio._OUTPUT_STREAMS)
    except coreaudio._Gone:
        # Torn down since the listing: a device that is gone could be neither.
        return _Bound(microphone=False, speakers=False)
    return _Bound(microphone=built_in and has_input, speakers=built_in and has_output)


def test_no_device_of_the_systems_reads_as_the_macs_own_that_cannot_be() -> None:
    built_ins = coreaudio.BUILT_INS
    wrong: list[tuple[int, bool, bool]] = []
    own = False
    for device in coreaudio._devices():
        bound = _bound(device)
        microphone, speakers = built_ins.microphone(device), built_ins.speakers(device)
        own = own or microphone or speakers
        if (microphone and not bound.microphone) or (speakers and not bound.speakers):
            wrong.append((device, microphone, speakers))
    assert not wrong, f"read as the Mac's own (device, microphone, speakers), and cannot be: {wrong}"
    if not own:
        pytest.skip("no device is on the Mac's own microphone or speakers now")


def test_each_device_of_the_systems_has_a_name_it_is_found_by() -> None:
    hardware = coreaudio.hardware()
    unfound: list[tuple[int, str]] = []
    for device in coreaudio._devices():
        try:
            name = coreaudio._name(device)
            assert name, f"device {device} is named nothing"
            for scope, find in ((coreaudio._INPUT_STREAMS, hardware.inputs), (coreaudio._OUTPUT_STREAMS, hardware.outputs)):
                # The first device of that name: this one, or another of the same name.
                if coreaudio._has_streams(device, scope) and ((found := find(name)) is None or coreaudio._name(found) != name):
                    unfound.append((device, name))
        except coreaudio._Gone:
            continue  # torn down while it was read: whether it was found says nothing
    # A device is unfound only where it has gone since: read from one listing, taken after every find.
    listed = coreaudio._devices()
    assert not [(device, name) for device, name in unfound if device in listed], f"not found by their names (device, name): {unfound}"


def test_a_device_the_system_does_not_have_is_neither() -> None:
    gone = max(coreaudio._devices(), default=0) + 1_000_000
    assert (coreaudio.BUILT_INS.microphone(gone), coreaudio.BUILT_INS.speakers(gone)) == (False, False)


class _Listened(NamedTuple):
    """CoreAudio's listener calls, faked: each (address, token) added and removed, in turn, and the status a removal of
    an address answers, 0 unless set."""

    added: list[tuple[object, int]]
    removed: list[tuple[object, int]]
    refusing: list[tuple[object, int]]


@pytest.fixture
def listened(monkeypatch: pytest.MonkeyPatch) -> _Listened:
    calls = _Listened([], [], [])

    def remove(_object: int, address: object, _listener: object, token: int) -> int:
        calls.removed.append((address, token))
        return next((status for refused, status in calls.refusing if refused is address), 0)

    monkeypatch.setattr(coreaudio.CoreAudio, "AudioObjectAddPropertyListener", lambda _object, address, _listener, token: calls.added.append((address, token)) or 0)
    monkeypatch.setattr(coreaudio.CoreAudio, "AudioObjectRemovePropertyListener", remove)
    monkeypatch.setattr(coreaudio, "_targets", {})
    return calls


@pytest.fixture
def said() -> Iterator[list[tuple[str, str]]]:
    """What the adapter logs, as each line's level and message."""
    lines: list[tuple[str, str]] = []
    sink = logger.add(lambda message: lines.append((message.record["level"].name, message.record["message"])), level="WARNING", filter="hands.voice.coreaudio")
    yield lines
    logger.remove(sink)


def _notice(token: int) -> None:
    coreaudio._listener(0, 1, None, token)


_LATE = ("WARNING", "a change to the audio devices came after listening for them had stopped")


def test_a_change_reaches_who_listens_until_listening_stops_and_one_after_is_said(listened: _Listened, said: list[tuple[str, str]]) -> None:
    changes: list[str] = []
    with coreaudio.device_changes(lambda: changes.append("changed")):
        token = listened.added[0][1]
        _notice(token)
    _notice(token)
    listening = [(coreaudio._INPUT, token), (coreaudio._OUTPUT, token), (coreaudio._DEVICES, token)]
    assert (listened.added, listened.removed, changes, said) == (listening, listening, ["changed"], [_LATE])


def test_a_listener_coreaudio_would_not_remove_is_said_and_calls_nobody(listened: _Listened, said: list[tuple[str, str]]) -> None:
    listened.refusing.append((coreaudio._DEVICES, -50))
    changes: list[str] = []
    with coreaudio.device_changes(lambda: changes.append("changed")):
        token = listened.added[0][1]
    _notice(token)
    assert changes == []
    assert said == [("ERROR", "CoreAudio would not stop listening for changes to the device list: OSStatus -50"), _LATE]


class _Contended:
    """A lock that says when one who takes it must wait for it."""

    def __init__(self, waited: threading.Event) -> None:
        self._lock = threading.Lock()
        self._waited = waited

    def __enter__(self) -> None:
        if not self._lock.acquire(blocking=False):
            self._waited.set()
            self._lock.acquire()

    def __exit__(self, *_: object) -> None:
        self._lock.release()


def test_a_stop_waits_for_a_change_being_passed_on_so_none_is_passed_on_once_it_has_stopped(listened: _Listened, said: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch) -> None:
    # [LAW:no-ambient-temporal-coupling] the interleaving forced, not hoped for: a change is held in who listens while
    # the stop runs, and is let go only once the stop has either waited for it or finished without it.
    waited_or_stopped, passing, let_go = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(coreaudio, "_calling", _Contended(waited_or_stopped))
    happened: list[str] = []

    def held() -> None:
        passing.set()
        let_go.wait()
        happened.append("passed on")

    listening = coreaudio.device_changes(held)
    listening.__enter__()
    notice = threading.Thread(target=_notice, args=(listened.added[0][1],))
    notice.start()
    passing.wait()

    def stop() -> None:
        listening.__exit__(None, None, None)
        happened.append("stopped")
        waited_or_stopped.set()

    stopper = threading.Thread(target=stop)
    stopper.start()
    waited_or_stopped.wait()
    while_held = list(happened)
    let_go.set()
    notice.join()
    stopper.join()
    assert (while_held, happened, said) == ([], ["passed on", "stopped"], [])


def test_a_change_who_listens_cannot_take_is_said_where_it_happened(listened: _Listened, said: list[tuple[str, str]]) -> None:
    def broken() -> None:
        raise RuntimeError("Event loop is closed")

    with coreaudio.device_changes(broken):
        _notice(listened.added[0][1])
    assert said == [("ERROR", "an audio device change could not be passed on")]
