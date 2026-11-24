# pyright: basic, reportAttributeAccessIssue=false
# [LAW:types-are-the-program] exception: PyObjC ships no type stubs, as in coreaudio.py.
"""CoreAudio's adapter: how it tells the Mac's own microphone and speakers, on devices of the test's own and against
the system's."""

import struct
from collections.abc import Iterator
from typing import NamedTuple

import CoreAudio
import pytest
from loguru import logger

from hands.voice import coreaudio

_TRANSPORT = coreaudio._address(CoreAudio.kAudioDevicePropertyTransportType)  # pyright: ignore[reportPrivateUsage]
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
}
_SIDES = {id(coreaudio._INPUT_SOURCE): "input", id(coreaudio._OUTPUT_SOURCE): "output"}  # pyright: ignore[reportPrivateUsage]


def _read(device: int, address: object) -> tuple[int, bytes]:
    # A read of anything the table does not hold is a KeyError: the fake answers only what it was written to.
    if device == _TORN_DOWN:
        return CoreAudio.kAudioHardwareBadObjectError, b""
    return _SOURCES[device, _SIDES[id(address)]]


def test_a_device_is_the_macs_own_microphone_or_speakers_only_on_their_data_source() -> None:
    built_ins = coreaudio._built_ins(_read)  # pyright: ignore[reportPrivateUsage]
    read = {device: (built_ins.microphone(device), built_ins.speakers(device)) for device in (1, 2, 3, 4, 5, _TORN_DOWN)}
    assert read == {1: (True, False), 2: (False, True), 3: (False, False), 4: (False, False), 5: (False, False), _TORN_DOWN: (False, False)}


def test_a_data_source_coreaudio_will_not_say_is_an_error() -> None:
    with pytest.raises(OSError, match="say which data source an audio device is on"):
        coreaudio._built_ins(_read).microphone(6)  # pyright: ignore[reportPrivateUsage]


class _Bound(NamedTuple):
    """Whether a device could be the Mac's own microphone, and its own speakers."""

    microphone: bool
    speakers: bool


def _bound(device: int) -> _Bound:
    # [LAW:verifiable-goals] a bound by facts the adapter does not decide by: the Mac's own pair has the built-in
    # transport type, and streams in the scope it faces. It is no more than a bound, since the headphone jack has the
    # built-in transport type too, and a name is no better: a virtual Mac's own pair, CI's runner's, is "Apple Virtual
    # Sound Device". The transport is read by the adapter's raw read, and compared here: the adapter compares no
    # transport, so no fault in its comparisons widens the bound with the verdicts.
    try:
        status, data = coreaudio._read(device, _TRANSPORT)  # pyright: ignore[reportPrivateUsage]
        coreaudio._of_device(status, "say how an audio device is connected")  # pyright: ignore[reportPrivateUsage]
        built_in = struct.unpack("I", data)[0] == CoreAudio.kAudioDeviceTransportTypeBuiltIn
        has_input = coreaudio._has_streams(device, coreaudio._INPUT_STREAMS)  # pyright: ignore[reportPrivateUsage]
        has_output = coreaudio._has_streams(device, coreaudio._OUTPUT_STREAMS)  # pyright: ignore[reportPrivateUsage]
    except coreaudio._Gone:  # pyright: ignore[reportPrivateUsage]
        # Torn down since the listing: a device that is gone could be neither.
        return _Bound(microphone=False, speakers=False)
    return _Bound(microphone=built_in and has_input, speakers=built_in and has_output)


def test_no_device_of_the_systems_reads_as_the_macs_own_that_cannot_be() -> None:
    built_ins = coreaudio.BUILT_INS
    wrong: list[tuple[int, bool, bool]] = []
    own = False
    for device in coreaudio._devices():  # pyright: ignore[reportPrivateUsage]
        bound = _bound(device)
        microphone, speakers = built_ins.microphone(device), built_ins.speakers(device)
        own = own or microphone or speakers
        if (microphone and not bound.microphone) or (speakers and not bound.speakers):
            wrong.append((device, microphone, speakers))
    assert not wrong, f"read as the Mac's own (device, microphone, speakers), and cannot be: {wrong}"
    if not own:
        pytest.skip("no device is on the Mac's own microphone or speakers now")


def test_a_device_the_system_does_not_have_is_neither() -> None:
    gone = max(coreaudio._devices(), default=0) + 1_000_000  # pyright: ignore[reportPrivateUsage]
    assert (coreaudio.BUILT_INS.microphone(gone), coreaudio.BUILT_INS.speakers(gone)) == (False, False)


@pytest.fixture
def listened(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[list[int], dict[int, int]]]:
    """CoreAudio's listener calls, faked: the tokens added, and the status each removal answers by its address's turn,
    0 unless set. Whatever the adapter's targets hold once a test is done is cleared, so none outlives it."""
    added: list[int] = []
    refusing: dict[int, int] = {}
    removed = iter(range(3))
    monkeypatch.setattr(coreaudio.CoreAudio, "AudioObjectAddPropertyListener", lambda _object, _address, _listener, token: added.append(token) or 0)
    monkeypatch.setattr(coreaudio.CoreAudio, "AudioObjectRemovePropertyListener", lambda *_: refusing.get(next(removed), 0))
    held = dict(coreaudio._targets)  # pyright: ignore[reportPrivateUsage]
    yield added, refusing
    coreaudio._targets.clear()  # pyright: ignore[reportPrivateUsage]
    coreaudio._targets.update(held)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def said() -> Iterator[list[tuple[str, str]]]:
    """What the adapter logs, as each line's level and message."""
    lines: list[tuple[str, str]] = []
    sink = logger.add(lambda message: lines.append((message.record["level"].name, message.record["message"])), level="WARNING", filter="hands.voice.coreaudio")
    yield lines
    logger.remove(sink)


def _notice(token: int) -> None:
    coreaudio._listener(0, 1, None, token)  # pyright: ignore[reportPrivateUsage]


def test_a_change_reaches_who_listens_until_listening_stops_and_one_after_is_said(listened: tuple[list[int], dict[int, int]], said: list[tuple[str, str]]) -> None:
    added, _refusing = listened
    changes: list[str] = []
    with coreaudio.device_changes(lambda: changes.append("changed")):
        _notice(added[0])
    _notice(added[0])
    assert (added, changes, said) == ([added[0]] * 3, ["changed"], [("WARNING", "a change to the audio devices came after listening for them had stopped")])


def test_a_listener_coreaudio_would_not_remove_still_calls_who_listened_and_is_said(listened: tuple[list[int], dict[int, int]], said: list[tuple[str, str]]) -> None:
    added, refusing = listened
    refusing[2] = -50
    changes: list[str] = []
    with coreaudio.device_changes(lambda: changes.append("changed")):
        pass
    _notice(added[0])
    assert changes == ["changed"]
    assert said == [("ERROR", "CoreAudio would not stop listening for changes to the audio devices: OSStatus [0, 0, -50], for the default input, the default output and the device list in turn")]


def test_a_change_who_listens_cannot_take_is_said_where_it_happened(listened: tuple[list[int], dict[int, int]], said: list[tuple[str, str]]) -> None:
    added, _refusing = listened

    def broken() -> None:
        raise RuntimeError("Event loop is closed")

    with coreaudio.device_changes(broken):
        _notice(added[0])
    assert said == [("ERROR", "an audio device change could not be passed on")]
