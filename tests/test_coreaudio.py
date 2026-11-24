# pyright: basic, reportAttributeAccessIssue=false
# [LAW:types-are-the-program] exception: PyObjC ships no type stubs, as in coreaudio.py.
"""CoreAudio's adapter: how it tells the Mac's own microphone and speakers, on devices of the test's own and against
the system's."""

import struct
from typing import NamedTuple

import CoreAudio
import pytest

from hands.voice import coreaudio

_TRANSPORT = coreaudio._address(CoreAudio.kAudioDevicePropertyTransportType)  # pyright: ignore[reportPrivateUsage]
_GONE = 9


def _source(code: bytes) -> tuple[int, bytes]:
    return 0, struct.pack("I", int.from_bytes(code))


# Devices of the test's own, by the data source each side is on: IOAudio's port subtypes. A side with none answers
# CoreAudio's unknown property, as Bluetooth's and a virtual device's do.
_SOURCES = {
    (1, "input"): _source(b"imic"),  # the Mac's own microphone
    (2, "output"): _source(b"ispk"),  # the Mac's own speakers
    (3, "output"): _source(b"hdpn"),  # the headphone jack, built in too
    (4, "input"): _source(b"emic"),  # a microphone plugged into the jack
    (6, "input"): (CoreAudio.kAudioHardwareUnspecifiedError, b""),
}


def _read(device: int, address: object) -> tuple[int, bytes]:
    if device == _GONE:
        return CoreAudio.kAudioHardwareBadObjectError, b""
    side = "input" if address is coreaudio._INPUT_SOURCE else "output"  # pyright: ignore[reportPrivateUsage]
    return _SOURCES.get((device, side), (CoreAudio.kAudioHardwareUnknownPropertyError, b""))


def test_a_device_is_the_macs_own_microphone_or_speakers_only_on_their_data_source() -> None:
    built_ins = coreaudio._built_ins(_read)  # pyright: ignore[reportPrivateUsage]
    read = {device: (built_ins.microphone(device), built_ins.speakers(device)) for device in (1, 2, 3, 4, 5, _GONE)}
    assert read == {1: (True, False), 2: (False, True), 3: (False, False), 4: (False, False), 5: (False, False), _GONE: (False, False)}


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
    # Sound Device". The transport is compared here, not by the adapter's comparison, which a fault would widen along
    # with the verdicts it bounds.
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
