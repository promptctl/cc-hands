# pyright: basic, reportAttributeAccessIssue=false
# [LAW:types-are-the-program] exception: PyObjC ships no type stubs, as in coreaudio.py.
"""CoreAudio's adapter, against the system's own audio devices."""

import struct

import CoreAudio
import pytest

from hands.voice import coreaudio

_TRANSPORT = coreaudio._address(CoreAudio.kAudioDevicePropertyTransportType)  # pyright: ignore[reportPrivateUsage]


def _built_in(device: int) -> bool:
    # Read here and not by the adapter's reader, which a fault would widen along with the verdicts it bounds.
    status, _size, data = CoreAudio.AudioObjectGetPropertyData(device, _TRANSPORT, 0, b"", 4, None)
    coreaudio._of_device(status, "say how an audio device is connected")  # pyright: ignore[reportPrivateUsage]
    return struct.unpack("I", data)[0] == CoreAudio.kAudioDeviceTransportTypeBuiltIn


def _could_be(device: int) -> tuple[str, bool, bool] | None:
    """The device's name, and whether it could be the Mac's own microphone and speakers, or None once it is gone."""
    # [LAW:verifiable-goals] a bound by facts the adapter does not decide by: the Mac's own pair is built in, and carries
    # sound the way it faces. It is no more than a bound, since the headphone jack is built in too, and a name is no
    # better: a virtual Mac's own pair, CI's runner's, is "Apple Virtual Sound Device".
    try:
        built_in = _built_in(device)
        return (
            coreaudio._name(device),  # pyright: ignore[reportPrivateUsage]
            built_in and coreaudio._has_streams(device, coreaudio._INPUT_STREAMS),  # pyright: ignore[reportPrivateUsage]
            built_in and coreaudio._has_streams(device, coreaudio._OUTPUT_STREAMS),  # pyright: ignore[reportPrivateUsage]
        )
    except coreaudio._Gone:  # pyright: ignore[reportPrivateUsage]
        # Torn down since the listing, as the adapter's own lookup takes it: no longer a device to judge.
        return None


def test_only_the_macs_own_microphone_and_speakers_read_as_its_own() -> None:
    built_ins = coreaudio.BUILT_INS
    seen = {device: _could_be(device) for device in coreaudio._devices()}  # pyright: ignore[reportPrivateUsage]
    sides = {device: (built_ins.microphone(device), built_ins.speakers(device)) for device in seen}
    wrong = [
        (could[0], sides[device])
        for device, could in seen.items()
        if could is not None and not all(read <= could_be for read, could_be in zip(sides[device], could[1:], strict=True))
    ]
    assert not wrong, f"read as the Mac's own (microphone, speakers), and cannot be: {wrong}"
    if not any(map(any, sides.values())):
        pytest.skip("no device is on the Mac's own microphone or speakers now")


def test_a_device_that_is_gone_is_neither() -> None:
    gone = max(coreaudio._devices()) + 1_000_000  # pyright: ignore[reportPrivateUsage]
    assert (coreaudio.BUILT_INS.microphone(gone), coreaudio.BUILT_INS.speakers(gone)) == (False, False)
