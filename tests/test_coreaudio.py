# pyright: basic, reportAttributeAccessIssue=false
# PyObjC ships no type stubs, as in coreaudio.py: the test reads a fact of each device the adapter does not read.
"""CoreAudio's adapter, against the system's own audio devices."""

import struct

import CoreAudio
import pytest

from hands.voice import coreaudio

_TRANSPORT = coreaudio._address(CoreAudio.kAudioDevicePropertyTransportType)  # pyright: ignore[reportPrivateUsage]


def _transport(device: int) -> bytes:
    status, _size, data = CoreAudio.AudioObjectGetPropertyData(device, _TRANSPORT, 0, b"", 4, None)
    assert status == 0, f"CoreAudio would not say how audio device {device} is connected: OSStatus {status}"
    return struct.pack(">I", struct.unpack("I", data)[0])


def test_only_the_macs_own_microphone_and_speakers_read_as_its_own_and_a_device_that_is_gone_is_neither() -> None:
    built_ins = coreaudio.BUILT_INS
    devices = coreaudio._devices()  # pyright: ignore[reportPrivateUsage]
    # The Mac's own microphone and speakers are built in, and carry sound the way they face; a USB, Bluetooth, virtual or
    # aggregate device is neither. The two facts say only what a device is not: the headphone jack is built in too. A name
    # says no more than they do: a virtual Mac's own pair, CI's runner's, is "Apple Virtual Sound Device".
    could_be = {
        device: (
            _transport(device) == b"bltn" and coreaudio._has_streams(device, coreaudio._INPUT_STREAMS),  # pyright: ignore[reportPrivateUsage]
            _transport(device) == b"bltn" and coreaudio._has_streams(device, coreaudio._OUTPUT_STREAMS),  # pyright: ignore[reportPrivateUsage]
        )
        for device in devices
    }
    sides = {device: (built_ins.microphone(device), built_ins.speakers(device)) for device in devices}
    wrong = [
        (coreaudio._name(device), _transport(device), sides[device])  # pyright: ignore[reportPrivateUsage]
        for device in devices
        if not all(read <= could for read, could in zip(sides[device], could_be[device], strict=True))
    ]
    assert not wrong, f"read as the Mac's own (microphone, speakers), and cannot be: {wrong}"
    if not any(map(any, sides.values())):
        pytest.skip("no device is on the Mac's own microphone or speakers now")
    gone = max(devices) + 1_000_000
    assert (built_ins.microphone(gone), built_ins.speakers(gone)) == (False, False)
