"""CoreAudio's adapter, against the system's own audio devices."""

import re

import pytest

from hands.voice import coreaudio

# How macOS names the Mac's own microphone and speakers: "MacBook Pro Microphone", "iMac Speakers".
_OWN = re.compile(r"(MacBook( Pro| Air)?|iMac|Mac mini|Mac Studio) (?P<side>Microphone|Speakers)")


def test_only_the_macs_own_microphone_and_speakers_read_as_its_own_and_a_device_that_is_gone_is_neither() -> None:
    built_ins = coreaudio.BUILT_INS
    devices = coreaudio._devices()  # pyright: ignore[reportPrivateUsage]
    named = {device: _OWN.fullmatch(coreaudio._name(device)) for device in devices}  # pyright: ignore[reportPrivateUsage]
    sides = {device: (built_ins.microphone(device), built_ins.speakers(device)) for device in devices}
    # A name says only what a device is not: the Mac's own speakers' device, on an Intel Mac with headphones in its jack,
    # is still named for the speakers and plays through the headphones.
    could_be = {device: (bool(match) and match["side"] == "Microphone", bool(match) and match["side"] == "Speakers") for device, match in named.items()}
    assert all(read <= could for device in devices for read, could in zip(sides[device], could_be[device], strict=True))
    if not any(map(any, sides.values())):
        pytest.skip("no device is on the Mac's own microphone or speakers now")
    gone = max(devices) + 1_000_000
    assert (built_ins.microphone(gone), built_ins.speakers(gone)) == (False, False)
