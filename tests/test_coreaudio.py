"""CoreAudio's adapter, against the system's own audio devices."""

import re

import pytest

from hands.voice import coreaudio

# How macOS names the Mac's own microphone and speakers: "MacBook Pro Microphone", "iMac Speakers".
_OWN = re.compile(r"(MacBook( Pro| Air)?|iMac|Mac mini|Mac Studio) (?P<side>Microphone|Speakers)")


@pytest.mark.outside
def test_the_macs_own_microphone_and_speakers_are_told_from_every_other_device_and_a_device_that_is_gone_is_neither() -> None:
    built_ins = coreaudio.BUILT_INS
    devices = coreaudio._devices()  # pyright: ignore[reportPrivateUsage]
    named = {device: _OWN.fullmatch(coreaudio._name(device)) for device in devices}  # pyright: ignore[reportPrivateUsage]
    own = {device: match["side"] for device, match in named.items() if match}
    if not own:
        pytest.skip("this Mac lists no microphone or speakers of its own")
    sides = {device: (built_ins.microphone(device), built_ins.speakers(device)) for device in devices}
    assert sides == {device: (own.get(device) == "Microphone", own.get(device) == "Speakers") for device in devices}
    gone = max(devices) + 1_000_000
    assert (built_ins.microphone(gone), built_ins.speakers(gone)) == (False, False)
