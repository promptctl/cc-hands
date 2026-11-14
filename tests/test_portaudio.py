"""`hands devices`: it prints the names hands matches a configured device by, and the system's default among them."""

from collections.abc import Iterator

import pytest

from hands.daemon import cli, exits
from hands.sessions.wide import begun, unit
from hands.virtualclock import VirtualClock
from hands.voice.portaudio import Listed, Lists, default_input, default_output, device_index, listing

CURLY = "Brandon’s AirPods"  # macOS spells the apostrophe in a device's name curly; a typed one is straight


@pytest.fixture(autouse=True)
def command_unit() -> Iterator[None]:
    """The unit of work a command runs inside, which the devices listed land on."""
    with unit("hands.command", lambda _event: None, begun(VirtualClock())):
        yield


def device(name: str, inputs: int, outputs: int) -> dict[str, object]:
    return {"name": name, "maxInputChannels": inputs, "maxOutputChannels": outputs}


class Listing:
    """PortAudio listing `devices`, with the system defaults at `heard` and `played` among them: None where it has none."""

    def __init__(self, devices: list[dict[str, object]], heard: str | None, played: str | None) -> None:
        self.devices = devices
        self.heard = heard
        self.played = played
        self.ended = False

    def get_device_count(self) -> int:
        return len(self.devices)

    def get_device_info_by_index(self, device_index: int) -> object:
        return self.devices[device_index]

    def get_default_input_device_info(self) -> object:
        if self.heard is None:
            raise OSError("No Default Input Device Available")
        return {"name": self.heard}

    def get_default_output_device_info(self) -> object:
        if self.played is None:
            raise OSError("No Default Output Device Available")
        return {"name": self.played}

    def terminate(self) -> None:
        self.ended = True


MAC = [
    device("MacBook Pro Microphone", 1, 0),
    device("MacBook Pro Speakers", 0, 2),
    device(CURLY, 1, 2),
    device("Aggregate Device", 0, 0),
]


def test_each_side_lists_the_devices_that_carry_sound_that_way_and_marks_the_default() -> None:
    mac = Listing(MAC, heard=CURLY, played="MacBook Pro Speakers")
    assert listing(mac, "maxInputChannels", default_input(mac)) == (Listed("MacBook Pro Microphone", False), Listed(CURLY, True))
    assert listing(mac, "maxOutputChannels", default_output(mac)) == (Listed("MacBook Pro Speakers", True), Listed(CURLY, False))


def test_every_name_listed_is_a_name_the_opener_finds_whatever_its_characters() -> None:
    mac = Listing(MAC, heard=None, played=None)
    for channels in ("maxInputChannels", "maxOutputChannels"):
        for listed in listing(mac, channels, None):
            assert device_index(mac, listed.name, channels) is not None
    # The straight apostrophe a user types is no device: the opener would say it is missing.
    assert device_index(mac, "Brandon's AirPods", "maxInputChannels") is None


def test_a_mac_with_no_default_marks_none_and_a_device_without_a_name_is_not_listed() -> None:
    bare = Listing([{"maxInputChannels": 1, "maxOutputChannels": 1}, device("Studio Display Microphone", 1, 0)], heard=None, played=None)
    assert listing(bare, "maxInputChannels", default_input(bare)) == (Listed("Studio Display Microphone", False),)


def test_devices_prints_what_can_be_heard_on_then_what_can_be_played_on(capsys: pytest.CaptureFixture[str]) -> None:
    mac = Listing(MAC, heard=CURLY, played="MacBook Pro Speakers")
    assert cli.list_devices(lambda: mac) == 0
    assert capsys.readouterr().out == (
        f"To hear on:\n  MacBook Pro Microphone\n  {CURLY} (default)\nTo play on:\n  MacBook Pro Speakers (default)\n  {CURLY}\n"
    )
    assert mac.ended


def test_a_side_with_no_device_prints_none(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.list_devices(lambda: Listing([device("Speakers", 0, 2)], heard=None, played="Speakers")) == 0
    assert capsys.readouterr().out == "To hear on:\n  none\nTo play on:\n  Speakers (default)\n"


def test_devices_fails_naming_the_install_when_portaudio_cannot_be_loaded(capsys: pytest.CaptureFixture[str]) -> None:
    def unloadable() -> Lists:
        raise ImportError("dlopen: libportaudio.2.dylib not found")

    ended = cli.list_devices(unloadable)
    assert isinstance(ended, exits.Failed) and ended.code == 1
    assert "brew install portaudio" in capsys.readouterr().err
