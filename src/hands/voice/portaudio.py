"""The audio devices PortAudio lists, which is what hands opens a stream on: the device config.toml names is found here, and `hands devices` prints from here."""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast

# PortAudio's field for how many channels a device has, which way: a device with some can be heard on, or played on.
Channels = Literal["maxInputChannels", "maxOutputChannels"]


class Lists(Protocol):
    """What is used of PyAudio to list devices, which is untyped."""

    def get_default_input_device_info(self) -> object: ...
    def get_default_output_device_info(self) -> object: ...
    def get_device_count(self) -> int: ...
    def get_device_info_by_index(self, device_index: int) -> object: ...
    def terminate(self) -> None: ...


@dataclass(frozen=True)
class Listed:
    """A device PortAudio lists, by the name config.toml names it by, and whether it is the system's default."""

    name: str
    default: bool


def named(py_audio: Lists, channels: Channels) -> Iterator[tuple[int, str]]:
    """The index and name of each device `py_audio` lists with `channels`, in its order: the one enumeration a name is matched against."""
    for index in range(py_audio.get_device_count()):
        # PyAudio's device info is an untyped mapping.
        info = cast(Mapping[str, object], py_audio.get_device_info_by_index(index))
        match info.get("name"), info.get(channels):
            case str(listed), int(count) if count > 0:
                yield index, listed
            case _:
                pass


def device_index(py_audio: Lists, name: str, channels: Channels) -> int | None:
    """The index `py_audio` lists the device called `name` at, among those with `channels`; None where it lists none."""
    return next((index for index, listed in named(py_audio, channels) if listed == name), None)


def listing(py_audio: Lists, channels: Channels, default: str | None) -> tuple[Listed, ...]:
    """Each name `named` finds, once, marked the default where it is `default`: a name opens the first device that bears it, so a second of the same name is not one config.toml can name."""
    return tuple(Listed(listed, listed == default) for listed in dict.fromkeys(listed for _index, listed in named(py_audio, channels)))


# PyAudio's words when PortAudio lists no default, raised with no errno to tell it by.
_NO_DEFAULT_INPUT = "No Default Input Device Available"
_NO_DEFAULT_OUTPUT = "No Default Output Device Available"


def default_input(py_audio: Lists) -> str | None:
    """The default input device's name, or None when the system has none: a Mac with no built-in microphone, its headset unplugged."""
    return _default(py_audio.get_default_input_device_info, _NO_DEFAULT_INPUT)


def default_output(py_audio: Lists) -> str | None:
    """The default output device's name, or None when the system has none."""
    return _default(py_audio.get_default_output_device_info, _NO_DEFAULT_OUTPUT)


def _default(get_info: Callable[[], object], no_default: str) -> str | None:
    # [LAW:parse-dont-validate] PortAudio says there is no default device by raising; here that becomes a typed absence.
    try:
        info = get_info()
    except OSError as error:
        # Asked only of a PortAudio that is running: an ended one says the same words whatever is plugged in.
        if error.args != (no_default,):
            raise
        return None
    return name_of(info)


def name_of(info: object) -> str:
    """The name in a device's info, which PortAudio always gives."""
    # PyAudio's device info is an untyped mapping; the name is the one field read from it.
    match info:
        case {"name": str(name)}:
            return name
        case _:
            raise AssertionError(f"PortAudio described a device without a name: {info!r}")
