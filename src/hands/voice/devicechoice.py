"""Which audio device each side of the desk is on, as config.toml chooses it, and where that puts it on the devices there are.

[LAW:one-type-per-behavior] a side follows the system's default (`Auto`), keeps to one device by name (`Named`), or is
given none (`Off`). A named device that is missing puts its side on the default until it is back, so what a side is on
is a function of the choice and the hardware alone (`placed`), which the follower compares as the hardware moves.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Auto:
    """The system's default device, followed as it changes."""


@dataclass(frozen=True)
class Named:
    """The device the system calls `name`, whatever the default is; the default while it is missing."""

    name: str


@dataclass(frozen=True)
class Off:
    """No device: nothing is opened for this side at the desk."""


Choice = Auto | Named | Off


@dataclass(frozen=True)
class Choices:
    input: Choice = Auto()
    output: Choice = Auto()


@dataclass(frozen=True)
class DefaultDevices:
    """CoreAudio's ids for the default input and output devices."""

    input: int
    output: int


# CoreAudio's id for the device with a name, among those that can be heard on or played on, if there is one. A device
# that goes and comes back has a new id.
Find = Callable[[str], int | None]


def _none_found(name: str) -> int | None:
    return None


@dataclass(frozen=True)
class Hardware:
    """What CoreAudio says is there: the defaults, and the way to find a device by name, which is asked only for a side
    that names one: a side on the default has no use for the list of devices."""

    defaults: DefaultDevices
    inputs: Find = _none_found
    outputs: Find = _none_found


@dataclass(frozen=True)
class OnDefault:
    """On the default device, by CoreAudio's id for it."""

    device: int


@dataclass(frozen=True)
class OnNamed:
    """On the device called `name`, by CoreAudio's id for it: one that left and came back is not the one a stream was opened on."""

    name: str
    device: int


@dataclass(frozen=True)
class Closed:
    """On no device."""


Place = OnDefault | OnNamed | Closed


@dataclass(frozen=True)
class Placement:
    """Where each side is, or should be."""

    input: Place
    output: Place


def placed(choices: Choices, hardware: Hardware) -> Placement:
    """Where `choices` put the sides on `hardware`."""
    return Placement(_place(choices.input, hardware.defaults.input, hardware.inputs), _place(choices.output, hardware.defaults.output, hardware.outputs))


def _place(choice: Choice, default: int, find: Find) -> Place:
    match choice:
        case Named(name=name):
            match find(name):
                case None:
                    return OnDefault(default)
                case device:
                    return OnNamed(name, device)
        case Auto():
            return OnDefault(default)
        case Off():
            return Closed()


@dataclass(frozen=True)
class Instead:
    """The default device, opened in place of the one config.toml names, which is missing."""

    missing: str
    device: str


@dataclass(frozen=True)
class Missing:
    """The microphone config.toml names is missing, and the system has no input device to open in its place."""

    missing: str


# What a side is on, by the names the system gives its devices: a device, the default in place of one missing, or none
# at all, which config.toml may choose (Off) and a microphone may be left with, None, where the system has no input device.
PlayedOn = str | Instead | Off
HeardOn = str | Instead | Missing | Off | None


def lost_microphone(heard: HeardOn) -> bool:
    """Whether a microphone on `heard` has no device to open, though config.toml does not say so: a fault."""
    match heard:
        case None | Missing():
            return True
        case str() | Instead() | Off():
            return False


def hears_nothing(heard: HeardOn) -> bool:
    """Whether a microphone on `heard` has no device to open, so a press to talk hears nothing, whoever chose it."""
    return lost_microphone(heard) or isinstance(heard, Off)
