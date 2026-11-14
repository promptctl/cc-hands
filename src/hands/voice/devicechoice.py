"""Which audio device each side of the desk is on, as config.toml chooses it, and where that puts it on the devices there are.

[LAW:one-type-per-behavior] a side follows the system's default (`Auto`), keeps to one device by name (`Named`), or is
given none (`Off`). A named device that is missing puts its side on the default until it is back, so what a side is on
is a function of the choice and the hardware alone (`placed`), which the follower compares as the hardware moves.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class Hardware:
    """What CoreAudio says is there: the defaults, and the ids of the devices that can be heard on and played on, by
    the names they go by. A device that goes and comes back has a new id."""

    defaults: DefaultDevices
    inputs: Mapping[str, int] = field(default_factory=dict[str, int])
    outputs: Mapping[str, int] = field(default_factory=dict[str, int])


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


def _place(choice: Choice, default: int, present: Mapping[str, int]) -> Place:
    match choice:
        case Named(name=name) if name in present:
            return OnNamed(name, present[name])
        case Auto() | Named():
            return OnDefault(default)
        case Off():
            return Closed()
