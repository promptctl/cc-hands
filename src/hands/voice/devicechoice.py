"""Which audio device each side of the desk is on, as config.toml chooses it, and where that puts it on the devices there are.

[LAW:one-type-per-behavior] a side follows the system's default (`Auto`) or keeps to one device by name (`Named`). A named device that is missing puts its side on the default until it is back, so what a side
is on is a function of the choice and the hardware alone (`placed`), which the follower compares as the hardware moves.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Auto:
    """The system's default device, followed as it changes."""


@dataclass(frozen=True)
class Named:
    """The device the system calls `name`, whatever the default is; the default while it is missing."""

    name: str


Choice = Auto | Named


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
    """What CoreAudio says is there: the defaults, and the names of the devices that can be heard on and played on."""

    defaults: DefaultDevices
    inputs: frozenset[str] = frozenset()
    outputs: frozenset[str] = frozenset()


@dataclass(frozen=True)
class OnDefault:
    """On the default device, by CoreAudio's id for it."""

    device: int


@dataclass(frozen=True)
class OnNamed:
    name: str


Place = OnDefault | OnNamed


@dataclass(frozen=True)
class Placement:
    """Where each side is, or should be."""

    input: Place
    output: Place


def placed(choices: Choices, hardware: Hardware) -> Placement:
    """Where `choices` put the sides on `hardware`."""
    return Placement(_place(choices.input, hardware.defaults.input, hardware.inputs), _place(choices.output, hardware.defaults.output, hardware.outputs))


def _place(choice: Choice, default: int, present: frozenset[str]) -> Place:
    match choice:
        case Named(name=name) if name in present:
            return OnNamed(name)
        case Auto() | Named():
            return OnDefault(default)
