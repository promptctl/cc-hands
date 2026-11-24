# pyright: basic, reportAttributeAccessIssue=false
# [LAW:types-are-the-program] exception: PyObjC ships no type stubs and loads its names lazily, so a static check
# cannot see them.
"""The system's audio devices, from CoreAudio: the defaults and the names of those there are, and word when they change.
In hands, this file alone calls CoreAudio."""

import itertools
import struct
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial

import CoreAudio
import CoreFoundation
import objc
from loguru import logger

from hands.voice.devicechoice import BuiltIns, DefaultDevices, Hardware

_SYSTEM = CoreAudio.kAudioObjectSystemObject


def _address(selector: int, scope: int = CoreAudio.kAudioObjectPropertyScopeGlobal) -> object:
    return CoreAudio.AudioObjectPropertyAddress(selector, scope, CoreAudio.kAudioObjectPropertyElementMain)


_INPUT = _address(CoreAudio.kAudioHardwarePropertyDefaultInputDevice)
_OUTPUT = _address(CoreAudio.kAudioHardwarePropertyDefaultOutputDevice)
_DEVICES = _address(CoreAudio.kAudioHardwarePropertyDevices)
_NAME = _address(CoreAudio.kAudioObjectPropertyName)
_INPUT_STREAMS = _address(CoreAudio.kAudioDevicePropertyStreams, CoreAudio.kAudioObjectPropertyScopeInput)
_OUTPUT_STREAMS = _address(CoreAudio.kAudioDevicePropertyStreams, CoreAudio.kAudioObjectPropertyScopeOutput)
_INPUT_SOURCE = _address(CoreAudio.kAudioDevicePropertyDataSource, CoreAudio.kAudioObjectPropertyScopeInput)
_OUTPUT_SOURCE = _address(CoreAudio.kAudioDevicePropertyDataSource, CoreAudio.kAudioObjectPropertyScopeOutput)
# What device_changes listens for, each by the name a refusal to stop is said in.
_LISTENED = (("the default input", _INPUT), ("the default output", _OUTPUT), ("the device list", _DEVICES))

# The data sources of the Mac's own microphone and speakers, IOAudio's port subtypes 'imic' and 'ispk'. Their transport
# alone, built-in, does not say so: the headphone jack is built in too, a device of its own on Apple silicon and, on an
# Intel Mac, the speakers' own device with its data source moved to 'hdpn'.
_INTERNAL_MICROPHONE = int.from_bytes(b"imic")
_INTERNAL_SPEAKERS = int.from_bytes(b"ispk")


def hardware() -> Hardware:
    """The default devices, and how to find a device by name. The devices are listed and named only when a side names
    one: reading them is no part of following the defaults, and a device mid-teardown can fail the read."""
    return Hardware(DefaultDevices(_device(_INPUT), _device(_OUTPUT)), partial(_found, _INPUT_STREAMS), partial(_found, _OUTPUT_STREAMS))


def _found(scope: object, name: str) -> int | None:
    """The first device listed called `name` that carries sound `scope` ways, as PortAudio lists them: the one a stream is opened on."""
    for device in _devices():
        try:
            if _name(device) == name and _has_streams(device, scope):
                return device
        except _Gone:
            # Torn down between the listing and the reading: it is not there, and its going is a change of its own.
            continue
    return None


def _device(address: object) -> int:
    status, data = _read(_SYSTEM, address)
    _check(status, "say which device is the default")
    return struct.unpack("I", data)[0]


def _devices() -> tuple[int, ...]:
    status, size = CoreAudio.AudioObjectGetPropertyDataSize(_SYSTEM, _DEVICES, 0, b"", None)
    _check(status, "say how many audio devices there are")
    status, size, data = CoreAudio.AudioObjectGetPropertyData(_SYSTEM, _DEVICES, 0, b"", size, None)
    _check(status, "list the audio devices")
    return struct.unpack(f"{size // 4}I", data[:size])


class _Gone(Exception):
    """A device CoreAudio no longer knows."""


# What CoreAudio says of a device that is not there: no such object, or no such device.
_GONE = (CoreAudio.kAudioHardwareBadObjectError, CoreAudio.kAudioHardwareBadDeviceError)


def _of_device(status: int, doing: str) -> None:
    if status in _GONE:
        raise _Gone()
    _check(status, doing)


def _name(device: int) -> str:
    # The name comes as a CFString the caller owns, as pointer bytes PyObjC leaves untyped: wrapped, which retains it
    # again, read, then released once for the reference the read gave.
    status, _size, data = CoreAudio.AudioObjectGetPropertyData(device, _NAME, 0, b"", 8, None)
    _of_device(status, "say an audio device's name")
    pointer = struct.unpack("Q", data)[0]
    # [LAW:no-silent-failure] a null CFString wraps as None, and releasing it ends the process.
    if pointer == 0:
        raise OSError(f"CoreAudio gave no name for audio device {device}")
    named = objc.objc_object(c_void_p=pointer)
    try:
        return str(named)
    finally:
        CoreFoundation.CFRelease(named)


# A four-byte property of a device, read: CoreAudio's status, and the bytes it read.
type _Read = Callable[[int, object], tuple[int, bytes]]


def _read(device: int, address: object) -> tuple[int, bytes]:
    status, _size, data = CoreAudio.AudioObjectGetPropertyData(device, address, 0, b"", 4, None)
    return status, data


def _on_source(read: _Read, address: object, source: int, device: int) -> bool:
    """Whether `device` is on the data source `source` now: read as it is asked, since a headphone plug can move it with
    no new device. A device with no data source, Bluetooth's or a virtual one, is on none, and so is one gone."""
    status, data = read(device, address)
    if status == CoreAudio.kAudioHardwareUnknownPropertyError:
        return False
    try:
        _of_device(status, "say which data source an audio device is on")
    except _Gone:
        return False
    return struct.unpack("I", data)[0] == source


def _built_ins(read: _Read) -> BuiltIns:
    """The Mac's own microphone and speakers, by the data source each device is on as `read` reads it."""
    return BuiltIns(partial(_on_source, read, _INPUT_SOURCE, _INTERNAL_MICROPHONE), partial(_on_source, read, _OUTPUT_SOURCE, _INTERNAL_SPEAKERS))


BUILT_INS = _built_ins(_read)


def _has_streams(device: int, scope: object) -> bool:
    status, size = CoreAudio.AudioObjectGetPropertyDataSize(device, scope, 0, b"", None)
    _of_device(status, "say which way an audio device carries sound")
    return size > 0


# [LAW:no-shared-mutable-globals] exception: CoreAudio carries only an integer to its listener, a module-level
# function, so a call finds its target here by token, and the tokens are counted here. device_changes alone writes
# them: it takes a token and adds its target before it listens, so nothing can call it yet, and deletes the target as
# it stops, under `_calling`, which a target is called under too.
_targets: dict[int, Callable[[], object]] = {}
_tokens = itertools.count(1)
_calling = threading.Lock()


@objc.callbackFor(CoreAudio.AudioObjectAddPropertyListener)
def _listener(_object: int, _count: int, _addresses: object, token: int) -> int:
    # A module-level function and not a block: PyObjC makes a new block for every call it is passed to, so a
    # block cannot be removed again (measured: it went on being called after its removal returned 0).
    # [LAW:no-ambient-temporal-coupling] looked up and called under the lock its deletion takes: a target is not called
    # once its listening has stopped.
    with _calling:
        if (target := _targets.get(token)) is not None:
            try:
                target()
            except Exception:
                # [LAW:no-silent-failure] this runs on CoreAudio's thread, under C frames an exception must not cross,
                # so it is logged where it happened.
                logger.exception("an audio device change could not be passed on")
            return 0
    # [LAW:no-silent-failure] a notice after its listening stopped, one in flight as it stopped or one from a listener
    # CoreAudio would not remove, is for nobody: said, so none is dropped unseen.
    logger.warning("a change to the audio devices came after listening for them had stopped")
    return 0


@contextmanager
def device_changes(changed: Callable[[], object]) -> Iterator[None]:
    """Call `changed` whenever the default input or output device changes, or a device comes or goes, for as long as
    the context is open, and never once it has closed.

    It is called on a CoreAudio thread, within milliseconds: measured at 17 ms from an aggregate device being
    destroyed to the call, once for the input and once for the output. It is called holding the lock the context's
    close waits on, so it returns at once and never closes the context itself.
    """
    token = next(_tokens)
    _targets[token] = changed
    listening: list[tuple[str, object]] = []
    try:
        for name, address in _LISTENED:
            _check(CoreAudio.AudioObjectAddPropertyListener(_SYSTEM, address, _listener, token), "listen for changes to the audio devices")
            listening.append((name, address))
        yield
    finally:
        # Only what was added is removed, so a refused registration is the error that is reported. Removed without the
        # lock, which a notice in flight may be waiting on while CoreAudio waits on it; then the target goes, whatever
        # CoreAudio said, so `changed` is not called once the context has closed, and a listener CoreAudio keeps finds
        # no target and says so.
        refused: list[str] = []
        try:
            for name, address in listening:
                if status := CoreAudio.AudioObjectRemovePropertyListener(_SYSTEM, address, _listener, token):
                    refused.append(f"{name}: OSStatus {status}")
        finally:
            with _calling:
                del _targets[token]
        if refused:
            # [LAW:no-silent-failure] exception: logged, not raised. A raise would take the place of whatever ends the
            # context, as a cancel ends hands' (voice/devices.py follows until cancelled), and leave the caller nothing
            # it could do: what CoreAudio kept is out of its reach, and says each notice it brings.
            logger.error(f"CoreAudio would not stop listening for changes to {'; '.join(refused)}")


def _check(status: int, doing: str) -> None:
    # [LAW:no-silent-failure] a listener CoreAudio refused is a device loss nobody would hear about.
    if status != 0:
        raise OSError(f"CoreAudio would not {doing}: OSStatus {status}")
