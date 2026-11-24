"""Throwaway probe for promptctl-voice-oalb.589: every CoreAudio device with its name and data sources."""

import struct

import CoreAudio
import objc

from hands.voice import coreaudio as ca


def fourcc(v):
    return repr(struct.pack(">I", v)) if v is not None else None


def u32(dev, sel, scope):
    addr = ca._address(sel, scope)
    st, _s, data = CoreAudio.AudioObjectGetPropertyData(dev, addr, 0, b"", 4, None)
    return (None if st != 0 else struct.unpack("I", data)[0]), st


def cfstr(dev, sel):
    addr = ca._address(sel)
    st, _s, data = CoreAudio.AudioObjectGetPropertyData(dev, addr, 0, b"", 8, None)
    if st != 0:
        return f"<status {st}>"
    p = struct.unpack("Q", data)[0]
    return str(objc.objc_object(c_void_p=p)) if p else None


G, I, O = CoreAudio.kAudioObjectPropertyScopeGlobal, CoreAudio.kAudioObjectPropertyScopeInput, CoreAudio.kAudioObjectPropertyScopeOutput
for d in ca._devices():
    ins, ist = u32(d, CoreAudio.kAudioDevicePropertyDataSource, I)
    outs, ost = u32(d, CoreAudio.kAudioDevicePropertyDataSource, O)
    tr, _ = u32(d, CoreAudio.kAudioDevicePropertyTransportType, G)
    print(
        "PROBE589", d, repr(ca._name(d)),
        "uid=", cfstr(d, CoreAudio.kAudioDevicePropertyDeviceUID),
        "model=", cfstr(d, CoreAudio.kAudioDevicePropertyModelUID),
        "maker=", cfstr(d, CoreAudio.kAudioObjectPropertyManufacturer),
        "transport=", fourcc(tr),
        "in_streams=", ca._has_streams(d, ca._INPUT_STREAMS), "out_streams=", ca._has_streams(d, ca._OUTPUT_STREAMS),
        "in_source=", fourcc(ins), f"(status {ist})", "out_source=", fourcc(outs), f"(status {ost})",
        "BUILT_INS=", (ca.BUILT_INS.microphone(d), ca.BUILT_INS.speakers(d)),
        flush=True,
    )
