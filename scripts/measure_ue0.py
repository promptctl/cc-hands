"""Throwaway: where MLX's time goes on a CI runner (promptctl-testing-px1f.ue0). Never merged."""
import cProfile
import pstats
import subprocess
import time

import mlx.core as mx
import numpy as np
import soundfile

from hands.voice.streaming import GATING, MEL_HOP, STEP, load

print("sysctl", subprocess.run("sysctl -n hw.ncpu hw.memsize machdep.cpu.brand_string; sw_vers -productVersion", shell=True, capture_output=True, text=True).stdout.split("\n"), flush=True)
print("metal", mx.metal.device_info(), mx.default_device(), flush=True)


def clocked(name, work, times):
    started = time.perf_counter()
    for _ in range(times):
        work()
    print(f"sync {name}: {1000 * (time.perf_counter() - started) / times:.3f} ms each over {times}", flush=True)


small = mx.zeros((320,))
clocked("eval small add", lambda: mx.eval(small + 1), 2000)
clocked("all isfinite", lambda: bool(mx.all(mx.isfinite(small))), 2000)
clocked("any item", lambda: mx.any(small > 1).item(), 2000)

started = time.perf_counter()
pair = load(GATING)
print(f"load {time.perf_counter() - started:.1f} s", flush=True)
audio, _ = soundfile.read("tests/fixtures/hands_echo_uncancelled.flac", dtype="float32")
frames = [320, STEP * MEL_HOP]


def decoded(frame):
    pair.session.reset()
    started, words = time.perf_counter(), []
    for start in range(0, len(audio), frame):
        words += list(pair.decoded(audio[start : start + frame]).words)
    words += list(pair.closed().words)
    return time.perf_counter() - started, "".join(said.text for said in words)


print(f"first decode {decoded(320)[0]:.1f} s", flush=True)
base = None
for round_ in range(2):
    for frame in frames:
        seconds, text = decoded(frame)
        base = base or text
        print(f"round {round_} frame {frame}: {seconds:.1f} s, same words {text == base}", flush=True)

profile = cProfile.Profile()
profile.enable()
decoded(320)
profile.disable()
stats = pstats.Stats(profile)
for tottime, cumtime, calls, key in sorted(((v[2], v[3], v[0], k) for k, v in stats.stats.items()), reverse=True)[:8]:
    print(f"{tottime:7.2f} tot {cumtime:7.2f} cum {calls:6d} calls {key[0].split('/')[-1]}:{key[2]}", flush=True)
