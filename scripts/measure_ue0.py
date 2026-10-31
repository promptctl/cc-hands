"""Throwaway: `_Paired` at 20 ms against the session fed directly at 20 ms, on one runner (promptctl-testing-px1f.ue0). Never merged."""
import statistics
import time

import numpy as np
import soundfile

from hands.voice.streaming import GATING, load

pair = load(GATING)
audio, _ = soundfile.read("tests/fixtures/hands_echo_uncancelled.flac", dtype="float32")
frames = [audio[start : start + 320] for start in range(0, len(audio), 320)]


def direct():
    pair.reset()
    started = time.perf_counter()
    text = "".join(delta.text for frame in frames for delta in pair.session.feed(frame))
    return time.perf_counter() - started, text



def through_paired():
    pair.reset()
    started = time.perf_counter()
    text = "".join(said.text for frame in frames for said in pair.decoded(frame).words)
    return time.perf_counter() - started, text


direct()
ratios = []
for round_ in range(3):
    fed, base = direct()
    paired, words = through_paired()
    ratios.append(paired / fed)
    print(f"round {round_}: direct {fed:.1f} s, paired {paired:.1f} s, ratio {paired / fed:.2f}, same words {words.replace(' ', '') == base.replace(' ', '') or len(words)}", flush=True)
print(f"median ratio {statistics.median(ratios):.2f}", flush=True)
