"""`python -m wake_training.trainer --training_config ... --<step>`: openWakeWord's trainer, `openwakeword.train`,
run unchanged, in the process it was written for, which is Linux's and not the Mac's. Three things differ:

- Worker processes. The trainer feeds its model through a DataLoader whose workers are handed lambdas. Linux forks
  workers, and a forked worker has them already; the Mac spawns them, and must pickle what it hands over, which a
  lambda cannot be. The trainer's workers are forked here, as on Linux.
- TorchScript. The trainer imports speechbrain, whose import puts TorchScript on its legacy executor, and that
  executor's fuser fails on Apple's GPU ("Unknown device for graph fuser") at piper-sample-generator's scripted
  functions. TorchScript is off here, so those functions run as plain PyTorch.
- TFLite. The trainer's `--convert_to_tflite` is a switch whose default is the string "False", which is true, so it
  converts every model it trains, through TensorFlow. hands hears ONNX: the switch's default is False here, so it is
  off unless given.
"""

import argparse
import multiprocessing
import os
import runpy
from typing import Any

_add = argparse.ArgumentParser.add_argument


def _added(parser: argparse.ArgumentParser, *names: str, **settings: Any) -> argparse.Action:
    # The trainer's "False" is meant as False; every other argument, of any parser, is declared as it is written.
    if names == ("--convert_to_tflite",):
        settings["default"] = False
    return _add(parser, *names, **settings)


if __name__ == "__main__":
    os.environ["PYTORCH_JIT"] = "0"  # read once, as torch is imported, which the trainer is about to do
    multiprocessing.set_start_method("fork")
    argparse.ArgumentParser.add_argument = _added  # pyright: ignore[reportAttributeAccessIssue]
    runpy.run_module("openwakeword.train", run_name="__main__", alter_sys=True)
