"""`python -m wake_training.trainer --training_config ... --<step>`: openWakeWord's trainer, `openwakeword.train`,
run unchanged, in the process it was written for, which is Linux's and not the Mac's. Two things differ:

- Worker processes. The trainer feeds its model through a DataLoader whose workers are handed lambdas. Linux forks
  workers, and a forked worker has them already; the Mac spawns them, and must pickle what it hands over, which a
  lambda cannot be. The trainer's workers are forked here, as on Linux.
- TorchScript. The trainer imports speechbrain, whose import puts TorchScript on its legacy executor, and that
  executor's fuser fails on Apple's GPU ("Unknown device for graph fuser") at piper-sample-generator's scripted
  functions. TorchScript is off here, so those functions run as plain PyTorch.
"""

import multiprocessing
import os
import runpy

if __name__ == "__main__":
    os.environ["PYTORCH_JIT"] = "0"  # read once, as torch is imported, which the trainer is about to do
    multiprocessing.set_start_method("fork")
    runpy.run_module("openwakeword.train", run_name="__main__", alter_sys=True)
