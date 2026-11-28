"""The build hook: hands' package carries the fritter binary, built from fritter/ for macOS arm64."""

import os
import runpy
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Go 1.25's oldest supported macOS is 12, so that is the oldest a wheel with its binary in it installs on.
TAG = "py3-none-macosx_12_0_arm64"


class FritterHook(BuildHookInterface[Any]):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)
        # No cgo: Go links fritter itself, so the binary's oldest macOS is Go's, which TAG names, not the build host's SDK.
        environ = {**os.environ, "GOOS": "darwin", "GOARCH": "arm64", "CGO_ENABLED": "0"}
        # Where in the package fritter lands, as hands finds it.
        fritter: PurePosixPath = runpy.run_path(str(root / "src" / "hands" / "packaged.py"))["FRITTER"]
        packaged = "hands" / fritter
        # [LAW:one-source-of-truth] one build, into the package's own source tree: a checkout's editable install runs
        # hands from src/, so it finds fritter where an installed wheel does, and a wheel takes the same file in.
        built = root / "src" / packaged
        # [LAW:no-silent-failure] no Go, or a failed build, fails the build: a hands without fritter cannot install it.
        # No VCS stamp: Go stamps a build inside a checkout with the checkout's HEAD, so every commit would change
        # fritter's bytes, which hands.sessions.wrapper.carried compares, and call every session's fritter stale.
        # Without it, fritter's bytes are its sources' alone.
        subprocess.run(["go", "build", "-trimpath", "-buildvcs=false", "-o", str(built), "."], cwd=root / "fritter", env=environ, check=True)
        if version == "editable":
            return
        # The binary is ignored by git, which the wheel's file selection follows, so it is taken in by name.
        build_data["force_include"][str(built)] = str(packaged)
        build_data["pure_python"] = False
        build_data["tag"] = TAG
