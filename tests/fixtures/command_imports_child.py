"""`hands <command>` in a fresh interpreter, writing which of Pipecat and torch it loaded to a file.

    python command_imports_child.py <home> <command> <out>

Each command's reach past this machine has a stand-in: `log` follows until Ctrl-C, which its first wait for a new line
stands in for; `phone` asks Tailscale for the tailnet's name and the machine for its LAN addresses; `login` runs Claude
Code's own login at the terminal."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from standins import replace

from hands.clock import SYSTEM, SystemClock
from hands.daemon import cli

# Only what each command loads itself is loaded here: a stand-in's module is imported where the stand-in is put in place.
if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from hands.brain import pinned, process
    from hands.sessions.child import Children
    from hands.voice import phoneaddress


class Interrupting(SystemClock):
    def sleep_blocking(self, seconds: float) -> None:
        raise KeyboardInterrupt


async def untailed(children: Children) -> str | phoneaddress.Untailed:
    from hands.voice.phoneaddress import Untailed

    return Untailed("a stand-in for Tailscale")


def lan_addresses() -> list[str]:
    return ["192.0.2.1"]


def logged_in(children: Children, config_dir: Path, pin: pinned.Pin, base_url: str, inherited: Mapping[str, str], method: process.Method) -> process.Login:
    from hands.brain.process import Login
    from hands.voice.backends import Account

    return Login(Account("claude.ai", "someone@example.com"), None)


def main() -> None:
    home, command, out = sys.argv[1:]
    match command:
        case "phone":
            from hands.voice import phoneaddress

            replace(phoneaddress.tailnet_name, untailed)
            replace(phoneaddress.lan_addresses, lan_addresses)
        case "login":
            from hands.brain import process

            replace(process.login, logged_in)
        case _:
            pass
    code = cli.main(["--home", home, command], Interrupting() if command == "log" else SYSTEM)
    with open(out, "w") as file:
        json.dump({"code": code, "loaded": sorted({module.split(".")[0] for module in sys.modules} & {"pipecat", "torch"})}, file)


main()
