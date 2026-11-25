"""`hands <command>` in a fresh interpreter, writing which of Pipecat, torch and cryptography it loaded to a file.

    python command_imports_child.py <home> <command> <out>

Each command's reach past this machine has a stand-in: `log` follows until Ctrl-C, which the clock every command runs on
makes its first wait; `phone` asks Tailscale for the tailnet's name and the machine for its LAN addresses; `login` runs Claude
Code's own login at the terminal."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from clocks import Interrupting
from standins import replace

from hands.daemon import cli
from hands.virtualclock import VirtualClock

# Only what each command loads itself is loaded here: a stand-in's module is imported where the stand-in is put in place.
if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from hands.brain import pinned, process
    from hands.sessions.child import Children
    from hands.core.wire import ApiRoute
    from hands.voice import phoneaddress


async def untailed(children: Children) -> str | phoneaddress.Untailed:
    from hands.voice.phoneaddress import Untailed

    return Untailed("a stand-in for Tailscale")


def lan_addresses() -> list[str]:
    return ["192.0.2.1"]


def logged_in(children: Children, config_dir: Path, pin: pinned.Pin, route: ApiRoute, inherited: Mapping[str, str], method: process.Method) -> process.Login:
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
    # [LAW:dataflow-not-control-flow] every command on the one clock, so `log`, which follows until Ctrl-C, ends where any
    # command that waits would.
    code = cli.main(["--home", home, command], Interrupting(VirtualClock()), processes=cli.KERNEL)
    with open(out, "w") as file:
        json.dump({"code": code, "loaded": sorted({module.split(".")[0] for module in sys.modules} & {"pipecat", "torch", "cryptography"})}, file)


main()
