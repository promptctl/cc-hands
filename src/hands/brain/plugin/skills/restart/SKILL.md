---
name: restart
description: How to restart hands without installing anything. Load it whenever the user asks hands to restart itself, or hands is overloaded, choppy, or stuck and a fresh start would help.
allowed-tools: Bash(*-m hands.daemon --home * restart)
---

# Restarting hands

The restart command your instructions name starts hands again in place, on the code, prompt, and setup on disk now: the
same terminal and menu-bar item, with every running session kept. It installs nothing and works wherever hands runs
from. The restart ends your turn, so nothing you say after the command is heard.

Say, in one short sentence, that you are restarting and will be back. Then run the restart command, with a timeout of
600000.

The user said: "restart yourself"
WRONG: you run the restart command, then say "Restarting now."
RIGHT: you say "Restarting, back in a moment.", then run the restart command.
