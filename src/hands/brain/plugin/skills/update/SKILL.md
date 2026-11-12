---
name: update
description: How to update hands and restart it. Load it whenever the user asks hands to update or reinstall itself, or to take the newest hands.
allowed-tools: Bash(*-m hands.daemon --home * update)
---

# Updating hands

The update command your instructions name brings the code hands runs from up to date and restarts it in place: the same
terminal and menu-bar item, with every running session kept. A hands run from a development checkout pulls that
checkout; any other gets the newest master. The restart ends your turn, so nothing you say after the command is heard.

Say, in one short sentence, that you are updating and will be back. Then run the update command, with a timeout of
600000: the install can take minutes. When a step fails, it says which one and why, and hands was not restarted: tell
the user that in a sentence.

The user said: "update yourself"
WRONG: you run the update command, then say "Updating now."
RIGHT: you say "Updating, back in a minute.", then run the update command.
