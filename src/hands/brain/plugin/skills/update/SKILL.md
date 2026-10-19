---
name: update
description: How to update hands and restart it. Load it whenever the user asks hands to update, reinstall, or restart itself from master.
allowed-tools: Bash(hands update)
---

# Updating hands

`hands update` installs the newest hands and restarts it in place: the same terminal and menu-bar item, with every
running session kept. The restart ends your turn, so nothing you say after the command is heard.

Say, in one short sentence, that you are updating and will be back. Then run `hands update`, with a timeout of 600000:
the install can take minutes. When a step fails, it says which one and why, and hands was not restarted: tell the
user that in a sentence.

The user said: "update yourself"
WRONG: you run `hands update`, then say "Updating now."
RIGHT: you say "Updating, back in a minute.", then run `hands update`.
