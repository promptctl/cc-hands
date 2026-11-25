"""What the user hears about a draft, a command, or an interrupt, built from what happened to it, never from the model repeating itself.

A draft is read back so the user can check what will be typed before it is, so its text and resolutions are
`spelled` rather than left to the speaker's filter, which says a path as its file name and would have the user
confirm "fix auth" for `src/auth.py` and `lib/auth.py` alike.
"""

import difflib
import itertools
import re
from collections.abc import Iterable, Mapping, Sequence

from hands.core.drafts import DraftAmended, DraftDiscarded, DraftOutcome, DraftStaged, NothingStaged
from hands.core.effects import Command, Key, MaybeTyped, NotTyped, Text, Typed
from hands.core.keyboard import InBackground, KeyboardOutcome, NothingRunning
from hands.core.reach import AtItsDialog, SessionEnded, UnknownSession, Unwrapped
from hands.core import session
from hands.core.session import Known, Mode, PermissionMode, Resolution, SessionId, UnknownMode
from hands.core.spoken import counted, spelled
from hands.core.tmux import Behind, NotInTmux, PaneUnread
from hands.sessions.registry import Listing, Sessions

_WORDS = re.compile(r"[^\s]+|\n")


def readback(outcome: DraftOutcome, name: str) -> str:
    """One spoken reply for the outcome; `name` is how the user knows the session."""
    match outcome:
        case DraftStaged(draft=draft, replaced=None):
            return f"Draft for {name}{_reading(draft.resolutions)}: {_said(draft.text)}"
        case DraftStaged(draft=draft):
            return f"New draft for {name}, replacing the last one{_reading(draft.resolutions)}: {_said(draft.text)}"
        case DraftAmended(before=before, after=after):
            added = [resolution for resolution in after.resolutions if resolution not in before.resolutions]
            return _amended(before.text, after.text, name, _reading(added))
        case DraftDiscarded():
            return f"Discarded the draft for {name}."
        case UnknownSession(session=session):
            return f"There is no session {session}."
        case NothingStaged():
            return f"There is no draft for {name}."
        case SessionEnded():
            return f"{name} has ended, so its draft cannot be staged, changed, or sent."
        case Typed():
            return f"Sent the draft to {name}."
        case NotTyped(input=Text(prompt=text), reason=reason):
            return f"The draft for {name} was not sent, and is no longer staged: {reason}. It said: {_said(text)}"
        case MaybeTyped(input=Text(prompt=text), reason=reason):
            return f"The draft for {name} may have been sent, and is no longer staged: {reason}. Read {name} before sending it again, which may send it twice. It said: {_said(text)}"
        case Unwrapped() as unwrapped:
            return f"{_unreachable(unwrapped, name)} The draft is still staged."
        case AtItsDialog():
            return f"{name} is waiting at a dialog, which would take the draft as its answer. The draft is still staged."


def keyboard_readback(outcome: KeyboardOutcome, name: str) -> str:
    """One spoken reply for what came of a command or an interrupt; `name` is how the user knows the session."""
    match outcome:
        case Typed(input=input):
            return f"Typed {_spoken_input(input)} into {name}."
        case NotTyped(input=input, reason=reason):
            return f"{_spoken_input(input)} was not typed into {name}: {reason}."
        case MaybeTyped(input=input, reason=reason):
            return f"{_spoken_input(input)} may have been typed into {name}: {reason}. Read {name} before typing it again."
        case NothingRunning():
            return f"{name} is at its prompt, so there is nothing to interrupt."
        case InBackground(subagents=subagents):
            return f"{name} is at its prompt, so there is nothing to interrupt. It has {counted(subagents, 'subagent')} working in the background, which Escape does not stop, so nothing was typed."
        case UnknownSession(session=session):
            return f"There is no session {session}."
        case SessionEnded():
            return f"{name} has ended."
        case Unwrapped() as unwrapped:
            return _unreachable(unwrapped, name)
        case AtItsDialog():
            return f"{name} is waiting at a dialog, which would take the command as its answer. Nothing was sent."


def _unreachable(unwrapped: Unwrapped, name: str) -> str:
    match unwrapped.pane:
        case Behind(pane=pane):
            return f"{name} was not started under fritter, and another program has the keyboard of its tmux pane {pane.id}, so hands cannot type into it: it is stopped, runs inside another program such as an editor or ssh, or was started from inside another session."
        case NotInTmux():
            return f"{name} was not started under fritter and runs in no tmux pane, so hands cannot type into it."
        case PaneUnread(reason=reason):
            return f"{name} was not started under fritter, and which tmux pane it runs in could not be read, so hands cannot type into it: {reason}."


def _spoken_input(input: Command | Key) -> str:
    match input:
        case Command():
            return input.typed
        case Key(key=key):
            return key.replace("_", " ").capitalize()


def identifier(listing: Listing[Known]) -> str:
    """A listed session as it is spoken and addressed."""
    return session.identifier(listing.session.membership.cwd, listing.name)


# Each mode as the footer of a session's own screen names it, so what is heard is what the user would read there:
# The footer shows the default mode as "manual mode on".
_MODES: Mapping[PermissionMode, str] = {
    "default": "manual mode",
    "acceptEdits": "accept edits mode",
    "plan": "plan mode",
    "auto": "auto mode",
    "dontAsk": "don't ask mode",
    "bypassPermissions": "bypass permissions mode",
}


def spoken_mode(mode: Mode) -> str:
    match mode:
        case UnknownMode(name=name):
            return f"a mode hands does not know, named {name}"
        case known:
            return _MODES[known]


def spoken_name(sessions: Sessions, session: SessionId) -> str:
    """How the user knows a session: its identifier, or its id when the registry has never heard of it."""
    listing = sessions.listing(session)
    return session if listing is None else identifier(listing)


def _reading(resolutions: Iterable[Resolution]) -> str:
    return "".join(f", reading '{resolution.heard}' as {spelled(resolution.meant)}" for resolution in resolutions)


# A draft longer than this, in words, is read back as the sentences that changed: past it, hearing the whole draft
# again to find one edit costs the user more than hearing the edit in its sentence.
_LONG = 60

# A sentence ends at a line break, or at a word ending in . ! or ?, closing quotes and brackets after it allowed; a
# list's number, "1.", is not a word that ends one.
_SENTENCE_END = re.compile(r"""[^\d.][.!?]["')\]]*$""")

type _Opcode = tuple[str, int, int, int, int]


def _amended(before: str, after: str, name: str, reading: str) -> str:
    # A line break is a word here, so moving text onto its own line is heard as a change.
    old, new = _WORDS.findall(before), _WORDS.findall(after)
    changes = [op for op in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes() if op[0] != "equal"]
    match (changes, before == after, len(new) - new.count("\n") > _LONG):
        case ([], True, _):
            return f"In the draft for {name}{reading}: nothing changed"
        case ([], False, _):
            return f"In the draft for {name}{reading}: only the spacing changed"
        case (_, _, False):
            return f"New draft for {name}{reading}: {_said(after)}"
        case (_, _, True):
            return f"In the draft for {name}{reading}, {_changed(old, new, changes)}"


def _changed(old: list[str], new: list[str], changes: Sequence[_Opcode]) -> str:
    """A long draft's changes as they now read: each sentence holding a change, whole, in draft order; and a removal
    that took whole sentences with it as what was removed, since no sentence left in the draft holds it."""
    was, now = _sentences(old), _sentences(new)
    # Keyed by where each part sits in the new draft, a removal ahead of the sentence it precedes; a sentence holding
    # several changes is one key, so it is read once.
    parts: dict[tuple[int, int], tuple[str, str]] = {}
    for tag, i1, i2, j1, j2 in changes:
        match tag:
            case "delete" if _between(was, i1) and _between(was, i2):
                parts[(j1, 0)] = ("removed", _spoken([word for word in old[i1:i2] if word != "\n"]))
            case "delete":
                # The sentence the cut was made in: the words after the cut when it began a sentence, else before.
                sentence = now[j1 if _between(was, i1) else j1 - 1]
                parts[(now.index(sentence), 1)] = ("it now says", _sentence(new, now, sentence))
            case _:
                for sentence in sorted(set(now[j1:j2])):
                    parts[(now.index(sentence), 1)] = ("it now says", _sentence(new, now, sentence))
    return " And ".join(
        f"{label}: {' '.join(said for _, said in run)}"
        for label, run in itertools.groupby((parts[key] for key in sorted(parts)), key=lambda part: part[0])
    )


def _sentences(words: list[str]) -> list[int]:
    """Which sentence each word is in, numbered from 0. A line break belongs to the sentence it ends, and a sentence
    starts only at a word, so a run of line breaks never makes a sentence with nothing to say."""
    numbers: list[int] = []
    sentence, ended = 0, False
    for word in words:
        sentence, ended = (sentence + 1, False) if ended and word != "\n" else (sentence, ended)
        numbers.append(sentence)
        ended = ended or word == "\n" or bool(_SENTENCE_END.search(word))
    return numbers


def _between(numbers: list[int], index: int) -> bool:
    """Whether a cut before the word at `index` falls between two sentences, the ends of the draft included."""
    return index in (0, len(numbers)) or numbers[index - 1] != numbers[index]


def _sentence(words: list[str], numbers: list[int], sentence: int) -> str:
    """One sentence as it is read back: its words spelled, the line break that ends it left unsaid."""
    return _spoken([word for word, number in zip(words, numbers) if number == sentence and word != "\n"])


def _said(text: str) -> str:
    """A draft as it is read back: its words `spelled`, its line breaks as words, so the speaker's filter finds no
    list, quote or heading in it to say instead of what will be typed."""
    return _spoken(_WORDS.findall(text))


def _spoken(words: list[str]) -> str:
    return " ".join("a line break" if word == "\n" else spelled(word) for word in words)
