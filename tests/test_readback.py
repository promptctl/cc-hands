"""What the user hears about a draft, from the outcome alone."""

import pytest

from hands.core.drafts import DraftAmended, DraftStaged
from hands.core.session import PromptText, Resolution, SessionId, Staged
from hands.core.spoken import spelled, spoken
from hands.voice.readback import readback

HELPER = Resolution("token helper", "tokenHelper.ts")


def amended(before: str, after: str, *resolutions: Resolution) -> str:
    was = Staged(PromptText(before), ())
    now = Staged(PromptText(after), resolutions)
    return readback(DraftAmended(SessionId("s1"), before=was, after=now), "cc-hands")


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("fix the old bug", "fix the new bug"),
        ("fix the bug", "fix the bug and test it"),
        ("fix the bug quickly", "fix the bug"),
    ],
)
def test_a_short_amended_draft_is_heard_whole_as_it_will_be_typed(before: str, after: str) -> None:
    assert amended(before, after) == f"New draft for cc-hands: {after}"


@pytest.mark.parametrize(
    ("before", "after", "heard"),
    [
        ("fix the  bug", "fix the bug", "only the spacing changed"),
        ("fix the bug", "fix the bug", "nothing changed"),
    ],
)
def test_an_amend_that_changed_no_words_says_so(before: str, after: str, heard: str) -> None:
    assert amended(before, after) == f"In the draft for cc-hands: {heard}"


def test_an_amend_names_only_the_resolutions_it_added() -> None:
    assert amended("use the helper", "use the token helper", HELPER) == (
        "New draft for cc-hands, reading 'token helper' as token Helper dot ts: use the token helper"
    )


# Long enough, at over sixty words, that only what changed in it is read back.
LONG = " ".join(
    [
        "Rename the helper in src/auth.py.",
        *["Keep the rest of the module exactly as it is now."] * 6,
        "Then run the tests.",
        "Tell me when it is done.",
    ]
)


@pytest.mark.parametrize(
    ("after", "heard"),
    [
        (LONG.replace("run the tests", "run the unit tests"), "it now says: Then run the unit tests."),
        # Each sentence holding a change is read whole, in the draft's order, and once however many changes it holds.
        (
            LONG.replace("Rename the", "Move the").replace("Tell me", "Please tell me").replace("it is done", "it is all done"),
            "it now says: Move the helper in src slash auth dot py. Please tell me when it is all done.",
        ),
        # A cut inside a sentence is heard as the sentence that is left.
        (LONG.replace("the tests.", "the"), "it now says: Then run the Tell me when it is done."),
        # A cut that took whole sentences leaves no sentence to read, so what was removed is said.
        (LONG.replace(" Then run the tests.", ""), "removed: Then run the tests."),
        (
            LONG.replace(" Then run the tests.", "").replace("Rename", "Move"),
            "it now says: Move the helper in src slash auth dot py. And removed: Then run the tests.",
        ),
        (LONG + "\nAlso lint it.", "it now says: Tell me when it is done. Also lint it."),
    ],
)
def test_a_long_amended_draft_is_heard_as_the_sentences_that_changed(after: str, heard: str) -> None:
    assert len(LONG.split()) > 60
    assert amended(LONG, after) == f"In the draft for cc-hands, {heard}"


def heard(text: str, *resolutions: Resolution) -> str:
    """A staged draft's readback as the speaker receives it, through the filter every utterance crosses."""
    return spoken(readback(DraftStaged(SessionId("s1"), Staged(PromptText(text), resolutions), replaced=None), "cc-hands")).text


@pytest.mark.parametrize(
    ("draft", "said"),
    [
        ("fix src/auth.py", "fix src slash auth dot py"),
        ("fix lib/auth.py.", "fix lib slash auth dot py."),
        ("fix src/auth.test.ts", "fix src slash auth dot test dot ts"),
        ("edit ~/code/x.py and ~/.zshrc", "edit tilde slash code slash x dot py and tilde slash dot zshrc"),
        ("edit src/auth.py:42", "edit src slash auth dot py colon 42"),
        ("fix scripts/run", "fix scripts slash run"),
        ("see https://x.com/a.py", "see https colon slash slash x dot com slash a dot py"),
        ("rename user_id to userId", "rename user underscore id to user Id"),
        ("run with --force", "run with dash dash force"),
        ("revert a1b2c3d", "revert a 1 b 2 c 3 d"),
        ("check session abcDEF123456xyz", "check session abc DEF 123456 xyz"),
        ("(see notes.md)", "(see notes dot md)"),
        ("call foo.bar() then cd ..", "call foo dot bar open paren close paren then cd dot dot"),
        ("fix `src/auth.py`", "fix backtick src slash auth dot py backtick"),
        ("rm *.py", "rm star dot py"),
        ("echo $HOME/x a|b", "echo dollar HOME slash x a pipe b"),
        ("open parseXMLFile.ts", "open parse XML File dot ts"),
        # Prose the filter leaves alone is left alone: spelling it would only be noise.
        ("the 2nd well-known test, e.g. at 3:30 for 1,000 users", "the 2nd well-known test, e.g. at 3:30 for 1,000 users"),
        # Line breaks are words, so a list, a fence or a quote in a draft is heard as typed, not as its shape.
        ("steps:\n1. fix src/a.py\n2. run it", "steps: a line break 1. fix src slash a dot py a line break 2. run it"),
        ("fix it\n```\nx\n```", "fix it a line break backtick backtick backtick a line break x a line break backtick backtick backtick"),
        ("run x\n\n> quoted", "run x a line break a line break greater than quoted"),
    ],
)
def test_a_draft_is_heard_as_what_will_be_typed(draft: str, said: str) -> None:
    assert heard(draft) == f"Draft for cc-hands: {said}"


@pytest.mark.parametrize(
    ("meant", "said"),
    [
        ("tokenHelper.ts", "token Helper dot ts"),
        ("token_helper.ts", "token underscore helper dot ts"),
        ("token-helper.ts", "token dash helper dot ts"),
        ("/Users/bmf/.hands/wire.sock", "slash Users slash bmf slash dot hands slash wire dot sock"),
        ("a1b2c3d4", "a 1 b 2 c 3 d 4"),
        ("sessionBase64encoder.ts", "session Base 64 encoder dot ts"),
    ],
)
def test_a_resolution_is_heard_as_the_exact_token_it_resolved_to(meant: str, said: str) -> None:
    assert heard("use it", Resolution("the helper", meant)) == f"Draft for cc-hands, reading 'the helper' as {said}: use it"


def test_an_amended_path_is_heard_whole() -> None:
    assert spoken(amended("fix src/auth.py", "fix lib/auth.py")).text == "New draft for cc-hands: fix lib slash auth dot py"
    assert spoken(amended(LONG, LONG.replace("src/auth.py", "lib/auth.py"))).text == (
        "In the draft for cc-hands, it now says: Rename the helper in lib slash auth dot py."
    )


# Each one is something `spoken` rewrites when it is not spelled: a path, a name, a flag, a sha, an id, a
# uuid, an id after the word that names it, a URL.
@pytest.mark.parametrize(
    "token",
    [
        "src/auth.py", "user_id", "HTTP2Handler", "~/code/cc-hands", "a+b@c:d", "tokenHelper", "a1b2c3d4",
        "Xyzabcdefghijklmnop12", "build/a1b2c3d.js", "session sessionBase64encoder.ts", "--max-count=5",
        "0f8fad5b-d9cb-469f-a165-70867728950e", "request abcDEF123456xyz", "https://x.com/a.py", "foo.d.ts",
        "```", "a|b", "foo.bar()", "`src/auth.py`", "| a | b |", "**bold**", "<https://x.com>", "[a](b.md)",
    ],
)
def test_the_speaker_filter_leaves_a_spelled_token_as_written(token: str) -> None:
    assert spoken(spelled(token)).text == spelled(token)
