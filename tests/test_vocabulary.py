"""Whisper's vocabulary: read from the focused session's repository and the running sessions, newest last."""

import asyncio
import os
import subprocess
from pathlib import Path

import pytest
from mlx_whisper.tokenizer import get_encoding

from hands.core.session import Membership, Running, Session, SessionId
from hands.core.status import Busy, Stamp
from hands.sessions.audit import Entry, Primed, level
from hands.sessions.focus import Unreadable, set_focus
from hands.sessions.home import Home
from hands.sessions.registry import Listing
from hands.voice import vocabulary as lexicon
from hands.virtualclock import VirtualClock
from hands.voice.vocabulary import READING, TOKENS, WORDS, Lexicon, prompt, vocabulary
from conftest import new_sessions


def git(repository: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)


def commit(repository: Path, *names: str) -> None:
    for name in names:
        (repository / name).parent.mkdir(parents=True, exist_ok=True)
        (repository / name).write_text(name)
    git(repository, "add", "-A")
    git(repository, "commit", "-qm", "work")


@pytest.fixture
def clock() -> VirtualClock:
    """A clock no test moves unless it says so: git's real run spends none of the vocabulary's budget, so what git says
    decides, not how fast."""
    return VirtualClock()


def session(cwd: Path, name: str = "s1") -> Session:
    return Session(Membership(SessionId(name), pid=4242, cwd=cwd, transcript=cwd / "t.jsonl"), Running(Busy(), Stamp(1000), None), mode=None)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "shop"
    root.mkdir()
    git(root, "init", "-q", "-b", "auth-rework")
    return root


async def test_the_focused_sessions_newest_files_and_branch_come_before_the_sessions_names(repository: Path, tmp_path: Path, clock: VirtualClock) -> None:
    commit(repository, "src/old_billing.ts")
    commit(repository, "src/authMiddleware.ts", "README.md")
    # Uncommitted work is newer than any commit, and a file renamed is named where it went.
    (repository / "src/sessionStore.ts").write_text("new")
    git(repository, "mv", "README.md", "GUIDE.md")
    focus = session(repository / "src")
    other = session(tmp_path / "cc-hands", "s2")

    primed = await vocabulary(clock, [Listing(focus, None), Listing(other, "dictation bias")], focus, os.environ, clock.now())

    assert primed.words.index("old_billing") < primed.words.index("authMiddleware") < primed.words.index("sessionStore")
    assert "GUIDE" in primed.words and primed.words.count("README") == 1
    assert primed.words[-3:] == ("auth-rework", "src", "cc-hands, dictation bias")
    assert (primed.focus, primed.failed) == (SessionId("s1"), None)
    assert level(primed) == "info"


async def test_a_rename_not_yet_staged_is_named_where_it_went(repository: Path, clock: VirtualClock) -> None:
    commit(repository, "billing.ts")
    (repository / "billing.ts").rename(repository / "ledger.ts")
    git(repository, "add", "-N", "ledger.ts")
    focus = session(repository)

    primed = await vocabulary(clock, [], focus, os.environ, clock.now())

    assert primed.words == ("billing", "ledger", "auth-rework")


async def test_a_word_whisper_refuses_in_a_prompt_is_left_out(repository: Path, clock: VirtualClock) -> None:
    commit(repository, "<|endoftext|>.ts", "ledger.ts")
    focus = session(repository)

    primed = await vocabulary(clock, [], focus, os.environ, clock.now())

    assert primed.words == ("ledger", "auth-rework")
    # As MLX Whisper encodes an initial prompt, which raises on a special token's text.
    get_encoding("multilingual").encode(" " + prompt(primed.words))


async def test_only_the_newest_words_are_kept(repository: Path, monkeypatch: pytest.MonkeyPatch, clock: VirtualClock) -> None:
    # Room for every word, so the count alone decides.
    monkeypatch.setattr(lexicon, "TOKENS", 10_000)
    commit(repository, *(f"old_{n}.py" for n in range(WORDS + 10)))
    commit(repository, "authMiddleware.ts")
    focus = session(repository)

    primed = await vocabulary(clock, [], focus, os.environ, clock.now())

    assert len(primed.words) == WORDS
    assert primed.words[-2:] == ("authMiddleware", "auth-rework")


async def test_the_oldest_words_are_dropped_until_the_rest_fit_the_prompt_tokens_whisper_keeps(repository: Path, clock: VirtualClock) -> None:
    named = [f"hands_dictation_{n}_transcription_server.py" for n in range(WORDS)]
    commit(repository, *named)
    focus = session(repository)

    primed = await vocabulary(clock, [], focus, os.environ, clock.now())

    # The newest words, as many as fit: one more of the older ones would not.
    assert primed.words[-1] == "auth-rework" and 0 < len(primed.words) < WORDS
    assert primed.tokens <= TOKENS < lexicon._tokens((named[-len(primed.words)].removesuffix(".py"), *primed.words))  # pyright: ignore[reportPrivateUsage]


def test_the_prompt_is_counted_as_whisper_encodes_it() -> None:
    # " Brynleigh Fryslie Jaxxon": a leading space, as Whisper encodes an initial prompt, and 10 tokens of its BPE.
    assert lexicon._tokens(("Brynleigh", "Fryslie", "Jaxxon")) == 10  # pyright: ignore[reportPrivateUsage]
    assert lexicon._tokens(()) == 0  # pyright: ignore[reportPrivateUsage]
    # Space-joined: no punctuation of the prompt's own to be read back into what was heard.
    assert prompt(("Brynleigh", "Fryslie", "Jaxxon")) == "Brynleigh Fryslie Jaxxon"


async def test_a_session_outside_any_repository_is_primed_with_the_sessions_alone(tmp_path: Path, clock: VirtualClock) -> None:
    focus = session(tmp_path / "notes")
    (tmp_path / "notes").mkdir()

    primed = await vocabulary(clock, [Listing(focus, "planning")], focus, {**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path)}, clock.now())

    assert (primed.words, primed.failed) == (("notes, planning",), None)


async def test_a_repository_with_no_commits_yet_is_primed_with_what_is_waiting_to_be_committed(repository: Path, clock: VirtualClock) -> None:
    (repository / "authMiddleware.ts").write_text("new")
    focus = session(repository)

    primed = await vocabulary(clock, [Listing(focus, None)], focus, os.environ, clock.now())

    assert (primed.words, primed.failed) == (("authMiddleware", "auth-rework", "shop"), None)


async def test_a_repository_git_cannot_read_primes_nothing_of_its_own_and_says_why(repository: Path, clock: VirtualClock) -> None:
    focus = session(repository)

    primed = await vocabulary(clock, [Listing(focus, None)], focus, {"PATH": "/nonexistent"}, clock.now())

    assert primed.words == ("shop",)
    assert primed.failed is not None and "cannot run git" in primed.failed
    assert level(primed) == "error"


async def test_a_budget_spent_before_git_runs_primes_the_sessions_alone_and_says_why(repository: Path, clock: VirtualClock) -> None:
    commit(repository, "ledger.ts")
    focus = session(repository)

    primed = await vocabulary(clock, [Listing(focus, None)], focus, os.environ, clock.now() - READING)

    assert primed.words == ("shop",)
    assert primed.failed is not None and f"spent the vocabulary's {READING:.1f}s" in primed.failed
    assert primed.seconds == READING
    assert level(primed) == "error"


async def test_a_git_still_running_when_the_budget_is_spent_primes_the_sessions_alone_and_says_why(repository: Path, tmp_path: Path, clock: VirtualClock) -> None:
    stuck = tmp_path / "bin"
    stuck.mkdir()
    (stuck / "git").write_text("#!/bin/sh\nexec /bin/sleep 60\n")
    (stuck / "git").chmod(0o755)
    focus = session(repository)

    reading = asyncio.create_task(vocabulary(clock, [Listing(focus, None)], focus, {**os.environ, "PATH": str(stuck)}, clock.now()))
    await clock.sleeping(READING, count=3)
    clock.advance(READING)
    primed = await reading

    assert primed.words == ("shop",)
    assert primed.failed is not None and f"was still running when reading the vocabulary had spent its {READING:.1f}s" in primed.failed
    assert primed.seconds == READING
    assert level(primed) == "error"


async def test_a_git_that_fails_ends_the_gits_still_running_before_the_vocabulary_is_read(repository: Path, tmp_path: Path, clock: VirtualClock) -> None:
    failing = tmp_path / "bin"
    failing.mkdir()
    (failing / "git").write_text('#!/bin/sh\ncase "$*" in *status*) echo broken >&2; exit 1;; esac\nexec /bin/sleep 60\n')
    (failing / "git").chmod(0o755)
    focus = session(repository)

    reading = asyncio.create_task(vocabulary(clock, [Listing(focus, None)], focus, {**os.environ, "PATH": str(failing)}, clock.now()))
    # git status fails at once, ending its wait; git log and git branch run on to the budget's end, and the vocabulary is
    # not read before they are ended.
    await clock.down_to(READING, count=2, of=3)
    assert not reading.done()
    clock.advance(READING)
    primed = await reading

    assert primed.failed is not None and "git status" in primed.failed and "exited 1: broken" in primed.failed
    assert "git log" in primed.failed and "git branch" in primed.failed
    assert clock.waking == ()


async def test_a_focus_that_cannot_be_read_says_so(tmp_path: Path, clock: VirtualClock) -> None:
    primed = await vocabulary(clock, [], Unreadable("garbled"), os.environ, clock.now())

    assert (primed.focus, primed.words, primed.failed) == (None, (), "the focus: garbled")
    assert level(primed) == "error"


async def test_with_nothing_focused_and_nothing_running_whisper_is_unprimed_and_that_is_recorded(tmp_path: Path, clock: VirtualClock) -> None:
    recorded: list[Entry] = []
    home = Home(tmp_path)
    set_focus(home, None)
    lexicon = Lexicon(new_sessions(clock=clock, record=recorded.append), home, os.environ, recorded.append)

    assert await lexicon() is None
    assert [(entry.focus, entry.words, entry.failed) for entry in recorded if isinstance(entry, Primed)] == [(None, (), None)]
