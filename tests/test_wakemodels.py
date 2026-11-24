"""The files the wake word is heard with: the three a word needs, each fetched from openWakeWord's release as the digest
pinned for it says, once however many ask. The fetch is the test's (`hands.voice.pinnedfile` holds a file to its digest,
tested there): no socket opens and no model loads."""

import asyncio
from pathlib import Path

import pytest
from fakefetch import Fetching

from hands.voice.pinnedfile import Mismatch
from hands.voice.wakemodels import DIGESTS, EMBEDDING, MELSPECTROGRAM, RELEASE, ROLES, VERSION, Files, Unheard, WakeModels
from hands.voice.wakeword import PRETRAINED, Pretrained, Trained

JARVIS = PRETRAINED["Hey Jarvis"]


def test_every_wake_word_of_openwakeword_hands_offers_is_pinned_with_the_two_models_that_hear_it() -> None:
    assert set(DIGESTS) == {MELSPECTROGRAM, EMBEDDING, *PRETRAINED.values()}


def test_a_files_fields_are_the_roles_the_page_asks_the_models_by() -> None:
    files = Files(Path("a"), Path("b"), Path("c"))
    assert (ROLES, files.by_role()) == (("melspectrogram", "embedding", "word"), {"melspectrogram": Path("a"), "embedding": Path("b"), "word": Path("c")})


async def test_a_pretrained_words_three_models_are_fetched_from_the_release_at_their_digests(tmp_path: Path) -> None:
    fetching = Fetching()
    files = await WakeModels(tmp_path, fetching).ensured(Pretrained())
    assert files == Files(tmp_path / MELSPECTROGRAM, tmp_path / EMBEDDING, tmp_path / JARVIS)
    assert [(pin.name, pin.url, pin.digest, pin.at) for _, pin in fetching.calls] == [
        (name, f"{RELEASE}/{name}", DIGESTS[name], f"openWakeWord {VERSION}") for name in (MELSPECTROGRAM, EMBEDDING, JARVIS)
    ]


async def test_another_pretrained_word_is_its_own_file_and_no_other_words_model_is_fetched(tmp_path: Path) -> None:
    fetching = Fetching()
    files = await WakeModels(tmp_path, fetching).ensured(Pretrained("Hey Mycroft"))
    assert (files.word, fetching.names) == (tmp_path / PRETRAINED["Hey Mycroft"], [MELSPECTROGRAM, EMBEDDING, PRETRAINED["Hey Mycroft"]])


async def test_a_word_the_user_trained_is_heard_with_their_file_where_they_keep_it_and_only_the_two_feature_models_are_fetched(tmp_path: Path) -> None:
    mine = tmp_path / "mine" / "hey_hands.onnx"
    mine.parent.mkdir()
    mine.write_bytes(b"my model")
    fetching = Fetching()
    files = await WakeModels(tmp_path / "models", fetching).ensured(Trained("Hey Hands", mine))
    assert (files.word, fetching.names) == (mine, [MELSPECTROGRAM, EMBEDDING])


async def test_a_word_the_user_trained_whose_model_is_not_there_is_unheard_naming_it(tmp_path: Path) -> None:
    missing = tmp_path / "mine" / "hey_hands.onnx"
    with pytest.raises(Unheard, match=f"no wake word model at {missing}"):
        await WakeModels(tmp_path / "models", Fetching()).ensured(Trained("Hey Hands", missing))


@pytest.mark.parametrize("failure", [OSError("the network is down"), Mismatch("embedding_model.onnx at openWakeWord v0.5.1 is aaa, not the bbb pinned")])
async def test_a_model_that_cannot_be_had_is_unheard_with_the_reason_a_fetch_gave(tmp_path: Path, failure: Exception) -> None:
    fetching = Fetching()
    fetching.failure = failure
    with pytest.raises(Unheard) as unheard:
        await WakeModels(tmp_path, fetching).ensured(Pretrained())
    assert (f"openWakeWord {VERSION}'s {MELSPECTROGRAM}" in str(unheard.value), str(failure) in str(unheard.value)) == (True, True)


async def test_asks_that_arrive_while_the_models_are_being_fetched_wait_for_that_fetch_and_each_file_is_fetched_once(tmp_path: Path) -> None:
    fetching = Fetching()
    fetching.release.clear()
    models = WakeModels(tmp_path, fetching)
    first = asyncio.create_task(models.ensured(Pretrained()))
    await fetching.begun.wait()
    second = asyncio.create_task(models.ensured(Pretrained()))
    await asyncio.sleep(0)
    assert not second.done()
    fetching.release.set()
    assert await first == await second
    assert fetching.names == [MELSPECTROGRAM, EMBEDDING, JARVIS]
