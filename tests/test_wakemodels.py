"""The files the wake word is heard with: the three a word needs, each fetched from openWakeWord's release as the digest
pinned for it says, once however many ask. The fetch is the test's (`hands.voice.pinnedfile` holds a file to its digest,
tested there): no socket opens and no model loads."""

import asyncio
from pathlib import Path
from typing import get_args

import pytest
from fakefetch import Fetching

from hands.virtualclock import VirtualClock
from hands.voice.pinnedfile import Mismatch, Unserved
from hands.voice.wakemodels import EMBEDDING, MELSPECTROGRAM, PRETRAINED, RELEASE, ROLES, VERSION, Files, WakeModels
from hands.voice.wakeword import Phrase, Pretrained, Trained

JARVIS = PRETRAINED["Hey Jarvis"]


def test_every_wake_word_of_openwakeword_hands_offers_has_its_model_pinned() -> None:
    assert set(PRETRAINED) == set(get_args(Phrase.__value__))


def test_each_file_is_pinned_to_the_release_at_its_version_and_a_digest_of_a_file() -> None:
    pins = [MELSPECTROGRAM, EMBEDDING, *PRETRAINED.values()]
    assert [(pin.url, pin.at, len(pin.digest)) for pin in pins] == [(f"{RELEASE}/{pin.name}", f"openWakeWord {VERSION}", 64) for pin in pins]
    assert len({pin.name for pin in pins}) == len(pins)


def test_a_files_fields_are_the_roles_the_page_asks_the_models_by() -> None:
    files = Files(Path("a"), Path("b"), Path("c"))
    assert (ROLES, files.by_role()) == (("melspectrogram", "embedding", "word"), {"melspectrogram": Path("a"), "embedding": Path("b"), "word": Path("c")})


async def test_a_pretrained_words_three_models_are_fetched_as_their_pins_say(tmp_path: Path) -> None:
    fetching = Fetching()
    models = WakeModels(tmp_path, VirtualClock(), fetching)
    ensured = await models.ensured(Pretrained())
    assert ensured.files == Files(tmp_path / MELSPECTROGRAM.name, tmp_path / EMBEDDING.name, tmp_path / JARVIS.name)
    assert [pin for _, pin in fetching.calls] == [MELSPECTROGRAM, EMBEDDING, JARVIS]
    assert ensured.facts() == {"files": "melspectrogram.onnx, embedding_model.onnx, hey_jarvis_v0.1.onnx", "fetched": "melspectrogram.onnx, embedding_model.onnx, hey_jarvis_v0.1.onnx", "replaced": ""}
    # The next ask is its own: it fetched nothing, and its facts say so.
    assert (await models.ensured(Pretrained())).facts()["fetched"] == ""


async def test_another_pretrained_word_is_its_own_file_and_no_other_words_model_is_fetched(tmp_path: Path) -> None:
    fetching = Fetching()
    ensured = await WakeModels(tmp_path, VirtualClock(), fetching).ensured(Pretrained("Hey Mycroft"))
    assert (ensured.files.word, [pin for _, pin in fetching.calls]) == (tmp_path / PRETRAINED["Hey Mycroft"].name, [MELSPECTROGRAM, EMBEDDING, PRETRAINED["Hey Mycroft"]])


async def test_a_word_the_user_trained_is_heard_with_their_file_where_they_keep_it_and_only_the_two_feature_models_are_fetched(tmp_path: Path) -> None:
    mine = tmp_path / "mine" / "hey_hands.onnx"
    mine.parent.mkdir()
    mine.write_bytes(b"my model")
    fetching = Fetching()
    ensured = await WakeModels(tmp_path / "models", VirtualClock(), fetching).ensured(Trained("Hey Hands", mine))
    assert (ensured.files.word, [pin for _, pin in fetching.calls]) == (mine, [MELSPECTROGRAM, EMBEDDING])


async def test_a_word_the_user_trained_whose_model_is_not_there_is_unserved_naming_it(tmp_path: Path) -> None:
    missing = tmp_path / "mine" / "hey_hands.onnx"
    with pytest.raises(Unserved, match=f"no wake word model at {missing}"):
        await WakeModels(tmp_path / "models", VirtualClock(), Fetching()).ensured(Trained("Hey Hands", missing))


@pytest.mark.parametrize("failure", [OSError("the network is down"), Mismatch("embedding_model.onnx at openWakeWord v0.5.1 is aaa, not the bbb pinned")])
async def test_a_model_that_cannot_be_had_is_unserved_with_the_reason_a_fetch_gave(tmp_path: Path, failure: Exception) -> None:
    fetching = Fetching()
    fetching.failure = failure
    with pytest.raises(Unserved) as unserved:
        await WakeModels(tmp_path, VirtualClock(), fetching).ensured(Pretrained())
    assert (f"openWakeWord {VERSION}'s {MELSPECTROGRAM.name}" in str(unserved.value), str(failure) in str(unserved.value)) == (True, True)


async def test_asks_that_arrive_while_the_models_are_being_fetched_wait_for_that_fetch_and_each_file_is_fetched_once(tmp_path: Path) -> None:
    fetching = Fetching()
    fetching.release.clear()
    models = WakeModels(tmp_path, VirtualClock(), fetching)
    first = asyncio.create_task(models.ensured(Pretrained()))
    await fetching.begun.wait()
    second = asyncio.create_task(models.ensured(Pretrained()))
    await asyncio.sleep(0)
    assert not second.done()
    fetching.release.set()
    assert (await first).files == (await second).files
    assert [pin for _, pin in fetching.calls] == [MELSPECTROGRAM, EMBEDDING, JARVIS]
