"""The desk's end-of-turn detection on the real models. Smart Turn's own code reads the machine's time as it takes audio, so
this module stays in UNMOVED in tests/clockwatch.py, and test_engaged.py, which drives the desk on scripted ears, moves."""

import pytest
from conftest import model_facts
from pipecat.audio.vad.vad_analyzer import VADState
from speech import quiet, spoken

from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice.engaged import SMART_TURN, SMART_TURN_REVISION, STOP_SECS, loaded
from test_engaged import Rig, stopped


class _Ticking(VirtualClock):
    """A clock that passes a tenth of a second at every read, so what is timed between two reads is never zero, and no real
    second is waited."""

    def now(self) -> float:
        self.at += 0.1
        return self.at


async def test_loading_the_models_is_its_own_unit_of_work() -> None:
    events: list[WideEvent] = []
    async with loaded(_Ticking(), 16000, events.append) as ears:
        assert await ears.detect(bytes(1024)) == VADState.QUIET
    [loaded_event] = events
    facts = model_facts(loaded_event)
    # Smart Turn's warm-up took some time on the clock it was given, and how much is on the load.
    warmup = facts.pop("smart_turn_warmup_secs")
    assert isinstance(warmup, float) and warmup > 0
    assert (loaded_event.event, loaded_event.outcome, facts) == (
        "desk.loaded",
        "ok",
        {"smart_turn": SMART_TURN, "smart_turn_revision": SMART_TURN_REVISION, "vad_stop_secs": STOP_SECS},
    )


@pytest.mark.parametrize(("said", "complete"), [("create a ticket to read session history.", True), ("can you tell me about the", False)])
async def test_smart_turn_judges_a_finished_sentence_complete_and_one_broken_off_not(said: str, complete: bool) -> None:
    # As the trigger gives it: the speech as speech, then the silence the detector ends it at, then the question.
    audio = spoken(said)
    async with loaded(VirtualClock(), 16000, lambda event: None) as models:
        for at in range(0, len(audio), 640):
            models.heard(audio[at : at + 640], True)
        models.heard(quiet(STOP_SECS), False)
        assert (await models.judge())[0] is complete


async def test_a_pause_between_words_is_inside_the_turn_heard_by_the_real_models() -> None:
    rig = Rig(verdicts=[])
    # Two phrases with a breath between them, then silence past Smart Turn's own stop: one turn, whatever it judges.
    audio = spoken("create a ticket") + quiet(0.4) + spoken("to read session history") + quiet(4.0)
    async with loaded(rig.clock, 16000, rig.events.append) as models:
        driving = rig.start(models)
        await rig.choose("engaged")
        await rig.hear(*(audio[at : at + 640] for at in range(0, len(audio), 640)))
        # Disengaging is heard after every buffer, so the engagement ends once all of it is heard.
        await rig.drained()
        await rig.choose("off")
        await rig.until(lambda: "deafen" in rig.made)
        await stopped(driving)
    assert rig.made == ["listen", "arm", "start", "stop", "deafen"]
    # The turn ended on its own, by a verdict or by the silence after it, not by the level that disengaged.
    desk = next(event for event in rig.events if event.event == "desk" and event.facts["level"] == "engaged")
    verdicts = [event.facts["complete"] for event in rig.events if event.event == "desk.judged"]
    assert verdicts.count(True) + desk.counts["ended_on_silence"] == 1
