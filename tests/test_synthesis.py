"""Each synthesis of hands' TTS is followed in band from the TTS to the device, and is one `speech` event: what of it
played, the gaps inside it, and how fast the TTS made it."""

import pytest
from pipecat.frames.frames import Frame, TTSAudioRawFrame

from hands.sessions.wide import WideEvent, begun
from hands.virtualclock import VirtualClock
from hands.voice.clips import Sentence
from hands.voice.microphone import Playout
from hands.voice.synthesis import Chunking, Cutter, Ended, Made, SpeakingIn, Synthesis, Syntheses, Taken
from test_microphone import Device, Room, _unfailing  # pyright: ignore[reportPrivateUsage]

CHUNKING = Chunking(size=8, rate=16_000)


CLOCK = VirtualClock()


def _synthesis(voice: str = "alba") -> Synthesis:
    return Synthesis(Sentence(voice, f"said in {voice}"), begun(CLOCK))


def _audio(length: int, context: str = "reply") -> TTSAudioRawFrame:
    return TTSAudioRawFrame(b"\x01" * length, 16_000, 1, context_id=context)


def _shape(frames: list[Frame]) -> list[object]:
    """Each frame as what it says: a chunk's audio, a synthesis's start, or its end and how."""
    return [_said(frame) for frame in frames]


def _said(frame: Frame) -> object:
    match frame:
        case TTSAudioRawFrame(audio=audio):
            return audio
        case SpeakingIn(synthesis=synthesis):
            return ("speaking in", synthesis)
        case Ended(synthesis=synthesis, how=how):
            return ("ended", synthesis, how)
        case _:
            raise AssertionError(f"the cutter passed {frame}")


def test_the_cutter_passes_whole_chunks_of_one_synthesis_each_and_ends_each_behind_its_tail_padded_with_silence() -> None:
    cutter, alba, mary = Cutter(), _synthesis("alba"), _synthesis("mary")
    made = Made(alba, 0.01, 0.1, None)
    passed = [
        *cutter.speaking_in(SpeakingIn(alba), CHUNKING),
        *cutter.heard(_audio(5), CHUNKING),
        *cutter.heard(_audio(7), CHUNKING),
        *cutter.made(made, CHUNKING),
        *cutter.speaking_in(SpeakingIn(mary), CHUNKING),
        *cutter.heard(_audio(8), CHUNKING),
    ]
    assert _shape(passed) == [
        ("speaking in", alba),
        b"\x01" * 8,
        b"\x01" * 4 + bytes(4),  # alba's tail never shares a chunk with mary's start
        ("ended", alba, made),
        ("speaking in", mary),
        b"\x01" * 8,
    ]


def test_a_chunk_the_cutter_passes_keeps_the_context_and_destination_of_the_audio_it_came_from() -> None:
    cutter = Cutter()
    cutter.speaking_in(SpeakingIn(_synthesis()), CHUNKING)
    frame = _audio(8, context="turn 3")
    frame.transport_destination = "desk"
    [chunk] = cutter.heard(frame, CHUNKING)
    assert isinstance(chunk, TTSAudioRawFrame) and (chunk.context_id, chunk.transport_destination) == ("turn 3", "desk")


def test_a_barge_in_drops_what_the_cutter_holds_and_ends_the_open_synthesis_once_whatever_of_it_straggles_in() -> None:
    cutter, alba, mary = Cutter(), _synthesis("alba"), _synthesis("mary")
    cutter.speaking_in(SpeakingIn(alba), CHUNKING)
    cutter.heard(_audio(12), CHUNKING)
    assert _shape(cutter.interrupted()) == [("ended", alba, "cut")]
    # The barge-in's stragglers: alba's audio and its Made, already behind the interruption.
    assert cutter.heard(_audio(8), CHUNKING) == []
    assert cutter.made(Made(alba, 0.01, 0.1, None), CHUNKING) == []
    assert cutter.interrupted() == []
    assert _shape(cutter.speaking_in(SpeakingIn(mary), CHUNKING)) == [("speaking in", mary)]


def test_a_synthesis_still_open_as_the_next_opens_was_cut_short() -> None:
    cutter, alba, mary = Cutter(), _synthesis("alba"), _synthesis("mary")
    cutter.speaking_in(SpeakingIn(alba), CHUNKING)
    cutter.heard(_audio(3), CHUNKING)
    assert _shape(cutter.speaking_in(SpeakingIn(mary), CHUNKING)) == [b"\x01" * 3 + bytes(5), ("ended", alba, "cut"), ("speaking in", mary)]


def test_tts_audio_at_another_rate_than_the_speaker_s_is_refused() -> None:
    cutter = Cutter()
    cutter.speaking_in(SpeakingIn(_synthesis()), CHUNKING)
    with pytest.raises(AssertionError, match="24000 Hz reached a speaker at 16000 Hz"):
        cutter.heard(TTSAudioRawFrame(b"\x01" * 8, 24_000, 1), CHUNKING)


def speech(events: list[WideEvent]) -> list[dict[str, object]]:
    """The facts of each of `events`, every one a synthesis's."""
    assert all(event.event == "speech" for event in events)
    return [dict(event.facts) for event in events]


def test_what_goes_missing_between_two_bytes_of_one_synthesis_is_a_gap_and_between_two_syntheses_is_none() -> None:
    events: list[WideEvent] = []
    syntheses, alba, mary = Syntheses(events.append), _synthesis("alba"), _synthesis("mary")
    syntheses.opened(alba)
    syntheses.opened(mary)
    syntheses.took(Taken(((alba, 0.04),), (), missing=0.5, late=2))  # the device ran dry before alba began: no gap
    syntheses.took(Taken(((alba, 0.04),), (), missing=0.02, late=1))  # ran dry inside alba: a gap
    syntheses.took(Taken(((mary, 0.04),), (Ended(alba, Made(alba, 0.5, 2.0, None)),), missing=0.3, late=0))  # between the two
    syntheses.took(Taken(((mary, 0.04),), (Ended(mary, "cut"),), missing=0, late=0))
    assert speech(events) == [
        {"voice": "alba", "played_s": 0.08, "phone_s": 0.0, "gaps": 1, "gap_s": 0.02, "late": 1, "how": "made", "made_s": 2.0, "rtf": 0.25},
        {"voice": "mary", "played_s": 0.08, "phone_s": 0.0, "gaps": 0, "gap_s": 0.0, "late": 0, "how": "cut", "made_s": None, "rtf": None},
    ]


def test_a_synthesis_the_tts_said_it_failed_is_a_failed_event_and_one_that_made_no_sound_has_no_real_time_factor() -> None:
    events: list[WideEvent] = []
    syntheses, alba = Syntheses(events.append), _synthesis()
    syntheses.opened(alba)
    syntheses.took(Taken((), (Ended(alba, Made(alba, 0.3, 0.0, "Unknown error occurred: no voice state")),), 0, 0))
    [event] = events
    assert (event.outcome, event.error, event.facts["how"], event.facts["made_s"], event.facts["rtf"]) == ("failed", "Unknown error occurred: no voice state", "made", 0.0, None)


def test_a_cue_inside_a_synthesis_does_not_hide_the_gap_after_it() -> None:
    events: list[WideEvent] = []
    syntheses, alba = Syntheses(events.append), _synthesis()
    device = Device()
    playout = Playout(device.open, Room(), 16_000, 1, _unfailing, syntheses.took)
    syntheses.opened(alba)
    playout.write(b"\x01" * 640, alba)
    device.take()
    playout.write(b"\x02" * 640, None)  # the talk key's cue, written between two chunks of alba
    device.take()
    device.take()  # nothing written: the device ran dry inside alba
    playout.write(b"\x01" * 640, alba)
    playout.end(Ended(alba, "cut"))
    device.take()
    [spoken] = speech(events)
    assert (spoken["played_s"], spoken["gaps"], spoken["gap_s"]) == (0.04, 1, 0.02)


def test_a_sentence_carried_on_at_the_phone_and_back_has_no_gap_for_its_stretch_on_the_phone() -> None:
    events: list[WideEvent] = []
    syntheses, alba = Syntheses(events.append), _synthesis()
    syntheses.opened(alba)
    syntheses.took(Taken(((alba, 0.04),), (), 0, 0))
    syntheses.phoned(alba, 0.5)
    syntheses.took(Taken(((alba, 0.04),), (Ended(alba, "cut"),), missing=0.5, late=0))
    [spoken] = speech(events)
    assert (spoken["played_s"], spoken["phone_s"], spoken["gaps"]) == (0.08, 0.5, 0)


def test_a_sentence_carried_on_onto_a_reopened_device_has_no_gap_for_the_move() -> None:
    events: list[WideEvent] = []
    syntheses, alba = Syntheses(events.append), _synthesis()
    syntheses.opened(alba)
    syntheses.took(Taken(((alba, 0.04),), (), 0, 0))
    syntheses.moved()  # the speaker attached a new stream, whose device ran without hands' speech until it came
    syntheses.took(Taken(((alba, 0.04),), (Ended(alba, "cut"),), missing=0.3, late=0))
    [spoken] = speech(events)
    assert (spoken["played_s"], spoken["gaps"]) == (0.08, 0)


def test_a_synthesis_whose_end_never_came_out_is_still_one_event_as_the_speaker_shuts() -> None:
    events: list[WideEvent] = []
    syntheses, alba, mary = Syntheses(events.append), _synthesis("alba"), _synthesis("mary")
    syntheses.opened(alba)
    syntheses.opened(mary)
    syntheses.took(Taken(((alba, 0.04),), (Ended(alba, "cut"),), 0, 0))
    syntheses.took(Taken(((mary, 0.04),), (), 0, 0))
    syntheses.close()
    syntheses.close()
    assert [(spoken["voice"], spoken["how"], spoken["played_s"]) for spoken in speech(events)] == [("alba", "cut", 0.04), ("mary", "cut", 0.04)]


def test_the_playout_says_whose_each_byte_taken_is_and_lets_a_synthesis_s_end_out_behind_its_last_byte() -> None:
    reports: list[Taken] = []
    device = Device()
    playout = Playout(device.open, Room(), 16_000, 1, _unfailing, reports.append)
    alba, mary = _synthesis("alba"), _synthesis("mary")
    ended = Ended(alba, "cut")
    device.take()  # nothing written yet: the device plays silence, and nothing is said of it
    assert reports == []
    playout.write(b"\x01" * 400, alba)
    playout.end(ended)
    playout.write(b"\x01" * 400, mary)
    device.take()  # 640 bytes: all of alba's, her end, and 240 of mary's
    device.take()  # the rest of mary's, then silence
    device.take()  # silence, and nothing to say
    assert reports == [Taken(((alba, 0.0125), (mary, 0.0075)), (ended,), missing=0.02, late=0), Taken(((mary, 0.005),), (), missing=0.0, late=0)]


def test_a_playout_closed_with_ends_still_waiting_lets_each_out() -> None:
    reports: list[Taken] = []
    device = Device()
    playout = Playout(device.open, Room(), 16_000, 1, _unfailing, reports.append)
    alba = _synthesis()
    ended = Ended(alba, "cut")
    playout.write(b"\x01" * 400, alba)
    playout.end(ended)
    playout.close()
    assert reports == [Taken((), (ended,), 0.0, 0)]
