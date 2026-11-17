"""Each synthesis followed from the TTS handing it over to the device taking its last byte, in bytes."""

from hands.sessions.wide import begun
from hands.virtualclock import VirtualClock
from hands.voice.synthesis import Delivered, Made, Piece, Syntheses, Synthesis


def _synthesis(voice: str) -> Synthesis:
    return Synthesis(voice, begun(VirtualClock()))


ALBA, MARY = _synthesis("alba"), _synthesis("mary")


class Followed:
    def __init__(self) -> None:
        self.delivered: list[Delivered] = []
        self.syntheses = Syntheses(self.delivered.append)

    def made(self, synthesis: Synthesis, length: int, making: float = 0.5) -> None:
        """The TTS hands the speaker `synthesis` whole: ahead, its audio, then behind it that it is made."""
        self.syntheses.speaking_in(synthesis)
        self.syntheses.heard(length)
        self.syntheses.made(Made(synthesis, making))


def test_a_stall_inside_a_synthesis_is_a_gap_and_the_silence_after_its_last_byte_is_not() -> None:
    followed = Followed()
    followed.made(ALBA, 300)
    first = followed.syntheses.written(100)
    followed.syntheses.took(first, 0, False)
    followed.syntheses.took([], 50, False)  # the device found nothing written while more of alba's was to come
    rest = followed.syntheses.written(200)
    followed.syntheses.took(rest, 40, False)  # her last byte, then the quiet after it
    assert followed.delivered == [Delivered(ALBA, 0.5, audio=300, played=300, phone=0, unplayed=0, gaps=1, gap=50, late=0)]


def test_a_chunk_pipecat_joined_across_two_syntheses_is_split_where_one_ends() -> None:
    followed = Followed()
    followed.made(ALBA, 150)
    followed.made(MARY, 150)
    written = [followed.syntheses.written(100) for _ in range(4)]
    assert written == [
        [Piece(ALBA, 100)],
        [Piece(ALBA, 50), Piece(MARY, 50)],
        [Piece(MARY, 100)],
        [Piece(None, 100)],  # the sender's padding: nobody's
    ]
    for pieces in written:
        followed.syntheses.took(pieces, 0, False)
    assert [(each.synthesis, each.played) for each in followed.delivered] == [(ALBA, 150), (MARY, 150)]


def test_a_replys_last_synthesis_is_delivered_as_its_last_byte_plays_with_no_next_reply() -> None:
    followed = Followed()
    followed.made(ALBA, 100)
    followed.syntheses.took(followed.syntheses.written(100), 0, False)
    assert [each.synthesis for each in followed.delivered] == [ALBA]
    followed.syntheses.took([], 10_000, True)  # minutes of quiet, a late callback in it: nobody's
    assert len(followed.delivered) == 1


def test_a_device_run_dry_by_a_tts_still_making_the_synthesis_is_a_gap_in_it() -> None:
    followed = Followed()
    followed.syntheses.speaking_in(ALBA)
    followed.syntheses.heard(100)
    followed.syntheses.took(followed.syntheses.written(100), 0, False)
    followed.syntheses.took([], 60, True)  # PortAudio late too: inside alba, as more of her comes
    followed.syntheses.heard(100)
    followed.syntheses.made(Made(ALBA, 2.0))
    assert followed.delivered == []
    followed.syntheses.took(followed.syntheses.written(100), 0, False)
    assert followed.delivered == [Delivered(ALBA, 2.0, audio=200, played=200, phone=0, unplayed=0, gaps=1, gap=60, late=1)]


def test_a_late_callback_that_takes_a_synthesis_is_counted_in_it() -> None:
    followed = Followed()
    followed.made(ALBA, 200)
    followed.syntheses.took(followed.syntheses.written(100), 0, True)
    followed.syntheses.took(followed.syntheses.written(100), 0, False)
    assert [each.late for each in followed.delivered] == [1]


def test_an_interruption_ends_each_synthesis_where_it_was_written_and_audio_after_it_is_nobodys() -> None:
    followed = Followed()
    followed.made(ALBA, 100)
    followed.syntheses.speaking_in(MARY)
    followed.syntheses.heard(300)
    written = followed.syntheses.written(150)  # alba's, and 50 of mary's; the sender holds the rest
    followed.syntheses.interrupted()
    followed.syntheses.heard(80)  # sent before the cut reached the TTS
    followed.syntheses.took(written, 0, False)
    assert [(each.synthesis, each.making, each.audio, each.played) for each in followed.delivered] == [(ALBA, 0.5, 100, 100), (MARY, None, 50, 50)]
    assert followed.syntheses.written(80) == [Piece(None, 80)]


def test_a_synthesis_moved_to_a_new_stream_partway_is_one_delivery_with_what_the_old_one_never_played() -> None:
    followed = Followed()
    followed.made(ALBA, 300)
    to_old = followed.syntheses.written(200)
    followed.syntheses.took([Piece(ALBA, 50)], 0, False)
    followed.syntheses.unplayed([Piece(ALBA, 150)])  # the old stream closes on the rest of what it was written
    followed.syntheses.took(followed.syntheses.written(100), 0, False)  # the new one plays the rest of her
    assert to_old == [Piece(ALBA, 200)]
    assert followed.delivered == [Delivered(ALBA, 0.5, audio=300, played=150, phone=0, unplayed=150, gaps=0, gap=0, late=0)]


def test_a_synthesis_said_at_the_phone_is_delivered_as_handed_to_it_and_the_desk_finds_no_gap_in_it() -> None:
    followed = Followed()
    followed.made(ALBA, 200)
    followed.syntheses.phoned(followed.syntheses.written(100))
    followed.syntheses.took([], 500, True)  # the desk's device, playing nothing of hers
    followed.syntheses.phoned(followed.syntheses.written(100))
    assert followed.delivered == [Delivered(ALBA, 0.5, audio=200, played=0, phone=200, unplayed=0, gaps=0, gap=0, late=0)]
