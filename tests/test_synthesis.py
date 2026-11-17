"""Each synthesis hands' TTS makes, cut into the sender's chunks and followed to the device taking its last byte."""

import pytest

from hands.voice.synthesis import Chunk, Delivered, Made, Piece, Syntheses

from delivered import said

SIZE = 4  # the sender's chunk, in bytes


def followed() -> tuple[Syntheses, list[Delivered]]:
    delivered: list[Delivered] = []
    return Syntheses(delivered.append), delivered


def written(syntheses: Syntheses, chunks: list[Chunk]) -> list[Piece]:
    """The pieces of `chunks` as the sender writes them, in order."""
    return [piece for _ in chunks for piece in syntheses.written()]


def facts(delivered: Delivered) -> tuple[object, ...]:
    return (delivered.synthesis.voice, delivered.audio, delivered.played, delivered.phone, delivered.unplayed, delivered.gaps, delivered.gap, delivered.late)


def test_two_syntheses_never_share_a_chunk_and_each_tail_is_padded_to_a_whole_one() -> None:
    syntheses, _ = followed()
    alba, mary = said("alba"), said("mary")
    chunks = syntheses.speaking_in(alba, SIZE)
    chunks += syntheses.heard(b"aaaaaa", SIZE)
    chunks += syntheses.speaking_in(mary, SIZE)
    chunks += syntheses.heard(b"mmm", SIZE)
    chunks += syntheses.stopped(SIZE)
    assert [chunk.audio for chunk in chunks] == [b"aaaa", b"aa\0\0", b"mmm\0"]
    assert [chunk.pieces for chunk in chunks] == [
        (Piece(alba, 4),),
        (Piece(alba, 2), Piece(None, 2)),
        (Piece(mary, 3), Piece(None, 1)),
    ]
    assert written(syntheses, chunks) == [piece for chunk in chunks for piece in chunk.pieces]


def test_a_synthesis_is_delivered_as_its_last_byte_plays_with_what_the_tts_took_to_make_it() -> None:
    syntheses, delivered = followed()
    alba = said("alba")
    chunks = syntheses.speaking_in(alba, SIZE) + syntheses.heard(b"aaaaaa", SIZE) + syntheses.made(Made(alba, 0.5), SIZE)
    pieces = written(syntheses, chunks)
    syntheses.took(pieces[:1], 0, 0)
    assert delivered == []
    syntheses.took(pieces[1:], 0, 0)
    [only] = delivered
    assert facts(only) == ("alba", 6, 6, 0, 0, 0, 0, 0)
    assert only.making == 0.5


def test_a_stall_inside_a_synthesis_is_a_gap_and_the_silence_between_two_is_not() -> None:
    syntheses, delivered = followed()
    alba, mary = said("alba"), said("mary")
    chunks = syntheses.speaking_in(alba, SIZE) + syntheses.heard(b"aaaaaaaa", SIZE)
    chunks += syntheses.speaking_in(mary, SIZE) + syntheses.heard(b"mmmm", SIZE) + syntheses.stopped(SIZE)
    pieces = written(syntheses, chunks)
    syntheses.took(pieces[:1], 10, 0)  # silence before alba's first byte: hers to begin, not a gap
    syntheses.took(pieces[1:2], 6, 1)  # the device ran dry inside alba, and PortAudio said it was late
    syntheses.took(pieces[2:], 20, 0)  # the pause between alba and mary
    assert [facts(each) for each in delivered] == [("alba", 8, 8, 0, 0, 1, 6, 1), ("mary", 4, 4, 0, 0, 0, 0, 0)]


def test_portaudio_late_before_a_synthesis_begins_is_not_charged_to_it() -> None:
    syntheses, delivered = followed()
    alba = said("alba")
    pieces = written(syntheses, syntheses.speaking_in(alba, SIZE) + syntheses.heard(b"aaaa", SIZE) + syntheses.stopped(SIZE))
    syntheses.took(pieces, 4, 1)
    assert [facts(each) for each in delivered] == [("alba", 4, 4, 0, 0, 0, 0, 0)]


def test_a_sentence_moved_to_the_phone_and_back_books_no_gap_for_the_time_at_the_phone() -> None:
    syntheses, delivered = followed()
    alba = said("alba")
    pieces = written(syntheses, syntheses.speaking_in(alba, SIZE) + syntheses.heard(b"aaaaaaaaaaaa", SIZE) + syntheses.stopped(SIZE))
    syntheses.took(pieces[:1], 0, 0)
    syntheses.phoned(pieces[1:2])
    syntheses.took(pieces[2:], 2000, 0)  # what the desk found missing while the phone played
    assert [facts(each) for each in delivered] == [("alba", 12, 8, 4, 0, 0, 0, 0)]


def test_an_interruption_finishes_each_synthesis_it_cut_with_what_played_and_its_audio_whole() -> None:
    syntheses, delivered = followed()
    alba, mary = said("alba"), said("mary")
    chunks = syntheses.speaking_in(alba, SIZE) + syntheses.heard(b"aaaaaaaa", SIZE) + syntheses.made(Made(alba, 0.25), SIZE)
    chunks += syntheses.speaking_in(mary, SIZE) + syntheses.heard(b"mmmmmm", SIZE)  # mary's last two bytes held
    syntheses.took(list(syntheses.written()), 0, 0)  # one of alba's chunks played; the rest the sender drops
    syntheses.interrupted()
    assert [facts(each) for each in delivered] == [("alba", 8, 4, 0, 4, 0, 0, 0), ("mary", 6, 0, 0, 6, 0, 0, 0)]
    # Made before the cut: how long the TTS took is of all it made, as the audio is.
    assert [each.making for each in delivered] == [0.25, None]


def test_a_synthesis_cut_off_by_the_next_after_all_of_it_played_is_delivered_then() -> None:
    syntheses, delivered = followed()
    alba = said("alba")
    pieces = written(syntheses, syntheses.speaking_in(alba, SIZE) + syntheses.heard(b"aaaa", SIZE))
    syntheses.took(pieces, 0, 0)
    assert delivered == []  # the TTS may yet make more of it
    syntheses.speaking_in(said("mary"), SIZE)  # it never said Made: the TTS failed partway
    assert [(facts(each), each.making) for each in delivered] == [(("alba", 4, 4, 0, 0, 0, 0, 0), None)]


def test_audio_with_no_synthesis_open_is_nobodys() -> None:
    syntheses, _ = followed()
    assert [chunk.pieces for chunk in syntheses.heard(b"zzzz", SIZE)] == [(Piece(None, 4),)]


def test_tts_audio_the_speaker_never_cut_is_refused() -> None:
    syntheses, _ = followed()
    with pytest.raises(AssertionError, match="never passed"):
        syntheses.written()


def test_a_synthesis_is_followed_by_identity_not_by_voice() -> None:
    syntheses, delivered = followed()
    first, second = said("alba"), said("alba")
    assert first != second
    pieces = written(syntheses, syntheses.speaking_in(first, SIZE) + syntheses.heard(b"aaaa", SIZE))
    pieces += written(syntheses, syntheses.speaking_in(second, SIZE) + syntheses.heard(b"aaaa", SIZE) + syntheses.stopped(SIZE))
    syntheses.took(pieces[:1], 0, 0)
    syntheses.took(pieces[1:], 8, 0)  # the pause between two sentences in one voice
    assert [(each.synthesis, each.gaps) for each in delivered] == [(first, 0), (second, 0)]
