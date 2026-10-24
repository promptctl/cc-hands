import pytest

from wake_training.word import Rejected, Word, word


def test_a_phrase_names_its_model_and_what_is_said() -> None:
    hey = word("  Hey   Hands ")
    assert (hey.phrase, hey.spoken, hey.name) == ("Hey Hands", "hey hands", "hey_hands")
    assert word("Okay Hands").name == "okay_hands"
    assert word("What's Up").name == "whats_up"


@pytest.mark.parametrize("said", ["", "   ", "hey_hands", "hey hands!", "hey 2 hands", "'hey", "hey/../hands"])
def test_a_phrase_that_cannot_be_said_or_named_is_rejected(said: str) -> None:
    with pytest.raises(Rejected):
        word(said)


def test_a_word_made_directly_is_held_to_the_same_rule() -> None:
    with pytest.raises(Rejected):
        Word("hey  hands")
