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


def test_a_word_s_adversarial_negatives_are_its_parts_and_its_words_led_by_another() -> None:
    assert word("Hey Hands").adversarial_negatives == ("hands", "hey", "my hands", "okay hands", "the hands", "these hands", "your hands")
    assert "hey hands" in word("Okay Hands").adversarial_negatives and "okay hands" not in word("Okay Hands").adversarial_negatives
    # Each ending is led by another word, the shortest too.
    assert {"okay there hands", "okay hands", "there", "there hands"} <= set(word("Hey There Hands").adversarial_negatives)
    # One word has no parts but itself and no ending after its first word: no adversarial negatives, rather than bare leads that
    # share nothing with it.
    assert word("Computer").adversarial_negatives == ()
