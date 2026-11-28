"""tests/agreement.py holds the shards of a run to one collection: every shard of the run, each once, and one digest among them."""

import pytest

from agreement import digest, held_to_one


def recorded(shard: int, total: int, collected: str) -> dict[str, object]:
    return {"shard": shard, "total": total, "digest": collected}


def test_the_shards_of_a_run_that_collected_the_same_tests_agree() -> None:
    held_to_one({f"collected-{k}": recorded(k, 3, "abc") for k in (1, 2, 3)})


def test_a_digest_is_of_the_ids_collected_and_not_of_the_order_or_the_repeats() -> None:
    assert digest(["b", "a", "b"]) == digest(["a", "b"])
    assert digest(["a", "b"]) != digest(["a", "c"])


@pytest.mark.parametrize(
    "records, said",
    [
        ({}, "no shard recorded"),
        ({"collected-1": recorded(1, 3, "abc"), "collected-3": recorded(3, 3, "abc")}, r"\[1, 3\] of \[3\]"),
        ({"collected-1": recorded(1, 2, "abc"), "collected-2": recorded(1, 2, "abc")}, r"\[1, 1\] of \[2\]"),
        ({"collected-1": recorded(1, 2, "abc"), "collected-2": recorded(2, 3, "abc")}, r"of \[2, 3\]"),
        ({"collected-1": recorded(1, 2, "abc"), "collected-2": recorded(2, 2, "abd")}, "collected-2: shard 2 collected abd"),
        ({"collected-1": recorded(1, 2, ""), "collected-2": recorded(2, 2, "")}, "collected nothing"),
    ],
)
def test_anything_but_every_shard_once_with_one_digest_is_refused_naming_what_each_recorded(records: dict[str, dict[str, object]], said: str) -> None:
    with pytest.raises(ValueError, match=said):
        held_to_one(records)
