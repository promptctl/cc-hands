"""The second of the microphone a wake sends from before the word (`Ring` in src/hands/voice/detector.js), run in QuickJS: what it
keeps is the newest samples exactly, however the blocks the tap gave them fall across its length."""

from typing import cast

import pytest
from jsmodule import JsModule


class Ring:
    """A ring of `length` samples in QuickJS, fed blocks of a rising count so every sample says which it is."""

    def __init__(self, length: int) -> None:
        self._js = JsModule("detector.js", "Ring")
        self._js.run(f"var ring = new Ring({length}); var next = 0;")

    def push(self, size: int) -> None:
        self._js.run(f"ring.push(Float32Array.from({{length: {size}}}, () => next++))")

    def held(self) -> list[float]:
        """What the ring has, in order, as one run of samples."""
        return cast(list[float], self._js.value("ring.blocks.flatMap(block => Array.from(block))"))

    def emptied(self) -> list[float]:
        return cast(list[float], self._js.value("ring.empty().flatMap(block => Array.from(block))"))


@pytest.mark.parametrize("blocks", [[4, 4, 4], [3, 3, 3, 3], [10], [1] * 12, [5, 2, 9]])
def test_the_ring_holds_exactly_the_newest_samples_however_the_blocks_fall_across_its_length(blocks: list[int]) -> None:
    ring = Ring(8)
    for size in blocks:
        ring.push(size)
    made = sum(blocks)
    assert ring.held() == [float(sample) for sample in range(max(0, made - 8), made)]


def test_a_ring_not_yet_full_holds_all_it_was_given() -> None:
    ring = Ring(8)
    ring.push(3)
    assert ring.held() == [0.0, 1.0, 2.0]


def test_what_a_ring_is_emptied_of_is_in_order_and_it_holds_only_what_comes_after() -> None:
    ring = Ring(8)
    for size in (6, 6):
        ring.push(size)
    assert ring.emptied() == [float(sample) for sample in range(4, 12)]
    ring.push(2)
    assert ring.held() == [12.0, 13.0]
