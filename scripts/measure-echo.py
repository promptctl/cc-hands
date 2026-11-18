"""Score an echo-cancelling chain on recorded takes by two numbers: hands' words leaked and a person's words lost, with the
canceller's ERLE beside them (tests/echoscore.py has what each counts, and what a take and a chain are).

    uv run scripts/measure-echo.py [CHAIN ...]    # every chain of echoscore.CHAINS if none given

Each take of scripts/record-double-talk.py (tests/fixtures, `echoscore.TAKES`) is run through the chain, and the cleaned
sound is heard by the real streaming pair as one clip, begun afresh for each take. Nothing here plays or records sound: the
takes are what the desk heard once. `raw`, the microphone with no canceller, is the control: a harness that cannot hear it
leak hears nothing. The numbers are the pair's on this machine's MLX device, and are the same on every run.
"""

import sys
from pathlib import Path

# Last on the path, so nothing in tests/ stands in for a module of the same name installed.
sys.path.append(str(Path(__file__).resolve().parent.parent / "tests"))
from echoscore import CHAINS, TAKES, heard_words, recorded, scored, summary  # noqa: E402

from hands.voice import streaming  # noqa: E402


def main(names: list[str]) -> None:
    unknown = [name for name in names if name not in CHAINS]
    if unknown:
        sys.exit(f"no chain {', '.join(unknown)}; the chains are {', '.join(CHAINS)}")
    pair = streaming.load(streaming.GATING)
    takes = [recorded(fixture) for fixture in TAKES]
    for name in names or list(CHAINS):
        print(f"chain {name}")
        scores = []
        for take in takes:
            cleaned = CHAINS[name](take)
            scores.append(scored(take, cleaned, heard_words(pair, cleaned)))
            print("  " + summary(scores[-1]), flush=True)
        print(f"  {'total':36} leaked {sum(len(each.leaked) for each in scores):2}  lost {sum(len(each.lost) for each in scores):2} of {sum(each.said for each in scores):2}  stray {sum(len(each.stray) for each in scores):2}")


if __name__ == "__main__":
    main(sys.argv[1:])
