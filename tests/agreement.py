"""Whether the shards of a run collected the same tests, which is what puts each test in exactly one of them: every shard cuts
the collection alone (tests/shards.py), so a collection that differs between jobs loses or doubles a test and no job fails.

Each shard records its number, the number of shards, the groups of weights its tests load (None for a run not told), and a
digest of the test ids it collected; the verdict job of test.yml and outside.yml holds what they recorded to one with

    python3 -I tests/agreement.py RECORD...

It reads only the standard library, since the verdict job runs on a runner with no environment built.
"""

import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def digest(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(ids))).encode()).hexdigest()


def held_to_one(records: Mapping[str, Mapping[str, Any]]) -> None:
    """Raise unless `records`, each shard's by its name, are every shard of each set of groups of weights, each once, and one
    digest in each set: the shards of a set cut the tests that load those groups, and are another set's no more than a test
    is."""
    sets: dict[str, dict[str, Mapping[str, Any]]] = {}
    for name, record in records.items():
        sets.setdefault(json.dumps(record["weights"]), {})[name] = record
    if not sets:
        raise ValueError("no shard recorded what it collected")
    for held in sets.values():
        held_to_one_set(held)


def held_to_one_set(records: Mapping[str, Mapping[str, Any]]) -> None:
    totals = {record["total"] for record in records.values()}
    shards = sorted(record["shard"] for record in records.values())
    if len(totals) != 1 or shards != list(range(1, next(iter(totals)) + 1)):
        raise ValueError(f"the shards that recorded are {shards} of {sorted(totals)}, not each of 1 to N once")
    if len({record["digest"] for record in records.values()}) != 1 or not all(record["digest"] for record in records.values()):
        listing = "\n".join(f"{name}: shard {record['shard']} collected {record['digest'] or 'nothing'}" for name, record in sorted(records.items()))
        raise ValueError(f"the shards collected different tests, so one may run in none:\n{listing}")


if __name__ == "__main__":
    try:
        held_to_one({each: json.loads(Path(each).read_text()) for each in sys.argv[1:]})
    except (ValueError, KeyError, TypeError) as error:
        sys.exit(f"{type(error).__name__}: {error}")
