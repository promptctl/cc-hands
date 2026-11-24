.PHONY: check check-unit check-types check-pair check-rest check-outside check-fritter check-wake-training audit

# Each target below that runs pytest writes the report log `.pytest_cache/<target>-log.jsonl`: a run of one does not replace
# another's, and CI's cost report reads the log of the target its job ran (tests/costs.py).

# SHARD=K/N runs the Kth of N shards of the tests a target selects (tests/shards.py): CI's jobs, each a shard of one tier, end
# inside the 5 minutes a job has. Without it a target runs all of its tests, as the 1st of 1 shard.
SHARD ?= 1/1

# The unit layer, held to nothing outside the process by the OS (docs/testing.md), and the type checks: what an agent runs
# on every turn. CI runs the two as jobs of their own, the first in shards.
check: check-unit check-types

check-unit:
	uv run pytest --layer=unit --shard=$(SHARD) --report-log=.pytest_cache/$@-log.jsonl

check-types:
	uv run pyright
	cd wake-training && uv run --locked pyright

# The `outside` tests that load the real streaming pair, run in full on the schedule: tests/test_streaming.py.
check-pair:
	uv run pytest -m outside tests/test_streaming.py --shard=$(SHARD) --report-log=.pytest_cache/$@-log.jsonl

# Every other test marked `outside`, then fritter's Go tests and wake-training's suite. With `check` and `check-pair`, the
# whole suite: `check` runs the tests not marked `outside`, and the two together the ones that are. CI runs the three parts as
# jobs of their own, the first in shards.
check-rest: check-outside check-fritter check-wake-training

check-outside:
	uv run pytest -m outside --ignore=tests/test_streaming.py --shard=$(SHARD) --report-log=.pytest_cache/$@-log.jsonl

check-fritter:
	cd fritter && go test -race ./...

check-wake-training:
	cd wake-training && uv run --locked pytest

# The tests marked `outside` under the unit layer's hold, each expected to be refused: one that is not does not need its
# mark. Run it when a test moves down.
audit:
	uv run pytest --layer=audit --report-log=.pytest_cache/$@-log.jsonl
