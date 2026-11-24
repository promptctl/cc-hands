.PHONY: check check-unit check-types check-pair check-rest check-outside check-fritter check-wake-training audit

# [LAW:one-source-of-truth] where a target's report log is: each target below that runs pytest writes the report log
# `.pytest_cache/<target>-log.jsonl`, so a run of one does not replace another's, and `make costs-<target>` reads it
# (tests/costs.py): CI runs that after a job's tests, so the job's log holds what each test cost. --no-sync: the locked
# environment is already built, and a sync here would be time a cancelled run does not have.
log = .pytest_cache/$(1)-log.jsonl

costs-%:
	uv run --no-sync python tests/costs.py $(call log,$*)

# SHARD=K/N runs the Kth of N shards of the tests `check-unit`, `check-pair` and `check-outside` select (tests/shards.py), and
# not the fritter's or wake-training's suites: CI's jobs, each a shard, end inside the 5 minutes a job has. Without it a target
# runs all of its tests, as the 1st of 1 shard.
SHARD ?= 1/1

# The unit layer, held to nothing outside the process by the OS (docs/testing.md), and the type checks: what an agent runs
# on every turn. CI runs the two as jobs of their own, the first in shards.
check: check-unit check-types

check-unit:
	uv run pytest --layer=unit --shard=$(SHARD) --report-log=$(call log,$@)

check-types:
	uv run pyright
	cd wake-training && uv run --locked pyright

# The `outside` tests that load the real streaming pair, run in full on the schedule: tests/test_streaming.py.
check-pair:
	uv run pytest -m outside tests/test_streaming.py --shard=$(SHARD) --report-log=$(call log,$@)

# Every other test marked `outside`, then fritter's Go tests and wake-training's suite. With `check` and `check-pair`, the
# whole suite: `check` runs the tests not marked `outside`, and the two together the ones that are. CI runs the three parts as
# jobs of their own, the first in shards.
check-rest: check-outside check-fritter check-wake-training

check-outside:
	uv run pytest -m outside --ignore=tests/test_streaming.py --shard=$(SHARD) --report-log=$(call log,$@)

check-fritter:
	cd fritter && go test -race ./...

check-wake-training:
	cd wake-training && uv run --locked pytest

# The tests marked `outside` under the unit layer's hold, each expected to be refused: one that is not does not need its
# mark. Run it when a test moves down.
audit:
	uv run pytest --layer=audit --report-log=$(call log,$@)
