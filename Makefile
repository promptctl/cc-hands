.PHONY: check check-unit check-types check-outside check-fritter check-wake-training fetch-weights audit

# Each target below that runs pytest writes the report log `.pytest_cache/<target>-log.jsonl`: a run of one does not replace
# another's, and CI's cost report reads the log of the target its job ran (tests/costs.py).

# The unit layer, held to nothing outside the process by the OS (docs/testing.md), and the type checks: what an agent runs
# on every turn.
check: check-unit check-types

# The unit layer. SHARD=K/N runs the Kth of N shards of it (tests/shards.py): test.yml's `unit` jobs each run one, and a run that
# names none is the 1st of 1.
SHARD ?= 1/1
check-unit:
	uv run pytest --layer=unit --shard=$(SHARD) --report-log=.pytest_cache/$@-log.jsonl

# test.yml's `types` job.
check-types:
	uv run pyright
	cd wake-training && uv run --locked pyright

# The tests marked `outside`, with no hold: the speech models, the pinned Claude Code, a Chrome page. outside.yml's jobs each run
# one shard (SHARD=K/N) of the tests that load exactly the groups of weights WEIGHTS names (hands/voice/pins.py; `WEIGHTS=` none,
# and a WEIGHTS not given on the command line takes every test and fetches every group), as a developer's whole run does.
check-outside:
	uv run pytest --layer=outside --shard=$(SHARD) $(if $(filter command line,$(origin WEIGHTS)),--weights=$(WEIGHTS)) --report-log=.pytest_cache/$@-log.jsonl

# outside.yml's `weights` job: the weights of one group, into the hub's cache, which the job saves for the jobs that restore it.
fetch-weights:
	uv run python tests/fetches.py $(GROUP)

# fritter's Go tests and wake-training's suite, each a job of outside.yml. With `check` and `check-outside`, the whole suite.
check-fritter:
	cd fritter && go test -race ./...

check-wake-training:
	cd wake-training && uv run --locked pytest

# The tests marked `outside` under the unit layer's hold, each expected to be refused: one that is not does not need its
# mark. Run it when a test moves down.
audit:
	uv run pytest --layer=audit --report-log=.pytest_cache/$@-log.jsonl
