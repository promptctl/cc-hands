.PHONY: check check-pair check-rest audit

# Each target below that runs pytest writes the report log `.pytest_cache/<target>-log.jsonl`: a run of one does not replace
# another's, and CI's cost report reads the log of the target its tier ran (tests/costs.py).

# The unit layer, held to nothing outside the process by the OS (docs/testing.md), and the type checks: what an agent runs
# on every turn, and CI's `unit` tier.
check:
	uv run pytest --layer=unit --report-log=.pytest_cache/$@-log.jsonl
	uv run pyright
	cd wake-training && uv run --locked pyright

# CI's `pair` tier: the real streaming pair's cases, `outside` tests that load the speech models, run in full every time.
check-pair:
	uv run pytest -m outside tests/test_streaming.py --report-log=.pytest_cache/$@-log.jsonl

# CI's `rest` tier: every other test marked `outside`, then fritter's Go tests and wake-training's suite. With `check` and
# `check-pair`, the whole suite: `check` runs the tests not marked `outside`, and the two together the ones that are.
check-rest:
	uv run pytest -m outside --ignore=tests/test_streaming.py --report-log=.pytest_cache/$@-log.jsonl
	cd fritter && go test -race ./...
	cd wake-training && uv run --locked pytest

# The tests marked `outside` under the unit layer's hold, each expected to be refused: one that is not does not need its
# mark. Run it when a test moves down.
audit:
	uv run pytest --layer=audit --report-log=.pytest_cache/$@-log.jsonl
