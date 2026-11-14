.PHONY: check check-all audit

# The unit layer, held to nothing outside the process by the OS (docs/testing.md), and the type checks: what an agent runs
# on every turn. Green here is not green on CI, which runs `check-all`. A report log of its own, so it does not replace the
# full run's.
check:
	uv run pytest --layer=unit --report-log=.pytest_cache/unit-log.jsonl
	uv run pyright
	cd wake-training && uv run --locked pyright

# Everything, with no layer's hold: what CI runs, until it runs the layers as tiers.
check-all:
	uv run pytest
	uv run pyright
	cd fritter && go test -race ./...
	cd wake-training && uv run --locked pytest && uv run --locked pyright

# The tests marked `outside` under the unit layer's hold, each expected to be refused: one that is not does not need its
# mark. Run it when a test moves down. A report log of its own, so it does not replace the full run's.
audit:
	uv run pytest --layer=audit --report-log=.pytest_cache/audit-log.jsonl
