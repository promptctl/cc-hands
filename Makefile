.PHONY: check

check:
	uv run pytest
	uv run pyright
	cd fritter && go test -race ./...
	cd wake-training && uv run --locked pytest && uv run --locked pyright
