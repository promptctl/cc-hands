.PHONY: check

check:
	uv run pytest tests/test_brain_e2e.py
