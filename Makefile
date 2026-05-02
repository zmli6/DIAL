.PHONY: smoke test install clean help

help:
	@echo "make install   — install DIAL in editable mode + dev extras"
	@echo "make smoke     — 5-minute smoke test (no LLM, no env packages)"
	@echo "make test      — run pytest"
	@echo "make clean     — remove caches and build artifacts"

install:
	pip install -e ".[dev]"

smoke:
	@echo "Running smoke test (stub mode, ~30 seconds)..."
	python -m dial.benchmark \
	    --gate examples/threshold_gate.py:ThresholdGate \
	    --stub --episodes 20 --seeds 42 --verbose
	@echo
	@echo "Running smoke test on the calibrated example..."
	python -m dial.benchmark \
	    --gate examples/calibrated_gate.py:CalibratedGate \
	    --stub --episodes 20 --seeds 42

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/
