.PHONY: run dev test test-all test-cov lint format install

install:
	uv sync --all-extras

run:
	uv run uvicorn vcon_mac_wtf.main:app --host 0.0.0.0 --port 8000

dev:
	uv run uvicorn vcon_mac_wtf.main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest tests/ -m "not integration" -v

test-all:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -m "not integration" --cov=src/vcon_mac_wtf --cov-report=term-missing

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
