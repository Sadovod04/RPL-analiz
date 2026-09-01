.PHONY: install db-up db-down db-logs test lint fmt clean

install:            ## create venv + install all deps (dev group included)
	uv sync

db-up:              ## start Postgres
	docker compose up -d

db-down:            ## stop Postgres (keeps volume)
	docker compose down

db-logs:
	docker compose logs -f postgres

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
