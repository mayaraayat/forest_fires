.PHONY: lint format typecheck test coverage precommit docs-serve docs-build

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy src tests

test:
	uv run pytest -q

coverage:
	uv run pytest --cov=src --cov-report=term-missing

precommit:
	pre-commit run --all-files

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build
