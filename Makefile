.PHONY: lint format typecheck test coverage precommit docs-serve docs-build

precommit:
	uv run pre-commit run --all-files

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build
