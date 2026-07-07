.PHONY: install test lint format type web clean

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -q

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

type:
	mypy src/gdo/

web:
	cd web && npm run dev

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
