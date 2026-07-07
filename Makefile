.PHONY: install test lint format type app clean

install:
	pip install -e ".[dev,app]"
	pre-commit install

test:
	pytest tests/ -q

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/ app/

type:
	mypy src/gdo/

app:
	python app/app.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
