.PHONY: install lint format test

install:
python -m pip install --upgrade pip
python -m pip install -e .[dev]

lint:
ruff check .
black --check .

format:
black .

test:
pytest
