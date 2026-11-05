.PHONY: install download build-reference fit-lolo predict check test lint format

install:
	pip install -e ".[dev]"

download:
	python scripts/01_download_references.py

build-reference:
	python scripts/02_build_reference_effects.py

fit-lolo:
	python scripts/03_fit_lolo.py

predict:
	python scripts/04_predict.py --contexts D,E,F

check:
	python scripts/05_check_submission.py out/predictions.h5ad

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/
