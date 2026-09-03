.PHONY: install ui doctor test

install:
	uv venv --python 3.12 || uv venv
	uv pip install --python .venv/bin/python --system-certs -e ".[dev]" || uv pip install --python .venv/bin/python -e ".[dev]"

ui: install
	./run.sh

doctor:
	.venv/bin/htm doctor

test:
	.venv/bin/pytest
