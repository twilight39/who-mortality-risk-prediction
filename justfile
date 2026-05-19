help:
    just --list --justfile {{justfile()}}

# Install dependencies from uv.lock
sync:
    uv sync

# Run all tests
 test:
    uv run pytest

# Run type checker
typecheck:
    uv run basedpyright

# Run linter
lint:
    uv run ruff check .

# Run formatter
fmt:
    uv run ruff format .

# Convert all .py notebooks to .ipynb
notebook:
    #!/usr/bin/env bash
    find notebooks -name "*.py" | while read -r pyfile; do
        echo "Converting $pyfile to notebook..."
        uv run jupytext --to notebook "$pyfile"
    done

# Regenerate figures for README
figures:
    uv run python -m scripts.generate_figures

# Run all notebooks end-to-end
run-notebooks:
    #!/usr/bin/env bash
    for nb in notebooks/*.ipynb; do
        echo "Executing $nb..."
        uv run jupyter execute "$nb"
    done
