#!/usr/bin/env bash
# Apply black + isort formatting to the codebase
set -e

cd "$(dirname "$0")/.."

echo "Sorting imports..."
uv run isort backend/

echo "Formatting code..."
uv run black backend/

echo "Done."
