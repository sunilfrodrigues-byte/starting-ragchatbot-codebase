#!/usr/bin/env bash
# Check formatting and run tests — exits non-zero if anything fails
set -e

cd "$(dirname "$0")/.."

echo "Checking import order..."
uv run isort --check-only backend/

echo "Checking formatting..."
uv run black --check backend/

echo "Running tests..."
cd backend && uv run pytest tests/ -v

echo "All checks passed."
