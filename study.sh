#!/usr/bin/env bash
# Run the study helper without activating anything:  ./study.sh   ./study.sh quiz   ./study.sh notes
cd "$(dirname "$0")" || exit 1
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it once with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
exec uv run --all-packages --quiet study "$@"
