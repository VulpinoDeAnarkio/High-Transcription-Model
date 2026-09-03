#!/usr/bin/env bash
# Uruchom lokalny interfejs High Transcription Model.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Brak ffmpeg. Na Ubuntu/Debian:  sudo apt install ffmpeg" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  UV=uv
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  UV="${HOME}/.local/bin/uv"
else
  echo "Brak uv. Instalacja:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Tworzę środowisko .venv …"
  "$UV" venv --python 3.12 || "$UV" venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Instaluję zależności …"
if "$UV" pip install --python .venv/bin/python --system-certs -e ".[dev]" >/tmp/htm-install.log 2>&1; then
  true
else
  "$UV" pip install --python .venv/bin/python -e ".[dev]"
fi

exec .venv/bin/htm ui "$@"
