#!/usr/bin/env bash
# Uruchom lokalny interfejs High Transcription Model (Linux i macOS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Finder / Spotlight na macOS startuje z minimalnym PATH.
export PATH="/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

OS="$(uname -s)"

ffmpeg_hint() {
  if [[ "$OS" == "Darwin" ]]; then
    echo "Brak ffmpeg. Na macOS:  brew install ffmpeg" >&2
    echo "Homebrew: https://brew.sh" >&2
  else
    echo "Brak ffmpeg. Na Ubuntu/Debian:  sudo apt install ffmpeg" >&2
  fi
}

uv_hint() {
  if [[ "$OS" == "Darwin" ]]; then
    echo "Brak uv. Na macOS:  brew install uv" >&2
    echo "albo:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  else
    echo "Brak uv. Instalacja:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  fi
}

if ! command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg_hint
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  UV=uv
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  UV="${HOME}/.local/bin/uv"
elif [[ -x "${HOME}/.cargo/bin/uv" ]]; then
  UV="${HOME}/.cargo/bin/uv"
else
  uv_hint
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Tworzę środowisko .venv …"
  "$UV" venv --python 3.12 || "$UV" venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Instaluję zależności …"
LOG="${TMPDIR:-/tmp}/htm-install.log"
if "$UV" pip install --python .venv/bin/python --system-certs -e ".[dev]" >"$LOG" 2>&1; then
  true
else
  "$UV" pip install --python .venv/bin/python -e ".[dev]"
fi

exec .venv/bin/htm ui "$@"
