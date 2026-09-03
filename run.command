#!/bin/bash
# Dwuklik w Finderze (macOS): otwiera Terminal i startuje UI.
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
chmod +x ./run.sh 2>/dev/null || true
./run.sh
status=$?
if [[ $status -ne 0 ]]; then
  echo
  echo "Zakończono z błędem ($status)."
  read -r -p "Enter, aby zamknąć okno…"
fi
exit "$status"
