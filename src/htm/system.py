"""OS-specific helpers so the same app runs on Linux and macOS."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path


def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_apple_silicon() -> bool:
    if not is_macos():
        return False
    machine = platform.machine().lower()
    return machine in {"arm64", "aarch64"}


def extra_bin_dirs() -> list[Path]:
    home = Path.home()
    dirs = [
        Path("/opt/homebrew/bin"),
        Path("/opt/homebrew/sbin"),
        Path("/usr/local/bin"),
        home / ".local" / "bin",
        home / ".cargo" / "bin",
    ]
    return [p for p in dirs if p.is_dir()]


def extend_path() -> None:
    """Put Homebrew / uv on PATH (Finder-launched apps often miss them)."""
    current = os.environ.get("PATH", "")
    parts = [str(p) for p in extra_bin_dirs() if str(p) not in current.split(":")]
    if parts:
        os.environ["PATH"] = ":".join(parts + [current]) if current else ":".join(parts)


def which(cmd: str) -> str | None:
    extend_path()
    return shutil.which(cmd)


def ffmpeg_hint() -> str:
    if is_macos():
        return (
            "Nie znaleziono ffmpeg. Na macOS:  brew install ffmpeg\n"
            "Homebrew: https://brew.sh  — potem otwórz nowe okno Terminala."
        )
    if is_linux():
        return "Nie znaleziono ffmpeg. Na Ubuntu/Debian:  sudo apt install ffmpeg"
    return "Nie znaleziono ffmpeg. Zainstaluj z https://ffmpeg.org/download.html"


def uv_hint() -> str:
    if is_macos():
        return (
            "Brak uv. Na macOS:\n"
            "  brew install uv\n"
            "albo:  curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
    return (
        "Brak uv. Instalacja:\n"
        "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    )


def libsndfile_hint() -> str | None:
    if is_macos():
        return "Jeśli soundfile nie startuje:  brew install libsndfile"
    return None


def describe_host() -> str:
    system = platform.system()
    if is_macos():
        ver = platform.mac_ver()[0] or "?"
        chip = "Apple Silicon" if is_apple_silicon() else "Intel"
        return f"macOS {ver} ({chip}, {platform.machine()})"
    return f"{system} {platform.release()} ({platform.machine()})"


def apply_macos_runtime_env() -> None:
    """Avoid Accelerate/OpenMP oversubscription with Numba on Apple Silicon."""
    if not is_macos():
        return
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
    os.environ.setdefault("NUMBA_NUM_THREADS", "4")
