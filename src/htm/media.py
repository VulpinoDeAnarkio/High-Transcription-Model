from __future__ import annotations

import subprocess
from pathlib import Path

from htm.system import extend_path, ffmpeg_hint, which

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".aiff", ".aif"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".qt"}


class MediaError(RuntimeError):
    pass


def require_ffmpeg() -> str:
    extend_path()
    path = which("ffmpeg")
    if not path:
        raise MediaError(ffmpeg_hint())
    return path


def extract_wav(source: Path, dest: Path, sr: int = 16_000) -> Path:
    """Convert any ffmpeg-readable media to 16 kHz mono PCM WAV."""
    ffmpeg = require_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise MediaError("ffmpeg nie wyodrębnił audio:\n" + "\n".join(tail))
    if not dest.exists() or dest.stat().st_size < 1024:
        raise MediaError(f"ffmpeg nie zapisał pliku audio: {dest}")
    return dest


def is_media(path: Path) -> bool:
    suf = path.suffix.lower()
    return suf in AUDIO_SUFFIXES or suf in VIDEO_SUFFIXES
