from __future__ import annotations

import wave
from pathlib import Path

from htm.asr import transcribe_audio
from htm.media import wav_duration
from htm.progress import clamp_pct, queue_percent, render_bar
from htm.types import Settings
from htm.ui import _pack


def _silence_wav(path: Path, seconds: float = 2.0, sr: int = 16_000) -> Path:
    n = int(sr * seconds)
    with wave.open(str(path), "w") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(b"\x00\x00" * n)
    return path


def test_clamp_and_queue_percent() -> None:
    assert clamp_pct(-3) == 0
    assert clamp_pct(140) == 100
    assert queue_percent(0, 0.5, 2) == 25
    assert queue_percent(1, 0.5, 2) == 75
    assert queue_percent(2, 0, 2) == 100
    assert queue_percent(0, 0, 0) == 0


def test_render_bar_escapes_and_width() -> None:
    html = render_bar(42.4, "<b>plik</b>", "a & b")
    assert "width:42.4%" in html
    assert "42%" in html
    assert "<b>" not in html
    assert "&lt;b&gt;plik&lt;/b&gt;" in html
    assert "a &amp; b" in html
    assert 'aria-valuenow="42"' in html


def test_wav_duration(tmp_path: Path) -> None:
    wav = _silence_wav(tmp_path / "a.wav", seconds=1.5)
    dur = wav_duration(wav)
    assert dur is not None
    assert abs(dur - 1.5) < 0.02
    assert wav_duration(tmp_path / "missing.wav") is None


def test_asr_reports_time_progress(tmp_path: Path) -> None:
    wav = _silence_wav(tmp_path / "a.wav", seconds=2.0)

    class Seg:
        def __init__(self, start: float, end: float, text: str) -> None:
            self.start = start
            self.end = end
            self.text = text
            self.words = None
            self.avg_logprob = -0.2

    class Info:
        language = "pl"
        language_probability = 0.99
        duration = 2.0

    class Model:
        def transcribe(self, *args, **kwargs):
            return iter([Seg(0.0, 0.5, "raz"), Seg(0.5, 2.0, "dwa")]), Info()

    ticks: list[tuple[float, str]] = []

    def on_progress(frac: float, msg: str) -> None:
        ticks.append((frac, msg))

    rows, lang, _prob = transcribe_audio(wav, Settings(), model=Model(), progress=on_progress)
    assert lang == "pl"
    assert len(rows) == 2
    assert ticks[0] == (0.0, "Transkrypcja…")
    assert any(abs(f - 0.25) < 1e-6 and "00:00" in m and "00:02" in m for f, m in ticks)
    assert ticks[-1][0] == 1.0


def test_pack_includes_progress_updates() -> None:
    out = _pack([], "status", "log", show_file=True, show_queue=True)
    assert len(out) == 12
    file_u, queue_u = out[-2], out[-1]

    def _get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        if hasattr(obj, "get"):
            return obj.get(key)
        return getattr(obj, key, None)

    assert _get(file_u, "visible") is True
    assert _get(queue_u, "visible") is True
    value = _get(file_u, "value") or ""
    assert "htm-progress-fill" in value
