from __future__ import annotations

from htm.ui import build_ui


def test_build_ui_smoke() -> None:
    demo = build_ui()
    assert demo.title == "High Transcription Model"
