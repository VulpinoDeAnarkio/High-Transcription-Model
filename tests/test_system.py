from htm.system import ffmpeg_hint, uv_hint


def test_hints_mention_tools() -> None:
    assert "ffmpeg" in ffmpeg_hint().lower()
    assert "uv" in uv_hint().lower()
