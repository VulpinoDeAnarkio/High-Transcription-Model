"""High Transcription Model — lokalna transkrypcja z identyfikacją mówców."""

__version__ = "1.2.1"


def _patch_pyav_metadata_errors() -> None:
    """Kompatybilność z PyAV >= 19 dla biblioteki faster-whisper.

    faster-whisper przekazuje argument `metadata_errors="ignore"` do `av.open()`.
    W PyAV 19.0.0 usunięto ten parametr, co powodowało:
    TypeError: open() got an unexpected keyword argument 'metadata_errors'.
    """
    try:
        import av

        if getattr(av.open, "_is_htm_patched", False):
            return

        orig_open = av.open

        def _safe_av_open(*args, **kwargs):
            try:
                return orig_open(*args, **kwargs)
            except TypeError as exc:
                if "metadata_errors" in str(exc) and "metadata_errors" in kwargs:
                    kwargs.pop("metadata_errors", None)
                    return orig_open(*args, **kwargs)
                raise

        _safe_av_open._is_htm_patched = True
        av.open = _safe_av_open
    except Exception:
        pass


_patch_pyav_metadata_errors()

