from __future__ import annotations

import os
from pathlib import Path

from htm.ssl_util import apply_insecure_ssl, ensure_ssl
from htm.types import Segment, Settings, Word


class AsrError(RuntimeError):
    pass


def _resolve_device(settings: Settings) -> tuple[str, str]:
    device = settings.device
    compute = settings.compute_type
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def _cache_dir(settings: Settings) -> Path:
    if settings.cache_dir:
        return settings.cache_dir
    env = os.environ.get("HTM_CACHE") or os.environ.get("HF_HOME")
    if env:
        return Path(env)
    return Path.cwd() / ".htms-cache"


def load_model(settings: Settings):
    from faster_whisper import WhisperModel

    ensure_ssl(settings.insecure_ssl)
    cache = _cache_dir(settings)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))
    device, compute = _resolve_device(settings)
    try:
        return WhisperModel(
            settings.model,
            device=device,
            compute_type=compute,
            download_root=str(cache),
        )
    except Exception as first:
        if settings.insecure_ssl is False:
            raise AsrError(f"Nie udało się wczytać modelu {settings.model}: {first}") from first
        apply_insecure_ssl()
        try:
            return WhisperModel(
                settings.model,
                device=device,
                compute_type=compute,
                download_root=str(cache),
            )
        except Exception as second:
            raise AsrError(
                f"Nie udało się wczytać modelu {settings.model}. "
                f"Sprawdź sieć / TLS (HTM_INSECURE_SSL=1). Szczegóły: {second}"
            ) from second


def transcribe_audio(audio: Path, settings: Settings, model=None) -> tuple[list[Segment], str | None, float | None]:
    if model is None:
        model = load_model(settings)
    language = settings.language if settings.language and settings.language != "auto" else None
    segments_iter, info = model.transcribe(
        str(audio),
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": settings.vad_min_silence_ms},
        word_timestamps=settings.word_timestamps,
        beam_size=settings.beam_size,
        condition_on_previous_text=settings.condition_on_previous_text,
    )
    rows: list[Segment] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        words: list[Word] = []
        if seg.words:
            words = [
                Word(start=w.start, end=w.end, word=w.word, prob=getattr(w, "probability", None))
                for w in seg.words
            ]
        rows.append(
            Segment(
                start=float(seg.start),
                end=float(seg.end),
                text=text,
                avg_logprob=getattr(seg, "avg_logprob", None),
                words=words,
            )
        )
    lang = getattr(info, "language", None)
    prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    return rows, lang, prob
