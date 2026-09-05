from __future__ import annotations

import os
from pathlib import Path

from htm.models import cache_dir, spec_for
from htm.ssl_util import apply_insecure_ssl, ensure_ssl
from htm.system import apply_macos_runtime_env, is_macos
from htm.types import Segment, Settings, Word


class AsrError(RuntimeError):
    pass


def _resolve_device(settings: Settings) -> tuple[str, str]:
    device = settings.device
    compute = settings.compute_type
    if device == "auto":
        device = "cpu"
        if not is_macos():
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
            except Exception:
                device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def _compute_fallbacks(preferred: str, device: str) -> list[str]:
    if device == "cuda":
        return [preferred, "float16", "float32"]
    # Apple Silicon / CPU: int8 sometimes rejects a model; fall back.
    ordered = [preferred, "int8", "int8_float32", "int8_float16", "float32"]
    seen: list[str] = []
    for item in ordered:
        if item not in seen:
            seen.append(item)
    return seen


def _cache_dir(settings: Settings) -> Path:
    return cache_dir(settings.cache_dir)


def load_model(settings: Settings):
    from faster_whisper import WhisperModel

    apply_macos_runtime_env()
    ensure_ssl(settings.insecure_ssl)
    spec = spec_for(settings.model)
    cache = _cache_dir(settings)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))
    device, compute = _resolve_device(settings)
    cpu_threads = os.cpu_count() or 4
    last_error: Exception | None = None
    ssl_retried = False

    for compute_type in _compute_fallbacks(compute, device):
        try:
            return WhisperModel(
                spec.id,
                device=device,
                compute_type=compute_type,
                download_root=str(cache),
                cpu_threads=cpu_threads,
                num_workers=1,
            )
        except Exception as err:
            last_error = err
            msg = str(err).lower()
            tls = "certificate" in msg or "ssl" in msg or "localentrynotfound" in msg
            if tls and not ssl_retried and settings.insecure_ssl is not False:
                apply_insecure_ssl()
                ssl_retried = True
                try:
                    return WhisperModel(
                        spec.id,
                        device=device,
                        compute_type=compute_type,
                        download_root=str(cache),
                        cpu_threads=cpu_threads,
                        num_workers=1,
                    )
                except Exception as err2:
                    last_error = err2
                    continue
            if "compute type" in msg or "compute_type" in msg or "not supported" in msg:
                continue
            if tls:
                continue
            break

    raise AsrError(
        f"Nie udało się wczytać modelu {spec.id} ({device}/{compute}). "
        f"Na macOS CTranslate2 działa na CPU (nie Metal). "
        f"Szczegóły: {last_error}"
    ) from last_error


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
