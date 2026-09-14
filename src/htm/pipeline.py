from __future__ import annotations

import tempfile
from pathlib import Path

from htm.asr import load_model, transcribe_audio
from htm.diarize import assign_speakers
from htm.export import write_outputs
from htm.media import extract_wav
from htm.models import spec_for
from htm.system import apply_macos_runtime_env, extend_path
from htm.types import ProgressFn, Settings, Transcript


def transcribe(
    source: str | Path,
    out_dir: str | Path | None = None,
    settings: Settings | None = None,
    progress: ProgressFn | None = None,
    model=None,
) -> tuple[Transcript, dict[str, Path]]:
    settings = settings or Settings()
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    extend_path()
    apply_macos_runtime_env()

    def report(pct: float, msg: str) -> None:
        if progress:
            progress(pct, msg)

    work = Path(out_dir).expanduser().resolve() if out_dir else Path(tempfile.mkdtemp(prefix="htm-"))
    work.mkdir(parents=True, exist_ok=True)
    if settings.keep_audio:
        audio = work / (source.stem + ".16k.wav")
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="htm-", suffix=".16k.wav", delete=False)
        tmp.close()
        audio = Path(tmp.name)

    report(0.04, "Wyodrębnianie ścieżki audio…")
    extract_wav(source, audio)

    spec = spec_for(settings.model)
    if model is None:
        report(0.12, f"Ładowanie modelu {spec.id} ({spec.title.lower()})…")
        model = load_model(settings)
    else:
        report(0.12, f"Model {spec.id} już w pamięci…")

    report(0.18, "Transkrypcja (to może potrwać)…")

    def asr_progress(frac: float, msg: str) -> None:
        report(0.18 + 0.58 * max(0.0, min(float(frac), 1.0)), msg)

    segments, language, lang_p = transcribe_audio(
        audio, settings, model=model, progress=asr_progress
    )

    report(0.78, "Identyfikacja mówców…")
    merged, speakers, notes = assign_speakers(
        audio,
        segments,
        n_speakers=settings.speakers,
        names=settings.names,
        merge_gap=settings.merge_gap,
    )

    result = Transcript(
        language=language,
        language_probability=lang_p,
        source=source,
        audio=audio,
        model=settings.model,
        speakers=speakers,
        segments=merged,
        notes=notes,
    )

    if not settings.keep_audio:
        try:
            audio.unlink(missing_ok=True)
        except OSError:
            pass
        result.audio = source

    report(0.92, "Zapis wyników…")
    paths = write_outputs(result, work)
    report(1.0, "Gotowe.")
    return result, paths
