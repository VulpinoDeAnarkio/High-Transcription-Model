from __future__ import annotations

import argparse
import sys
from pathlib import Path

from htm import __version__
from htm.media import MediaError, require_ffmpeg
from htm.models import DEFAULT_MODEL, cli_help, describe_cache, normalize_model
from htm.system import describe_host, extend_path, is_macos, libsndfile_hint, uv_hint, which
from htm.types import Settings


def _parse_names(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]


def _model_arg(raw: str) -> str:
    try:
        return normalize_model(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="htm",
        description="High Transcription Model — lokalna transkrypcja z identyfikacją mówców.",
    )
    p.add_argument("--version", action="version", version=f"High Transcription Model {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    ui = sub.add_parser("ui", help="Uruchom interfejs lokalny w przeglądarce")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--share", action="store_true", help="Tymczasowy publiczny URL Gradio (niezalecane)")

    tr = sub.add_parser("transcribe", help="Transkrybuj plik audio lub wideo")
    tr.add_argument("source", type=Path, help="Plik .mp4 / .wav / .mp3 / …")
    tr.add_argument("-o", "--out", type=Path, default=Path("output"), help="Katalog wyników")
    tr.add_argument("-l", "--language", default="auto", help="Kod języka (pl, en) albo auto")
    tr.add_argument("-m", "--model", type=_model_arg, default=DEFAULT_MODEL, help=cli_help())
    tr.add_argument("-s", "--speakers", type=int, default=None, help="Liczba mówców (domyślnie: auto)")
    tr.add_argument("-n", "--names", default="", help='Imiona, przecinkami, w kolejności pierwszego zabrania głosu')
    tr.add_argument("--device", default="auto", help="cpu / cuda / auto")
    tr.add_argument("--compute-type", default="auto")
    tr.add_argument("--beam-size", type=int, default=5)
    tr.add_argument("--keep-audio", action="store_true")
    tr.add_argument("--insecure-ssl", action="store_true", help="Wyłącz weryfikację TLS (sieć korporacyjna)")
    tr.add_argument("--no-previous-text", action="store_true", help="Wyłącz kontekst poprzedniego zdania (lepsze przy zmianach mówcy)")

    doc = sub.add_parser("doctor", help="Sprawdź zależności lokalne")
    doc.add_argument("--insecure-ssl", action="store_true")
    return p


def cmd_doctor(insecure_ssl: bool) -> int:
    ok = True
    print(f"High Transcription Model {__version__}")
    print(f"system: {describe_host()}")
    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    if is_macos():
        print("uwaga: na Macu faster-whisper liczy na CPU (CTranslate2), nie na Metal/ANE")
    try:
        ff = require_ffmpeg()
        print(f"ffmpeg: {ff}")
    except MediaError as e:
        print(f"BŁĄD: {e}")
        ok = False
    uv = which("uv")
    print(f"uv: {uv or 'brak (można użyć pip)'}")
    if not uv:
        print(uv_hint())
    for pkg in ("faster_whisper", "librosa", "sklearn", "gradio"):
        try:
            __import__(pkg)
            print(f"{pkg}: OK")
        except ImportError:
            print(f"{pkg}: BRAK — w katalogu aplikacji: uv pip install -e .")
            ok = False
    try:
        import soundfile  # noqa: F401

        print("soundfile: OK")
    except Exception as e:
        print(f"soundfile: BŁĄD ({e})")
        extra = libsndfile_hint()
        if extra:
            print(f"  {extra}")
        ok = False
    for line in describe_cache():
        print(line)
    if insecure_ssl:
        from htm.ssl_util import apply_insecure_ssl

        apply_insecure_ssl()
        print("TLS: tryb HTM_INSECURE_SSL")
    return 0 if ok else 1


def cmd_transcribe(args: argparse.Namespace) -> int:
    from htm.pipeline import transcribe

    lang = None if args.language in {"", "auto"} else args.language
    settings = Settings(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=lang,
        beam_size=args.beam_size,
        speakers=args.speakers,
        names=_parse_names(args.names),
        keep_audio=args.keep_audio,
        insecure_ssl=True if args.insecure_ssl else None,
        condition_on_previous_text=not args.no_previous_text,
    )

    def progress(pct: float, msg: str) -> None:
        print(f"[{pct:5.0%}] {msg}", flush=True)

    result, paths = transcribe(args.source, out_dir=args.out, settings=settings, progress=progress)
    print()
    print(f"Model: {result.model}")
    print(f"Język: {result.language}")
    print(f"Mówcy ({len(result.speakers)}):")
    for sp in result.speakers:
        print(f"  - {sp.label}")
    print("Pliki:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from htm.ui import launch

    launch(host=args.host, port=args.port, share=args.share)
    return 0


def main(argv: list[str] | None = None) -> int:
    extend_path()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "doctor":
            return cmd_doctor(args.insecure_ssl)
        if args.cmd == "ui":
            return cmd_ui(args)
        if args.cmd == "transcribe":
            return cmd_transcribe(args)
        parser.print_help()
        return 2
    except KeyboardInterrupt:
        print("Przerwano.", file=sys.stderr)
        return 130
    except FileNotFoundError as e:
        print(f"Nie znaleziono pliku: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Błąd: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
