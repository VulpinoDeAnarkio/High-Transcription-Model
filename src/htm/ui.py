"""Lokalny interfejs (przeglądarka) — Linux i macOS, kolejka plików."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import gradio as gr

from htm import __version__
from htm.asr import load_model
from htm.export import to_markdown
from htm.models import DEFAULT_MODEL, radio_choices, spec_for
from htm.pipeline import transcribe
from htm.progress import queue_percent, render_bar
from htm.queue import (
    CANCELLED,
    DONE,
    ERROR,
    RUNNING,
    WAITING,
    add_paths,
    collect_from_folder,
    format_queue,
    jobs_from_state,
    jobs_to_state,
    mark_skips,
    output_dir_for,
)
from htm.system import describe_host
from htm.types import Settings

LANGUAGES = ["auto", "pl", "en", "de", "uk", "cs", "sk", "fr", "es"]
OUT_SHARED = "Wspólny katalog"
OUT_BESIDE = "Obok każdego pliku źródłowego"

_stop = threading.Event()

CSS = """
.gradio-container { max-width: 1180px !important; }
footer { visibility: hidden; height: 0; }
.htm-progress { margin: 0.15rem 0 0.65rem; }
.htm-progress-row { display: flex; justify-content: space-between; gap: 0.75rem; align-items: baseline; }
.htm-progress-title { font-size: 0.92rem; font-weight: 600; }
.htm-progress-pct { font-variant-numeric: tabular-nums; font-size: 0.92rem; opacity: 0.8; }
.htm-progress-sub { font-size: 0.82rem; opacity: 0.75; margin: 0.15rem 0 0.35rem; }
.htm-progress-track { height: 10px; border-radius: 4px; overflow: hidden; background: var(--block-border-color, #d4d4d4); }
.htm-progress-fill { height: 100%; background: var(--button-primary-background-fill, #4f46e5); }
"""


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _append_log(log: str, line: str) -> str:
    row = f"{_now()}  {line}"
    prev = (log or "").strip()
    text = f"{prev}\n{row}" if prev else row
    lines = text.splitlines()
    if len(lines) > 120:
        lines = lines[-120:]
    return "\n".join(lines)


def _paths_from_upload(file_obj) -> list[Path]:
    if file_obj is None:
        return []
    items = file_obj if isinstance(file_obj, (list, tuple)) else [file_obj]
    out: list[Path] = []
    for item in items:
        if item is None:
            continue
        out.append(Path(item if isinstance(item, str) else item.name))
    return out


def _settings(language: str, n_speakers: float, names: str, model: str, insecure: bool) -> Settings:
    names_list = [p.strip() for p in names.replace(";", ",").split(",") if p.strip()]
    n = int(n_speakers)
    return Settings(
        model=model,
        language=None if language == "auto" else language,
        speakers=None if n <= 0 else n,
        names=names_list,
        insecure_ssl=True if insecure else None,
    )


def _bars(
    file_pct: float,
    file_title: str,
    file_sub: str,
    queue_pct: float,
    queue_title: str,
    queue_sub: str,
    *,
    show_file: bool,
    show_queue: bool,
):
    return (
        gr.update(value=render_bar(file_pct, file_title, file_sub), visible=show_file),
        gr.update(value=render_bar(queue_pct, queue_title, queue_sub), visible=show_queue),
    )


def _pack(
    jobs,
    status: str,
    log: str,
    summary: str = "",
    transcript: str = "",
    paths: dict | None = None,
    file_pct: float = 0,
    file_title: str = "Bieżący plik",
    file_sub: str = "",
    queue_pct: float = 0,
    queue_title: str = "Kolejka",
    queue_sub: str = "",
    show_file: bool = False,
    show_queue: bool = False,
):
    paths = paths or {}
    file_u, queue_u = _bars(
        file_pct,
        file_title,
        file_sub,
        queue_pct,
        queue_title,
        queue_sub,
        show_file=show_file,
        show_queue=show_queue,
    )
    return (
        format_queue(jobs),
        jobs_to_state(jobs),
        status,
        log,
        summary,
        transcript,
        paths.get("md"),
        paths.get("txt"),
        paths.get("srt"),
        paths.get("json"),
        file_u,
        queue_u,
    )


def add_uploads(file_obj, queue_state, log):
    jobs = jobs_from_state(queue_state)
    paths = _paths_from_upload(file_obj)
    jobs, n = add_paths(jobs, paths)
    if n == 0:
        gr.Warning("Nic nie dodano — brak plików audio/wideo albo już są w kolejce.")
        msg = "Nie dodano nowych plików."
    else:
        msg = f"Dodano {n} plik(ów) z wrzucenia."
        gr.Info(msg)
    log = _append_log(log, msg)
    return format_queue(jobs), jobs_to_state(jobs), msg, log, gr.update(value=None)


def add_folder(folder: str, recursive: bool, queue_state, log):
    jobs = jobs_from_state(queue_state)
    folder = (folder or "").strip()
    if not folder:
        raise gr.Error("Wklej ścieżkę do folderu (Finder / Files: prawy przycisk → kopiuj ścieżkę).")
    try:
        found = collect_from_folder(folder, recursive=bool(recursive))
    except FileNotFoundError as e:
        raise gr.Error(str(e)) from e
    jobs, n = add_paths(jobs, found)
    skipped = len(found) - n
    msg = f"Z folderu: +{n} nagranie/nagrań."
    if skipped:
        msg += f" Pominięto {skipped} (duplikat albo nie-media)."
    if n == 0 and not found:
        msg = "W folderze nie ma plików audio/wideo."
        gr.Warning(msg)
    else:
        gr.Info(msg)
    log = _append_log(log, msg + f"  ({folder})")
    return format_queue(jobs), jobs_to_state(jobs), msg, log


def clear_done(queue_state, log):
    jobs = jobs_from_state(queue_state)
    before = len(jobs)
    jobs = [j for j in jobs if j.status in {WAITING, RUNNING}]
    removed = before - len(jobs)
    msg = f"Usunięto {removed} zakończonych pozycji." if removed else "Nie było zakończonych pozycji."
    log = _append_log(log, msg)
    return format_queue(jobs), jobs_to_state(jobs), msg, log


def clear_all(queue_state, log):
    jobs = jobs_from_state(queue_state)
    if any(j.status == RUNNING for j in jobs):
        raise gr.Error("Najpierw zatrzymaj kolejkę (po bieżącym pliku), potem wyczyść.")
    log = _append_log(log, "Wyczyszczono kolejkę.")
    return format_queue([]), [], "Kolejka pusta.", log


def ask_stop():
    _stop.set()
    return "Zatrzymam po bieżącym pliku — Whisper musi dokończyć to nagranie."


def run_queue(
    queue_state,
    language,
    n_speakers,
    names,
    model,
    insecure,
    out_mode,
    out_dir_raw,
    skip_existing,
    log,
    progress=gr.Progress(),
):
    jobs = jobs_from_state(queue_state)
    if not jobs:
        raise gr.Error("Kolejka jest pusta.")
    if any(j.status == RUNNING for j in jobs):
        raise gr.Error("Kolejka już działa.")

    beside = out_mode == OUT_BESIDE
    out_root = Path(out_dir_raw or "output").expanduser()
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if skip_existing:
        mark_skips(jobs, beside=beside, out_dir=out_root)

    waiting = [j for j in jobs if j.status == WAITING]
    if not waiting:
        msg = "Nie ma plików do zrobienia (wszystkie gotowe albo pominięte)."
        log = _append_log(log, msg)
        yield _pack(jobs, msg, log)
        return

    _stop.clear()
    settings = _settings(language, n_speakers, names, model, insecure)
    spec = spec_for(settings.model)
    total = len(waiting)
    show_queue = total > 1
    log = _append_log(log, f"Start kolejki: {total} plików · {spec.id} · {describe_host()}")
    yield _pack(
        jobs,
        f"Ładowanie modelu {spec.id}…",
        log,
        file_pct=2,
        file_title="Bieżący plik",
        file_sub=f"Ładowanie modelu {spec.id}",
        queue_pct=0,
        queue_title="Kolejka",
        queue_sub=f"0 / {total} plików" if show_queue else "",
        show_file=True,
        show_queue=show_queue,
    )

    try:
        model_obj = load_model(settings)
    except Exception as e:
        log = _append_log(log, f"Błąd modelu: {e}")
        yield _pack(jobs, f"Nie udało się wczytać modelu: {e}", log, show_file=True, show_queue=show_queue)
        return

    last_paths: dict = {}
    last_summary = ""
    last_md = ""
    done_i = 0

    def emit(status: str, file_frac: float, file_sub: str, job_name: str = ""):
        finished = max(done_i - 1, 0)
        q_pct = queue_percent(finished, file_frac, total)
        if show_queue:
            q_sub = f"plik {done_i} / {total}" if done_i else f"0 / {total} plików"
        else:
            q_sub = ""
        try:
            progress(
                (finished + file_frac) / total if total else file_frac,
                desc=file_sub or status,
            )
        except Exception:
            pass
        return _pack(
            jobs,
            status,
            log,
            last_summary,
            last_md,
            last_paths,
            file_pct=file_frac * 100,
            file_title=job_name or "Bieżący plik",
            file_sub=file_sub,
            queue_pct=q_pct,
            queue_title="Kolejka",
            queue_sub=q_sub,
            show_file=True,
            show_queue=show_queue,
        )

    for job in jobs:
        if job.status != WAITING:
            continue
        if _stop.is_set():
            job.status = CANCELLED
            job.detail = "zatrzymane przed startem"
            for rest in jobs:
                if rest.status == WAITING:
                    rest.status = CANCELLED
                    rest.detail = "zatrzymane"
            log = _append_log(log, "Kolejka zatrzymana.")
            yield emit("Zatrzymano.", 0, "Zatrzymano", job.name)
            return

        done_i += 1
        job.status = RUNNING
        job.detail = f"{done_i}/{total}"
        status = f"W toku {done_i}/{total}: {job.name}"
        log = _append_log(log, status)
        yield emit(status, 0.0, "Start…", job.name)

        dest = output_dir_for(job, beside=beside, out_dir=out_root)
        dest.mkdir(parents=True, exist_ok=True)

        snap = {"frac": 0.0, "msg": "Start…"}
        holder: list = []
        errors: list[BaseException] = []

        def on_progress(pct: float, msg: str) -> None:
            snap["frac"] = max(0.0, min(1.0, float(pct)))
            snap["msg"] = msg

        def work() -> None:
            try:
                holder.append(
                    transcribe(
                        job.source,
                        out_dir=dest,
                        settings=settings,
                        progress=on_progress,
                        model=model_obj,
                    )
                )
            except BaseException as exc:  # noqa: BLE001 — przekazać do pętli UI
                errors.append(exc)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        while worker.is_alive():
            yield emit(status, snap["frac"], snap["msg"], job.name)
            worker.join(timeout=0.45)
        worker.join()

        if errors:
            err = errors[0]
            job.status = ERROR
            job.detail = str(err).splitlines()[-1][:200]
            log = _append_log(log, f"BŁĄD {job.name}: {job.detail}")
            yield emit(f"Błąd: {job.name}", snap["frac"], job.detail, job.name)
            continue

        result, paths = holder[0]
        job.status = DONE
        job.md = str(paths["md"])
        job.detail = f"{len(result.speakers)} mówców · {len(result.segments)} kwestii"
        last_paths = {k: str(v) for k, v in paths.items()}
        last_md = to_markdown(result)
        last_summary = (
            f"**{job.name}** · {len(result.speakers)} mówców · "
            f"język `{result.language}` · **{spec.id}** ({spec.title.lower()})\n\n"
            + "\n".join(
                f"- **{s.label}**"
                + (f" (F0 ≈ {s.median_f0:.0f} Hz)" if s.median_f0 else "")
                for s in result.speakers
            )
        )
        log = _append_log(log, f"OK {job.name} → {paths['md'].name}")
        yield emit(f"Gotowe {done_i}/{total}: {job.name}", 1.0, "Gotowe.", job.name)

    c_err = sum(1 for j in jobs if j.status == ERROR)
    msg = f"Koniec kolejki. Gotowe {sum(1 for j in jobs if j.status == DONE)}/{total}."
    if c_err:
        msg += f" Błędów: {c_err}."
    log = _append_log(log, msg)
    yield _pack(
        jobs,
        msg,
        log,
        last_summary,
        last_md,
        last_paths,
        file_pct=100,
        file_title="Bieżący plik",
        file_sub="Zakończono.",
        queue_pct=100,
        queue_title="Kolejka",
        queue_sub=f"{total} / {total} plików" if show_queue else "",
        show_file=True,
        show_queue=show_queue,
    )


def build_ui() -> gr.Blocks:
    host = describe_host()
    with gr.Blocks(title="High Transcription Model") as demo:
        queue_state = gr.State([])
        gr.Markdown(
            f"# High Transcription Model\n"
            f"Lokalna transkrypcja z kolejką plików · v{__version__} · {host}\n\n"
            "Ten sam interfejs na **Linuxie** i **macOS**. "
            "Nagrania zostają na tym komputerze — nic nie idzie do chmury."
        )
        with gr.Row():
            with gr.Column(scale=6):
                gr.Markdown("### Kolejka")
                file_in = gr.File(
                    label="Wrzuć nagrania (kilka naraz)",
                    file_types=["audio", "video"],
                    file_count="multiple",
                    height=110,
                )
                add_files_btn = gr.Button("Dodaj wrzucone pliki do kolejki")
                with gr.Row():
                    folder_in = gr.Textbox(
                        label="Albo folder na dysku",
                        placeholder="/home/…/nagrania   lub   /Users/…/nagrania",
                        info="Dla dużych zbiorów (dziesiątki MP3/MP4) — bez kopiowania do przeglądarki.",
                        scale=4,
                    )
                    recursive = gr.Checkbox(label="Z podfolderami", value=False, scale=1)
                add_folder_btn = gr.Button("Dodaj folder do kolejki")
                queue_md = gr.Markdown(format_queue([]))
                with gr.Row():
                    start_btn = gr.Button("Uruchom kolejkę", variant="primary")
                    stop_btn = gr.Button("Zatrzymaj po bieżącym")
                with gr.Row():
                    clear_done_btn = gr.Button("Usuń zakończone")
                    clear_all_btn = gr.Button("Wyczyść kolejkę")
                status = gr.Markdown("Gotowy do dodania plików.")
                file_bar = gr.HTML(value="", visible=False)
                queue_bar = gr.HTML(value="", visible=False)
                log = gr.Textbox(label="Dziennik", lines=8, max_lines=16, interactive=False)
            with gr.Column(scale=4):
                gr.Markdown("### Ustawienia (dla całej kolejki)")
                language = gr.Dropdown(LANGUAGES, value="pl", label="Język")
                model = gr.Radio(
                    choices=radio_choices(),
                    value=DEFAULT_MODEL,
                    label="Model",
                    info="Turbo — rozmowy. large-v3 — wolniejszy, zwykle mniej pomyłek.",
                )
                n_speakers = gr.Slider(
                    0, 8, value=0, step=1, label="Liczba mówców (0 = automatycznie)"
                )
                names = gr.Textbox(
                    label="Imiona mówców",
                    placeholder="Anna Kowalska, Jan Nowak",
                    info="Opcjonalnie, w kolejności pierwszego zabrania głosu. Te same dla wszystkich plików.",
                )
                out_mode = gr.Radio(
                    [OUT_SHARED, OUT_BESIDE],
                    value=OUT_SHARED,
                    label="Gdzie zapisać wynik",
                )
                out_dir = gr.Textbox(value="output", label="Wspólny katalog wyników")
                skip_existing = gr.Checkbox(
                    label="Pomijaj pliki, które już mają transkrypcję",
                    value=True,
                    info="Szuka `{nazwa}_transkrypcja.md` obok źródła albo w katalogu wyników.",
                )
                insecure = gr.Checkbox(
                    label="Sieć korporacyjna: wyłącz weryfikację TLS",
                    value=False,
                )
                gr.Markdown(
                    "Na CPU godzina nagrania to zwykle kilkadziesiąt minut. "
                    "Kolejka idzie **po kolei** — tak jest szybciej niż dwa Whisper naraz. "
                    "Zatrzymanie czeka na koniec bieżącego pliku."
                )

        gr.Markdown("### Ostatni ukończony plik")
        summary = gr.Markdown()
        transcript = gr.Markdown()
        with gr.Row():
            md_file = gr.File(label="Markdown")
            txt_file = gr.File(label="TXT")
        with gr.Row():
            srt_file = gr.File(label="Napisy SRT")
            json_file = gr.File(label="JSON")

        add_files_btn.click(
            add_uploads,
            inputs=[file_in, queue_state, log],
            outputs=[queue_md, queue_state, status, log, file_in],
        )
        add_folder_btn.click(
            add_folder,
            inputs=[folder_in, recursive, queue_state, log],
            outputs=[queue_md, queue_state, status, log],
        )
        clear_done_btn.click(
            clear_done,
            inputs=[queue_state, log],
            outputs=[queue_md, queue_state, status, log],
        )
        clear_all_btn.click(
            clear_all,
            inputs=[queue_state, log],
            outputs=[queue_md, queue_state, status, log],
        )
        stop_btn.click(ask_stop, outputs=[status], queue=False)
        start_btn.click(
            run_queue,
            inputs=[
                queue_state,
                language,
                n_speakers,
                names,
                model,
                insecure,
                out_mode,
                out_dir,
                skip_existing,
                log,
            ],
            outputs=[
                queue_md,
                queue_state,
                status,
                log,
                summary,
                transcript,
                md_file,
                txt_file,
                srt_file,
                json_file,
                file_bar,
                queue_bar,
            ],
        )
    return demo


def launch(*, host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    demo = build_ui()
    demo.queue(default_concurrency_limit=2)
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=CSS,
        max_file_size="4gb",
        allowed_paths=[str(Path.home()), str(Path.cwd())],
    )
