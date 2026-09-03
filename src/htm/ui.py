from __future__ import annotations

from pathlib import Path

import gradio as gr

from htm import __version__
from htm.export import to_markdown
from htm.pipeline import transcribe
from htm.types import Settings

MODELS = [
    "large-v3-turbo",
    "large-v3",
    "medium",
    "small",
    "base",
]
LANGUAGES = ["auto", "pl", "en", "de", "uk", "cs", "sk", "fr", "es"]


def _run(
    file_obj,
    language: str,
    n_speakers: float,
    names: str,
    model: str,
    insecure_ssl: bool,
    progress=gr.Progress(),
):
    if file_obj is None:
        raise gr.Error("Wskaż plik audio lub wideo.")
    path = Path(file_obj if isinstance(file_obj, str) else file_obj.name)
    names_list = [p.strip() for p in names.replace(";", ",").split(",") if p.strip()]
    n = int(n_speakers)
    settings = Settings(
        model=model,
        language=None if language == "auto" else language,
        speakers=None if n <= 0 else n,
        names=names_list,
        insecure_ssl=True if insecure_ssl else None,
    )
    out = Path.cwd() / "output"
    out.mkdir(exist_ok=True)

    def on_progress(pct: float, msg: str) -> None:
        progress(pct, desc=msg)

    result, paths = transcribe(path, out_dir=out, settings=settings, progress=on_progress)
    md = to_markdown(result)
    summary = (
        f"**{len(result.speakers)} mówców** · język: `{result.language}` · "
        f"{len(result.segments)} kwestii\n\n"
        + "\n".join(
            f"- **{s.label}**"
            + (f" (F0 ≈ {s.median_f0:.0f} Hz)" if s.median_f0 else "")
            for s in result.speakers
        )
    )
    return (
        summary,
        md,
        str(paths["md"]),
        str(paths["txt"]),
        str(paths["srt"]),
        str(paths["json"]),
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="High Transcription Model") as demo:
        gr.Markdown(
            f"# High Transcription Model\n"
            f"Lokalna transkrypcja audio/wideo z identyfikacją mówców · v{__version__}"
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(
                    label="Nagranie (mp4, wav, mp3, m4a, mkv…)",
                    file_types=["audio", "video"],
                )
                language = gr.Dropdown(LANGUAGES, value="pl", label="Język")
                model = gr.Dropdown(MODELS, value="large-v3-turbo", label="Model Whisper")
                n_speakers = gr.Slider(
                    0,
                    8,
                    value=0,
                    step=1,
                    label="Liczba mówców (0 = wykryj automatycznie)",
                )
                names = gr.Textbox(
                    label="Imiona mówców (opcjonalnie)",
                    placeholder="Anna Kowalska, Jan Nowak, Maria Wiśniewska",
                    info="W kolejności pierwszego zabrania głosu na nagraniu.",
                )
                insecure = gr.Checkbox(
                    label="Sieć korporacyjna: wyłącz weryfikację TLS",
                    value=False,
                )
                run = gr.Button("Transkrybuj", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown(label="Podsumowanie")
                transcript = gr.Markdown(label="Transkrypcja")
                with gr.Row():
                    md_file = gr.File(label="Markdown")
                    txt_file = gr.File(label="TXT")
                with gr.Row():
                    srt_file = gr.File(label="Napisy SRT")
                    json_file = gr.File(label="JSON")
        run.click(
            _run,
            inputs=[file_in, language, n_speakers, names, model, insecure],
            outputs=[summary, transcript, md_file, txt_file, srt_file, json_file],
        )
        gr.Markdown(
            "Modele pobierane są raz do katalogu `.htms-cache`. "
            "Na CPU nagranie 30-minutowe trwa zwykle kilkanaście–kilkadziesiąt minut. "
            "Krótkie wtrącenia („mhm”, „tak”) mogą być przypisane sąsiedniemu mówcy."
        )
    return demo


def launch(*, host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    demo = build_ui()
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=True,
        theme=gr.themes.Soft(),
        max_file_size="2gb",
    )
