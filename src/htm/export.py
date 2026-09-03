from __future__ import annotations

import json
from pathlib import Path

from htm.diarize import format_timestamp
from htm.types import Transcript


def to_markdown(result: Transcript) -> str:
    src = result.source.name
    lines = [
        f"# Transkrypcja — {src}",
        "",
        f"Plik: `{src}`",
        f"Model: `{result.model}`",
    ]
    if result.language:
        p = f" ({result.language_probability:.0%})" if result.language_probability else ""
        lines.append(f"Język: {result.language}{p}")
    lines += ["", "## Mówcy", ""]
    for sp in result.speakers:
        f0 = f", F0 ≈ {sp.median_f0:.0f} Hz" if sp.median_f0 else ""
        g = {"M": "mężczyzna", "K": "kobieta"}.get(sp.gender or "", sp.gender or "nieokreślona")
        lines.append(f"- **{sp.label}** — {g}{f0}, {sp.speaking_seconds:.0f} s wypowiedzi")
        for sample in sp.samples[:2]:
            snippet = sample if len(sample) < 140 else sample[:137] + "…"
            lines.append(f"  - _{snippet}_")
    if result.notes:
        lines += ["", "## Uwagi modelu", ""]
        for note in result.notes:
            lines.append(f"- {note}")
    lines += ["", "---", ""]
    for seg in result.segments:
        speaker = seg.speaker or "Mówca"
        lines.append(f"**[{format_timestamp(seg.start)}] {speaker}**")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_txt(result: Transcript) -> str:
    lines = []
    for seg in result.segments:
        speaker = seg.speaker or "Mówca"
        lines.append(f"[{format_timestamp(seg.start)}] {speaker}: {seg.text}")
    return "\n".join(lines) + "\n"


def _srt_stamp(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(result: Transcript) -> str:
    blocks = []
    for i, seg in enumerate(result.segments, 1):
        speaker = seg.speaker or "Mówca"
        blocks.append(
            f"{i}\n{_srt_stamp(seg.start)} --> {_srt_stamp(seg.end)}\n{speaker}: {seg.text}\n"
        )
    return "\n".join(blocks) + "\n"


def to_json(result: Transcript) -> str:
    payload = {
        "source": str(result.source),
        "audio": str(result.audio),
        "model": result.model,
        "language": result.language,
        "language_probability": result.language_probability,
        "notes": result.notes,
        "speakers": [
            {
                "label": s.label,
                "cluster": s.cluster,
                "median_f0": s.median_f0,
                "gender": s.gender,
                "speaking_seconds": round(s.speaking_seconds, 2),
                "samples": s.samples,
            }
            for s in result.speakers
        ],
        "segments": [
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "speaker": seg.speaker,
                "text": seg.text,
                "median_f0": seg.median_f0,
                "cluster": seg.cluster,
            }
            for seg in result.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_outputs(result: Transcript, out_dir: Path, stem: str | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or (result.source.stem + "_transkrypcja")
    paths = {
        "md": out_dir / f"{stem}.md",
        "txt": out_dir / f"{stem}.txt",
        "srt": out_dir / f"{stem}.srt",
        "json": out_dir / f"{stem}.json",
    }
    paths["md"].write_text(to_markdown(result), encoding="utf-8")
    paths["txt"].write_text(to_txt(result), encoding="utf-8")
    paths["srt"].write_text(to_srt(result), encoding="utf-8")
    paths["json"].write_text(to_json(result), encoding="utf-8")
    return paths
