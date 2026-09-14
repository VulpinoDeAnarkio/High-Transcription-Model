"""Kolejka transkrypcji — czysta logika, bez Gradio.

Jedna sesja, sekwencyjnie (CPU i tak nie uniesie dwóch Whisperów naraz).
Działa tak samo na Linuxie i macOS: ścieżki lokalne albo wrzucone pliki.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from htm.media import AUDIO_SUFFIXES, VIDEO_SUFFIXES, is_media

WAITING = "waiting"
RUNNING = "running"
DONE = "done"
ERROR = "error"
SKIPPED = "skipped"
CANCELLED = "cancelled"

STATUS_PL = {
    WAITING: "oczekuje",
    RUNNING: "w toku",
    DONE: "gotowe",
    ERROR: "błąd",
    SKIPPED: "pominięte",
    CANCELLED: "zatrzymane",
}

MEDIA_SUFFIXES = AUDIO_SUFFIXES | VIDEO_SUFFIXES


@dataclass
class Job:
    path: str
    name: str
    status: str = WAITING
    detail: str = ""
    size: str = ""
    md: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> Job:
        known = set(cls.__dataclass_fields__)
        data = {k: v for k, v in raw.items() if k in known}
        return cls(**data)

    @property
    def source(self) -> Path:
        return Path(self.path)


def size_label(path: Path) -> str:
    try:
        n = path.stat().st_size
    except OSError:
        return "?"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.0f} kB"
    return f"{n} B"


def job_from_path(path: Path) -> Job:
    path = path.expanduser().resolve()
    return Job(path=str(path), name=path.name, size=size_label(path))


def collect_from_folder(folder: str | Path, *, recursive: bool = False) -> list[Path]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Nie ma takiego folderu: {root}")
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and is_media(p)]
    else:
        files = [p for p in root.iterdir() if p.is_file() and is_media(p)]
    files.sort(key=lambda p: p.name.casefold())
    return files


def add_paths(jobs: list[Job], paths: list[Path]) -> tuple[list[Job], int]:
    """Dopisz nowe nagrania. Pomija nie-media i duplikaty (ta sama ścieżka)."""
    seen = {Path(j.path).resolve() for j in jobs}
    added = 0
    out = list(jobs)
    for raw in paths:
        path = Path(raw).expanduser()
        try:
            path = path.resolve()
        except OSError:
            continue
        if not path.is_file() or not is_media(path) or path in seen:
            continue
        seen.add(path)
        out.append(job_from_path(path))
        added += 1
    return out, added


def output_dir_for(job: Job, *, beside: bool, out_dir: Path) -> Path:
    if beside:
        return Path(job.path).parent
    return out_dir


def expected_markdown(job: Job, *, beside: bool, out_dir: Path) -> Path:
    dest = output_dir_for(job, beside=beside, out_dir=out_dir)
    return dest / f"{Path(job.path).stem}_transkrypcja.md"


def mark_skips(jobs: list[Job], *, beside: bool, out_dir: Path) -> list[Job]:
    for job in jobs:
        if job.status not in {WAITING, SKIPPED}:
            continue
        md = expected_markdown(job, beside=beside, out_dir=out_dir)
        if md.is_file() and md.stat().st_size > 80:
            job.status = SKIPPED
            job.detail = "transkrypcja już istnieje"
            job.md = str(md)
        elif job.status == SKIPPED and job.detail == "transkrypcja już istnieje":
            job.status = WAITING
            job.detail = ""
            job.md = ""
    return jobs


def pending(jobs: list[Job]) -> list[Job]:
    return [j for j in jobs if j.status == WAITING]


def clear_finished(jobs: list[Job]) -> list[Job]:
    return [j for j in jobs if j.status not in {DONE, SKIPPED, ERROR, CANCELLED}]


def counts(jobs: list[Job]) -> dict[str, int]:
    out = {k: 0 for k in STATUS_PL}
    for j in jobs:
        out[j.status] = out.get(j.status, 0) + 1
    out["total"] = len(jobs)
    return out


def format_queue(jobs: list[Job]) -> str:
    if not jobs:
        return "_Kolejka pusta. Wrzuć pliki albo wklej ścieżkę folderu._"
    c = counts(jobs)
    head = (
        f"**{c['total']}** w kolejce"
        f" · oczekuje {c[WAITING]}"
        f" · w toku {c[RUNNING]}"
        f" · gotowe {c[DONE]}"
        f" · pominięte {c[SKIPPED]}"
        f" · błąd {c[ERROR]}"
    )
    if c[CANCELLED]:
        head += f" · zatrzymane {c[CANCELLED]}"
    lines = [
        head,
        "",
        "| # | Plik | Rozmiar | Status | Uwagi |",
        "|---:|---|---|---|---|",
    ]
    for i, job in enumerate(jobs, 1):
        note = (job.detail or "").replace("|", "/")
        if len(note) > 80:
            note = note[:77] + "…"
        lines.append(
            f"| {i} | `{job.name}` | {job.size} | {STATUS_PL.get(job.status, job.status)} | {note} |"
        )
    return "\n".join(lines)


def jobs_to_state(jobs: list[Job]) -> list[dict]:
    return [j.to_dict() for j in jobs]


def jobs_from_state(raw: list[dict] | None) -> list[Job]:
    if not raw:
        return []
    return [Job.from_dict(item) for item in raw]
