from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from htm.models import DEFAULT_MODEL, normalize_model

ProgressFn = Callable[[float, str], None]


@dataclass
class Word:
    start: float
    end: float
    word: str
    prob: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    median_f0: float | None = None
    cluster: int | None = None
    avg_logprob: float | None = None
    words: list[Word] = field(default_factory=list)


@dataclass
class SpeakerInfo:
    label: str
    cluster: int
    median_f0: float | None
    gender: str | None
    speaking_seconds: float
    samples: list[str]


@dataclass
class Transcript:
    language: str | None
    language_probability: float | None
    source: Path
    audio: Path
    model: str
    speakers: list[SpeakerInfo]
    segments: list[Segment]
    notes: list[str] = field(default_factory=list)


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = None
    beam_size: int = 5
    speakers: int | None = None
    names: list[str] = field(default_factory=list)
    vad_min_silence_ms: int = 400
    merge_gap: float = 1.4
    condition_on_previous_text: bool = True
    insecure_ssl: bool | None = None
    cache_dir: Path | None = None
    keep_audio: bool = False
    word_timestamps: bool = True

    def __post_init__(self) -> None:
        self.model = normalize_model(self.model)
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
