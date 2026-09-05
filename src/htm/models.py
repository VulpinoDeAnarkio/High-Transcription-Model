"""Katalog modeli faster-whisper używanych przez High Transcription Model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "large-v3-turbo"
QUALITY_MODEL = "large-v3"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    title: str
    blurb: str
    hub_id: str
    size_hint: str
    primary: bool = True


MODELS: dict[str, ModelSpec] = {
    "large-v3-turbo": ModelSpec(
        id="large-v3-turbo",
        title="Szybszy",
        blurb="Rozmowy i spotkania. Ten sam wariant co przy transkrypcji HR.",
        hub_id="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        size_hint="~1,6 GB",
        primary=True,
    ),
    "large-v3": ModelSpec(
        id="large-v3",
        title="Dokładniejszy",
        blurb="Pełny model. Wolniejszy, zwykle mniej pomyłek — jak przy odzyskiwaniu czytanego tekstu.",
        hub_id="Systran/faster-whisper-large-v3",
        size_hint="~3 GB",
        primary=True,
    ),
    "medium": ModelSpec(
        id="medium",
        title="Podgląd medium",
        blurb="Szybszy podgląd, słabsza polszczyzna.",
        hub_id="Systran/faster-whisper-medium",
        size_hint="~1,5 GB",
        primary=False,
    ),
    "small": ModelSpec(
        id="small",
        title="Podgląd small",
        blurb="Bardzo szybki podgląd, wyraźnie gorsza jakość.",
        hub_id="Systran/faster-whisper-small",
        size_hint="~500 MB",
        primary=False,
    ),
    "base": ModelSpec(
        id="base",
        title="Podgląd base",
        blurb="Najszybszy, tylko do testu pipeline’u.",
        hub_id="Systran/faster-whisper-base",
        size_hint="~150 MB",
        primary=False,
    ),
}

_ALIASES = {
    "turbo": "large-v3-turbo",
    "large-v3-turbo": "large-v3-turbo",
    "large": "large-v3",
    "large-v3": "large-v3",
    "medium": "medium",
    "small": "small",
    "base": "base",
}


def primary_models() -> list[ModelSpec]:
    return [spec for spec in MODELS.values() if spec.primary]


def all_model_ids() -> tuple[str, ...]:
    return tuple(MODELS)


def spec_for(model_id: str) -> ModelSpec:
    return MODELS[normalize_model(model_id)]


def radio_choices() -> list[tuple[str, str]]:
    """Etykieta Gradio Radio → id modelu."""
    return [
        (f"{spec.title} — {spec.id} ({spec.size_hint})", spec.id)
        for spec in primary_models()
    ]


def cli_help() -> str:
    lines = [
        "Model faster-whisper. Dwa główne warianty:",
        "large-v3-turbo (domyślny, szybszy — rozmowy),",
        "large-v3 (dokładniejszy, wolniejszy — czytanie / odzysk tekstu).",
        "Aliasy: turbo, large. Dodatkowo: medium, small, base.",
    ]
    return " ".join(lines)


def normalize_model(raw: str | None) -> str:
    if raw is None or not str(raw).strip():
        return DEFAULT_MODEL
    key = str(raw).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # Etykieta z UI Radio: „Szybszy — large-v3-turbo (~1,6 GB)”
    if "large-v3-turbo" in key or (key.endswith("turbo") and "large" in key):
        return "large-v3-turbo"
    if "large-v3" in key:
        return "large-v3"
    raise ValueError(
        f"Nieznany model {raw!r}. Wybierz large-v3-turbo albo large-v3 "
        f"(albo turbo / large)."
    )


def cache_dir(explicit: Path | None = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("HTM_CACHE") or os.environ.get("HF_HOME")
    if env:
        return Path(env)
    return Path.cwd() / ".htms-cache"


def _hub_cache_folder(hub_id: str) -> str:
    return "models--" + hub_id.replace("/", "--")


def is_model_cached(model_id: str, root: Path | None = None) -> bool:
    spec = spec_for(model_id)
    base = cache_dir(root)
    folder = base / _hub_cache_folder(spec.hub_id)
    if not folder.exists():
        return False
    return any(folder.rglob("model.bin"))


def describe_cache(root: Path | None = None) -> list[str]:
    base = cache_dir(root)
    lines = [f"cache: {base} ({'istnieje' if base.exists() else 'zostanie utworzony'})"]
    for spec in primary_models():
        state = "w cache" if is_model_cached(spec.id, base) else f"do pobrania ({spec.size_hint})"
        lines.append(f"  {spec.id} ({spec.title.lower()}): {state}")
    return lines
