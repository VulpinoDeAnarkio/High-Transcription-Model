from __future__ import annotations

import pytest

from htm.cli import build_parser
from htm.models import (
    DEFAULT_MODEL,
    QUALITY_MODEL,
    normalize_model,
    primary_models,
    radio_choices,
    spec_for,
)
from htm.types import Settings


def test_default_is_turbo() -> None:
    assert DEFAULT_MODEL == "large-v3-turbo"
    assert Settings().model == DEFAULT_MODEL


def test_normalize_ui_label() -> None:
    assert normalize_model("Szybszy — large-v3-turbo (~1,6 GB)") == "large-v3-turbo"
    assert normalize_model("Dokładniejszy — large-v3 (~3 GB)") == "large-v3"
    assert normalize_model("turbo") == "large-v3-turbo"
    assert normalize_model("large") == QUALITY_MODEL
    assert normalize_model("large-v3") == "large-v3"
    assert normalize_model(None) == DEFAULT_MODEL


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Nieznany model"):
        normalize_model("whisper-xyz")
    with pytest.raises(ValueError, match="Nieznany model"):
        Settings(model="nie-ma-takiego")


def test_primary_choice_is_exactly_two() -> None:
    ids = [s.id for s in primary_models()]
    assert ids == ["large-v3-turbo", "large-v3"]
    values = [value for _label, value in radio_choices()]
    assert values == ids


def test_quality_spec() -> None:
    spec = spec_for("large")
    assert spec.id == "large-v3"
    assert "Dokładniejszy" in spec.title


def test_cli_accepts_both_models() -> None:
    parser = build_parser()
    turbo = parser.parse_args(["transcribe", "a.mp3"])
    assert turbo.model == "large-v3-turbo"
    quality = parser.parse_args(["transcribe", "a.mp3", "--model", "large-v3"])
    assert quality.model == "large-v3"
    alias = parser.parse_args(["transcribe", "a.mp3", "-m", "large"])
    assert alias.model == "large-v3"


def test_cli_rejects_unknown_model() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "a.mp3", "--model", "xyz"])
