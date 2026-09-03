from __future__ import annotations

import numpy as np
import soundfile as sf

from htm.diarize import assign_speakers
from htm.types import Segment


def test_gender_split_on_two_tones(tmp_path) -> None:
    sr = 16_000
    t = np.linspace(0, 1.4, int(sr * 1.4), endpoint=False)
    male = 0.25 * np.sin(2 * np.pi * 110 * t)
    female = 0.25 * np.sin(2 * np.pi * 230 * t)
    gap = np.zeros(int(0.4 * sr))
    y = np.concatenate([male, gap, female])
    wav = tmp_path / "two.wav"
    sf.write(wav, y, sr)

    text = "raz dwa trzy cztery pięć sześć siedem osiem"
    segs = [
        Segment(0.05, 1.3, text),
        Segment(1.85, 3.2, text),
    ]
    merged, infos, notes = assign_speakers(wav, segs, n_speakers=2, names=["Mężczyzna", "Kobieta"])
    assert len(merged) == 2
    assert {s.speaker for s in merged} == {"Mężczyzna", "Kobieta"}
    assert any("F0" in n for n in notes)
