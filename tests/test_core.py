from __future__ import annotations

from htm.diarize import format_timestamp, merge_adjacent
from htm.export import to_markdown, to_srt, to_txt
from htm.polish import first_person_gender
from htm.types import Segment, SpeakerInfo, Transcript


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3723) == "1:02:03"


def test_first_person_gender() -> None:
    assert first_person_gender("Nie zdążyłem chyba wprowadzić.") == "M"
    assert first_person_gender("Chciałabym Ci przekazać kilka informacji.") == "K"
    assert first_person_gender("Proszę bardzo.") is None


def test_merge_adjacent() -> None:
    segs = [
        Segment(0.0, 1.0, "Ala", speaker="A"),
        Segment(1.2, 2.0, "ma kota", speaker="A"),
        Segment(4.0, 5.0, "Nie.", speaker="B"),
    ]
    out = merge_adjacent(segs, merge_gap=1.4)
    assert len(out) == 2
    assert out[0].text == "Ala ma kota"
    assert out[1].speaker == "B"


def test_txt_and_srt() -> None:
    result = Transcript(
        language="pl",
        language_probability=1.0,
        source=__import__("pathlib").Path("spotkanie.mp4"),
        audio=__import__("pathlib").Path("a.wav"),
        model="large-v3-turbo",
        speakers=[
            SpeakerInfo(
                label="Jan",
                cluster=0,
                median_f0=110.0,
                gender="M",
                speaking_seconds=2.0,
                samples=["Dzień dobry."],
            )
        ],
        segments=[
            Segment(1.0, 2.5, "Dzień dobry.", speaker="Jan"),
        ],
    )
    txt = to_txt(result)
    assert "Jan: Dzień dobry." in txt
    srt = to_srt(result)
    assert "00:00:01,000 --> 00:00:02,500" in srt
    assert "Jan: Dzień dobry." in srt
    md = to_markdown(result)
    assert "large-v3-turbo" in md
    assert "szybszy" in md


def test_av_metadata_errors_compatibility() -> None:
    import io
    import av

    try:
        av.open(io.BytesIO(b""), mode="r", metadata_errors="ignore")
    except TypeError as e:
        assert "metadata_errors" not in str(e)
    except Exception:
        # Oczekiwany brak poprawnego kontenera danych, ale NIE TypeError o metadata_errors
        pass

