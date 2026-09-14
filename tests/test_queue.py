from __future__ import annotations

from pathlib import Path

from htm.queue import (
    DONE,
    SKIPPED,
    WAITING,
    add_paths,
    collect_from_folder,
    expected_markdown,
    format_queue,
    job_from_path,
    jobs_from_state,
    jobs_to_state,
    mark_skips,
    pending,
)


def test_collect_folder_non_recursive(tmp_path: Path) -> None:
    (tmp_path / "a.mp3").write_bytes(b"x" * 100)
    (tmp_path / "b.wav").write_bytes(b"x" * 100)
    (tmp_path / "readme.txt").write_text("nope")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "c.mp4").write_bytes(b"x" * 100)
    found = collect_from_folder(tmp_path, recursive=False)
    names = [p.name for p in found]
    assert names == ["a.mp3", "b.wav"]
    deep = collect_from_folder(tmp_path, recursive=True)
    assert {p.name for p in deep} == {"a.mp3", "b.wav", "c.mp4"}


def test_add_paths_skips_duplicates_and_non_media(tmp_path: Path) -> None:
    a = tmp_path / "spotkanie.mp4"
    a.write_bytes(b"x" * 50)
    (tmp_path / "notes.md").write_text("x")
    jobs, n = add_paths([], [a, a, tmp_path / "notes.md"])
    assert n == 1
    assert len(jobs) == 1
    jobs2, n2 = add_paths(jobs, [a])
    assert n2 == 0
    assert len(jobs2) == 1


def test_mark_skips_existing_markdown(tmp_path: Path) -> None:
    src = tmp_path / "rozmowa.mp3"
    src.write_bytes(b"x" * 200)
    md = tmp_path / "rozmowa_transkrypcja.md"
    md.write_text("# already\n" + "x" * 100)
    job = job_from_path(src)
    marked = mark_skips([job], beside=True, out_dir=tmp_path / "output")
    assert marked[0].status == SKIPPED
    assert Path(marked[0].md) == md
    assert expected_markdown(job, beside=True, out_dir=tmp_path / "out") == md


def test_queue_state_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "a.m4a"
    src.write_bytes(b"abc")
    jobs, _ = add_paths([], [src])
    jobs[0].status = DONE
    raw = jobs_to_state(jobs)
    back = jobs_from_state(raw)
    assert back[0].name == "a.m4a"
    assert back[0].status == DONE
    assert pending(back) == []


def test_format_queue_empty_and_rows(tmp_path: Path) -> None:
    assert "pusta" in format_queue([])
    src = tmp_path / "nagranie.mkv"
    src.write_bytes(b"x" * 2048)
    jobs, _ = add_paths([], [src])
    text = format_queue(jobs)
    assert "nagranie.mkv" in text
    assert "oczekuje" in text
    assert jobs[0].status == WAITING
