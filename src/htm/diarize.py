"""Acoustic speaker diarization on top of ASR segments.

Pipeline:
1. YIN pitch track for the whole file.
2. Per-segment embedding (MFCC + spectral + F0).
3. Optional gender split when F0 is clearly bimodal (e.g. 1 man + 2 women).
4. Agglomerative clustering within each gender group.
5. Temporal smoothing of short backchannels.
6. Merge adjacent turns of the same speaker.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from htm.polish import first_person_gender
from htm.types import Segment, SpeakerInfo

MALE_F0_MAX = 155.0
FEMALE_F0_MIN = 165.0
MIN_EMBED_SEC = 0.35
SHORT_SEC = 0.55
MAX_SHORT_WORDS = 2


def format_timestamp(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def median_f0(times: np.ndarray, f0: np.ndarray, start: float, end: float) -> float:
    mask = (times >= start) & (times < end) & np.isfinite(f0)
    vals = f0[mask]
    if vals.size < 3:
        return float("nan")
    lo, hi = np.percentile(vals, [15, 85])
    core = vals[(vals >= lo) & (vals <= hi)]
    if core.size < 2:
        core = vals
    return float(np.median(core))


def _embed(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray | None:
    a = max(0, int(start * sr))
    b = min(len(y), int(end * sr))
    if b - a < int(MIN_EMBED_SEC * sr):
        return None
    clip = y[a:b]
    mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    centroid = float(librosa.feature.spectral_centroid(y=clip, sr=sr).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(y=clip, sr=sr).mean())
    rolloff = float(librosa.feature.spectral_rolloff(y=clip, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(clip).mean())
    rms = float(librosa.feature.rms(y=clip).mean())
    return np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            delta.mean(axis=1),
            np.array([centroid, bandwidth, rolloff, zcr, rms], dtype=np.float64),
        ]
    )


def _pitch_track(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    hop = 256
    f0 = librosa.yin(y, fmin=60, fmax=350, sr=sr, hop_length=hop, frame_length=2048)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    f0 = f0.astype(np.float64)
    f0[(f0 < 60) | (f0 > 350)] = np.nan
    return times, f0


def _f0_threshold(f0_values: list[float]) -> float:
    vals = np.array([v for v in f0_values if v == v], dtype=np.float64)
    if vals.size < 8:
        return MALE_F0_MAX
    # Midpoint between low and high modes if the distribution is split.
    low = vals[vals < 160]
    high = vals[vals >= 170]
    if low.size >= 3 and high.size >= 3:
        return float((np.median(low) + np.median(high)) / 2)
    return MALE_F0_MAX


def _cluster_vectors(vectors: np.ndarray, n: int) -> np.ndarray:
    n = max(1, min(n, len(vectors)))
    if n == 1:
        return np.zeros(len(vectors), dtype=int)
    scaled = StandardScaler().fit_transform(vectors)
    return AgglomerativeClustering(n_clusters=n, linkage="ward").fit_predict(scaled)


def _best_n(vectors: np.ndarray, k_max: int) -> int:
    if len(vectors) < 4:
        return 1
    scaled = StandardScaler().fit_transform(vectors)
    k_max = max(1, min(k_max, len(vectors) - 1))
    best_k, best_score = 1, -1.0
    for k in range(2, k_max + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled)
        if len(set(labels)) < 2:
            continue
        try:
            score = float(silhouette_score(scaled, labels))
        except Exception:
            continue
        if score > best_score:
            best_k, best_score = k, score
    return best_k if best_score >= 0.12 else 1


def _gender_of_f0(f0: float | None, threshold: float) -> str | None:
    if f0 is None or not np.isfinite(f0):
        return None
    if f0 < threshold:
        return "M"
    if f0 >= max(threshold, FEMALE_F0_MIN - 5):
        return "K"
    return None


def assign_speakers(
    audio_path: str | Path,
    segments: list[Segment],
    n_speakers: int | None = None,
    names: list[str] | None = None,
    merge_gap: float = 1.4,
) -> tuple[list[Segment], list[SpeakerInfo], list[str]]:
    notes: list[str] = []
    if not segments:
        return [], [], notes

    y, sr = librosa.load(str(audio_path), sr=16_000, mono=True)
    times, f0 = _pitch_track(y, sr)

    for seg in segments:
        med = median_f0(times, f0, seg.start, seg.end)
        seg.median_f0 = None if not np.isfinite(med) else med

    f0s = [s.median_f0 for s in segments if s.median_f0 is not None]
    threshold = _f0_threshold([v for v in f0s if v is not None])
    notes.append(f"Próg F0 mężczyzna/kobieta ≈ {threshold:.0f} Hz.")

    indexed_vecs: list[tuple[int, np.ndarray, str | None]] = []
    for i, seg in enumerate(segments):
        vec = _embed(y, sr, seg.start, seg.end)
        gender = _gender_of_f0(seg.median_f0, threshold)
        ling = first_person_gender(seg.text)
        if gender is None and ling:
            gender = ling
        if vec is None:
            continue
        extra = np.array(
            [
                0.0 if seg.median_f0 is None else seg.median_f0,
                1.0 if gender == "M" else (2.0 if gender == "K" else 0.0),
            ]
        )
        indexed_vecs.append((i, np.concatenate([vec, extra]), gender))

    male_idx = [i for i, _, g in indexed_vecs if g == "M"]
    female_idx = [i for i, _, g in indexed_vecs if g == "K"]
    other_idx = [i for i, _, g in indexed_vecs if g not in {"M", "K"}]
    use_gender_split = len(male_idx) >= 2 and len(female_idx) >= 2

    cluster_of: dict[int, int] = {}
    next_cluster = 0

    if use_gender_split:
        male_time = sum(segments[i].end - segments[i].start for i in male_idx)
        female_time = sum(segments[i].end - segments[i].start for i in female_idx)
        notes.append(
            f"Wykryto dwa rejestry głosu (męski {male_time:.0f}s / żeński {female_time:.0f}s)."
        )
        if n_speakers is None:
            n_male = _best_n(np.vstack([v for i, v, g in indexed_vecs if g == "M"]), 3)
            n_female = _best_n(np.vstack([v for i, v, g in indexed_vecs if g == "K"]), 4)
        else:
            n_male = 1 if male_time >= female_time * 0.08 else max(1, n_speakers // 3)
            n_male = min(n_male, max(1, n_speakers - 1))
            n_female = max(1, n_speakers - n_male)

        vec_by_i = {i: v for i, v, _g in indexed_vecs}

        def _assign_group(indices: list[int], n: int) -> None:
            nonlocal next_cluster
            if not indices:
                return
            vecs = np.vstack([vec_by_i[i] for i in indices])
            labels = _cluster_vectors(vecs, n)
            remap = {old: next_cluster + j for j, old in enumerate(sorted(set(labels)))}
            next_cluster += len(remap)
            for i, lab in zip(indices, labels):
                cluster_of[i] = remap[int(lab)]

        _assign_group(male_idx, n_male)
        _assign_group(female_idx, n_female)
        # Ambiguous segments: nearest cluster centroid of any gender.
        if other_idx:
            known = [(i, v) for i, v, g in indexed_vecs if i in cluster_of]
            cents: dict[int, list[np.ndarray]] = defaultdict(list)
            for i, v in known:
                cents[cluster_of[i]].append(v)
            mean_c = {c: np.mean(np.vstack(vs), axis=0) for c, vs in cents.items()}
            vec_by_i = {i: v for i, v, g in indexed_vecs}
            for i in other_idx:
                v = vec_by_i[i]
                cluster_of[i] = min(mean_c, key=lambda c: float(np.linalg.norm(v - mean_c[c])))
    else:
        vecs = np.vstack([v for _, v, _ in indexed_vecs])
        k = n_speakers if n_speakers else _best_n(vecs, 6)
        labels = _cluster_vectors(vecs, k)
        for (i, _, _), lab in zip(indexed_vecs, labels):
            cluster_of[i] = int(lab)
        notes.append(f"Klasteryzacja bez podziału płci, k={len(set(cluster_of.values()))}.")

    # Fill remaining segments from neighbours.
    last = 0
    for i, seg in enumerate(segments):
        if i in cluster_of:
            last = cluster_of[i]
            continue
        cluster_of[i] = last

    # Temporal smoothing for short backchannels.
    last_cluster = cluster_of.get(0, 0)
    last_gender = _gender_of_f0(segments[0].median_f0, threshold)
    for i, seg in enumerate(segments):
        c = cluster_of[i]
        g = _gender_of_f0(seg.median_f0, threshold) or first_person_gender(seg.text)
        n_words = len(seg.text.split())
        short = (seg.end - seg.start) < SHORT_SEC or n_words <= MAX_SHORT_WORDS
        if short and last_gender and g in {None, last_gender}:
            c = last_cluster
        cluster_of[i] = c
        last_cluster = c
        if g:
            last_gender = g

    # Cluster → gender by median F0 + morphology votes
    f0_by_c: dict[int, list[float]] = defaultdict(list)
    gender_votes: dict[int, list[str]] = defaultdict(list)
    for i, seg in enumerate(segments):
        c = cluster_of[i]
        if seg.median_f0 is not None:
            f0_by_c[c].append(seg.median_f0)
        ling = first_person_gender(seg.text)
        if ling:
            gender_votes[c].append(ling)

    def cluster_gender(c: int) -> str | None:
        votes = gender_votes.get(c, [])
        if votes:
            if votes.count("M") > votes.count("K"):
                return "M"
            if votes.count("K") > votes.count("M"):
                return "K"
        vals = f0_by_c.get(c, [])
        if not vals:
            return None
        return "M" if float(np.median(vals)) < threshold else "K"

    # Labels: Mówca M1 / K1… then optional user names by first appearance
    first_time: dict[int, float] = {}
    for i, seg in enumerate(segments):
        c = cluster_of[i]
        first_time.setdefault(c, seg.start)
    order = sorted(first_time, key=lambda c: first_time[c])

    gender_count = {"M": 0, "K": 0, "?": 0}
    auto_label: dict[int, str] = {}
    for c in order:
        g = cluster_gender(c) or "?"
        if g == "M":
            gender_count["M"] += 1
            auto_label[c] = f"Mówca M{gender_count['M']}"
        elif g == "K":
            gender_count["K"] += 1
            auto_label[c] = f"Mówca K{gender_count['K']}"
        else:
            gender_count["?"] += 1
            auto_label[c] = f"Mówca {gender_count['?']}"

    names = [n.strip() for n in (names or []) if n and n.strip()]
    label_of: dict[int, str] = dict(auto_label)
    if names:
        for c, name in zip(order, names):
            label_of[c] = name
        if len(names) != len(order):
            notes.append(
                f"Podano {len(names)} imion, wykryto {len(order)} mówców. "
                "Imiona mapowane w kolejności pierwszego zabrania głosu."
            )

    for i, seg in enumerate(segments):
        c = cluster_of[i]
        seg.cluster = c
        seg.speaker = label_of[c]

    merged = merge_adjacent(segments, merge_gap=merge_gap)

    infos: list[SpeakerInfo] = []
    for c in order:
        segs_c = [s for s in merged if s.cluster == c]
        speaking = sum(s.end - s.start for s in segs_c)
        samples = [s.text for s in segs_c if len(s.text.split()) >= 6][:3]
        if not samples:
            samples = [s.text for s in segs_c[:3]]
        f0_med = float(np.median(f0_by_c[c])) if f0_by_c.get(c) else None
        infos.append(
            SpeakerInfo(
                label=label_of[c],
                cluster=c,
                median_f0=f0_med,
                gender=cluster_gender(c),
                speaking_seconds=speaking,
                samples=samples,
            )
        )
    return merged, infos, notes


def merge_adjacent(segments: list[Segment], merge_gap: float = 1.4) -> list[Segment]:
    if not segments:
        return []
    out = [
        Segment(
            start=segments[0].start,
            end=segments[0].end,
            text=segments[0].text,
            speaker=segments[0].speaker,
            median_f0=segments[0].median_f0,
            cluster=segments[0].cluster,
            avg_logprob=segments[0].avg_logprob,
            words=list(segments[0].words),
        )
    ]
    for seg in segments[1:]:
        prev = out[-1]
        same = seg.speaker == prev.speaker and (seg.start - prev.end) <= merge_gap
        if same:
            prev.end = seg.end
            prev.text = (prev.text + " " + seg.text).strip()
            f0s = [x for x in (prev.median_f0, seg.median_f0) if x is not None]
            prev.median_f0 = float(np.median(f0s)) if f0s else prev.median_f0
            prev.words.extend(seg.words)
        else:
            out.append(
                Segment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker=seg.speaker,
                    median_f0=seg.median_f0,
                    cluster=seg.cluster,
                    avg_logprob=seg.avg_logprob,
                    words=list(seg.words),
                )
            )
    for s in out:
        s.text = " ".join(s.text.split())
    return [s for s in out if s.text and s.text not in {".", ",", "-", "…"}]
