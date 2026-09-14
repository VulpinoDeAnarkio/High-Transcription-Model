"""Paski postępu (plik + kolejka) — czysty HTML, bez Gradio."""

from __future__ import annotations

import html


def clamp_pct(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def queue_percent(finished_before: int, file_frac: float, total: int) -> float:
    """finished_before = ile plików już domknięte; file_frac 0–1 bieżącego."""
    if total <= 0:
        return 0.0
    frac = max(0.0, min(1.0, float(file_frac)))
    return clamp_pct(100.0 * (finished_before + frac) / total)


def render_bar(pct: float, title: str, subtitle: str = "") -> str:
    pct = clamp_pct(pct)
    title_h = html.escape(title or "")
    sub_h = html.escape(subtitle or "")
    extra = f'<div class="htm-progress-sub">{sub_h}</div>' if sub_h else ""
    return (
        f'<div class="htm-progress">'
        f'<div class="htm-progress-row"><span class="htm-progress-title">{title_h}</span>'
        f'<span class="htm-progress-pct">{pct:.0f}%</span></div>'
        f"{extra}"
        f'<div class="htm-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="{pct:.0f}">'
        f'<div class="htm-progress-fill" style="width:{pct:.1f}%"></div>'
        f"</div></div>"
    )
