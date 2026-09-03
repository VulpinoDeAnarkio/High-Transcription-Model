# Jak działa pipeline

```
plik mp4/wav
    │
    ▼
ffmpeg  ──►  WAV 16 kHz mono
    │
    ▼
faster-whisper (VAD + beam search + word timestamps)
    │
    ▼
YIN F0  +  MFCC/widmo  ──►  klastry mówców
    │                         (opcjonalny split M/K)
    ▼
wygładzenie krótkich wtrąceń + scalanie sąsiednich kwestii
    │
    ▼
Markdown / TXT / SRT / JSON
```

## Diarizacja

1. Dla całego nagrania liczona jest wysokość głosu (YIN, 60–350 Hz).
2. Każdy segment Whisper dostaje medianę F0 (obcięte percentyle 15–85, żeby piknięcia YIN nie psuły etykiety).
3. Jeśli rozkład F0 jest dwumodalny (jest głos męski i żeński), najpierw dzielimy po rejestrze, potem klastrujemy **wewnątrz** płci. Dzięki temu dwóch kobiet nie miesza się z jedynym mężczyzną.
4. W przeciwnym razie jedna klasteryzacja Ward na wektorach MFCC (20 pasm + delta + centroid/bandwidth/rolloff/ZCR/RMS + F0).
5. Segmenty krótsze niż 0,55 s albo ≤ 2 słowa dziedziczą poprzedniego mówcę tej samej płci.
6. Sąsiednie kwestie tego samego mówcy z przerwą ≤ 1,4 s są sklejane.

Liczba mówców: `--speakers N` albo silhouette (2–6) w trybie auto.

Na macOS CTranslate2 **nie używa Metal** — transkrypcja idzie na CPU (M-series i tak jest szybki). `int8` ma fallback na `int8_float32` / `float32`.

## Czego tu nie ma (świadomie)

- **pyannote.audio** — wymaga tokenu Hugging Face, PyTorch i zwykle GPU; na CPU bywa wolniejszy niż ten pipeline, a próg wejścia jest wyższy.
- **Separacja nakładających się głosów** — Whisper i tak zwraca jedną ścieżkę; nakładki trzeba czytać ostrożnie.
- **OCR kafelków Teams** — nazwy z obrazu bywają ucięte; imiona podaje użytkownik.
