# High Transcription Model

Lokalna aplikacja do transkrypcji nagrań audio/wideo i **identyfikacji mówców po głosie**. Działa offline po pierwszym pobraniu modelu — nic nie idzie do chmury.

Powstała z pipeline’u użytego do transkrypcji rozmowy HR (faster-whisper `large-v3-turbo` + diarizacja akustyczna: wysokość głosu, MFCC i ciągłość czasowa).

## Co robi

1. Wyciąga ścieżkę dźwiękową (`ffmpeg` → WAV 16 kHz, mono).
2. Transkrybuje **faster-whisper**. Wybierasz model:
   - **`large-v3-turbo`** (domyślny) — szybszy, dobry do rozmów i spotkań;
   - **`large-v3`** — pełny, dokładniejszy, wolniejszy (czytanie, odzyskiwanie tekstu).
3. Dzieli wypowiedzi na mówców:
   - śledzenie F0 (YIN) — pewne rozróżnienie mężczyzna / kobieta,
   - wektory MFCC + cechy widmowe,
   - klasteryzacja hierarchiczna,
   - wygładzanie krótkich wtrąceń,
   - opcjonalnie polskie formy 1. os. (*powiedziałem* / *chciałabym*) jako słaby prior płci.
4. Zapisuje **Markdown, TXT, SRT i JSON**. W interfejsie możesz złożyć **kolejkę** wielu nagrań (wrzucenie albo folder na dysku) i odpalić je jedno po drugim.

Imiona nie są zgadywane z treści spotkania. Podajesz je sam (`--names`) w kolejności **pierwszego zabrania głosu**, albo zostawiasz etykiety `Mówca M1` / `Mówca K1`.

## Wymagania

- **Linux** albo **macOS** 13+ (Apple Silicon i Intel)
- Python **3.11+** (zalecane 3.12)
- [ffmpeg](https://ffmpeg.org/)
- [uv](https://docs.astral.sh/uv/) (zalecane) albo zwykły `pip`
- ~2 GB na `large-v3-turbo` albo ~3 GB na `large-v3` przy pierwszym uruchomieniu danego modelu
- CPU wystarczy. GPU NVIDIA (`--device cuda`) tylko poza macOS — na Macu CTranslate2 liczy na CPU, nie na Metal.

## Szybki start (interfejs w przeglądarce)

**Linux**

```bash
cd "High Transcription Model"
chmod +x run.sh
./run.sh
```

**macOS** — [docs/MACOS.md](docs/MACOS.md)

```bash
brew install ffmpeg uv
chmod +x run.sh run.command
./run.sh
# albo dwuklik run.command w Finderze
```

Otworzy się http://127.0.0.1:7860. Ten sam widok na Linuxie i macOS. Możesz wrzucić **wiele plików** albo wkleić **ścieżkę folderu** (lepsze przy dużych MP3/MP4 — nic nie kopiuje się do przeglądarki). Język, model i imiona ustawiasz raz dla całej kolejki, potem **Uruchom kolejkę**. Widać pasek postępu bieżącego pliku (czas transkrypcji); przy więcej niż jednym nagraniu dochodzi pasek całej kolejki. Pliki idą po kolei; **Zatrzymaj po bieżącym** nie ubija Whisper w połowie nagrania.

Krok po kroku: [docs/LOKALNE-URUCHOMIENIE.md](docs/LOKALNE-URUCHOMIENIE.md) (Linux) · [docs/MACOS.md](docs/MACOS.md).

## Linia poleceń

```bash
source .venv/bin/activate

# diagnostyka
htm doctor

# transkrypcja (domyślnie large-v3-turbo)
htm transcribe nagranie.mp4 \
  --language pl \
  --speakers 3 \
  --names "Astryda Lisienko,Paweł Bernach,Marcelina Dusza" \
  --out ./output

# dokładniejszy, wolniejszy model
htm transcribe nagranie.mp3 --language pl --model large-v3 --out ./output
```

Wyniki:

| plik | zawartość |
|---|---|
| `*_transkrypcja.md` | transkrypcja z mówcami i timestampami |
| `*_transkrypcja.txt` | wersja linia-po-linii |
| `*_transkrypcja.srt` | napisy |
| `*_transkrypcja.json` | pełne dane (klastry, F0) |

Przydatne flagi:

- `--model large-v3` — pełny model, wyższa jakość, wolniej (alias: `--model large`)
- `--model large-v3-turbo` — szybszy wariant (domyślny; alias: `--model turbo`)
- `--model small` — szybki podgląd
- `--speakers 3` — gdy wiesz, ile osób mówi
- `--insecure-ssl` — gdy korporacyjne TLS blokuje Hugging Face
- `--no-previous-text` — czasem lepiej przy częstych zmianach mówcy
- `--device cuda` — GPU NVIDIA (nie dotyczy macOS)

## Jak rozpoznawani są mówcy

| sygnał | do czego służy |
|---|---|
| wysokość głosu (F0) | mężczyzna ~80–150 Hz, kobieta ~170–280 Hz |
| MFCC + widmo | rozróżnienie dwóch osób tej samej płci |
| czas | krótkie „tak / mhm” dziedziczą poprzedniego mówcę tej samej płci |
| morfologia PL | *powiedziałem* vs *zostałam* jako wskazówka płci, nie imienia |

**Ograniczenia:** nakładające się głosy Whisper zlewa w jedną ścieżkę; bardzo krótkie kwestie bywają przypisane sąsiadowi; dwie osoby o podobnej barwie wymagają `--speakers` i ręcznego sprawdzenia imion.

## Cache modeli

Domyślnie: `./.htms-cache`. Możesz wskazać istniejący cache Hugging Face:

```bash
export HTM_CACHE="/ścieżka/do/hf-cache"
htm transcribe nagranie.mp4
```

## Licencja

Kod aplikacji: MIT. Modele Whisper podlegają licencjom autorów (faster-whisper / OpenAI Whisper).
