# macOS — High Transcription Model

Ta sama aplikacja co na Linuxie. Na Macu **faster-whisper / CTranslate2 liczy na CPU**, nie na Metal ani Apple Neural Engine. M1/M2/M3/M4 i tak są zwykle wyraźnie szybsze niż typowy laptopowy CPU x86.

Obsługiwane: macOS 13+ · Apple Silicon i Intel.

## 1. Homebrew

Jeśli nie masz Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Na Apple Silicon dopisz do `~/.zprofile` (instalator zwykle sam to podpowie):

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

## 2. ffmpeg i uv

```bash
brew install ffmpeg uv
ffmpeg -version | head -1
uv --version
```

Gdy `soundfile` zgłasza brak biblioteki:

```bash
brew install libsndfile
```

## 3. Start z Findera albo Terminala

W katalogu `High Transcription Model`:

```bash
chmod +x run.sh run.command
./run.sh
```

Albo **dwuklik** `run.command` — otworzy Terminal i UI pod http://127.0.0.1:7860

Pierwsze uruchomienie tworzy `.venv`, dociąga zależności i wybrany model Whisper (`large-v3-turbo` ~1,6 GB, `large-v3` ~3 GB).

## 4. Diagnostyka

```bash
source .venv/bin/activate
htm doctor
```

Spodziewany fragment:

```
system: macOS … (Apple Silicon, arm64)
ffmpeg: /opt/homebrew/bin/ffmpeg
faster_whisper: OK
```

Jeśli `ffmpeg: BRAK` przy zainstalowanym brew — zamknij Terminal i otwórz **nowe** okno (PATH). `run.sh` i `run.command` i tak dokładają `/opt/homebrew/bin`.

## 5. Transkrypcja z Terminala

```bash
cd "/ścieżka/do/High Transcription Model"
source .venv/bin/activate

htm transcribe ~/Desktop/nagranie.mp4 \
  --language pl \
  --speakers 3 \
  --names "Osoba A, Osoba B, Osoba C" \
  --out ./output
```

`--device cuda` na Macu **nie działa** (brak NVIDIA). Zostaw `auto` — aplikacja wybierze CPU. Gdy `int8` nie wstanie na Twoim CTranslate2, silnik sam spadnie na `int8_float32` / `float32`.

## 6. Uprawnienia macOS

- **Prywatność → Pełny dostęp do dysku / Pliki i foldery**: Terminal (albo Cursor) musi móc czytać katalog z nagraniem (Biurko, iCloud, „Pobrane”).
- **Gatekeeper**: przy pierwszym dwukliku `run.command` → prawy przycisk → **Otwórz**.
- Nagrania z iPhone (`.mov`) są obsługiwane tak samo jak `.mp4`.

## 7. TLS / sieć firmowa

Tak samo jak na Linuxie:

```bash
export HTM_INSECURE_SSL=1
htm transcribe nagranie.mp4 --insecure-ssl
```

albo checkbox w UI.

## 8. Wydajność (orientacyjnie)

| sprzęt | ~28 min nagrania, `large-v3-turbo` |
|---|---|
| M1 / M2 | zwykle kilkanaście–30 min |
| Intel Mac | bliżej 30–50 min |
| NVIDIA GPU (tylko Linux/Windows) | kilka–kilkanaście min |

`large-v3` na tym samym sprzęcie bywa wyraźnie wolniejszy (orientacyjnie 2–3×).

Nie instaluj CUDA na Macu. Nie oczekuj przyspieszenia z `mps` / PyTorch Metal — ten stack idzie przez CTranslate2 na CPU.
