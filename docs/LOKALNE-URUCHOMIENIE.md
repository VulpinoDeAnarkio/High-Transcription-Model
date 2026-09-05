# Uruchomienie lokalne — High Transcription Model

- **macOS** (Homebrew, Finder, Apple Silicon): [MACOS.md](MACOS.md)
- Poniżej: **Ubuntu / Debian**. WSL2 jak Linux.

## 1. Zależności systemowe

```bash
sudo apt update
sudo apt install -y ffmpeg python3 python3-venv python3-pip
ffmpeg -version | head -1
```

`ffmpeg` jest **wymagany** — bez niego aplikacja nie wyciągnie dźwięku z MP4.

Na macOS zamiast `apt`: `brew install ffmpeg uv` — szczegóły w [MACOS.md](MACOS.md).

## 2. Instalator pakietów Python: uv (zalecane)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Bez uv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 3. Instalacja aplikacji

W katalogu `High Transcription Model`:

```bash
cd "/ścieżka/do/High Transcription Model"
uv venv --python 3.12
source .venv/bin/activate
uv pip install --system-certs -e ".[dev]"
```

Jeśli instalacja pada na certyfikacie TLS (`UnknownIssuer`, `CERTIFICATE_VERIFY_FAILED`):

```bash
uv pip install --system-certs -e ".[dev]"
# albo
export HTM_INSECURE_SSL=1
uv pip install -e ".[dev]"
```

Sprawdzenie:

```bash
htm doctor
```

Oczekiwane: `ffmpeg: …`, `faster_whisper: OK`, `gradio: OK`.

## 4. Interfejs graficzny

```bash
./run.sh
# albo po aktywacji venv:
htm ui
```

Domyślnie: [http://127.0.0.1:7860](http://127.0.0.1:7860) (tylko lokalny host).

Opcje:

```bash
htm ui --port 7861
htm ui --host 127.0.0.1 --port 7860
```

`--share` tworzy tymczasowy publiczny URL Gradio — **nie używaj** dla nagrań poufnych (HR, medyczne, wewnętrzne).

## 5. Transkrypcja z terminala

```bash
source .venv/bin/activate

htm transcribe "/ścieżka/nagranie.mp4" \
  --language pl \
  --speakers 3 \
  --names "Osoba A, Osoba B, Osoba C" \
  --out ./output
```

Pliki pojawią się w `./output/`:

- `*_transkrypcja.md`
- `*_transkrypcja.txt`
- `*_transkrypcja.srt`
- `*_transkrypcja.json`

### Kolejność imion

`--names` mapuje klastry **w kolejności pierwszego zabrania głosu** na nagraniu, nie alfabetycznie. Jeśli pierwsza mówi Marcelina, drugie imię w liście nie może być jej, jeśli Paweł odezwał się wcześniej.

Po transkrypcji zajrzyj do nagłówka Markdown (próbki wypowiedzi przy każdym mówcy) i w razie pomyłki popraw etykiety w pliku albo odpal ponownie z inną kolejnością.

### Dobór modelu

W interfejsie są **dwa** warianty (przełącznik):

| model | kiedy |
|---|---|
| `large-v3-turbo` (domyślny, **Szybszy**) | rozmowy, spotkania; ten sam co przy transkrypcji HR |
| `large-v3` (**Dokładniejszy**) | czytanie, odzyskiwanie tekstu; wolniejszy, zwykle mniej pomyłek |

Z linii poleceń: `--model large-v3-turbo` albo `--model large-v3` (aliasy: `turbo`, `large`). Dodatkowo `medium` / `small` / `base` tylko do podglądu.

Na CPU nagranie ~28 min przy `large-v3-turbo` trwa zwykle **ok. 20–40 min**. `large-v3` bywa 2–3× wolniejszy. GPU NVIDIA (Linux) skraca to kilkukrotnie (`--device cuda`). Na macOS zostaje CPU — M-series i tak bywa szybszy niż typowy laptop x86.

## 6. Hugging Face / TLS

Pierwsze uruchomienie pobiera wybrany model (`large-v3-turbo` ~1,6 GB, `large-v3` ~3 GB). Typowe błędy:

```
CERTIFICATE_VERIFY_FAILED
LocalEntryNotFoundError
```

Rozwiązanie:

```bash
export HTM_INSECURE_SSL=1
htm transcribe nagranie.mp4 --language pl --insecure-ssl
```

Albo w UI zaznacz *Sieć korporacyjna: wyłącz weryfikację TLS*.

Gotowy cache (np. z wcześniejszej transkrypcji) możesz podłączyć:

```bash
export HTM_CACHE="/home/…/hf-cache"
htm transcribe nagranie.mp4
```

## 7. Testy jednostkowe (bez modelu)

```bash
source .venv/bin/activate
pytest
```

Nie pobierają Whisper — sprawdzają znaczniki czasu, scalanie kwestii i formy gramatyczne.

## 8. Poufność

Wszystko liczy się lokalnie. Nie włączaj `htm ui --share` dla nagrań służbowych. Katalog `output/` i `.htms-cache/` trzymaj poza chmurą, jeśli nagranie jest wrażliwe.
