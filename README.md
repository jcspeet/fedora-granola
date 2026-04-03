# eatmo

A local, privacy-first meeting transcription and notes app for Fedora Linux.
Works like [Granola](https://granola.ai/) — no meeting bots, no cloud uploads for audio.

## Features

- Captures **microphone** + **system audio output** simultaneously (via PipeWire monitor)
- Transcribes in real-time using **Whisper** (runs locally, no data leaves your machine)
- Generates structured **meeting notes** via your choice of AI provider
- **Chat** with your meeting notes or across all past meetings
- Native **GTK4 / libadwaita** UI that fits right in on GNOME/Fedora
- Launches from the **application grid** like a normal app
- **Save** sessions as Markdown files

## Requirements

- Fedora 38+ (uses PipeWire as audio server)
- Python 3.11+
- One of the following for summarization:
  - [Anthropic API key](https://console.anthropic.com/) (default)
  - [OpenAI API key](https://platform.openai.com/)
  - [Ollama](https://ollama.com/) running locally

## Installation

```bash
./install.sh
```

This installs system packages via `dnf`, Python dependencies into a virtualenv, and registers the app icon and desktop launcher so eatmo appears in your application grid.

## Configuration

API keys and provider settings can be set in the **Settings** dialog inside the app (gear icon in the header bar), or manually:

**Environment variable:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Config file** (persists across sessions):
```bash
mkdir -p ~/.config/eatmo
echo 'ANTHROPIC_API_KEY=sk-ant-...' > ~/.config/eatmo/config.env
```

## Running

From the application grid, or:

```bash
./launch.sh
```

## Usage

1. Launch the app — it loads the Whisper model in the background (~5–15s)
2. Once **"Ready"** appears in the status bar, click **⏺ Start Recording**
3. Speak and/or join a meeting — transcription appears live
4. Click **⏹ Stop Recording** when done
5. Click **Summarize** to generate structured meeting notes
6. Click **Save** to write a `.md` file to `~/.local/share/eatmo/`

You can re-summarize at any time — if notes already exist, the app will ask for confirmation first.

## AI Providers

Switch providers in Settings (gear icon). Changes take effect immediately without restarting.

| Provider  | Model picker | Requires |
|-----------|-------------|----------|
| Anthropic | Yes (live fetch) | API key |
| OpenAI    | Yes (live fetch) | API key |
| Ollama    | Yes (local fetch) | Ollama running + model pulled |

For Ollama:
```bash
ollama serve
ollama pull llama3.1
```

## Whisper Model Selection

Set via environment variable. Larger = more accurate, slower:

| Model      | Size   | Speed (CPU) | Use case         |
|------------|--------|-------------|------------------|
| `tiny`     | 39 MB  | Very fast   | Quick tests      |
| `base`     | 74 MB  | Fast        | **Default**      |
| `small`    | 244 MB | Moderate    | Better accuracy  |
| `medium`   | 769 MB | Slow        | High accuracy    |
| `large-v3` | 1.5 GB | Very slow   | Best (needs GPU) |

```bash
GRANOLA_MODEL=small ./launch.sh
```

## How System Audio Capture Works

Fedora uses PipeWire as its audio server. PipeWire exposes a **monitor source** for each
output sink — a loopback of everything played through your speakers/headphones. The app
detects this monitor automatically via `pactl get-default-sink` and opens it as a second
input stream, mixing it with the mic before feeding to Whisper.

If the monitor isn't found, the app falls back to mic-only capture.

## Saved Files

Sessions are saved to `~/.local/share/eatmo/meeting_YYYY-MM-DD_HH-MM-SS.md`
with the meeting notes followed by the raw transcript.
