# Fedora Granola

A local, privacy-first meeting transcription and notes app for Fedora Linux.
Works like [Granola](https://granola.ai/) — no meeting bots, no cloud uploads for audio.

## Features

- Captures **microphone** + **system audio output** simultaneously (via PipeWire monitor)
- Transcribes in real-time using **Whisper** (runs locally, no data leaves your machine)
- Generates structured **meeting notes** via Claude API (the only network call)
- Native **GTK4 / libadwaita** UI that fits right in on GNOME/Fedora
- **Save** sessions as Markdown files

## Requirements

- Fedora 38+ (uses PipeWire as audio server)
- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) for summarization

## Installation

```bash
./install.sh
```

This installs system packages via `dnf` and Python dependencies into a virtualenv.

## Configuration

Set your API key in one of two ways:

**Option A — environment variable:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Option B — config file** (persists across sessions):
```bash
mkdir -p ~/.config/fedora-granola
echo 'ANTHROPIC_API_KEY=sk-ant-...' > ~/.config/fedora-granola/config.env
```

## Running

```bash
venv/bin/python main.py
```

## Usage

1. Launch the app — it loads the Whisper model in the background (~5–15s)
2. Once **"Ready"** appears in the status bar, click **⏺ Start Recording**
3. Speak and/or join a meeting — transcription appears live in the top pane
4. Click **⏹ Stop Recording** when done
5. Click **Summarize** to generate meeting notes with Claude
6. Click **Save** to write a `.md` file to `~/.local/share/fedora-granola/`

## Whisper Model Selection

Set via environment variable. Larger = more accurate, slower:

| Model      | Size   | Speed (CPU) | Use case              |
|------------|--------|-------------|-----------------------|
| `tiny`     | 39 MB  | Very fast   | Quick tests           |
| `base`     | 74 MB  | Fast        | **Default**           |
| `small`    | 244 MB | Moderate    | Better accuracy       |
| `medium`   | 769 MB | Slow        | High accuracy         |
| `large-v3` | 1.5 GB | Very slow   | Best (needs GPU)      |

```bash
GRANOLA_MODEL=small venv/bin/python main.py
```

## How System Audio Capture Works

Fedora uses PipeWire as its audio server. PipeWire (like PulseAudio before it)
exposes a **monitor source** for each output sink — a loopback of everything
being played through your speakers/headphones. The app detects this monitor
automatically via `pactl get-default-sink` and opens it as a second input stream.

Both streams (mic + monitor) are mixed together before being fed to Whisper.
If the monitor isn't found, the app falls back to mic-only capture.

## Saved Files

Sessions are saved to `~/.local/share/fedora-granola/meeting_YYYY-MM-DD_HH-MM-SS.md`
with the meeting notes followed by the raw transcript.
