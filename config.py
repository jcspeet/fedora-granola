import os
from pathlib import Path

# --- Audio ---
SAMPLE_RATE = 16000        # Hz — Whisper expects 16kHz
CHANNELS = 1
CHUNK_SECONDS = 10         # seconds of audio per transcription chunk
MIC_GAIN = 0.7             # mic mix weight
MONITOR_GAIN = 0.7         # system audio mix weight

# --- Whisper ---
# Models: tiny, base, small, medium, large-v3
# Smaller = faster but less accurate. 'base' is a good default for CPU.
WHISPER_MODEL = os.environ.get("GRANOLA_MODEL", "base")
WHISPER_DEVICE = "auto"    # auto, cpu, cuda
WHISPER_COMPUTE_TYPE = "int8"  # int8 for CPU, float16 for CUDA

# --- LLM Provider ---
# Set GRANOLA_PROVIDER to "openai" to use OpenAI instead of Anthropic
LLM_PROVIDER = os.environ.get("GRANOLA_PROVIDER", "anthropic").lower()

# --- Anthropic / Claude ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"

# --- OpenAI ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("GRANOLA_OPENAI_MODEL", "gpt-4o")

SUMMARY_SYSTEM_PROMPT = """\
You are an expert meeting note-taker. Given a raw transcript, produce clean, \
structured meeting notes in Markdown with these sections:
- **Summary** (2-3 sentences)
- **Key Points** (bullet list)
- **Action Items** (bullet list with owner if mentioned, else "?")
- **Decisions Made** (bullet list, or "None" if absent)

Be concise. Do not pad or repeat. If the transcript is short or unclear, \
do your best with what's there.\
"""

# --- Storage ---
DATA_DIR = Path.home() / ".local" / "share" / "fedora-granola"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_DIR = Path.home() / ".config" / "fedora-granola"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Load keys from config file if not in env
_config_env = CONFIG_DIR / "config.env"
if _config_env.exists():
    for line in _config_env.read_text().splitlines():
        line = line.strip()
        if not ANTHROPIC_API_KEY and line.startswith("ANTHROPIC_API_KEY="):
            ANTHROPIC_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
        elif not OPENAI_API_KEY and line.startswith("OPENAI_API_KEY="):
            OPENAI_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("GRANOLA_PROVIDER="):
            LLM_PROVIDER = line.split("=", 1)[1].strip().strip('"').strip("'").lower()
