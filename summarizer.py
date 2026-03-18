"""
Meeting summarization via Claude API.
"""

import logging
import threading

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, SUMMARY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Summarizer:
    """
    Sends a transcript to Claude and streams the summary back via on_token.

    on_token:    callable(str)  — called for each streamed token.
    on_complete: callable(str)  — called with the full summary when done.
    on_error:    callable(str)  — called on errors.
    """

    def __init__(self, on_token=None, on_complete=None, on_error=None):
        self.on_token = on_token
        self.on_complete = on_complete
        self.on_error = on_error
        self._thread: threading.Thread | None = None

    def _run(self, transcript: str):
        if not ANTHROPIC_API_KEY:
            msg = (
                "ANTHROPIC_API_KEY is not set.\n"
                "Set it via: export ANTHROPIC_API_KEY=your_key\n"
                "Or add it to ~/.config/fedora-granola/config.env"
            )
            if self.on_error:
                self.on_error(msg)
            return

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        full_text = []

        try:
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=SUMMARY_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Please summarize this meeting transcript:\n\n{transcript}",
                    }
                ],
            ) as stream:
                for text in stream.text_stream:
                    full_text.append(text)
                    if self.on_token:
                        self.on_token(text)

            summary = "".join(full_text)
            if self.on_complete:
                self.on_complete(summary)

        except anthropic.AuthenticationError:
            if self.on_error:
                self.on_error("Invalid API key. Check your ANTHROPIC_API_KEY.")
        except Exception as e:
            logger.error("Summarization error: %s", e)
            if self.on_error:
                self.on_error(str(e))

    def summarize(self, transcript: str):
        """Start summarization in a background thread."""
        if not transcript.strip():
            if self.on_error:
                self.on_error("Transcript is empty — nothing to summarize.")
            return
        self._thread = threading.Thread(target=self._run, args=(transcript,), daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
