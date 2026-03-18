"""
Meeting summarization via Claude or OpenAI API.
"""

import logging
import threading

from config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SUMMARY_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class Summarizer:
    """
    Sends a transcript to Claude or OpenAI and streams the summary back via on_token.

    on_token:    callable(str)  — called for each streamed token.
    on_complete: callable(str)  — called with the full summary when done.
    on_error:    callable(str)  — called on errors.
    """

    def __init__(self, on_token=None, on_complete=None, on_error=None):
        self.on_token = on_token
        self.on_complete = on_complete
        self.on_error = on_error
        self._thread: threading.Thread | None = None

    def _run_anthropic(self, transcript: str):
        if not ANTHROPIC_API_KEY:
            msg = (
                "ANTHROPIC_API_KEY is not set.\n"
                "Set it via: export ANTHROPIC_API_KEY=your_key\n"
                "Or add it to ~/.config/fedora-granola/config.env"
            )
            if self.on_error:
                self.on_error(msg)
            return

        import anthropic

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

    def _run_openai(self, transcript: str):
        if not OPENAI_API_KEY:
            msg = (
                "OPENAI_API_KEY is not set.\n"
                "Set it via: export OPENAI_API_KEY=your_key\n"
                "Or add it to ~/.config/fedora-granola/config.env"
            )
            if self.on_error:
                self.on_error(msg)
            return

        try:
            from openai import OpenAI, AuthenticationError
        except ImportError:
            if self.on_error:
                self.on_error("openai package not installed. Run: pip install openai")
            return

        client = OpenAI(api_key=OPENAI_API_KEY)
        full_text = []

        try:
            with client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=1024,
                stream=True,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Please summarize this meeting transcript:\n\n{transcript}",
                    },
                ],
            ) as stream:
                for chunk in stream:
                    text = chunk.choices[0].delta.content or ""
                    if text:
                        full_text.append(text)
                        if self.on_token:
                            self.on_token(text)

            summary = "".join(full_text)
            if self.on_complete:
                self.on_complete(summary)

        except AuthenticationError:
            if self.on_error:
                self.on_error("Invalid API key. Check your OPENAI_API_KEY.")
        except Exception as e:
            logger.error("Summarization error: %s", e)
            if self.on_error:
                self.on_error(str(e))

    def _run(self, transcript: str):
        if LLM_PROVIDER == "openai":
            self._run_openai(transcript)
        else:
            self._run_anthropic(transcript)

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
