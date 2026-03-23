"""
Meeting summarization via Claude, OpenAI, or local Ollama.
"""

import logging
import threading

import config

logger = logging.getLogger(__name__)


class Summarizer:
    """
    Sends a transcript to the configured LLM and streams the summary back via on_token.

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
        if not config.ANTHROPIC_API_KEY:
            if self.on_error:
                self.on_error(
                    "ANTHROPIC_API_KEY is not set.\n"
                    "Open Settings and enter your Anthropic API key."
                )
            return

        import anthropic

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        full_text = []

        try:
            with client.messages.stream(
                model=config.CLAUDE_MODEL,
                max_tokens=1024,
                system=config.SUMMARY_SYSTEM_PROMPT,
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

            if self.on_complete:
                self.on_complete("".join(full_text))

        except anthropic.AuthenticationError:
            if self.on_error:
                self.on_error("Invalid Anthropic API key. Check Settings.")
        except Exception as e:
            logger.error("Summarization error: %s", e)
            if self.on_error:
                self.on_error(str(e))

    def _run_openai(self, transcript: str):
        if not config.OPENAI_API_KEY:
            if self.on_error:
                self.on_error(
                    "OPENAI_API_KEY is not set.\n"
                    "Open Settings and enter your OpenAI API key."
                )
            return

        try:
            from openai import OpenAI, AuthenticationError
        except ImportError:
            if self.on_error:
                self.on_error("openai package not installed. Run: pip install openai")
            return

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        full_text = []

        try:
            with client.chat.completions.create(
                model=config.OPENAI_MODEL,
                max_tokens=1024,
                stream=True,
                messages=[
                    {"role": "system", "content": config.SUMMARY_SYSTEM_PROMPT},
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

            if self.on_complete:
                self.on_complete("".join(full_text))

        except AuthenticationError:
            if self.on_error:
                self.on_error("Invalid OpenAI API key. Check Settings.")
        except Exception as e:
            logger.error("Summarization error: %s", e)
            if self.on_error:
                self.on_error(str(e))

    def _run_ollama(self, transcript: str):
        try:
            from openai import OpenAI
        except ImportError:
            if self.on_error:
                self.on_error("openai package not installed. Run: pip install openai")
            return

        client = OpenAI(api_key="ollama", base_url=config.OLLAMA_BASE_URL)
        full_text = []

        try:
            with client.chat.completions.create(
                model=config.OLLAMA_MODEL,
                max_tokens=1024,
                stream=True,
                messages=[
                    {"role": "system", "content": config.SUMMARY_SYSTEM_PROMPT},
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

            if self.on_complete:
                self.on_complete("".join(full_text))

        except Exception as e:
            logger.error("Ollama summarization error: %s", e)
            if self.on_error:
                self.on_error(
                    f"Ollama error: {e}\n\n"
                    "Make sure Ollama is running (ollama serve) and the model is pulled."
                )

    def _run(self, transcript: str):
        provider = config.LLM_PROVIDER
        if provider == "openai":
            self._run_openai(transcript)
        elif provider == "ollama":
            self._run_ollama(transcript)
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
