"""
Chat with meeting context via Claude or OpenAI.
"""

import logging
import threading

from config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)


class Chatter:
    """
    Streams a chat response given a message history and system prompt.

    on_token:    callable(str)  — called for each streamed token.
    on_complete: callable(str)  — called with the full response when done.
    on_error:    callable(str)  — called on errors.
    """

    def __init__(self, on_token=None, on_complete=None, on_error=None):
        self.on_token = on_token
        self.on_complete = on_complete
        self.on_error = on_error
        self._thread: threading.Thread | None = None

    def chat(self, messages: list[dict], system_prompt: str):
        """Start a chat turn in a background thread.

        messages: full conversation history as list of {"role": ..., "content": ...}
        system_prompt: injected as system context
        """
        self._thread = threading.Thread(
            target=self._run,
            args=(messages, system_prompt),
            daemon=True,
        )
        self._thread.start()

    def _run(self, messages: list[dict], system_prompt: str):
        if LLM_PROVIDER == "openai":
            self._run_openai(messages, system_prompt)
        else:
            self._run_anthropic(messages, system_prompt)

    def _run_anthropic(self, messages: list[dict], system_prompt: str):
        if not ANTHROPIC_API_KEY:
            if self.on_error:
                self.on_error("ANTHROPIC_API_KEY is not set.")
            return

        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        full_text = []

        try:
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    full_text.append(text)
                    if self.on_token:
                        self.on_token(text)

            if self.on_complete:
                self.on_complete("".join(full_text))

        except anthropic.AuthenticationError:
            if self.on_error:
                self.on_error("Invalid API key. Check your ANTHROPIC_API_KEY.")
        except Exception as e:
            logger.error("Chat error: %s", e)
            if self.on_error:
                self.on_error(str(e))

    def _run_openai(self, messages: list[dict], system_prompt: str):
        if not OPENAI_API_KEY:
            if self.on_error:
                self.on_error("OPENAI_API_KEY is not set.")
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
                max_tokens=2048,
                stream=True,
                messages=[{"role": "system", "content": system_prompt}] + messages,
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
                self.on_error("Invalid API key. Check your OPENAI_API_KEY.")
        except Exception as e:
            logger.error("Chat error: %s", e)
            if self.on_error:
                self.on_error(str(e))
