"""
Real-time transcription using faster-whisper.

Runs inference in a background thread so the UI stays responsive.
"""

import logging
import queue
import threading

import numpy as np

from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, SAMPLE_RATE

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Receives audio chunks via queue, transcribes with faster-whisper,
    and delivers segments via on_segment callback.

    on_segment: callable(str) — receives transcribed text segments.
    on_ready:   callable()    — called when the model is loaded and ready.
    on_error:   callable(str) — called on fatal errors.
    """

    def __init__(self, on_segment, on_ready=None, on_error=None):
        self.on_segment = on_segment
        self.on_ready = on_ready
        self.on_error = on_error

        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._model = None
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self):
        from faster_whisper import WhisperModel
        logger.info("Loading Whisper model '%s'…", WHISPER_MODEL)
        device = WHISPER_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute_type = WHISPER_COMPUTE_TYPE
        if device == "cuda":
            compute_type = "float16"

        self._model = WhisperModel(
            WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper model loaded (device=%s, compute=%s)", device, compute_type)

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """Transcribe a numpy float32 array at SAMPLE_RATE Hz."""
        if self._model is None:
            return ""

        # faster-whisper wants float32 numpy at 16kHz
        audio = audio.astype(np.float32)

        segments, info = self._model.transcribe(
            audio,
            beam_size=3,
            language=None,        # auto-detect
            vad_filter=True,      # skip silence
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()

    def _worker(self):
        try:
            self._load_model()
        except Exception as e:
            logger.error("Model load failed: %s", e)
            if self.on_error:
                self.on_error(str(e))
            return

        if self.on_ready:
            self.on_ready()

        while self._running:
            try:
                chunk = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if chunk is None:  # stop sentinel
                break

            try:
                text = self._transcribe_chunk(chunk)
                if text:
                    self.on_segment(text)
            except Exception as e:
                logger.error("Transcription error: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the background transcription thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def push(self, audio: np.ndarray):
        """Enqueue an audio chunk for transcription."""
        self._queue.put(audio)

    def stop(self):
        """Stop transcription. Blocks until the thread exits."""
        self._running = False
        self._queue.put(None)  # unblock the worker
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()
