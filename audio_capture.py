"""
Audio capture: mic input + system audio output (via PipeWire/PulseAudio monitor).

Both streams are captured at 16kHz mono, mixed together, and delivered in
CHUNK_SECONDS-sized numpy arrays via the on_chunk callback.
"""

import subprocess
import threading
import time
import logging

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from math import gcd

from config import SAMPLE_RATE, CHANNELS, CHUNK_SECONDS, MIC_GAIN, MONITOR_GAIN

logger = logging.getLogger(__name__)


def find_monitor_source() -> str | int | None:
    """
    Find the system audio monitor source for the current default output device.
    Returns a sounddevice device index, or None if not found.
    Works with speakers, Bluetooth headphones, etc.
    """
    try:
        devices = sd.query_devices()

        # 1. Prefer Easy Effects sink monitor (post-EQ audio)
        sources_result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True, timeout=3
        )
        if "easyeffects_sink.monitor" in sources_result.stdout:
            for i, dev in enumerate(devices):
                if "easy effects sink" in dev["name"].lower() and dev["max_input_channels"] > 0:
                    logger.info("Found Easy Effects monitor (index %d)", i)
                    return i

        # 2. Get the default sink description from pactl
        sink_result = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True, text=True, timeout=3
        )
        default_sink = sink_result.stdout.strip()

        # Get the sink's human-readable description
        sinks_result = subprocess.run(
            ["pactl", "list", "sinks"],
            capture_output=True, text=True, timeout=3
        )
        sink_description = None
        current_name = None
        for line in sinks_result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Name:"):
                current_name = line.split(":", 1)[1].strip()
            elif line.startswith("Description:") and current_name == default_sink:
                sink_description = line.split(":", 1)[1].strip()
                break

        if sink_description:
            desc_lower = sink_description.lower()
            words = [w for w in desc_lower.split() if len(w) > 3]

            def is_monitor(dev):
                # PipeWire monitors have equal input and output channel counts
                return (dev["max_input_channels"] > 0
                        and dev["max_input_channels"] == dev["max_output_channels"])

            # Exact: description is substring of device name
            for i, dev in enumerate(devices):
                if is_monitor(dev) and desc_lower in dev["name"].lower():
                    logger.info("Found monitor by description: %s (index %d)", dev["name"], i)
                    return i

            # Fuzzy: at least 2 significant words match
            for i, dev in enumerate(devices):
                name = dev["name"].lower()
                if is_monitor(dev) and sum(w in name for w in words) >= 2:
                    logger.info("Found monitor by fuzzy match: %s (index %d)", dev["name"], i)
                    return i

    except Exception as e:
        logger.warning("Could not find monitor source: %s", e)
    return None


class AudioCapture:
    """
    Captures microphone and system audio simultaneously, mixes them, and
    delivers CHUNK_SECONDS-long numpy float32 arrays to on_chunk().
    """

    def __init__(self, on_chunk, on_level=None):
        """
        on_chunk: callable(np.ndarray) — receives mixed audio chunks at 16kHz mono.
        on_level: callable(float, float) — called with (mic_rms, monitor_rms) on each
                  audio callback, suitable for driving a level meter / waveform display.
        """
        self.on_chunk = on_chunk
        self.on_level = on_level
        self._lock = threading.Lock()
        self._mic_buf: list[np.ndarray] = []
        self._mon_buf: list[np.ndarray] = []
        self._running = False
        self._streams: list[sd.InputStream] = []
        self._chunk_thread: threading.Thread | None = None
        self.monitor_available = False
        self._monitor_native_rate: int = SAMPLE_RATE
        self._last_mic_rms: float = 0.0
        self._last_mon_rms: float = 0.0

    # ------------------------------------------------------------------
    # Callbacks (called from sounddevice's internal thread)
    # ------------------------------------------------------------------

    def _mic_callback(self, indata, frames, time_info, status):
        if status:
            logger.debug("Mic status: %s", status)
        audio = indata[:, 0].copy()
        self._last_mic_rms = float(np.sqrt(np.mean(audio ** 2)))
        if self.on_level:
            self.on_level(self._last_mic_rms, self._last_mon_rms)
        with self._lock:
            self._mic_buf.append(audio)

    def _monitor_callback(self, indata, frames, time_info, status):
        if status:
            logger.debug("Monitor status: %s", status)
        audio = indata[:, 0].copy()
        self._last_mon_rms = float(np.sqrt(np.mean(audio ** 2)))
        if self.on_level:
            self.on_level(self._last_mic_rms, self._last_mon_rms)
        if self._monitor_native_rate != SAMPLE_RATE:
            g = gcd(self._monitor_native_rate, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // g, self._monitor_native_rate // g).astype(np.float32)
        with self._lock:
            self._mon_buf.append(audio)

    # ------------------------------------------------------------------
    # Chunk delivery thread
    # ------------------------------------------------------------------

    def _chunk_loop(self):
        chunk_samples = int(SAMPLE_RATE * CHUNK_SECONDS)
        mic_acc = np.array([], dtype=np.float32)
        mon_acc = np.array([], dtype=np.float32)

        while self._running:
            time.sleep(0.2)  # drain interval

            with self._lock:
                if self._mic_buf:
                    mic_acc = np.concatenate([mic_acc] + self._mic_buf)
                    self._mic_buf.clear()
                if self._mon_buf:
                    mon_acc = np.concatenate([mon_acc] + self._mon_buf)
                    self._mon_buf.clear()

            while len(mic_acc) >= chunk_samples:
                mic_chunk = mic_acc[:chunk_samples]
                mic_acc = mic_acc[chunk_samples:]

                if len(mon_acc) >= chunk_samples:
                    mon_chunk = mon_acc[:chunk_samples]
                    mon_acc = mon_acc[chunk_samples:]
                else:
                    # Pad monitor with zeros if behind
                    mon_chunk = np.zeros(chunk_samples, dtype=np.float32)
                    if len(mon_acc) > 0:
                        mon_chunk[:len(mon_acc)] = mon_acc
                        mon_acc = np.array([], dtype=np.float32)

                mixed = mic_chunk * MIC_GAIN + mon_chunk * MONITOR_GAIN
                mixed = np.clip(mixed, -1.0, 1.0)
                try:
                    self.on_chunk(mixed)
                except Exception as e:
                    logger.error("on_chunk error: %s", e)

        # Flush remaining audio (may be less than a full chunk)
        if len(mic_acc) > 512:  # skip tiny leftover
            if len(mon_acc) > 0:
                n = max(len(mic_acc), len(mon_acc))
                m = np.zeros(n, dtype=np.float32)
                m[:len(mic_acc)] += mic_acc * MIC_GAIN
                m[:len(mon_acc)] += mon_acc * MONITOR_GAIN
                mixed = np.clip(m, -1.0, 1.0)
            else:
                mixed = np.clip(mic_acc * MIC_GAIN, -1.0, 1.0)
            try:
                self.on_chunk(mixed)
            except Exception as e:
                logger.error("Flush on_chunk error: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self._running:
            return
        self._running = True
        self._mic_buf.clear()
        self._mon_buf.clear()

        # Microphone stream
        try:
            mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self._mic_callback,
                blocksize=int(SAMPLE_RATE * 0.5),
            )
            mic_stream.start()
            self._streams.append(mic_stream)
            logger.info("Mic stream started")
        except Exception as e:
            logger.error("Could not open mic stream: %s", e)
            self._running = False
            raise

        # System audio monitor stream
        monitor_source = find_monitor_source()
        if monitor_source is not None:
            try:
                native_rate = int(sd.query_devices(monitor_source)["default_samplerate"])
                self._monitor_native_rate = native_rate
                mon_stream = sd.InputStream(
                    device=monitor_source,
                    samplerate=native_rate,
                    channels=CHANNELS,
                    dtype="float32",
                    callback=self._monitor_callback,
                    blocksize=int(native_rate * 0.5),
                )
                mon_stream.start()
                self._streams.append(mon_stream)
                self.monitor_available = True
                logger.info("Monitor stream started: %s at %dHz", monitor_source, native_rate)
            except Exception as e:
                logger.warning("Could not open monitor stream: %s — only mic will be captured", e)
        else:
            logger.warning("No monitor source found — only mic will be captured")

        self._chunk_thread = threading.Thread(target=self._chunk_loop, daemon=True)
        self._chunk_thread.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self._streams.clear()
        if self._chunk_thread:
            self._chunk_thread.join(timeout=5)
            self._chunk_thread = None

    def list_devices(self) -> str:
        """Return a human-readable list of audio devices for debugging."""
        lines = ["Available audio devices:"]
        for i, dev in enumerate(sd.query_devices()):
            marker = " [IN]" if dev["max_input_channels"] > 0 else "     "
            lines.append(f"  {i:2d}{marker} {dev['name']}")
        return "\n".join(lines)
