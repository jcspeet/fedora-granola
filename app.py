"""
Fedora Granola — GTK4 / libadwaita UI
"""

import collections
import datetime
import logging
import os
import threading
from pathlib import Path

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango

from audio_capture import AudioCapture
from chat import Chatter
import config
from config import DATA_DIR, CHAT_SYSTEM_PROMPT
from summarizer import Summarizer
from transcriber import Transcriber

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_meeting_file(path: Path) -> tuple[str, datetime.datetime, str, str]:
    """Parse a saved meeting .md file.
    Returns (title, datetime, transcript, summary).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return path.stem, datetime.datetime.fromtimestamp(path.stat().st_mtime), "", ""

    title = path.stem
    lines = text.split("\n")
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()

    transcript = ""
    summary = ""
    if "## Raw Transcript" in text:
        parts = text.split("## Raw Transcript", 1)
        transcript = parts[1].strip()
        first_part = parts[0]
        if "## Notes" in first_part:
            summary = first_part.split("## Notes", 1)[1].strip()

    try:
        stem = path.stem  # meeting_2025-03-18_14-30-00
        ts_part = stem[len("meeting_"):]
        dt = datetime.datetime.strptime(ts_part, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        dt = datetime.datetime.fromtimestamp(path.stat().st_mtime)

    return title, dt, transcript, summary


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

_PROVIDERS = ["Anthropic", "OpenAI", "Local (Ollama)"]
_PROVIDER_KEYS = ["anthropic", "openai", "ollama"]

# OpenAI model IDs that are not chat models
_OPENAI_EXCLUDED = (
    "text-embedding", "dall-e", "whisper", "tts",
    "babbage", "davinci", "ada", "curie",
)


def _fetch_models_bg(provider: str, key: str) -> list[str]:
    """Fetch available model IDs. Runs in a background thread; returns [] on error."""
    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            return [m.id for m in client.models.list().data]
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=key)
            all_models = client.models.list().data
            return sorted(
                m.id for m in all_models
                if not any(m.id.startswith(p) for p in _OPENAI_EXCLUDED)
            )
        else:  # ollama
            from openai import OpenAI
            client = OpenAI(api_key="ollama", base_url=config.OLLAMA_BASE_URL)
            return [m.id for m in client.models.list().data]
    except Exception as e:
        logger.warning("Could not fetch models for %s: %s", provider, e)
        return []


class SettingsDialog(Adw.Window):
    """Modal settings window for provider selection, credentials, and model choice."""

    def __init__(self, parent):
        super().__init__()
        self.set_transient_for(parent)
        self.set_modal(True)
        self.set_default_size(420, -1)
        self.set_title("Settings")
        self.set_resizable(False)

        self._fetched_models: list[str] = []

        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        header = Adw.HeaderBar()
        header.set_show_end_title_buttons(False)
        cancel_btn = Gtk.Button(label="Cancel")
        cancel_btn.connect("clicked", lambda _: self.close())
        header.pack_start(cancel_btn)
        self._save_btn = Gtk.Button(label="Save")
        self._save_btn.add_css_class("suggested-action")
        self._save_btn.connect("clicked", self._on_save)
        header.pack_end(self._save_btn)
        toolbar_view.add_top_bar(header)

        page = Adw.PreferencesPage()
        toolbar_view.set_content(page)

        group = Adw.PreferencesGroup(title="AI Provider")
        page.add(group)

        # Provider dropdown
        self._provider_row = Adw.ComboRow(title="Provider")
        self._provider_row.set_model(Gtk.StringList.new(_PROVIDERS))
        current = config.LLM_PROVIDER
        idx = _PROVIDER_KEYS.index(current) if current in _PROVIDER_KEYS else 0
        self._provider_row.set_selected(idx)
        self._provider_row.connect("notify::selected", self._on_provider_changed)
        group.add(self._provider_row)

        # API key row (masked) — hidden for Ollama
        self._key_row = Adw.PasswordEntryRow()
        group.add(self._key_row)

        # Model selector
        self._model_combo = Adw.ComboRow(title="Model")
        self._model_spinner = Gtk.Spinner()
        self._model_spinner.set_margin_top(4)
        self._model_spinner.set_margin_bottom(4)
        self._model_combo.add_suffix(self._model_spinner)
        refresh_btn = Gtk.Button()
        refresh_btn.set_icon_name("view-refresh-symbolic")
        refresh_btn.add_css_class("flat")
        refresh_btn.set_tooltip_text("Fetch available models")
        refresh_btn.connect("clicked", self._on_refresh_clicked)
        self._model_combo.add_suffix(refresh_btn)
        group.add(self._model_combo)

        # Hint label
        self._hint_label = Gtk.Label()
        self._hint_label.set_margin_start(12)
        self._hint_label.set_margin_end(12)
        self._hint_label.set_margin_top(4)
        self._hint_label.set_margin_bottom(8)
        self._hint_label.set_xalign(0)
        self._hint_label.add_css_class("caption")
        self._hint_label.add_css_class("dim-label")
        self._hint_label.set_wrap(True)
        group.add(self._hint_label)

        self._apply_provider(idx, fetch=True)

    # ------------------------------------------------------------------

    def _current_model(self, provider: str) -> str:
        if provider == "anthropic":
            return config.CLAUDE_MODEL
        elif provider == "openai":
            return config.OPENAI_MODEL
        return config.OLLAMA_MODEL

    def _apply_provider(self, provider_idx: int, fetch: bool = False):
        provider = _PROVIDER_KEYS[provider_idx]
        if provider == "anthropic":
            self._key_row.set_title("Anthropic API Key")
            self._key_row.set_text(config.ANTHROPIC_API_KEY)
            self._key_row.set_visible(True)
            self._hint_label.set_label("Get a key at console.anthropic.com")
            if fetch and config.ANTHROPIC_API_KEY:
                self._fetch_models(provider, config.ANTHROPIC_API_KEY)
            else:
                self._set_model_list([config.CLAUDE_MODEL])
        elif provider == "openai":
            self._key_row.set_title("OpenAI API Key")
            self._key_row.set_text(config.OPENAI_API_KEY)
            self._key_row.set_visible(True)
            self._hint_label.set_label("Get a key at platform.openai.com")
            if fetch and config.OPENAI_API_KEY:
                self._fetch_models(provider, config.OPENAI_API_KEY)
            else:
                self._set_model_list([config.OPENAI_MODEL])
        else:  # ollama
            self._key_row.set_visible(False)
            self._hint_label.set_label(
                "Install Ollama from ollama.com, then run: ollama pull llama3.1"
            )
            if fetch:
                self._fetch_models(provider, "ollama")
            else:
                self._set_model_list([config.OLLAMA_MODEL])

    def _fetch_models(self, provider: str, key: str):
        self._model_spinner.start()
        self._save_btn.set_sensitive(False)
        self._model_combo.set_sensitive(False)

        def _bg():
            models = _fetch_models_bg(provider, key)
            GLib.idle_add(self._ui_models_fetched, provider, models)

        threading.Thread(target=_bg, daemon=True).start()

    def _ui_models_fetched(self, provider: str, models: list[str]):
        self._model_spinner.stop()
        self._save_btn.set_sensitive(True)
        self._model_combo.set_sensitive(True)
        if not models:
            models = [self._current_model(provider)]
        self._set_model_list(models, self._current_model(provider))
        return False

    def _set_model_list(self, models: list[str], current: str | None = None):
        self._fetched_models = models
        self._model_combo.set_model(Gtk.StringList.new(models))
        target = current or (models[0] if models else "")
        idx = models.index(target) if target in models else 0
        self._model_combo.set_selected(idx)

    def _on_provider_changed(self, row, _param):
        self._apply_provider(row.get_selected(), fetch=False)

    def _on_refresh_clicked(self, _btn):
        provider_idx = self._provider_row.get_selected()
        provider = _PROVIDER_KEYS[provider_idx]
        key = self._key_row.get_text().strip() if provider != "ollama" else "ollama"
        if key:
            self._fetch_models(provider, key)

    def _on_save(self, _btn):
        provider_idx = self._provider_row.get_selected()
        provider = _PROVIDER_KEYS[provider_idx]

        config.save_setting("GRANOLA_PROVIDER", provider)

        if provider == "anthropic":
            config.save_setting("ANTHROPIC_API_KEY", self._key_row.get_text().strip())
            model_idx = self._model_combo.get_selected()
            if self._fetched_models and model_idx < len(self._fetched_models):
                config.save_setting("EATMO_ANTHROPIC_MODEL", self._fetched_models[model_idx])
        elif provider == "openai":
            config.save_setting("OPENAI_API_KEY", self._key_row.get_text().strip())
            model_idx = self._model_combo.get_selected()
            if self._fetched_models and model_idx < len(self._fetched_models):
                config.save_setting("EATMO_OPENAI_MODEL", self._fetched_models[model_idx])
        else:
            model_idx = self._model_combo.get_selected()
            if self._fetched_models and model_idx < len(self._fetched_models):
                config.save_setting("EATMO_OLLAMA_MODEL", self._fetched_models[model_idx])

        self.close()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class GranolaWindow(Adw.ApplicationWindow):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_title("Fedora Granola")
        self.set_default_size(1100, 700)

        self._transcript_parts: list[str] = []
        self._recording = False
        self._transcriber_ready = False
        self._session_start: datetime.datetime | None = None
        self._viewing_saved = False
        self._current_save_path: Path | None = None
        self._meeting_title: str | None = None
        self._summary_header_buf: list[str] = []
        self._title_found = False

        # Waveform data: rolling history of (mic_rms, mon_rms) samples
        self._waveform_history: collections.deque = collections.deque(maxlen=200)
        self._waveform_mic_rms: float = 0.0
        self._waveform_mon_rms: float = 0.0

        # Chat state
        self._chat_history: list[dict] = []
        self._chat_mode: str = "meeting"  # "meeting" or "global"
        self._chat_streaming: bool = False

        self._build_ui()
        self._setup_backend()
        self._refresh_meeting_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        # Header bar
        header = Adw.HeaderBar()
        toolbar_view.add_top_bar(header)

        self._record_btn = Gtk.Button()
        self._record_btn.set_css_classes(["suggested-action", "pill"])
        self._record_btn.set_sensitive(False)
        self._record_btn.connect("clicked", self._on_record_clicked)
        header.pack_start(self._record_btn)
        self._update_record_button()

        # Notes / Chat toggle
        seg_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        seg_box.add_css_class("linked")
        self._notes_btn = Gtk.ToggleButton(label="Notes")
        self._notes_btn.set_active(True)
        self._chat_btn = Gtk.ToggleButton(label="Chat")
        self._chat_btn.set_group(self._notes_btn)
        seg_box.append(self._notes_btn)
        seg_box.append(self._chat_btn)
        header.set_title_widget(seg_box)
        self._chat_btn.connect("notify::active", self._on_view_toggle)

        settings_btn = Gtk.Button()
        settings_btn.set_icon_name("preferences-system-symbolic")
        settings_btn.set_tooltip_text("Settings")
        settings_btn.connect("clicked", self._on_settings_clicked)
        header.pack_end(settings_btn)

        self._summarize_btn = Gtk.Button(label="Summarize")
        self._summarize_btn.set_css_classes(["pill"])
        self._summarize_btn.set_sensitive(False)
        self._summarize_btn.connect("clicked", self._on_summarize_clicked)
        header.pack_end(self._summarize_btn)

        self._save_btn = Gtk.Button(label="Save")
        self._save_btn.set_css_classes(["pill"])
        self._save_btn.set_sensitive(False)
        self._save_btn.connect("clicked", self._on_save_clicked)
        header.pack_end(self._save_btn)

        # Waveform bar
        self._waveform = Gtk.DrawingArea()
        self._waveform.set_size_request(-1, 48)
        self._waveform.set_hexpand(True)
        self._waveform.set_draw_func(self._draw_waveform)
        toolbar_view.add_top_bar(self._waveform)

        # Outer horizontal paned: sidebar | content
        outer_paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        outer_paned.set_position(230)
        outer_paned.set_shrink_start_child(False)
        outer_paned.set_shrink_end_child(False)
        outer_paned.set_resize_start_child(False)
        outer_paned.set_resize_end_child(True)
        toolbar_view.set_content(outer_paned)

        outer_paned.set_start_child(self._build_sidebar())

        self._content_stack = Gtk.Stack()
        self._content_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self._content_stack.set_transition_duration(150)
        self._content_stack.add_named(self._build_content(), "notes")
        self._content_stack.add_named(self._build_chat_panel(), "chat")
        outer_paned.set_end_child(self._content_stack)

        # Status bar
        self._status_bar = Gtk.Label(label="Loading Whisper model…")
        self._status_bar.set_css_classes(["caption", "dim-label"])
        self._status_bar.set_margin_start(12)
        self._status_bar.set_margin_end(12)
        self._status_bar.set_margin_top(4)
        self._status_bar.set_margin_bottom(6)
        self._status_bar.set_xalign(0)
        toolbar_view.add_bottom_bar(self._status_bar)

        css = Gtk.CssProvider()
        css.load_from_string("""
            textview { font-size: 13px; }
            textview.monospace { font-family: monospace; font-size: 12px; }
        """)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _build_sidebar(self) -> Gtk.Widget:
        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        sidebar_box.add_css_class("background")

        new_btn = Gtk.Button(label="New Recording")
        new_btn.set_css_classes(["suggested-action"])
        new_btn.set_margin_start(8)
        new_btn.set_margin_end(8)
        new_btn.set_margin_top(8)
        new_btn.set_margin_bottom(8)
        new_btn.connect("clicked", self._on_new_recording_clicked)
        sidebar_box.append(new_btn)

        global_chat_btn = Gtk.Button(label="Chat: All Meetings")
        global_chat_btn.set_css_classes(["flat"])
        global_chat_btn.set_margin_start(8)
        global_chat_btn.set_margin_end(8)
        global_chat_btn.set_margin_bottom(4)
        global_chat_btn.connect("clicked", self._on_global_chat_clicked)
        sidebar_box.append(global_chat_btn)

        sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        sidebar_box.append(sep)

        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._meeting_list = Gtk.ListBox()
        self._meeting_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._meeting_list.add_css_class("navigation-sidebar")
        self._meeting_list.connect("row-activated", self._on_meeting_row_activated)
        scroll.set_child(self._meeting_list)
        sidebar_box.append(scroll)

        return sidebar_box

    def _build_content(self) -> Gtk.Widget:
        paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        paned.set_position(320)

        # Transcript
        transcript_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        transcript_label = Gtk.Label(label="Transcript", xalign=0)
        transcript_label.set_css_classes(["heading"])
        transcript_label.set_margin_start(12)
        transcript_label.set_margin_top(8)
        transcript_label.set_margin_bottom(4)
        transcript_box.append(transcript_label)

        transcript_scroll = Gtk.ScrolledWindow()
        transcript_scroll.set_vexpand(True)
        transcript_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._transcript_view = Gtk.TextView()
        self._transcript_view.set_editable(True)
        self._transcript_view.set_wrap_mode(Gtk.WrapMode.WORD)
        self._transcript_view.set_margin_start(12)
        self._transcript_view.set_margin_end(12)
        self._transcript_view.set_margin_top(8)
        self._transcript_view.set_margin_bottom(8)
        self._transcript_view.set_css_classes(["monospace"])
        self._transcript_buf = self._transcript_view.get_buffer()
        transcript_scroll.set_child(self._transcript_view)
        transcript_box.append(transcript_scroll)

        paned.set_start_child(transcript_box)

        # Summary
        summary_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        summary_label = Gtk.Label(label="Meeting Notes", xalign=0)
        summary_label.set_css_classes(["heading"])
        summary_label.set_margin_start(12)
        summary_label.set_margin_top(8)
        summary_label.set_margin_bottom(4)
        summary_box.append(summary_label)

        summary_scroll = Gtk.ScrolledWindow()
        summary_scroll.set_vexpand(True)
        summary_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._summary_view = Gtk.TextView()
        self._summary_view.set_editable(True)
        self._summary_view.set_wrap_mode(Gtk.WrapMode.WORD)
        self._summary_view.set_margin_start(12)
        self._summary_view.set_margin_end(12)
        self._summary_view.set_margin_top(8)
        self._summary_view.set_margin_bottom(8)
        self._summary_buf = self._summary_view.get_buffer()
        summary_scroll.set_child(self._summary_view)
        summary_box.append(summary_scroll)

        paned.set_end_child(summary_box)

        return paned

    # ------------------------------------------------------------------
    # Waveform
    # ------------------------------------------------------------------

    def _draw_waveform(self, area, cr, width, height):
        # Background
        cr.set_source_rgb(0.08, 0.08, 0.08)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        if not self._waveform_history:
            return

        samples = list(self._waveform_history)
        n = len(samples)
        bar_w = max(1.0, width / self._waveform_history.maxlen)
        half = height / 2.0

        for i, (mic, mon) in enumerate(samples):
            x = (i / self._waveform_history.maxlen) * width

            # Mic — green, drawn upward from centre
            mic_h = min(mic * height * 4, half)
            cr.set_source_rgba(0.2, 0.85, 0.4, 0.9)
            cr.rectangle(x, half - mic_h, bar_w - 0.5, mic_h)
            cr.fill()

            # System audio — blue, drawn downward from centre
            mon_h = min(mon * height * 4, half)
            cr.set_source_rgba(0.3, 0.6, 1.0, 0.9)
            cr.rectangle(x, half, bar_w - 0.5, mon_h)
            cr.fill()

        # Centre line
        cr.set_source_rgba(1, 1, 1, 0.15)
        cr.set_line_width(1)
        cr.move_to(0, half)
        cr.line_to(width, half)
        cr.stroke()

    def _on_audio_level(self, mic_rms: float, mon_rms: float):
        self._waveform_mic_rms = mic_rms
        self._waveform_mon_rms = mon_rms
        GLib.idle_add(self._ui_push_waveform_sample, mic_rms, mon_rms)

    def _ui_push_waveform_sample(self, mic_rms: float, mon_rms: float):
        self._waveform_history.append((mic_rms, mon_rms))
        self._waveform.queue_draw()
        return False

    # ------------------------------------------------------------------
    # Backend setup
    # ------------------------------------------------------------------

    def _setup_backend(self):
        self._transcriber = Transcriber(
            on_segment=self._on_segment,
            on_ready=self._on_transcriber_ready,
            on_error=self._on_transcriber_error,
        )
        self._transcriber.start()
        self._capture = AudioCapture(
            on_chunk=self._on_audio_chunk,
            on_level=self._on_audio_level,
        )

    # ------------------------------------------------------------------
    # Meeting list
    # ------------------------------------------------------------------

    def _refresh_meeting_list(self):
        while (child := self._meeting_list.get_first_child()) is not None:
            self._meeting_list.remove(child)

        paths = sorted(DATA_DIR.glob("meeting_*.md"), reverse=True)
        for path in paths:
            title, dt, _, _ = _parse_meeting_file(path)
            row = self._make_meeting_row(path, title, dt)
            self._meeting_list.append(row)

    def _make_meeting_row(self, path: Path, title: str, dt: datetime.datetime) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        row.meeting_path = path

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(8)
        box.set_margin_bottom(8)

        title_label = Gtk.Label(label=title, xalign=0)
        title_label.set_ellipsize(Pango.EllipsizeMode.END)
        title_label.add_css_class("body")

        date_label = Gtk.Label(label=dt.strftime("%b %d, %Y  %H:%M"), xalign=0)
        date_label.add_css_class("caption")
        date_label.add_css_class("dim-label")

        box.append(title_label)
        box.append(date_label)
        row.set_child(box)
        return row

    def _on_meeting_row_activated(self, listbox, row):
        self._load_saved_meeting(row.meeting_path)

    def _load_saved_meeting(self, path: Path):
        title, dt, transcript, summary = _parse_meeting_file(path)
        self._transcript_buf.set_text(transcript)
        self._summary_buf.set_text(summary)
        self._viewing_saved = True
        self._current_save_path = path
        self.set_title(f"Fedora Granola — {title}")
        self._set_ui_editable(False)
        self._set_status(f"Viewing: {title}  ({dt.strftime('%b %d, %Y')})")
        self._chat_mode = "meeting"
        self._clear_chat()
        self._chat_entry.set_placeholder_text("Ask something about this meeting…")
        self._notes_btn.set_active(True)

    # ------------------------------------------------------------------
    # New recording
    # ------------------------------------------------------------------

    def _on_new_recording_clicked(self, btn):
        if self._recording:
            return
        self._viewing_saved = False
        self._current_save_path = None
        self._meeting_title = None
        self._title_found = False
        self._summary_header_buf = []
        self._transcript_parts = []
        self._transcript_buf.set_text("")
        self._summary_buf.set_text("")
        self._session_start = None
        self._meeting_list.unselect_all()
        self._chat_mode = "meeting"
        self._clear_chat()
        self._chat_entry.set_placeholder_text("Ask something about this meeting…")
        self._notes_btn.set_active(True)
        self._set_ui_editable(True)
        self._record_btn.set_sensitive(self._transcriber_ready)
        self._summarize_btn.set_sensitive(False)
        self._save_btn.set_sensitive(False)
        self.set_title("Fedora Granola")
        self._set_status("Ready — click Start Recording to begin")

    def _set_ui_editable(self, editable: bool):
        self._transcript_view.set_editable(editable)
        self._summary_view.set_editable(editable)
        if not editable:
            # Never disable the record button while recording — user must be able to stop
            if not self._recording:
                self._record_btn.set_sensitive(False)
            self._summarize_btn.set_sensitive(False)
            self._save_btn.set_sensitive(False)

    # ------------------------------------------------------------------
    # Record button
    # ------------------------------------------------------------------

    def _update_record_button(self):
        if self._recording:
            self._record_btn.set_label("⏹ Stop Recording")
            self._record_btn.set_css_classes(["destructive-action", "pill"])
        else:
            self._record_btn.set_label("⏺ Start Recording")
            self._record_btn.set_css_classes(["suggested-action", "pill"])

    # ------------------------------------------------------------------
    # Backend callbacks — all UI updates via GLib.idle_add
    # ------------------------------------------------------------------

    def _on_transcriber_ready(self):
        GLib.idle_add(self._ui_transcriber_ready)

    def _ui_transcriber_ready(self):
        self._transcriber_ready = True
        if not self._viewing_saved:
            self._record_btn.set_sensitive(True)
        self._set_status("Ready — click Start Recording to begin")
        return False

    def _on_transcriber_error(self, msg: str):
        GLib.idle_add(self._ui_show_error, "Transcription error", msg)

    def _on_audio_chunk(self, audio):
        self._transcriber.push(audio)

    def _on_segment(self, text: str):
        GLib.idle_add(self._ui_append_transcript, text)

    def _ui_append_transcript(self, text: str):
        self._transcript_parts.append(text)
        buf = self._transcript_buf
        end_iter = buf.get_end_iter()
        if buf.get_char_count() > 0:
            buf.insert(end_iter, " ")
            end_iter = buf.get_end_iter()
        buf.insert(end_iter, text)
        end_mark = buf.get_insert()
        self._transcript_view.scroll_mark_onscreen(end_mark)
        self._summarize_btn.set_sensitive(True)
        self._save_btn.set_sensitive(True)
        return False

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_settings_clicked(self, btn):
        dlg = SettingsDialog(self)
        dlg.present()

    def _on_record_clicked(self, btn):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self._recording = True
        self._session_start = datetime.datetime.now()
        self._update_record_button()
        try:
            self._capture.start()
            status = "Recording…"
            if self._capture.monitor_available:
                status += " (mic + system audio)"
            else:
                status += " (mic only — system audio monitor not found)"
            self._set_status(status)
        except Exception as e:
            self._recording = False
            self._update_record_button()
            self._ui_show_error("Audio error", str(e))

    def _stop_recording(self):
        self._recording = False
        self._update_record_button()
        self._capture.stop()
        self._waveform_history.clear()
        self._waveform.queue_draw()
        self._set_status(
            f"Recording stopped. {len(self._transcript_parts)} segments transcribed."
        )

    def _on_summarize_clicked(self, btn):
        transcript = self._get_full_transcript()
        if not transcript.strip():
            self._ui_show_error("Nothing to summarize", "The transcript is empty.")
            return

        self._summarize_btn.set_sensitive(False)
        self._summary_buf.set_text("")
        self._summary_header_buf = []
        self._title_found = False
        self._meeting_title = None
        self._set_status("Generating meeting notes…")

        summarizer = Summarizer(
            on_token=lambda t: GLib.idle_add(self._ui_stream_summary, t),
            on_complete=lambda s: GLib.idle_add(self._ui_summary_done, s),
            on_error=lambda e: GLib.idle_add(self._ui_show_error, "Summary error", e),
        )
        summarizer.summarize(transcript)

    def _ui_stream_summary(self, token: str):
        if not self._title_found:
            self._summary_header_buf.append(token)
            combined = "".join(self._summary_header_buf)
            if "\n" in combined:
                head, _, rest = combined.partition("\n")
                self._title_found = True
                if head.lower().startswith("title:"):
                    self._meeting_title = head[len("title:"):].strip()
                    rest = rest.lstrip("\n")
                else:
                    rest = combined  # not a title line, write it all
                if rest:
                    end_iter = self._summary_buf.get_end_iter()
                    self._summary_buf.insert(end_iter, rest)
            return False

        end_iter = self._summary_buf.get_end_iter()
        self._summary_buf.insert(end_iter, token)
        end_mark = self._summary_buf.get_insert()
        self._summary_view.scroll_mark_onscreen(end_mark)
        return False

    def _ui_summary_done(self, full_text: str):
        self._summarize_btn.set_sensitive(True)
        title = self._meeting_title or "Meeting Notes"
        self._auto_save(title)
        self._refresh_meeting_list()
        self.set_title(f"Fedora Granola — {title}")
        self._set_status(f'Meeting notes ready — saved as "{title}"')
        return False

    def _auto_save(self, title: str):
        transcript = self._get_full_transcript()
        summary = self._summary_buf.get_text(
            self._summary_buf.get_start_iter(),
            self._summary_buf.get_end_iter(),
            False,
        )
        ts = (self._session_start or datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
        self._current_save_path = DATA_DIR / f"meeting_{ts}.md"

        content = f"# {title}\n"
        content += f"*{ts.replace('_', ' ')}*\n\n"
        if summary.strip():
            content += "## Notes\n\n" + summary.strip() + "\n\n"
        content += "## Raw Transcript\n\n" + transcript.strip() + "\n"

        self._current_save_path.write_text(content, encoding="utf-8")

    def _on_save_clicked(self, btn):
        title = self._meeting_title or "Meeting Notes"
        self._auto_save(title)
        self._refresh_meeting_list()
        self._set_status(f'Saved as "{title}"')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_full_transcript(self) -> str:
        return self._transcript_buf.get_text(
            self._transcript_buf.get_start_iter(),
            self._transcript_buf.get_end_iter(),
            False,
        )

    def _set_status(self, msg: str):
        self._status_bar.set_label(msg)

    def _ui_show_error(self, title: str, msg: str):
        dialog = Adw.AlertDialog(heading=title, body=msg)
        dialog.add_response("ok", "OK")
        dialog.present(self)
        return False

    # ------------------------------------------------------------------
    # Chat panel
    # ------------------------------------------------------------------

    def _build_chat_panel(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._chat_view = Gtk.TextView()
        self._chat_view.set_editable(False)
        self._chat_view.set_wrap_mode(Gtk.WrapMode.WORD)
        self._chat_view.set_margin_start(12)
        self._chat_view.set_margin_end(12)
        self._chat_view.set_margin_top(8)
        self._chat_view.set_margin_bottom(8)
        self._chat_buf = self._chat_view.get_buffer()
        self._chat_end_mark = self._chat_buf.create_mark(
            "chat-end", self._chat_buf.get_end_iter(), left_gravity=False
        )
        scroll.set_child(self._chat_view)
        box.append(scroll)

        input_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        input_row.set_margin_start(12)
        input_row.set_margin_end(12)
        input_row.set_margin_top(4)
        input_row.set_margin_bottom(8)

        self._chat_entry = Gtk.Entry()
        self._chat_entry.set_hexpand(True)
        self._chat_entry.set_placeholder_text("Ask something about this meeting…")
        self._chat_entry.connect("activate", self._on_chat_send)
        input_row.append(self._chat_entry)

        self._send_btn = Gtk.Button(label="Send")
        self._send_btn.set_css_classes(["suggested-action"])
        self._send_btn.connect("clicked", self._on_chat_send)
        input_row.append(self._send_btn)

        box.append(input_row)
        return box

    def _on_view_toggle(self, btn, _param):
        if self._chat_btn.get_active():
            self._content_stack.set_visible_child_name("chat")
            self._chat_entry.grab_focus()
        else:
            self._content_stack.set_visible_child_name("notes")

    def _on_global_chat_clicked(self, btn):
        self._chat_mode = "global"
        self._meeting_list.unselect_all()
        self._clear_chat()
        self._chat_entry.set_placeholder_text("Ask something across all meetings…")
        self._chat_btn.set_active(True)
        self._set_status("Chat: All Meetings")

    def _clear_chat(self):
        self._chat_history = []
        self._chat_buf.set_text("")

    def _build_meeting_context(self) -> str:
        transcript = self._get_full_transcript()
        notes = self._summary_buf.get_text(
            self._summary_buf.get_start_iter(),
            self._summary_buf.get_end_iter(),
            False,
        )
        parts = []
        if notes.strip():
            parts.append(f"## Meeting Notes\n{notes.strip()}")
        if transcript.strip():
            parts.append(f"## Raw Transcript\n{transcript.strip()}")
        return "\n\n".join(parts)

    def _build_global_context(self) -> str:
        paths = sorted(DATA_DIR.glob("meeting_*.md"), reverse=True)
        chunks = []
        for path in paths:
            title, dt, transcript, summary = _parse_meeting_file(path)
            chunk = f"### {title} ({dt.strftime('%Y-%m-%d')})\n"
            if summary:
                chunk += summary[:800]
            chunks.append(chunk)
        return "All meeting notes:\n\n" + "\n\n".join(chunks) if chunks else ""

    def _on_chat_send(self, widget):
        if self._chat_streaming:
            return
        text = self._chat_entry.get_text().strip()
        if not text:
            return

        self._chat_entry.set_text("")
        self._send_btn.set_sensitive(False)
        self._chat_entry.set_sensitive(False)
        self._chat_streaming = True

        # Inject context into first message only
        if not self._chat_history:
            ctx = (self._build_global_context() if self._chat_mode == "global"
                   else self._build_meeting_context())
            user_content = f"Context:\n{ctx}\n\n---\n\n{text}" if ctx else text
        else:
            user_content = text

        self._chat_history.append({"role": "user", "content": user_content})
        self._append_chat("You: " + text + "\n")
        self._append_chat("Assistant: ")

        chatter = Chatter(
            on_token=lambda t: GLib.idle_add(self._ui_chat_token, t),
            on_complete=lambda s: GLib.idle_add(self._ui_chat_done, s),
            on_error=lambda e: GLib.idle_add(self._ui_chat_error, e),
        )
        chatter.chat(list(self._chat_history), CHAT_SYSTEM_PROMPT)

    def _append_chat(self, text: str):
        end_iter = self._chat_buf.get_end_iter()
        self._chat_buf.insert(end_iter, text)
        self._chat_buf.move_mark(self._chat_end_mark, self._chat_buf.get_end_iter())
        self._chat_view.scroll_mark_onscreen(self._chat_end_mark)

    def _ui_chat_token(self, token: str):
        self._append_chat(token)
        return False

    def _ui_chat_done(self, full_response: str):
        self._chat_history.append({"role": "assistant", "content": full_response})
        self._append_chat("\n\n")
        self._send_btn.set_sensitive(True)
        self._chat_entry.set_sensitive(True)
        self._chat_streaming = False
        self._chat_entry.grab_focus()
        return False

    def _ui_chat_error(self, msg: str):
        self._append_chat(f"\n[Error: {msg}]\n\n")
        self._send_btn.set_sensitive(True)
        self._chat_entry.set_sensitive(True)
        self._chat_streaming = False
        return False

    def do_close_request(self):
        if self._recording:
            self._capture.stop()
        self._transcriber.stop()
        return False


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class GranolaApp(Adw.Application):

    def __init__(self):
        super().__init__(
            application_id="io.github.eatmo",
            flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
        )
        self.connect("activate", self._on_activate)

    def _on_activate(self, app):
        win = GranolaWindow(application=app)
        win.present()
