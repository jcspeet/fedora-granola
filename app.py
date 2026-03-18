"""
Fedora Granola — GTK4 / libadwaita UI
"""

import datetime
import logging
import os
import threading

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk

from audio_capture import AudioCapture
from config import DATA_DIR
from summarizer import Summarizer
from transcriber import Transcriber

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class GranolaWindow(Adw.ApplicationWindow):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_title("Fedora Granola")
        self.set_default_size(900, 680)

        self._transcript_parts: list[str] = []
        self._recording = False
        self._transcriber_ready = False
        self._session_start: datetime.datetime | None = None

        self._build_ui()
        self._setup_backend()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Root layout
        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        # Header bar
        header = Adw.HeaderBar()
        toolbar_view.add_top_bar(header)

        # Record button (left side)
        self._record_btn = Gtk.Button()
        self._record_btn.set_css_classes(["suggested-action", "pill"])
        self._record_btn.set_sensitive(False)
        self._record_btn.connect("clicked", self._on_record_clicked)
        header.pack_start(self._record_btn)
        self._update_record_button()

        # Summarize button (right side)
        self._summarize_btn = Gtk.Button(label="Summarize")
        self._summarize_btn.set_css_classes(["pill"])
        self._summarize_btn.set_sensitive(False)
        self._summarize_btn.connect("clicked", self._on_summarize_clicked)
        header.pack_end(self._summarize_btn)

        # Save button (right side)
        self._save_btn = Gtk.Button(label="Save")
        self._save_btn.set_css_classes(["pill"])
        self._save_btn.set_sensitive(False)
        self._save_btn.connect("clicked", self._on_save_clicked)
        header.pack_end(self._save_btn)

        # Main paned layout
        paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        paned.set_position(320)
        toolbar_view.set_content(paned)

        # --- Top: Transcript ---
        transcript_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        transcript_header = Adw.PreferencesRow()
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

        # --- Bottom: Summary ---
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

        # --- Status bar ---
        self._status_bar = Gtk.Label(label="Loading Whisper model…")
        self._status_bar.set_css_classes(["caption", "dim-label"])
        self._status_bar.set_margin_start(12)
        self._status_bar.set_margin_end(12)
        self._status_bar.set_margin_top(4)
        self._status_bar.set_margin_bottom(6)
        self._status_bar.set_xalign(0)
        toolbar_view.add_bottom_bar(self._status_bar)

        # CSS for styling
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

        self._capture = AudioCapture(on_chunk=self._on_audio_chunk)

    # ------------------------------------------------------------------
    # Record button helpers
    # ------------------------------------------------------------------

    def _update_record_button(self):
        if self._recording:
            self._record_btn.set_label("⏹ Stop Recording")
            self._record_btn.set_css_classes(["destructive-action", "pill"])
        else:
            self._record_btn.set_label("⏺ Start Recording")
            self._record_btn.set_css_classes(["suggested-action", "pill"])

    # ------------------------------------------------------------------
    # Callbacks from backend threads — all UI updates via GLib.idle_add
    # ------------------------------------------------------------------

    def _on_transcriber_ready(self):
        GLib.idle_add(self._ui_transcriber_ready)

    def _ui_transcriber_ready(self):
        self._transcriber_ready = True
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

        # Auto-scroll to bottom
        end_mark = buf.get_insert()
        self._transcript_view.scroll_mark_onscreen(end_mark)

        self._summarize_btn.set_sensitive(True)
        self._save_btn.set_sensitive(True)
        return False

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

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
        self._set_status("Generating meeting notes with Claude…")

        summarizer = Summarizer(
            on_token=lambda t: GLib.idle_add(self._ui_stream_summary, t),
            on_complete=lambda s: GLib.idle_add(self._ui_summary_done),
            on_error=lambda e: GLib.idle_add(self._ui_show_error, "Summary error", e),
        )
        summarizer.summarize(transcript)

    def _ui_stream_summary(self, token: str):
        end_iter = self._summary_buf.get_end_iter()
        self._summary_buf.insert(end_iter, token)
        # Auto-scroll
        end_mark = self._summary_buf.get_insert()
        self._summary_view.scroll_mark_onscreen(end_mark)
        return False

    def _ui_summary_done(self):
        self._summarize_btn.set_sensitive(True)
        self._set_status("Meeting notes ready.")
        return False

    def _on_save_clicked(self, btn):
        transcript = self._get_full_transcript()
        summary = self._summary_buf.get_text(
            self._summary_buf.get_start_iter(),
            self._summary_buf.get_end_iter(),
            False,
        )

        ts = (self._session_start or datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
        path = DATA_DIR / f"meeting_{ts}.md"

        content = f"# Meeting — {ts.replace('_', ' ')}\n\n"
        if summary.strip():
            content += "## Notes\n\n" + summary.strip() + "\n\n"
        content += "## Raw Transcript\n\n" + transcript.strip() + "\n"

        path.write_text(content, encoding="utf-8")
        self._set_status(f"Saved to {path}")

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
        dialog = Adw.AlertDialog(
            heading=title,
            body=msg,
        )
        dialog.add_response("ok", "OK")
        dialog.present(self)
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
            application_id="io.github.fedora-granola",
            flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
        )
        self.connect("activate", self._on_activate)

    def _on_activate(self, app):
        win = GranolaWindow(application=app)
        win.present()
