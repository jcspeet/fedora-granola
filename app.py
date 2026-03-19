"""
Fedora Granola — GTK4 / libadwaita UI
"""

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
from config import DATA_DIR
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

        # Outer horizontal paned: sidebar | content
        outer_paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        outer_paned.set_position(230)
        outer_paned.set_shrink_start_child(False)
        outer_paned.set_shrink_end_child(False)
        outer_paned.set_resize_start_child(False)
        outer_paned.set_resize_end_child(True)
        toolbar_view.set_content(outer_paned)

        outer_paned.set_start_child(self._build_sidebar())
        outer_paned.set_end_child(self._build_content())

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
