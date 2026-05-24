import numpy as np
import matplotlib
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from utils.i18n import _


class EEGPlotWidget(QWidget):
    """Embedded matplotlib widget for stacked multi-channel EEG visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.raw = None
        self.duration = 10.0
        self.start = 0.0
        self.amplitude_scale = 1.0
        self.max_channels_display = 32
        self.view_changed_callback = None

        # Cached data for fast redraw during wheel interaction
        self._cached_data = None
        self._cached_times = None
        self._cached_ch_names = None
        self._cached_n_channels = 0
        self._cached_highlight_events = None
        self._base_offset = 50.0

        # Display mode: 'eeg' or 'psd'
        self._display_mode = 'eeg'

        # PSD cursor elements
        self._psd_cursor_line = None
        self._psd_cursor_text = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Matplotlib figure with dark theme
        self.fig = Figure(facecolor='#1e1e2e', tight_layout={'pad': 0.3, 'h_pad': 0.2, 'w_pad': 0.2})
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #1e1e2e;")

        # Navigation toolbar styled
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("""
            QToolBar { background-color: #181825; border: none; }
            QToolButton { color: #cdd6f4; background-color: #313244;
                          border-radius: 4px; margin: 2px; padding: 4px; }
            QToolButton:hover { background-color: #45475a; }
        """)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        # Intercept wheel events via Qt event filter for reliable modifier detection
        self.canvas.installEventFilter(self)

        # Mouse move for PSD cursor
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def eventFilter(self, obj, event):
        if obj == self.canvas and event.type() == QEvent.Wheel:
            # Wheel only works in EEG mode, not PSD mode
            if self._display_mode == 'eeg' and self.toolbar.mode == '':
                delta = event.angleDelta().y()
                modifiers = event.modifiers()
                self._handle_wheel(delta, modifiers)
                return True
        return super().eventFilter(obj, event)

    @staticmethod
    def _get_nice_scale(current, direction):
        """Return next/previous nice scale in 10-20-30...100-200-300 sequence."""
        import math
        if current <= 0:
            current = 10
        if current < 100:
            step = 10
            min_val = 10
        elif current < 1000:
            step = 100
            min_val = 10
        else:
            step = 1000
            min_val = 100
        base = math.floor(current / step) * step
        if direction > 0:
            return base + step
        else:
            if current > base + 0.01:
                return max(min_val, base)
            else:
                prev = base - step
                if base == 100 and prev < 100:
                    return 90
                elif base == 1000 and prev < 1000:
                    return 900
                else:
                    return max(min_val, prev)

    def _handle_wheel(self, delta, modifiers):
        """Handle mouse wheel: amplitude / time zoom / time pan."""
        if self.raw is None or self._cached_data is None:
            return
        step = 1 if delta > 0 else -1

        if modifiers == Qt.ControlModifier:
            # Ctrl + wheel: time zoom (change duration)
            factor = 1.1 if step > 0 else 0.9
            new_dur = self.duration * factor
            self.duration = max(1.0, min(60.0, new_dur))
        elif modifiers == Qt.AltModifier:
            # Alt + wheel: time pan (move window)
            shift = self.duration * 0.1 * step
            new_start = self.start - shift
            max_start = max(0, self.raw.times[-1] - self.duration)
            self.start = max(0.0, min(max_start, new_start))
        else:
            # Plain wheel: amplitude scale in nice 1-2-5 steps
            scale_uv = self._base_offset * self.amplitude_scale
            new_scale = self._get_nice_scale(scale_uv, step)
            self.amplitude_scale = new_scale / self._base_offset
            self.amplitude_scale = max(0.001, min(1000.0, self.amplitude_scale))

        if modifiers in (Qt.ControlModifier, Qt.AltModifier):
            # Time zoom/pan needs new data window — full re-fetch
            self.plot_eeg(self.raw, title=self.ax.get_title())
        else:
            # Amplitude scale only — fast cached redraw
            self._draw_traces(self.ax.get_title())
        if self.view_changed_callback:
            self.view_changed_callback(self.duration, self.start)

    def _on_mouse_move(self, event):
        """Show vertical cursor line with frequency readout in PSD mode."""
        if self._display_mode != 'psd':
            return
        if event.inaxes != self.ax:
            if self._psd_cursor_line:
                self._psd_cursor_line.set_visible(False)
            if self._psd_cursor_text:
                self._psd_cursor_text.set_visible(False)
            self.canvas.draw_idle()
            return

        freq = event.xdata
        if freq is None:
            return

        if self._psd_cursor_line:
            self._psd_cursor_line.set_xdata([freq, freq])
            self._psd_cursor_line.set_visible(True)
        if self._psd_cursor_text:
            self._psd_cursor_text.set_x(freq)
            self._psd_cursor_text.set_text(f'{freq:.1f} Hz')
            self._psd_cursor_text.set_visible(True)

        self.canvas.draw_idle()

    def _style_axes(self):
        self.ax.set_facecolor('#1e1e2e')
        self.ax.tick_params(colors='#cdd6f4', labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color('#45475a')
        self.ax.xaxis.label.set_color('#cdd6f4')
        self.ax.yaxis.label.set_color('#cdd6f4')
        self.ax.title.set_color('#89b4fa')
        self.ax.grid(True, alpha=0.2, color='#45475a')
        # Remove default horizontal margins so traces fill the full width
        self.ax.margins(x=0)

    def _generate_bright_colors(self, n):
        """Generate n bright colors with fixed high value, avoiding dark regions."""
        import matplotlib.colors as mcolors
        hues = np.linspace(0, 1, n, endpoint=False)
        hsv = np.zeros((n, 3))
        hsv[:, 0] = hues
        hsv[:, 1] = 0.85   # saturation
        hsv[:, 2] = 0.95   # value (brightness) — high so all colors are vivid
        return mcolors.hsv_to_rgb(hsv)

    def plot_eeg(self, raw, duration=None, start=None, title="EEG Signal", highlight_events=None):
        if raw is None:
            return
        if duration is not None:
            self.duration = duration
        if start is not None:
            self.start = start

        self._display_mode = 'eeg'
        self.raw = raw
        sfreq = raw.info['sfreq']
        start_sample = int(self.start * sfreq)
        n_samples = int(self.duration * sfreq)
        end_sample = min(start_sample + n_samples, raw.n_times)

        if start_sample >= end_sample:
            self.ax.clear()
            self._style_axes()
            self.ax.text(0.5, 0.5, "No data in selected range", ha='center', va='center',
                         transform=self.ax.transAxes, color='#6c7086', fontsize=14)
            self.canvas.draw()
            return

        data, times = raw[:, start_sample:end_sample]
        data = data * 1e6  # Convert to microvolts

        # Cache for fast redraw
        self._cached_data = data
        self._cached_times = times
        self._cached_n_channels = min(data.shape[0], self.max_channels_display)
        self._cached_ch_names = raw.info['ch_names'][:self._cached_n_channels]
        self._cached_highlight_events = highlight_events

        # Compute base offset from data
        n_channels = self._cached_n_channels
        channel_ranges = [np.max(np.abs(data[i])) for i in range(n_channels)]
        median_range = np.median(channel_ranges) if channel_ranges else 50
        if median_range == 0:
            median_range = 50
        self._base_offset = median_range * 2.8

        # Default display scale: ~200 µV per division
        self.amplitude_scale = 200.0 / self._base_offset
        self.amplitude_scale = max(0.001, min(1000.0, self.amplitude_scale))

        self._draw_traces(title)

    def _draw_traces(self, title):
        """Fast redraw using cached data. Only updates scale/offset/view params."""
        if self._cached_data is None:
            return

        data = self._cached_data
        times = self._cached_times
        n_channels = self._cached_n_channels
        ch_names = self._cached_ch_names
        highlight_events = self._cached_highlight_events

        offset = self._base_offset * self.amplitude_scale

        self.ax.clear()
        self._style_axes()

        colors = self._generate_bright_colors(n_channels)

        for i in range(n_channels):
            y = data[i] + i * offset
            self.ax.plot(times, y, color=colors[i], linewidth=0.7, alpha=0.85)

        # Y-axis with channel labels at each trace center
        self.ax.set_yticks([i * offset for i in range(n_channels)])
        self.ax.set_yticklabels(ch_names, fontsize=8, color='#cdd6f4')

        self.ax.set_xlabel("Time (s)", fontsize=10, color='#cdd6f4')
        self.ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color='#89b4fa')

        # Scale annotation: how many µV between adjacent channel baselines
        scale_uv = self._base_offset * self.amplitude_scale
        self.ax.text(
            0.99, 0.99,
            f"{_('每格')} {int(round(scale_uv))} µV",
            transform=self.ax.transAxes,
            fontsize=10,
            color='#a6e3a1',
            ha='right',
            va='top',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#181825',
                      edgecolor='#a6e3a1', alpha=0.9)
        )

        # Add event markers if provided: list of (time, desc, color)
        if highlight_events:
            y_min, y_max = self.ax.get_ylim()
            for ev_time, ev_desc, ev_color in highlight_events:
                if self.start <= ev_time <= self.start + self.duration:
                    self.ax.axvline(x=ev_time, color=ev_color, linestyle='--',
                                    alpha=0.8, linewidth=1.2)
                    # Event description at top
                    self.ax.text(ev_time, y_max * 0.98, str(ev_desc),
                                 color=ev_color, fontsize=8, ha='center', va='top',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e1e2e',
                                           edgecolor=ev_color, alpha=0.8))
                    # Event time at bottom
                    self.ax.text(ev_time, y_min + (y_max - y_min) * 0.01,
                                 f"{ev_time:.2f}s",
                                 color=ev_color, fontsize=8, ha='center', va='bottom',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e1e2e',
                                           edgecolor=ev_color, alpha=0.8))

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_psd(self, raw, title="Power Spectral Density"):
        """Plot PSD using MNE's built-in method."""
        if raw is None:
            return
        self._display_mode = 'psd'
        self.ax.clear()
        self._style_axes()

        try:
            psd = raw.compute_psd(fmax=65)
            freqs = psd.freqs
            spectrum = psd.get_data()
            # Average across channels for clarity
            mean_spectrum = np.mean(spectrum, axis=0)

            self.ax.plot(freqs, 10 * np.log10(mean_spectrum + 1e-12), color='#89b4fa', linewidth=1.2)
            self.ax.set_xlabel("Frequency (Hz)", fontsize=10)
            self.ax.set_ylabel("Power (dB)", fontsize=10)
            self.ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            self.ax.set_xlim((0, 65))

            # Create cursor elements (hidden initially)
            self._psd_cursor_line = self.ax.axvline(x=0, color='#f9e2af', linestyle='--',
                                                     alpha=0.8, linewidth=1, visible=False)
            y_min, y_max = self.ax.get_ylim()
            self._psd_cursor_text = self.ax.text(
                0, y_min + (y_max - y_min) * 0.02, '',
                color='#f9e2af', fontsize=9, ha='center', va='bottom', visible=False,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e1e2e',
                          edgecolor='#f9e2af', alpha=0.8)
            )

            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            self.ax.text(0.5, 0.5, f"PSD Error: {str(e)}", ha='center', va='center',
                         transform=self.ax.transAxes, color='#f38ba8', fontsize=10)
            self.canvas.draw()

    def clear(self):
        self.ax.clear()
        self._style_axes()
        self.canvas.draw()

