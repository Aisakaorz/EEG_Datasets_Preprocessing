import numpy as np
import mne
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QVBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QWidget
)
from PySide6.QtGui import QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import signal as sp_signal


class ICAAuditDialog(QDialog):
    """Dialog for manual review of ICA components before application."""

    def __init__(self, ica, raw, auto_exclude, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ICA 成分审核")
        self.setMinimumSize(700, 520)
        self.ica = ica
        self.raw = raw
        self.auto_exclude = set(auto_exclude or [])
        self._batch_updating = False

        # Identify artifact types (robust: may fail on some datasets)
        try:
            eog = self.ica.find_bads_eog(self.raw)[0]
            self.eog_indices = [int(x) for x in eog] if eog else []
        except Exception:
            try:
                eog = self.ica.find_bads_eog(self.raw, ch_name=self.raw.info['ch_names'][0])[0]
                self.eog_indices = [int(x) for x in eog] if eog else []
            except Exception:
                self.eog_indices = []
        try:
            mus = self.ica.find_bads_muscle(self.raw)[0]
            self.mus_indices = [int(x) for x in mus] if mus else []
        except Exception:
            self.mus_indices = []

        self._build_ui()

    def _build_ui(self):
        from PySide6.QtGui import QColor
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)

        # Left panel: component list
        left_layout = QVBoxLayout()
        left_layout.setSpacing(6)

        info = QLabel(
            "<b>ICA 成分审核</b><br>"
            f"总成分数: {self.ica.n_components_}<br>"
            f"Worker推荐排除: {sorted(self.auto_exclude)}<br>"
            f"本窗口重检: 眼动{self.eog_indices}, 肌电{self.mus_indices}"
        )
        info.setWordWrap(True)
        left_layout.addWidget(info)

        # Quick action buttons
        btn_row = QHBoxLayout()
        btn_sel_all = QPushButton("全排除")
        btn_sel_none = QPushButton("全保留")
        btn_sel_eog = QPushButton("仅眼动")
        btn_sel_mus = QPushButton("仅肌电")
        btn_sel_worker = QPushButton("Worker推荐")
        for b in (btn_sel_all, btn_sel_none, btn_sel_eog, btn_sel_mus, btn_sel_worker):
            b.setStyleSheet("font-size:11px;padding:2px 6px;")
            btn_row.addWidget(b)
        btn_sel_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        btn_sel_none.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        btn_sel_eog.clicked.connect(lambda: self._set_by_type('eog'))
        btn_sel_mus.clicked.connect(lambda: self._set_by_type('mus'))
        btn_sel_worker.clicked.connect(lambda: self._set_by_indices(self.auto_exclude))
        left_layout.addLayout(btn_row)

        self.list_widget = QListWidget()
        self.list_widget.setMaximumWidth(280)
        self.list_widget.currentItemChanged.connect(self._on_component_selected)
        for i in range(self.ica.n_components_):
            item = QListWidgetItem("placeholder")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            checked = i in self.auto_exclude
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            item.setData(Qt.UserRole, i)
            self.list_widget.addItem(item)
            self._update_item_text(item)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        left_layout.addWidget(self.list_widget)

        # Live stats
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        left_layout.addWidget(self.stats_label)
        self._update_stats()

        legend = QLabel(
            "<span style='color:#a6adc8'>红色底 = 将排除; 绿色底 = 将保留. "
            "点击成分右侧预览波形和功率谱</span>"
        )
        legend.setWordWrap(True)
        left_layout.addWidget(legend)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        left_layout.addWidget(btns)

        main_layout.addLayout(left_layout, 0)

        # Right panel: preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        preview_title = QLabel("<b>成分预览（点击左侧成分查看）</b>")
        right_layout.addWidget(preview_title)

        self.preview_fig = Figure(figsize=(5, 4), facecolor='#1e1e2e')
        self.preview_canvas = FigureCanvas(self.preview_fig)
        self.preview_canvas.setStyleSheet("background-color: #1e1e2e;")
        right_layout.addWidget(self.preview_canvas)

        self.preview_ax = self.preview_fig.add_subplot(111)
        self._style_preview_axes()
        self.preview_ax.text(0.5, 0.5, "请点击左侧成分\n查看波形预览",
                             ha='center', va='center', transform=self.preview_ax.transAxes,
                             color='#6c7086', fontsize=12)
        self.preview_canvas.draw()

        main_layout.addWidget(right_widget, 1)

    def _update_item_text(self, item):
        from PySide6.QtGui import QColor
        idx = item.data(Qt.UserRole)
        labels = []
        if idx in self.eog_indices:
            labels.append("眼动")
        if idx in self.mus_indices:
            labels.append("肌电")
        label_str = f" [{'+'.join(labels)}]" if labels else ""
        action = "[排除]" if item.checkState() == Qt.Checked else "[保留]"
        item.setText(_(f"{action} 成分{idx:02d}{label_str}"))
        if item.checkState() == Qt.Checked:
            item.setBackground(QColor('#452233'))
        else:
            item.setBackground(QColor('#1e1e2e'))

    def _update_stats(self):
        n_excl = sum(1 for i in range(self.list_widget.count())
                     if self.list_widget.item(i).checkState() == Qt.Checked)
        n_keep = self.ica.n_components_ - n_excl
        self.stats_label.setText(
            f"<b>将排除 <span style='color:#f38ba8'>{n_excl}</span> 个 | "
            f"保留 <span style='color:#a6e3a1'>{n_keep}</span> 个</b>"
        )

    def _on_item_changed(self, item):
        if self._batch_updating:
            return
        self._update_item_text(item)
        self._update_stats()

    def _set_all(self, state):
        self._batch_updating = True
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(state)
            self._update_item_text(item)
        self._batch_updating = False
        self._update_stats()

    def _set_by_type(self, typ):
        self._batch_updating = True
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            idx = item.data(Qt.UserRole)
            if typ == 'eog' and idx in self.eog_indices:
                item.setCheckState(Qt.Checked)
            elif typ == 'mus' and idx in self.mus_indices:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self._update_item_text(item)
        self._batch_updating = False
        self._update_stats()

    def _set_by_indices(self, indices):
        s = set(indices)
        self._batch_updating = True
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            idx = item.data(Qt.UserRole)
            item.setCheckState(Qt.Checked if idx in s else Qt.Unchecked)
            self._update_item_text(item)
        self._batch_updating = False
        self._update_stats()

    def _style_preview_axes(self):
        self.preview_ax.set_facecolor('#1e1e2e')
        self.preview_ax.tick_params(colors='#cdd6f4', labelsize=8)
        for spine in self.preview_ax.spines.values():
            spine.set_color('#45475a')
        self.preview_ax.xaxis.label.set_color('#cdd6f4')
        self.preview_ax.yaxis.label.set_color('#cdd6f4')
        self.preview_ax.title.set_color('#89b4fa')
        self.preview_ax.grid(True, alpha=0.2, color='#45475a')

    def _on_component_selected(self, current, previous):
        if current is None:
            return
        idx = current.data(Qt.UserRole)
        self._plot_component(idx)

    def _plot_component(self, idx):
        self.preview_fig.clear()

        def _style_ax(ax):
            ax.set_facecolor('#1e1e2e')
            ax.tick_params(colors='#cdd6f4', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#45475a')
            ax.xaxis.label.set_color('#cdd6f4')
            ax.yaxis.label.set_color('#cdd6f4')
            ax.title.set_color('#89b4fa')
            ax.grid(True, alpha=0.2, color='#45475a')

        try:
            ica_sources = self.ica.get_sources(self.raw)
            source_data = ica_sources.get_data(picks=[idx])[0]
            sfreq = self.raw.info['sfreq']

            # Show up to 5 seconds
            n_show = min(int(sfreq * 5), len(source_data))
            times = np.arange(n_show) / sfreq

            # Determine type label
            labels = []
            if idx in self.eog_indices:
                labels.append("眼动伪迹")
            if idx in self.mus_indices:
                labels.append("肌电伪迹")
            type_str = f" [{' / '.join(labels)}]" if labels else " [非伪迹]"

            # Top subplot: waveform (time domain)
            ax_wave = self.preview_fig.add_subplot(2, 1, 1)
            _style_ax(ax_wave)
            ax_wave.plot(times, source_data[:n_show], color='#89b4fa', linewidth=0.7)
            ax_wave.set_xlabel("时间 (秒)", fontsize=9)
            ax_wave.set_ylabel("幅度", fontsize=9)
            ax_wave.set_title(f"ICA 成分 {idx}{type_str} — 波形", fontsize=11, fontweight='bold')

            # Bottom subplot: PSD (frequency domain)
            freqs, psd = sp_signal.welch(source_data[:n_show], fs=sfreq, nperseg=min(256, n_show))
            psd_db = 10 * np.log10(psd + 1e-12)

            ax_psd = self.preview_fig.add_subplot(2, 1, 2)
            _style_ax(ax_psd)
            ax_psd.plot(freqs, psd_db, color='#f9e2af', linewidth=0.8, alpha=0.7)
            ax_psd.set_xlabel("频率 (Hz)", fontsize=9)
            ax_psd.set_ylabel("功率谱密度 (dB)", fontsize=9)
            ax_psd.set_title("功率谱密度 (PSD)", fontsize=10)

        except Exception as e:
            ax_err = self.preview_fig.add_subplot(111)
            ax_err.set_facecolor('#1e1e2e')
            ax_err.text(0.5, 0.5, f"预览加载失败:\n{str(e)}",
                        ha='center', va='center', transform=ax_err.transAxes,
                        color='#f38ba8', fontsize=10)

        self.preview_fig.tight_layout()
        self.preview_canvas.draw()

    def get_excluded(self):
        excluded = []
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).checkState() == Qt.Checked:
                excluded.append(self.list_widget.item(i).data(Qt.UserRole))
        return excluded

