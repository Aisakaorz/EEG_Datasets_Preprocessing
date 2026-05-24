import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QGroupBox,
    QSpinBox, QDoubleSpinBox, QProgressBar,
    QTextEdit, QMessageBox, QComboBox, QListWidget,
    QListWidgetItem, QGridLayout, QSizePolicy,
    QTabWidget, QRadioButton, QButtonGroup,
    QDialog, QDialogButtonBox
)

from gui.styles import DARK_STYLESHEET
from gui.plot_widget import EEGPlotWidget
from gui.ica_audit_dialog import ICAAuditDialog
from gui.paradigm_dialog import ParadigmDesignDialog, DEFAULT_PARADIGM
from workers.loader import CNTLoaderWorker
from workers.preprocess import PreprocessStepWorker, PreprocessAllWorker
from workers.de_extract import DEExtractWorker
from utils.i18n import _, set_language, LANG_EN, LANG_ZH

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("CNT EEG 预处理工作站"))
        self.setMinimumSize(1280, 960)
        self.resize(1440, 960)

        # Internal state
        self.raw_original = None
        self.raw_current = None
        self.raw_segment = None
        self.events = []
        self.de_tensor = None
        self.de_band_names = []
        self.worker = None
        self.ica_object = None
        self.ica_auto_exclude = []
        self.ica_manual_exclude = []
        self.raw_pre_ica = None
        self.paradigm_segments = []
        self.paradigm_config = self._load_paradigm_config()

        self._build_ui()
        self._apply_styles()

        # Busy animation timer for long-running operations
    def _start_busy(self, text=_("处理中...")):
        """Show busy text on progress bar; actual progress is driven by worker signals."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat(text)
        self.progress_bar.setValue(0)

    def _stop_busy(self):
        """Restore normal progress bar display."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setValue(0)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ==================== Left Panel (fixed width) ====================
        left_widget = QWidget()
        left_widget.setFixedWidth(440)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.setSpacing(4)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header_layout = QHBoxLayout()
        self.title_label = QLabel(_("CNT EEG 预处理工作站"))
        self.title_label.setObjectName("titleLabel")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        self.btn_lang = QPushButton("Switch to English")
        self.btn_lang.setMaximumHeight(30)
        self.btn_lang.setToolTip(_("切换语言"))
        self.btn_lang.clicked.connect(self._toggle_language)
        header_layout.addWidget(self.btn_lang)
        left_layout.addLayout(header_layout)

        # --- File Group ---
        self.file_group = QGroupBox(_("文件"))
        file_grid = QGridLayout(self.file_group)
        file_grid.setSpacing(4)
        file_grid.setContentsMargins(6, 8, 6, 6)

        self.line_cnt = QLineEdit()
        self.line_cnt.setPlaceholderText(_("请选择 .cnt 文件..."))
        self.line_cnt.setReadOnly(True)
        self.line_cnt.setMaximumHeight(26)

        self.btn_browse = QPushButton(_("浏览"))
        self.btn_browse.setObjectName("primaryButton")
        self.btn_browse.setMaximumHeight(26)
        self.btn_browse.clicked.connect(self._browse_cnt)

        self.btn_load = QPushButton(_("加载"))
        self.btn_load.setObjectName("successButton")
        self.btn_load.setMaximumHeight(26)
        self.btn_load.clicked.connect(self._load_cnt)

        self.lbl_cnt = QLabel(_("CNT:"))
        file_grid.addWidget(self.lbl_cnt, 0, 0)
        file_grid.addWidget(self.line_cnt, 0, 1)
        file_grid.addWidget(self.btn_browse, 0, 2)
        file_grid.addWidget(self.btn_load, 0, 3)

        self.line_out = QLineEdit()
        self.line_out.setText("./data/features/cnt")
        self.line_out.setMaximumHeight(26)

        self.btn_out = QPushButton(_("浏览"))
        self.btn_out.setMaximumHeight(26)
        self.btn_out.clicked.connect(self._browse_out)

        self.lbl_out = QLabel(_("输出:"))
        file_grid.addWidget(self.lbl_out, 1, 0)
        file_grid.addWidget(self.line_out, 1, 1)
        file_grid.addWidget(self.btn_out, 1, 2)

        left_layout.addWidget(self.file_group)

        # --- Events / Segmentation Group ---
        self.events_group = QGroupBox(_("事件与分段"))
        events_vlayout = QVBoxLayout(self.events_group)
        events_vlayout.setSpacing(4)
        events_vlayout.setContentsMargins(6, 8, 6, 6)

        self.events_list = QListWidget()
        self.events_list.setMinimumHeight(80)
        self.events_list.setMaximumHeight(120)
        self.events_list.itemClicked.connect(self._on_event_selected)
        events_vlayout.addWidget(self.events_list)

        # Seg controls: spinboxes on row 0, buttons on rows 1-2
        seg_grid = QGridLayout()
        seg_grid.setSpacing(6)
        for c in range(8):
            seg_grid.setColumnStretch(c, 1)

        self.lbl_seg_before = QLabel(_("前:"))
        seg_grid.addWidget(self.lbl_seg_before, 0, 0, Qt.AlignRight)
        self.spin_seg_before = QDoubleSpinBox()
        self.spin_seg_before.setRange(0, 300)
        self.spin_seg_before.setValue(0)
        self.spin_seg_before.setSuffix(" s")
        self.spin_seg_before.setMaximumHeight(24)
        seg_grid.addWidget(self.spin_seg_before, 0, 1)

        self.lbl_seg_after = QLabel(_("后:"))
        seg_grid.addWidget(self.lbl_seg_after, 0, 2, Qt.AlignRight)
        self.spin_seg_after = QDoubleSpinBox()
        self.spin_seg_after.setRange(0.1, 300)
        self.spin_seg_after.setValue(5)
        self.spin_seg_after.setSuffix(" s")
        self.spin_seg_after.setMaximumHeight(24)
        seg_grid.addWidget(self.spin_seg_after, 0, 3)

        self.btn_extract_seg = QPushButton(_("截取事件前后片段"))
        self.btn_extract_seg.setObjectName("primaryButton")
        self.btn_extract_seg.setMaximumHeight(26)
        self.btn_extract_seg.clicked.connect(self._extract_segment)
        seg_grid.addWidget(self.btn_extract_seg, 1, 0, 1, 4)

        self.btn_manual_seg = QPushButton(_("截取当前可视化脑电片段"))
        self.btn_manual_seg.setObjectName("primaryButton")
        self.btn_manual_seg.setMaximumHeight(26)
        self.btn_manual_seg.clicked.connect(self._manual_segment)
        seg_grid.addWidget(self.btn_manual_seg, 1, 4, 1, 4)

        self.btn_design_paradigm = QPushButton(_("设计范式"))
        self.btn_design_paradigm.setObjectName("successButton")
        self.btn_design_paradigm.setMaximumHeight(26)
        self.btn_design_paradigm.clicked.connect(self._design_paradigm)
        self.btn_design_paradigm.setEnabled(False)
        seg_grid.addWidget(self.btn_design_paradigm, 2, 0, 1, 4)

        self.btn_auto_paradigm = QPushButton(_("按范式提取"))
        self.btn_auto_paradigm.setObjectName("successButton")
        self.btn_auto_paradigm.setMaximumHeight(26)
        self.btn_auto_paradigm.clicked.connect(self._auto_extract_paradigm)
        self.btn_auto_paradigm.setEnabled(False)
        seg_grid.addWidget(self.btn_auto_paradigm, 2, 4, 1, 4)

        events_vlayout.addLayout(seg_grid)
        left_layout.addWidget(self.events_group)

        # --- Parameters Group ---
        self.param_group = QGroupBox(_("预处理参数"))
        param_layout = QVBoxLayout(self.param_group)
        param_layout.setSpacing(4)
        param_layout.setContentsMargins(6, 8, 6, 6)

        param_grid = QGridLayout()
        param_grid.setSpacing(6)
        param_grid.setColumnStretch(1, 1)
        param_grid.setColumnStretch(3, 1)

        # Row 0: Sampling Rate | High-pass
        self.lbl_sfreq = QLabel(_("采样率"))
        self.spin_sfreq = QSpinBox()
        self.spin_sfreq.setRange(50, 1000)
        self.spin_sfreq.setValue(200)
        self.spin_sfreq.setSuffix(" Hz")
        self.spin_sfreq.setMaximumHeight(24)
        self.spin_sfreq.setMaximumWidth(75)
        self.lbl_lfreq = QLabel(_("高通滤波"))
        self.spin_lfreq = QDoubleSpinBox()
        self.spin_lfreq.setRange(0.01, 10)
        self.spin_lfreq.setValue(0.1)
        self.spin_lfreq.setSuffix(" Hz")
        self.spin_lfreq.setSingleStep(0.01)
        self.spin_lfreq.setMaximumHeight(24)
        self.spin_lfreq.setMaximumWidth(75)
        param_grid.addWidget(self.lbl_sfreq, 0, 0, Qt.AlignRight)
        param_grid.addWidget(self.spin_sfreq, 0, 1)
        param_grid.addWidget(self.lbl_lfreq, 0, 2, Qt.AlignRight)
        param_grid.addWidget(self.spin_lfreq, 0, 3)

        # Row 1: Notch | Low-pass
        self.lbl_notch = QLabel(_("陷波滤波"))
        self.combo_notch = QComboBox()
        self.combo_notch.addItems(["50Hz", "60Hz", "50Hz, 60Hz", _("关")])
        self.combo_notch.setMaximumHeight(24)
        self.combo_notch.setMaximumWidth(75)
        self.lbl_hfreq = QLabel(_("低通滤波"))
        self.spin_hfreq = QDoubleSpinBox()
        self.spin_hfreq.setRange(10, 100)
        self.spin_hfreq.setValue(50)
        self.spin_hfreq.setSuffix(" Hz")
        self.spin_hfreq.setMaximumHeight(24)
        self.spin_hfreq.setMaximumWidth(75)
        param_grid.addWidget(self.lbl_notch, 1, 0, Qt.AlignRight)
        param_grid.addWidget(self.combo_notch, 1, 1)
        param_grid.addWidget(self.lbl_hfreq, 1, 2, Qt.AlignRight)
        param_grid.addWidget(self.spin_hfreq, 1, 3)

        # Row 2: Reference Electrode | ICA
        self.lbl_ref = QLabel(_("参考电极"))
        self.combo_ref = QComboBox()
        self.combo_ref.addItems(["average", "Cz", "Fp1"])
        self.combo_ref.setMaximumHeight(24)
        self.combo_ref.setMaximumWidth(75)
        self.lbl_ica = QLabel(_("ICA"))
        self.spin_ica = QSpinBox()
        self.spin_ica.setRange(5, 60)
        self.spin_ica.setValue(20)
        self.spin_ica.setMaximumHeight(24)
        self.spin_ica.setMaximumWidth(75)
        param_grid.addWidget(self.lbl_ref, 2, 0, Qt.AlignRight)
        param_grid.addWidget(self.combo_ref, 2, 1)
        param_grid.addWidget(self.lbl_ica, 2, 2, Qt.AlignRight)
        param_grid.addWidget(self.spin_ica, 2, 3)

        # Row 3: DE Window Length | DE Window Step
        self.lbl_win_len = QLabel(_("DE窗长"))
        self.spin_win_len = QDoubleSpinBox()
        self.spin_win_len.setRange(0.5, 10)
        self.spin_win_len.setValue(1.0)
        self.spin_win_len.setSuffix(" s")
        self.spin_win_len.setSingleStep(0.1)
        self.spin_win_len.setMaximumHeight(24)
        self.spin_win_len.setMaximumWidth(75)
        self.lbl_win_step = QLabel(_("DE步长"))
        self.spin_win_step = QDoubleSpinBox()
        self.spin_win_step.setRange(0.1, 10)
        self.spin_win_step.setValue(1.0)
        self.spin_win_step.setSuffix(" s")
        self.spin_win_step.setSingleStep(0.1)
        self.spin_win_step.setMaximumHeight(24)
        self.spin_win_step.setMaximumWidth(75)
        param_grid.addWidget(self.lbl_win_len, 3, 0, Qt.AlignRight)
        param_grid.addWidget(self.spin_win_len, 3, 1)
        param_grid.addWidget(self.lbl_win_step, 3, 2, Qt.AlignRight)
        param_grid.addWidget(self.spin_win_step, 3, 3)

        param_layout.addLayout(param_grid)

        left_layout.addWidget(self.param_group)

        # --- Step-by-step Preprocessing Group ---
        self.prep_group = QGroupBox(_("脑电预处理"))
        prep_layout = QVBoxLayout(self.prep_group)
        prep_layout.setSpacing(4)
        prep_layout.setContentsMargins(6, 8, 6, 6)

        steps_grid = QGridLayout()
        steps_grid.setSpacing(5)
        steps_grid.setColumnStretch(0, 1)
        steps_grid.setColumnStretch(1, 1)

        self.btn_step_bandpass = QPushButton(_("1. 带通滤波"))
        self.btn_step_bandpass.setMaximumHeight(28)
        self.btn_step_bandpass.clicked.connect(lambda: self._run_step('bandpass'))
        self.btn_step_bandpass.setEnabled(False)
        steps_grid.addWidget(self.btn_step_bandpass, 0, 0)

        self.btn_step_notch = QPushButton(_("2. 陷波滤波"))
        self.btn_step_notch.setMaximumHeight(28)
        self.btn_step_notch.clicked.connect(lambda: self._run_step('notch'))
        self.btn_step_notch.setEnabled(False)
        steps_grid.addWidget(self.btn_step_notch, 0, 1)

        self.btn_step_downsample = QPushButton(_("3. 降采样"))
        self.btn_step_downsample.setMaximumHeight(28)
        self.btn_step_downsample.clicked.connect(lambda: self._run_step('downsample'))
        self.btn_step_downsample.setEnabled(False)
        steps_grid.addWidget(self.btn_step_downsample, 1, 0)

        self.btn_step_ref = QPushButton(_("4. 重参考"))
        self.btn_step_ref.setMaximumHeight(28)
        self.btn_step_ref.clicked.connect(lambda: self._run_step('rereference'))
        self.btn_step_ref.setEnabled(False)
        steps_grid.addWidget(self.btn_step_ref, 1, 1)

        self.btn_step_badch = QPushButton(_("5. 坏导检测"))
        self.btn_step_badch.setMaximumHeight(28)
        self.btn_step_badch.clicked.connect(lambda: self._run_step('badch'))
        self.btn_step_badch.setEnabled(False)
        steps_grid.addWidget(self.btn_step_badch, 2, 0)

        self.btn_step_ica = QPushButton(_("6. ICA"))
        self.btn_step_ica.setMaximumHeight(28)
        self.btn_step_ica.clicked.connect(lambda: self._run_step('ica'))
        self.btn_step_ica.setEnabled(False)
        steps_grid.addWidget(self.btn_step_ica, 2, 1)

        self.btn_step_badseg = QPushButton(_("7. 坏段剔除"))
        self.btn_step_badseg.setMaximumHeight(28)
        self.btn_step_badseg.clicked.connect(lambda: self._run_step('badseg'))
        self.btn_step_badseg.setEnabled(False)
        steps_grid.addWidget(self.btn_step_badseg, 3, 0)

        self.btn_reset_prep = QPushButton(_("重置"))
        self.btn_reset_prep.setObjectName("dangerButton")
        self.btn_reset_prep.setMaximumHeight(28)
        self.btn_reset_prep.clicked.connect(self._reset_preprocessing)
        self.btn_reset_prep.setEnabled(False)
        steps_grid.addWidget(self.btn_reset_prep, 3, 1)

        prep_layout.addLayout(steps_grid)

        left_layout.addWidget(self.prep_group)

        # --- DE Extraction & Save Group ---
        self.de_save_group = QGroupBox(_("DE提取与保存"))
        de_save_layout = QVBoxLayout(self.de_save_group)
        de_save_layout.setSpacing(4)
        de_save_layout.setContentsMargins(6, 8, 6, 6)

        # DE extraction row
        de_row = QHBoxLayout()
        de_row.setSpacing(4)
        self.lbl_extract_de = QLabel(_("提取DE:"))
        de_row.addWidget(self.lbl_extract_de, 0)
        self.btn_extract_de = QPushButton(_("提取"))
        self.btn_extract_de.setObjectName("primaryButton")
        self.btn_extract_de.setMaximumHeight(26)
        self.btn_extract_de.clicked.connect(self._extract_de)
        self.btn_extract_de.setEnabled(False)
        de_row.addWidget(self.btn_extract_de, 1)
        de_row.addStretch(1)
        de_save_layout.addLayout(de_row)

        # Save row: evenly spread
        save_row = QHBoxLayout()
        save_row.setSpacing(4)
        self.lbl_label = QLabel(_("标签:"))
        save_row.addWidget(self.lbl_label, 0)

        self.label_group = QButtonGroup(self)
        self.radio_neg = QRadioButton(_("消极(0)"))
        self.radio_neu = QRadioButton(_("中性(1)"))
        self.radio_pos = QRadioButton(_("积极(2)"))
        self.radio_neu.setChecked(True)
        self.label_group.addButton(self.radio_neg, 0)
        self.label_group.addButton(self.radio_neu, 1)
        self.label_group.addButton(self.radio_pos, 2)

        save_row.addWidget(self.radio_neg, 1)
        save_row.addWidget(self.radio_neu, 1)
        save_row.addWidget(self.radio_pos, 1)

        self.btn_save_pt = QPushButton(_("保存 .pt"))
        self.btn_save_pt.setObjectName("primaryButton")
        self.btn_save_pt.setMaximumHeight(26)
        self.btn_save_pt.clicked.connect(self._save_pt)
        self.btn_save_pt.setEnabled(False)
        save_row.addWidget(self.btn_save_pt, 1)
        de_save_layout.addLayout(save_row)

        left_layout.addWidget(self.de_save_group)

        # --- Progress & Log ---
        self.prog_group = QGroupBox(_("进度与日志"))
        prog_layout = QVBoxLayout(self.prog_group)
        prog_layout.setSpacing(2)
        prog_layout.setContentsMargins(6, 6, 6, 6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumHeight(18)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        prog_layout.addWidget(self.progress_bar)

        self.log_edit = QTextEdit()
        self.log_edit.setMinimumHeight(110)
        self.log_edit.setMaximumHeight(140)
        self.log_edit.setPlaceholderText(_("运行日志..."))
        prog_layout.addWidget(self.log_edit)

        left_layout.addWidget(self.prog_group)

        main_layout.addWidget(left_widget)

        # ==================== Right Panel ====================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # View controls
        controls_layout = QHBoxLayout()
        controls_layout.addStretch()

        self.lbl_view_dur = QLabel(_("显示时长:"))
        controls_layout.addWidget(self.lbl_view_dur)
        self.spin_view_dur = QSpinBox()
        self.spin_view_dur.setRange(1, 60)
        self.spin_view_dur.setValue(10)
        self.spin_view_dur.setSuffix(" s")
        self.spin_view_dur.valueChanged.connect(self._refresh_plot)
        controls_layout.addWidget(self.spin_view_dur)

        self.lbl_view_start = QLabel(_("起始时间:"))
        controls_layout.addWidget(self.lbl_view_start)
        self.spin_view_start = QDoubleSpinBox()
        self.spin_view_start.setRange(0, 99999)
        self.spin_view_start.setValue(0)
        self.spin_view_start.setSuffix(" s")
        self.spin_view_start.valueChanged.connect(self._refresh_plot)
        controls_layout.addWidget(self.spin_view_start)

        self.btn_psd = QPushButton(_("查看 PSD"))
        self.btn_psd.clicked.connect(self._toggle_psd_view)
        controls_layout.addWidget(self.btn_psd)

        right_layout.addLayout(controls_layout)

        # Tabs
        self.right_tabs = QTabWidget()

        self.eeg_plot = EEGPlotWidget()
        self.eeg_plot.view_changed_callback = self._on_plot_view_changed
        self.right_tabs.addTab(self.eeg_plot, _("脑电可视化"))

        self.info_edit = QTextEdit()
        self.info_edit.setReadOnly(True)
        self.info_edit.setPlaceholderText(_("数据信息将显示在这里..."))
        self.right_tabs.addTab(self.info_edit, _("数据信息"))

        right_layout.addWidget(self.right_tabs)
        main_layout.addWidget(right_widget, 1)
    def _apply_styles(self):
        self.setStyleSheet(DARK_STYLESHEET)

    def _log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_edit.append(f"[{timestamp}] {msg}")

    def _browse_cnt(self):
        path = QFileDialog.getOpenFileName(self, _("选择 CNT 文件"), "", _("CNT 文件 (*.cnt);;所有文件 (*)"))[0]
        if path:
            self.line_cnt.setText(path)

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, _("选择输出目录"))
        if path:
            self.line_out.setText(path)

    def _get_cfg(self):
        notch_text = self.combo_notch.currentText()
        if notch_text in ("50 Hz", "50Hz"):
            notch_freqs = [50]
        elif notch_text in ("60 Hz", "60Hz"):
            notch_freqs = [60]
        elif notch_text in ("50+60 Hz", "50+60"):
            notch_freqs = [50, 60]
        else:
            notch_freqs = []

        return {
            'sfreq_target': self.spin_sfreq.value(),
            'l_freq': self.spin_lfreq.value(),
            'h_freq': self.spin_hfreq.value(),
            'notch_freqs': notch_freqs,
            'reference': self.combo_ref.currentText(),
            'run_ica': True,
            'ica_n_components': self.spin_ica.value(),
            'ica_random_state': 42,
            'window_length_sec': self.spin_win_len.value(),
            'window_step_sec': self.spin_win_step.value(),
            'bands': {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 14),
                'beta': (14, 31),
                'gamma': (31, 50),
            },
        }

    def _disable_all_buttons(self):
        self.btn_step_bandpass.setEnabled(False)
        self.btn_step_notch.setEnabled(False)
        self.btn_step_downsample.setEnabled(False)
        self.btn_step_ref.setEnabled(False)
        self.btn_step_badch.setEnabled(False)
        self.btn_step_ica.setEnabled(False)
        self.btn_step_badseg.setEnabled(False)
        self.btn_extract_de.setEnabled(False)
        self.btn_save_pt.setEnabled(False)
        self.btn_reset_prep.setEnabled(False)

    def _enable_step_buttons(self):
        has_data = self.raw_current is not None
        self.btn_step_bandpass.setEnabled(has_data)
        self.btn_step_notch.setEnabled(has_data)
        self.btn_step_downsample.setEnabled(has_data)
        self.btn_step_ref.setEnabled(has_data)
        self.btn_step_badch.setEnabled(has_data)
        self.btn_step_ica.setEnabled(has_data)
        self.btn_step_badseg.setEnabled(has_data)
        self.btn_reset_prep.setEnabled(has_data)
        self.btn_extract_de.setEnabled(has_data)
        self.btn_save_pt.setEnabled(has_data)
        # After ICA fitted, change the ICA button to audit mode (yellow highlight)
        if self.ica_object is not None:
            self.btn_step_ica.setText(_("6. 审核ICA ★"))
            self.btn_step_ica.setStyleSheet("background-color: #f9e2af; color: #1e1e2e; font-weight: bold; border: 2px solid #fab387;")
            try:
                self.btn_step_ica.clicked.disconnect()
            except Exception:
                pass
            self.btn_step_ica.clicked.connect(self._audit_ica)

    def _load_cnt(self):
        path = self.line_cnt.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, _("警告"), _("请先选择一个有效的 .cnt 文件"))
            return

        self._disable_all_buttons()
        self.log_edit.clear()
        self._log("🚀 " + _("开始加载 CNT 文件") + "...")
        self._start_busy(_("读取 CNT 中，请稍候") + "...")

        self.worker = CNTLoaderWorker(path)
        self.worker.log_signal.connect(self._log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.done_signal.connect(self._on_load_done)
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()

    def _on_load_done(self, raw, events_info):
        self.raw_original = raw
        self.raw_current = raw
        self.raw_segment = None
        self.raw_pre_ica = None
        self.ica_object = None
        self.ica_auto_exclude = []
        self.ica_manual_exclude = []
        self.events = events_info
        self.de_tensor = None

        # Update events list
        self.events_list.clear()
        self.paradigm_segments = []
        for i, (ev_time, ev_desc, ev_type) in enumerate(events_info):
            item_text = f"[{i+1:03d}] {ev_desc} ({ev_time:.3f}s)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, ('event', ev_time, ev_desc))
            self.events_list.addItem(item)
        if not events_info:
            self.events_list.addItem(_("未检测到事件标记"))

        # Update info tab
        info_text = (
            f"<h2>📋 {_("数据信息")}</h2>"
            f"<table cellspacing='8'>"
            f"<tr><td><b>{_("通道数")}</b></td><td>{len(raw.ch_names)}</td></tr>"
            f"<tr><td><b>{_("采样率")}</b></td><td>{raw.info['sfreq']:.1f} Hz</td></tr>"
            f"<tr><td><b>{_("总时长")}</b></td><td>{raw.times[-1]:.2f} s</td></tr>"
            f"<tr><td><b>{_("采样点数")}</b></td><td>{raw.n_times}</td></tr>"
            f"<tr><td><b>{_("事件数")}</b></td><td>{len(events_info)}</td></tr>"
            f"<tr><td><b>{_("通道列表")}</b></td><td>{', '.join(raw.info['ch_names'])}</td></tr>"
            f"</table>"
        )
        self.info_edit.setHtml(info_text)

        # Update view controls range
        self.spin_view_start.setRange(0, raw.times[-1])
        self.spin_view_start.setValue(0)

        # Plot
        self.eeg_plot.plot_eeg(raw, duration=self.spin_view_dur.value(), start=0,
                               title=_("EEG 信号"))

        self._stop_busy()
        # Enable paradigm buttons when data loaded
        self.btn_auto_paradigm.setEnabled(self.raw_original is not None)
        self.btn_design_paradigm.setEnabled(self.raw_original is not None)
        # Restore ICA button to original state when loading new data
        self.btn_step_ica.setText(_("6. ICA"))
        self.btn_step_ica.setStyleSheet("")
        try:
            self.btn_step_ica.clicked.disconnect()
        except Exception:
            pass
        self.btn_step_ica.clicked.connect(lambda: self._run_step('ica'))
        self._enable_step_buttons()
        self._log("✅ " + _("CNT 文件加载完成，可以开始预处理或选择事件截取片段"))

    def _on_event_selected(self, item):
        data = item.data(Qt.UserRole)
        if not data:
            return
        kind = data[0]
        if kind == 'event':
            _kind, ev_time, ev_desc = data
            self._log(f"{_('选中事件')}: {ev_desc} ({ev_time:.3f}s)")
            start = max(0, ev_time - 2)
            self.spin_view_start.setValue(start)
            self.eeg_plot.plot_eeg(
                self.raw_current if self.raw_current else self.raw_original,
                duration=self.spin_view_dur.value(),
                start=start,
                title=f"{_('事件')}: {ev_desc} ({ev_time:.2f}s)",
                highlight_events=[(ev_time, ev_desc, '#f38ba8')]
            )
        elif kind == 'trial':
            _kind, idx = data
            seg = self.paradigm_segments[idx]
            tmin, tmax = seg['tmin'], seg['tmax']
            code, label_name, label_id = seg['code'], seg['label_name'], seg['label_id']
            self._log(f"{_('选中片段')}: {code} {label_name} [{tmin:.1f}s ~ {tmax:.1f}s]")
            btn = self.label_group.button(label_id)
            if btn:
                btn.setChecked(True)
            self.raw_current = seg['raw'].copy()
            self.raw_segment = None
            self.raw_pre_ica = None
            self.ica_object = None
            self.ica_auto_exclude = []
            self.ica_manual_exclude = []
            self.de_tensor = None
            dur = int(tmax - tmin)
            self.spin_view_dur.setValue(dur)
            self.spin_view_start.setValue(tmin)
            self.eeg_plot.plot_eeg(
                self.raw_current,
                duration=dur,
                start=0,
                title=f"{code} {label_name} [{tmin:.1f}s ~ {tmax:.1f}s]",
                highlight_events=[(0, code, '#f38ba8')]
            )
            self._enable_step_buttons()
            self.btn_step_ica.setText(_("6. ICA"))
            self.btn_step_ica.setStyleSheet("")
            try:
                self.btn_step_ica.clicked.disconnect()
            except Exception:
                pass
            self.btn_step_ica.clicked.connect(lambda: self._run_step('ica'))

    def _extract_segment(self):
        if self.raw_current is None:
            QMessageBox.warning(self, _("警告"), _("请先加载 CNT 文件"))
            return

        item = self.events_list.currentItem()
        if not item or not item.data(Qt.UserRole):
            QMessageBox.warning(self, _("警告"), _("请先从事件列表中选择一个事件"))
            return

        data = item.data(Qt.UserRole)
        kind = data[0]

        if kind == 'event':
            _kind, ev_time, ev_desc = data
            before = self.spin_seg_before.value()
            after = self.spin_seg_after.value()
            tmin = ev_time - before
            tmax = ev_time + after
            self._log(f"📐 {_('截取片段')}: {ev_desc} [{tmin:.2f}s ~ {tmax:.2f}s]")
        elif kind == 'trial':
            _kind, idx = data
            seg = self.paradigm_segments[idx]
            tmin, tmax = seg['tmin'], seg['tmax']
            code, label_name = seg['code'], seg['label_name']
            self._log(f"📐 {_('截取片段')}: {code} {label_name} [{tmin:.1f}s ~ {tmax:.1f}s]")
            self._log(f"✅ {_('片段截取完成')}: {self.raw_current.n_times} {_('采样点')}, "
                      f"{self.raw_current.times[-1]:.2f}s")
            self._enable_step_buttons()
            return
        else:
            return

        try:
            self.raw_segment = self.raw_current.copy().crop(tmin=tmin, tmax=tmax)
            self.raw_current = self.raw_segment
            self.eeg_plot.plot_eeg(
                self.raw_current,
                duration=self.raw_current.times[-1],
                start=0,
                title=f"{_('截取片段')}: {ev_desc} ({self.raw_current.times[-1]:.2f}s)",
                highlight_events=[(tmin, _("事件"), '#f38ba8')] if tmin > 0 else None
            )
            self._log(f"✅ {_('片段截取完成')}: {self.raw_current.n_times} {_('采样点')}, "
                      f"{self.raw_current.times[-1]:.2f}s")
            self._enable_step_buttons()
        except Exception as e:
            QMessageBox.critical(self, _("错误"), f"{_('截取片段')} {_('失败')}: {str(e)}")

    def _manual_segment(self):
        if self.raw_current is None:
            QMessageBox.warning(self, _("警告"), _("请先加载 CNT 文件"))
            return
        # Use current view range as segment
        start = self.spin_view_start.value()
        duration = self.spin_view_dur.value()
        tmax = min(start + duration, self.raw_current.times[-1])
        self._log(f"📐 {_("手动截取")}: [{start:.2f}s ~ {tmax:.2f}s]")
        try:
            self.raw_segment = self.raw_current.copy().crop(tmin=start, tmax=tmax)
            self.raw_current = self.raw_segment
            self.eeg_plot.plot_eeg(self.raw_current, duration=self.raw_current.times[-1],
                                   start=0, title=f"{_("手动截取片段")} ({self.raw_current.times[-1]:.2f}s)")
            self._log(f"✅ {_("手动截取完成")}: {self.raw_current.n_times} {_("采样点")}")
            self._enable_step_buttons()
        except Exception as e:
            QMessageBox.critical(self, _("错误"), f"{_("截取")} {_("失败")}: {str(e)}")

    def _run_step(self, step_name):
        if self.raw_current is None:
            return
        self._disable_all_buttons()
        self.progress_bar.setValue(0)
        self._log(f"▶️ {_("开始执行")}: {step_name}")

        cfg = self._get_cfg()
        self.worker = PreprocessStepWorker(self.raw_current, step_name, cfg)
        self.worker.log_signal.connect(self._log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.done_signal.connect(lambda raw, ica_obj, ica_excl: self._on_step_done(raw, step_name, ica_obj, ica_excl))
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()

    def _run_all_steps(self):
        if self.raw_current is None:
            return
        self._disable_all_buttons()
        self.progress_bar.setValue(0)
        self._log("🚀 " + _("一键执行全部预处理步骤（自动模式，ICA直接应用）..."))

        cfg = self._get_cfg()
        self.worker = PreprocessAllWorker(self.raw_current, cfg)
        self.worker.log_signal.connect(self._log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.done_signal.connect(lambda raw, ica_obj, ica_excl: self._on_step_done(raw, _("全部步骤"), ica_obj, ica_excl))
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()

    def _on_step_done(self, raw, step_name, ica_obj=None, ica_excl=None):
        self.raw_current = raw
        self.progress_bar.setValue(100)

        if ica_obj is not None:
            self.ica_object = ica_obj
            self.ica_auto_exclude = ica_excl or []
            self.ica_manual_exclude = list(self.ica_auto_exclude)
            self.raw_pre_ica = raw.copy()  # snapshot before ICA application

        title = f"{_('预处理后')} — {step_name}"
        if step_name == 'ica' and self.ica_object is not None:
            title += f" ({_("请审核ICA")})"

        if self.eeg_plot._display_mode == 'psd':
            self.eeg_plot.plot_psd(self.raw_current, title=title)
        else:
            self.eeg_plot.plot_eeg(
                self.raw_current,
                duration=min(self.spin_view_dur.value(), self.raw_current.times[-1]),
                start=0,
                title=title
            )
        self._enable_step_buttons()
        self._log(f"✅ {step_name} {_('完成')}, {_('波形已更新')}")


    def _toggle_language(self):
        """Toggle between Chinese and English."""
        from utils.i18n import get_language, set_language, LANG_EN, LANG_ZH
        current = get_language()
        new_lang = LANG_EN if current == LANG_ZH else LANG_ZH
        set_language(new_lang)
        self._retranslate_ui()
        self._log(f"Language switched to {'English' if new_lang == LANG_EN else 'Chinese'}")

    def _retranslate_ui(self):
        """Update all translatable texts after language switch."""
        self.setWindowTitle(_("CNT EEG 预处理工作站"))
        self.title_label.setText(_("CNT EEG 预处理工作站"))
        # Group boxes
        self.file_group.setTitle(_("文件"))
        self.events_group.setTitle(_("事件与分段"))
        self.param_group.setTitle(_("预处理参数"))
        self.prep_group.setTitle(_("脑电预处理"))
        self.de_save_group.setTitle(_("DE提取与保存"))
        self.prog_group.setTitle(_("进度与日志"))
        # Labels
        self.lbl_cnt.setText(_("CNT:"))
        self.lbl_out.setText(_("输出:"))
        self.lbl_sfreq.setText(_("采样率"))
        self.lbl_lfreq.setText(_("高通滤波"))
        self.lbl_hfreq.setText(_("低通滤波"))
        self.lbl_notch.setText(_("陷波滤波"))
        self.lbl_ref.setText(_("参考电极"))
        self.lbl_ica.setText(_("ICA"))
        self.lbl_win_len.setText(_("DE窗长"))
        self.lbl_win_step.setText(_("DE步长"))
        self.lbl_view_dur.setText(_("显示时长:"))
        self.lbl_view_start.setText(_("起始时间:"))
        self.lbl_seg_before.setText(_("前:"))
        self.lbl_seg_after.setText(_("后:"))
        self.lbl_extract_de.setText(_("提取DE:"))
        self.lbl_label.setText(_("标签:"))
        # Placeholders
        self.line_cnt.setPlaceholderText(_("请选择 .cnt 文件..."))
        self.log_edit.setPlaceholderText(_("运行日志..."))
        self.info_edit.setPlaceholderText(_("数据信息将显示在这里..."))
        # Buttons
        self.btn_browse.setText(_("浏览"))
        self.btn_load.setText(_("加载"))
        self.btn_out.setText(_("浏览"))
        self.btn_auto_paradigm.setText(_("按范式提取"))
        self.btn_design_paradigm.setText(_("设计范式"))
        self.btn_extract_seg.setText(_("截取事件前后片段"))
        self.btn_manual_seg.setText(_("截取当前可视化脑电片段"))
        self.btn_step_bandpass.setText(_("1. 带通滤波"))
        self.btn_step_notch.setText(_("2. 陷波滤波"))
        self.btn_step_downsample.setText(_("3. 降采样"))
        self.btn_step_ref.setText(_("4. 重参考"))
        self.btn_step_badch.setText(_("5. 坏导检测"))
        self.btn_step_badseg.setText(_("7. 坏段剔除"))
        self.btn_reset_prep.setText(_("重置"))
        self.btn_extract_de.setText(_("提取"))
        self.btn_save_pt.setText(_("保存 .pt"))
        self.radio_neg.setText(_("消极(0)"))
        self.radio_neu.setText(_("中性(1)"))
        self.radio_pos.setText(_("积极(2)"))
        # Tabs
        self.right_tabs.setTabText(0, _("脑电可视化"))
        self.right_tabs.setTabText(1, _("数据信息"))
        # Dynamic: PSD toggle
        current_psd = self.btn_psd.text()
        if "PSD" in current_psd or "功率谱" in current_psd:
            self.btn_psd.setText(_("查看 EEG"))
        else:
            self.btn_psd.setText(_("查看 PSD"))
        # Dynamic: ICA button state
        if self.ica_object is not None:
            self.btn_step_ica.setText(_("6. 审核ICA ★"))
        else:
            self.btn_step_ica.setText(_("6. ICA"))
        # Dynamic: language switch button
        from utils.i18n import get_language, LANG_EN, LANG_ZH
        if get_language() == LANG_ZH:
            self.btn_lang.setText("Switch to English")
        else:
            self.btn_lang.setText("切换为中文")

        # Refresh trial labels in events list
        for i in range(self.events_list.count()):
            item = self.events_list.item(i)
            data = item.data(Qt.UserRole)
            if data and data[0] == 'trial':
                idx = data[1]
                if idx < len(self.paradigm_segments):
                    seg = self.paradigm_segments[idx]
                    label_name = self._get_label_name(seg['label_id'])
                    tmin, tmax = seg['tmin'], seg['tmax']
                    code = seg['code']
                    item.setText(f"[R{seg['round_idx']+1}-{code}] {label_name} ({tmin:.1f}s ~ {tmax:.1f}s)")

        # Refresh plot titles for known translatable strings
        if self.raw_current is not None:
            current_title = self.eeg_plot.ax.get_title()
            title_map = {
                "EEG 信号": _("EEG 信号"),
                "EEG Signal": _("EEG 信号"),
                "功率谱密度 (PSD)": _("功率谱密度 (PSD)"),
                "Power Spectral Density (PSD)": _("功率谱密度 (PSD)"),
                "已重置为原始信号": _("已重置为原始信号"),
                "Reset to Raw Signal": _("已重置为原始信号"),
                "ICA 应用后": _("ICA 应用后"),
                "After ICA": _("ICA 应用后"),
            }
            new_title = title_map.get(current_title, current_title)
            if self.eeg_plot._display_mode == 'psd':
                self.eeg_plot.plot_psd(self.raw_current, title=new_title)
            else:
                duration = self.spin_view_dur.value()
                start = self.spin_view_start.value()
                max_start = max(0, self.raw_current.times[-1] - duration)
                start = max(0.0, min(start, max_start))
                self.eeg_plot.plot_eeg(self.raw_current,
                                       duration=duration,
                                       start=start,
                                       title=new_title)

    def _load_paradigm_config(self):
        """Load default paradigm config (can be extended to load from file)."""
        import copy
        return copy.deepcopy(DEFAULT_PARADIGM)

    def _get_label_name(self, label_id):
        names = {0: _("消极"), 1: _("中性"), 2: _("积极")}
        return names.get(label_id, _("未知"))

    def _design_paradigm(self):
        """Open paradigm design dialog."""
        dialog = ParadigmDesignDialog(self.paradigm_config, parent=self)
        if dialog.exec() == QDialog.Accepted:
            self.paradigm_config = dialog.get_paradigm()
            self._log(_("范式配置已更新"))

    def _auto_extract_paradigm(self):
        """Auto-extract trials from all rounds based on paradigm config."""
        if self.raw_original is None:
            QMessageBox.warning(self, _("警告"), _("未加载数据"))
            return
        cfg = self.paradigm_config
        start_marker = str(cfg['start_marker'])
        # Find all round start times from events
        round_starts = []
        for ev_time, ev_desc, ev_type in self.events:
            if ev_desc == start_marker:
                round_starts.append(ev_time)
        if not round_starts:
            self._log(_("未检测到起始标记，跳过自动提取"))
            return

        self._start_busy(_("按范式提取中..."))
        try:
            # Extract trials for each round
            self.paradigm_segments = []
            self.events_list.clear()
            seg_idx = 0
            total_duration = self.raw_original.times[-1]
            total_rounds = len(round_starts)
            for r_idx, r_start in enumerate(round_starts):
                self.progress_bar.setValue(int((r_idx / total_rounds) * 100))
                for t_idx, trial_cfg in enumerate(cfg['trials']):
                    tmin = r_start + trial_cfg['offset']
                    tmax = tmin + trial_cfg['duration']
                    # 鲁棒性：若该 trial 超出数据范围则跳过，且同轮后续 trial 偏移更大，直接中断本轮
                    if tmax > total_duration:
                        self._log(f"Round {r_idx+1} {trial_cfg['name']} {_('跳过')}: "
                                  f"{_('超出数据范围')} ({tmin:.1f}s ~ {tmax:.1f}s > {total_duration:.1f}s)")
                        break
                    try:
                        seg_raw = self.raw_original.copy().crop(tmin=tmin, tmax=tmax)
                    except Exception as e:
                        self._log(f"Round {r_idx+1} {trial_cfg['name']} {_('失败')}: {e}")
                        continue
                    label_id = trial_cfg['label_id']
                    label_name = self._get_label_name(label_id)
                    self.paradigm_segments.append({
                        'round_idx': r_idx, 'trial_idx': t_idx,
                        'round_start': r_start, 'tmin': tmin, 'tmax': tmax,
                        'label_id': label_id, 'label_name': label_name,
                        'code': trial_cfg['name'], 'trial_name': trial_cfg['name'],
                        'raw': seg_raw,
                    })
                    item_text = f"[R{r_idx+1}-{trial_cfg['name']}] {label_name} ({tmin:.1f}s ~ {tmax:.1f}s)"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, ('trial', seg_idx))
                    self.events_list.addItem(item)
                    seg_idx += 1
            self.progress_bar.setValue(100)
            self._log(f"{_('已自动提取')}{len(self.paradigm_segments)}{_('个 trial')}")
            if self.events_list.count() > 0:
                self.events_list.setCurrentRow(0)
                self._on_event_selected(self.events_list.item(0))
        finally:
            self._stop_busy()

    def _audit_ica(self):
        """Open ICA audit dialog for manual component review."""
        if self.ica_object is None or self.raw_current is None:
            QMessageBox.warning(self, _("警告"), _("请先执行 ICA 步骤"))
            return

        # Use raw_pre_ica if available, otherwise current raw (ICA may not have been applied yet)
        raw_for_ica = getattr(self, 'raw_pre_ica', self.raw_current)

        dialog = ICAAuditDialog(self.ica_object, raw_for_ica, self.ica_auto_exclude, parent=self)
        if dialog.exec() == QDialog.Accepted:
            self.ica_manual_exclude = dialog.get_excluded()
            self._apply_ica()
        else:
            self._log(_("ICA 审核已取消"))

    def _apply_ica(self):
        """Apply ICA with manually selected exclusions."""
        if self.ica_object is None or self.raw_current is None:
            return
        try:
            raw_for_ica = getattr(self, 'raw_pre_ica', self.raw_current).copy()
            self.ica_object.exclude = list(self.ica_manual_exclude)
            self.raw_current = self.ica_object.apply(raw_for_ica)
            self._log(f"✅ ICA {_("已应用")}, {_("排除成分")}: {self.ica_manual_exclude}")
            if self.eeg_plot._display_mode == 'psd':
                self.eeg_plot.plot_psd(self.raw_current, title=_("ICA 应用后"))
            else:
                self.eeg_plot.plot_eeg(
                    self.raw_current,
                    duration=min(self.spin_view_dur.value(), self.raw_current.times[-1]),
                    start=0,
                    title=_("ICA 应用后")
                )
        except Exception as e:
            self._log(f"❌ ICA {_("应用失败")}: {str(e)}")
            QMessageBox.critical(self, _("错误"), f"ICA {_("应用失败")}: {str(e)}")

    def _reset_preprocessing(self):
        if self.raw_original is None:
            return
        self.raw_current = self.raw_original.copy()
        self.raw_segment = None
        self.raw_pre_ica = None
        self.ica_object = None
        self.ica_auto_exclude = []
        self.ica_manual_exclude = []
        self.de_tensor = None
        if self.eeg_plot._display_mode == 'psd':
            self.eeg_plot.plot_psd(self.raw_current, title=_("已重置为原始信号"))
        else:
            self.eeg_plot.plot_eeg(self.raw_current, duration=self.spin_view_dur.value(),
                                   start=0, title=_("已重置为原始信号"))
        self._log("🔄 " + _("已重置为原始加载数据"))
        # Restore ICA button to original state
        self.btn_step_ica.setText(_("6. ICA"))
        self.btn_step_ica.setStyleSheet("")
        try:
            self.btn_step_ica.clicked.disconnect()
        except Exception:
            pass
        self.btn_step_ica.clicked.connect(lambda: self._run_step('ica'))
        self._enable_step_buttons()

    def _extract_de(self):
        if self.raw_current is None:
            return
        self._disable_all_buttons()
        self.progress_bar.setValue(0)
        self._log("🔬 " + _("开始提取 DE 特征") + "...")

        cfg = self._get_cfg()
        self.worker = DEExtractWorker(self.raw_current, cfg)
        self.worker.log_signal.connect(self._log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.done_signal.connect(self._on_de_done)
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()

    def _on_de_done(self, de_tensor, band_names):
        self.de_tensor = de_tensor
        self.de_band_names = band_names
        self.progress_bar.setValue(100)

        info_text = (
            f"<h2>📊 {_("DE 特征提取完成")}</h2>"
            f"<table cellspacing='8'>"
            f"<tr><td><b>{_("张量形状")}</b></td><td>{tuple(de_tensor.shape)}</td></tr>"
            f"<tr><td><b>通道数</b></td><td>{de_tensor.shape[0]}</td></tr>"
            f"<tr><td><b>时间窗数</b></td><td>{de_tensor.shape[1]}</td></tr>"
            f"<tr><td><b>频段</b></td><td>{', '.join(band_names)}</td></tr>"
            f"</table>"
        )
        self.info_edit.setHtml(info_text)
        self.right_tabs.setCurrentIndex(1)  # Switch to info tab

        self._enable_step_buttons()
        self._log("✅ " + _("DE 特征提取完成"))
        QMessageBox.information(self, _("完成"), f"DE {_("特征提取完成")}！\n{_("张量形状")}: {tuple(de_tensor.shape)}")

    def _save_pt(self):
        if self.raw_current is None:
            QMessageBox.warning(self, _("警告"), _("没有可保存的数据"))
            return

        out_dir = self.line_out.text().strip()
        os.makedirs(out_dir, exist_ok=True)

        label_id = self.label_group.checkedId()
        label_names = {0: "negative", 1: "neutral", 2: "positive"}
        label_name = label_names[label_id]

        default_name = f"eeg_segment_{label_name}.pt"
        path = QFileDialog.getSaveFileName(
            self, _("保存 .pt 文件"),
            os.path.join(out_dir, default_name),
            "PyTorch Files (*.pt)"
        )[0]
        if not path:
            return

        try:
            data = torch.from_numpy(self.raw_current.get_data()).float()
            sfreq = self.raw_current.info['sfreq']
            ch_names = self.raw_current.info['ch_names']

            save_dict = {
                'eeg_data': data,               # (n_channels, n_times)
                'label': label_id,              # 0=negative, 1=neutral, 2=positive
                'label_name': label_name,
                'sfreq': sfreq,
                'ch_names': ch_names,
            }

            if self.de_tensor is not None:
                save_dict['de_features'] = self.de_tensor
                save_dict['de_band_names'] = self.de_band_names

            torch.save(save_dict, path)
            self._log(f"💾 {_("已保存")}: {path}")
            self._log(f"   {_("标签")}: {label_name} ({label_id})")
            self._log(f"   {_("EEG 形状")}: {tuple(data.shape)}")
            if self.de_tensor is not None:
                self._log(f"   {_("DE 形状")}: {tuple(self.de_tensor.shape)}")

            QMessageBox.information(self, _("保存成功"),
                                    f"{_("数据已成功保存至")}:\n{path}\n\n"
                                    f"{_("标签")}: {label_name} ({label_id})")

        except Exception as e:
            QMessageBox.critical(self, _("保存失败"), str(e))

    def _on_plot_view_changed(self, duration, start):
        """Sync spinboxes when plot view changes via mouse wheel."""
        self.spin_view_dur.blockSignals(True)
        self.spin_view_start.blockSignals(True)
        self.spin_view_dur.setValue(int(duration))
        self.spin_view_start.setValue(start)
        self.spin_view_dur.blockSignals(False)
        self.spin_view_start.blockSignals(False)

    def _refresh_plot(self):
        if self.raw_current is None:
            return
        duration = self.spin_view_dur.value()
        start = self.spin_view_start.value()
        max_start = max(0, self.raw_current.times[-1] - duration)
        start = max(0.0, min(start, max_start))
        current_title = self.eeg_plot.ax.get_title()
        if self.eeg_plot._display_mode == 'psd':
            self.eeg_plot.plot_psd(self.raw_current, title=current_title or _("功率谱密度 (PSD)"))
        else:
            self.eeg_plot.plot_eeg(self.raw_current, duration=duration, start=start,
                                   title=current_title or _("EEG 信号"))

    def _toggle_psd_view(self):
        if self.raw_current is None:
            return
        if "PSD" in self.btn_psd.text() or "功率谱" in self.btn_psd.text():
            self.eeg_plot.plot_psd(self.raw_current, title=_("功率谱密度 (PSD)"))
            self.btn_psd.setText(_("查看 EEG"))
        else:
            duration = self.spin_view_dur.value()
            start = self.spin_view_start.value()
            self.eeg_plot.plot_eeg(self.raw_current, duration=duration, start=start,
                                   title=_("EEG 信号"))
            self.btn_psd.setText(_("查看 PSD"))

    def _on_error(self, msg):
        self._log(f"❌ [{_('[错误]')}] {msg}")
        self._stop_busy()
        self._enable_step_buttons()
        QMessageBox.critical(self, _("错误"), msg)
