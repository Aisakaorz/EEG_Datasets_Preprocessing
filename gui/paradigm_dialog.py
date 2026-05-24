"""Paradigm design dialog for multi-round EEG trial extraction."""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTableWidget, QTableWidgetItem, QComboBox, QPushButton,
    QLabel, QSpinBox, QDoubleSpinBox, QMessageBox, QAbstractItemView,
)
from PySide6.QtCore import Qt
from utils.i18n import _

# Default SEED-like 288s paradigm: 6 trials per round
DEFAULT_PARADIGM = {
    'start_marker': 80,
    'end_marker': 88,
    'trials': [
        {'name': 'Pos-1', 'label_id': 2, 'label_name_zh': '积极', 'offset': 18, 'duration': 30},
        {'name': 'Pos-2', 'label_id': 2, 'label_name_zh': '积极', 'offset': 63, 'duration': 30},
        {'name': 'Neu-1', 'label_id': 1, 'label_name_zh': '中性', 'offset': 108, 'duration': 30},
        {'name': 'Neu-2', 'label_id': 1, 'label_name_zh': '中性', 'offset': 153, 'duration': 30},
        {'name': 'Neg-1', 'label_id': 0, 'label_name_zh': '消极', 'offset': 198, 'duration': 30},
        {'name': 'Neg-2', 'label_id': 0, 'label_name_zh': '消极', 'offset': 243, 'duration': 30},
    ]
}

def get_label_options():
    return [
        (_("消极(0)"), 0),
        (_("中性(1)"), 1),
        (_("积极(2)"), 2),
    ]


class ParadigmDesignDialog(QDialog):
    """Dialog to design/edit the paradigm trial extraction schema."""

    def __init__(self, paradigm_config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(_("范式设计"))
        self.resize(560, 420)
        self._config = self._deep_copy(paradigm_config or DEFAULT_PARADIGM)
        self._build_ui()
        self._load_config()

    def _deep_copy(self, cfg):
        import copy
        return copy.deepcopy(cfg)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Top: markers
        form = QFormLayout()
        self.spin_start_marker = QSpinBox()
        self.spin_start_marker.setRange(0, 999)
        self.spin_start_marker.setValue(self._config.get('start_marker', 80))
        form.addRow(_("起始标记:"), self.spin_start_marker)

        self.spin_end_marker = QSpinBox()
        self.spin_end_marker.setRange(0, 999)
        self.spin_end_marker.setValue(self._config.get('end_marker', 88))
        form.addRow(_("结束标记:"), self.spin_end_marker)
        layout.addLayout(form)

        # Table
        layout.addWidget(QLabel(f"<b>{_('Trial 截取规范')}</b>"))
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([_("Trial名称"), _("情绪标签"), _("起始偏移(s)"), _("持续时长(s)")])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setCornerButtonEnabled(False)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        layout.addWidget(self.table)

        # Table controls
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton(_("+ 添加"))
        self.btn_add.clicked.connect(self._add_trial)
        self.btn_del = QPushButton(_("- 删除"))
        self.btn_del.clicked.connect(self._del_trial)
        self.btn_up = QPushButton(_("↑ 上移"))
        self.btn_up.clicked.connect(self._move_up)
        self.btn_down = QPushButton(_("↓ 下移"))
        self.btn_down.clicked.connect(self._move_down)
        self.btn_default = QPushButton(_("恢复默认"))
        self.btn_default.clicked.connect(self._restore_default)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_del)
        btn_layout.addWidget(self.btn_up)
        btn_layout.addWidget(self.btn_down)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_default)
        layout.addLayout(btn_layout)

        # Dialog buttons
        dlg_btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton(_("确认"))
        self.btn_ok.setObjectName("primaryButton")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton(_("取消"))
        self.btn_cancel.clicked.connect(self.reject)
        dlg_btn_layout.addStretch()
        dlg_btn_layout.addWidget(self.btn_ok)
        dlg_btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(dlg_btn_layout)

    def _load_config(self):
        trials = self._config.get('trials', [])
        self.table.setRowCount(len(trials))
        for i, t in enumerate(trials):
            self._set_row(i, t)
        self.table.resizeColumnsToContents()
        # Ensure emotion label column is wide enough for translated text
        self.table.setColumnWidth(1, max(self.table.columnWidth(1), 110))

    def _set_row(self, row, t):
        # Name
        item_name = QTableWidgetItem(t.get('name', ''))
        self.table.setItem(row, 0, item_name)
        # Label combo (disable wheel to prevent accidental changes)
        combo = QComboBox()
        combo.setFocusPolicy(Qt.StrongFocus)
        combo.wheelEvent = lambda event: event.ignore()
        combo.setProperty('_row', row)
        for txt, val in get_label_options():
            combo.addItem(txt, val)
        combo.setCurrentIndex(t.get('label_id', 1))
        self.table.setCellWidget(row, 1, combo)
        # Offset
        item_off = QTableWidgetItem(str(t.get('offset', 0)))
        self.table.setItem(row, 2, item_off)
        # Duration
        item_dur = QTableWidgetItem(str(t.get('duration', 30)))
        self.table.setItem(row, 3, item_dur)

    def _get_row_data(self, row):
        name = self.table.item(row, 0).text() if self.table.item(row, 0) else ''
        combo = self.table.cellWidget(row, 1)
        label_id = combo.currentData() if combo else 1
        offset = float(self.table.item(row, 2).text()) if self.table.item(row, 2) else 0
        duration = float(self.table.item(row, 3).text()) if self.table.item(row, 3) else 30
        return {'name': name, 'label_id': label_id, 'offset': offset, 'duration': duration}

    def _add_trial(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._set_row(row, {'name': f'Trial-{row+1}', 'label_id': 1, 'offset': 0, 'duration': 30})
        self.table.selectRow(row)

    def _del_trial(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def _move_up(self):
        row = self.table.currentRow()
        if row > 0:
            data = self._get_row_data(row)
            self.table.removeRow(row)
            self.table.insertRow(row - 1)
            self._set_row(row - 1, data)
            self.table.selectRow(row - 1)

    def _move_down(self):
        row = self.table.currentRow()
        if row >= 0 and row < self.table.rowCount() - 1:
            data = self._get_row_data(row)
            self.table.removeRow(row)
            self.table.insertRow(row + 1)
            self._set_row(row + 1, data)
            self.table.selectRow(row + 1)

    def _restore_default(self):
        reply = QMessageBox.question(self, _("恢复默认"), _("确定恢复为默认 288s SEED 范式？"))
        if reply == QMessageBox.Yes:
            self._config = self._deep_copy(DEFAULT_PARADIGM)
            self.spin_start_marker.setValue(self._config['start_marker'])
            self.spin_end_marker.setValue(self._config['end_marker'])
            self._load_config()

    def get_paradigm(self):
        """Return the configured paradigm dict."""
        trials = []
        for row in range(self.table.rowCount()):
            trials.append(self._get_row_data(row))
        return {
            'start_marker': self.spin_start_marker.value(),
            'end_marker': self.spin_end_marker.value(),
            'trials': trials,
        }
