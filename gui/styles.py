DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: bold;
    color: #89b4fa;
    font-size: 12px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}

QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 5px 12px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #45475a;
    border-color: #585b70;
}

QPushButton:pressed {
    background-color: #585b70;
}

QPushButton:disabled {
    background-color: #181825;
    color: #6c7086;
    border-color: #313244;
}

QPushButton#primaryButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}

QPushButton#primaryButton:hover {
    background-color: #b4befe;
}

QPushButton#primaryButton:pressed {
    background-color: #74c7ec;
}

QPushButton#successButton {
    background-color: #a6e3a1;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}

QPushButton#successButton:hover {
    background-color: #81c8be;
}

QPushButton#warningButton {
    background-color: #f9e2af;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}

QPushButton#warningButton:hover {
    background-color: #fab387;
}

QPushButton#dangerButton {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}

QPushButton#dangerButton:hover {
    background-color: #eba0ac;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px 5px;
    font-size: 12px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #89b4fa;
}

QListWidget {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px;
    font-size: 12px;
}

QListWidget::item:selected {
    background-color: #585b70;
    color: #cdd6f4;
    border-radius: 4px;
}

QListWidget::item:hover {
    background-color: #45475a;
}

QTableWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    gridline-color: #45475a;
}

QTableWidget::item {
    background-color: #1e1e2e;
    color: #cdd6f4;
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}

QHeaderView::section {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    padding: 5px;
    font-weight: bold;
}

QTableCornerButton::section {
    background-color: #313244;
    border: 1px solid #45475a;
}

QProgressBar {
    border: 1px solid #45475a;
    border-radius: 6px;
    text-align: center;
    color: #cdd6f4;
    background-color: #313244;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #89b4fa, stop:1 #b4befe);
    border-radius: 5px;
}

QTextEdit {
    background-color: #181825;
    color: #a6e3a1;
    border: 1px solid #313244;
    border-radius: 5px;
    padding: 5px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}

QRadioButton, QCheckBox {
    color: #cdd6f4;
    spacing: 6px;
}

QLabel#titleLabel {
    font-size: 18px;
    font-weight: bold;
    color: #89b4fa;
}

QLabel#subtitleLabel {
    font-size: 12px;
    color: #6c7086;
}

QSplitter::handle {
    background-color: #45475a;
}

QScrollArea {
    border: none;
}

QSlider::groove:horizontal {
    height: 6px;
    background-color: #313244;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #89b4fa;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::sub-page:horizontal {
    background-color: #585b70;
    border-radius: 3px;
}

QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 6px;
    background-color: #1e1e2e;
}

QTabBar::tab {
    background-color: #313244;
    color: #cdd6f4;
    padding: 8px 16px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #45475a;
}
"""
