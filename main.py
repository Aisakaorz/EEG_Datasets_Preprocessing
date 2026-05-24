#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNT EEG Preprocessing GUI — Entry Point
========================================
Usage:
    python main.py

Dependencies:
    pip install mne matplotlib numpy scipy torch PySide6
"""

import sys

import matplotlib
matplotlib.use('QtAgg')

from PySide6.QtWidgets import QApplication
from gui.styles import DARK_STYLESHEET
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
