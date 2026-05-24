# nuamps-eeg-preproc

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-1.5+-green.svg)](https://mne.tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **NeuroScan NuAmps (32ch) EEG Preprocessing & DE Feature Extraction Workstation**

A PySide6-based GUI application for preprocessing EEG data recorded with NeuroScan NuAmps 32-channel amplifier and extracting Differential Entropy (DE) features.

---

## ✨ Features

| Module | Description |
|--------|-------------|
| **CNT Loader** | Load `.cnt` files with automatic event/annotation extraction |
| **Interactive Visualization** | Real-time EEG waveform & PSD plotting with event highlighting and mouse-wheel zoom/pan |
| **Step-by-Step Preprocessing** | Bandpass → Notch → Downsample → Re-reference → Bad Channel → ICA → Bad Segment |
| **ICA Audit** | Visual ICA component review with automatic EOG/muscle artifact detection |
| **Paradigm Auto-Extraction** | Design custom trial schemas and auto-extract trials across multiple rounds with robust bounds checking |
| **DE Extraction** | Band-wise Differential Entropy feature computation with sliding windows |
| **Export** | Save preprocessed data & DE tensors as PyTorch `.pt` files with emotion labels |
| **Bilingual UI** | Full Chinese / English interface support |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Aisakaorz/nuamps-eeg-preproc.git
cd nuamps-eeg-preproc
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install mne matplotlib numpy scipy torch PySide6
```

> **Note:** This project is developed and tested with Python 3.10+ and MNE-Python 1.5+.

---

## 🚀 Quick Start

Launch the GUI application:

```bash
python main.py
```

### Typical Workflow

1. **Load CNT** — Select your `.cnt` file and click **加载** (Load).
2. **Preprocess** — Execute preprocessing steps one by one or run the full pipeline:
   - **Bandpass Filter** (High-pass / Low-pass)
   - **Notch Filter** (50 / 60 / 50+60 Hz)
   - **Downsample**
   - **Re-reference** (Average / Cz / Fp1)
   - **Bad Channel Detection & Interpolation**
   - **ICA + Artifact Audit**
   - **Bad Segment Rejection**
3. **Segment by Paradigm** — Click **按范式提取** (Auto Extract) to slice trials automatically based on your custom paradigm configuration. Incomplete rounds are gracefully skipped.
4. **Extract DE** — Click **提取** (Extract) to compute Differential Entropy features.
5. **Save** — Choose the emotion label and save as `.pt`.

### Visualization Shortcuts

| Action | Control |
|--------|---------|
| Amplitude scale up/down | Mouse wheel |
| Time zoom in/out | Ctrl + Mouse wheel |
| Time pan left/right | Alt + Mouse wheel |
| Toggle PSD / EEG view | Click **查看 PSD** button |

---

## 🖥️ Device Setup (Tested Configuration)

| Item | Specification |
|------|---------------|
| Amplifier | NeuroScan NuAmps (32-channel) |
| Cap | 10-20 system 32-channel EEG cap |
| Reference | Average Reference |
| Sampling Rate | 1000 Hz |
| Recording Software | Curry 8 |

The tool is built around `mne.io.read_raw_cnt`, so other CNT-compatible devices may also work.

---

## 📁 Project Structure

```
nuamps-eeg-preproc/
├── main.py                      # GUI entry point
├── gui/
│   ├── main_window.py           # Main application window
│   ├── paradigm_dialog.py       # Paradigm designer dialog
│   ├── ica_audit_dialog.py      # ICA component review dialog
│   ├── plot_widget.py           # EEG/PSD plotting widget
│   └── styles.py                # Dark theme stylesheet
├── workers/
│   ├── loader.py                # CNT loading worker (QThread)
│   ├── preprocess.py            # Preprocessing workers (QThread)
│   └── de_extract.py            # DE extraction worker (QThread)
├── utils/
│   ├── i18n.py                  # Internationalization (zh/en)
│   └── compat.py                # MNE compatibility helpers
├── scripts/                           # Public dataset DE feature extraction (not for .cnt)
│   ├── extract_seed_de_features.py    # SEED dataset
│   └── extract_seed_iv_de_features.py # SEED-IV dataset
├── data/
│   ├── raw/                     # Place raw .cnt files here
│   └── features/                # Extracted features output
└── LICENSE                      # MIT License
```

---

## 🧪 Paradigm Support

The built-in default paradigm follows the protocol (6 trials per round × 30s emotion clips). You can customize:

- Start / end marker codes
- Trial names, emotion labels, offsets, and durations
- Number of trials per round

Open **设计范式** (Design Paradigm) to edit the schema, then click **按范式提取** to batch-extract all valid trials across rounds. Incomplete rounds (where recording stopped mid-trial) are gracefully skipped.

---

## ⚙️ DE Feature Parameters

Default frequency bands for DE extraction:

| Band | Range (Hz) |
|------|------------|
| Delta | 1 – 4 |
| Theta | 4 – 8 |
| Alpha | 8 – 14 |
| Beta | 14 – 31 |
| Gamma | 31 – 50 |

Window parameters (configurable in GUI):
- **Window Length:** 1.0 s
- **Window Step:** 1.0 s

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [MNE-Python](https://mne.tools/) — Open-source Python package for exploring, visualizing, and analyzing human neurophysiological data
- [PySide6](https://wiki.qt.io/Qt_for_Python) — Official Python bindings for Qt6
