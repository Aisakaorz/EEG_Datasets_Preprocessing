import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
import mne


class DEExtractWorker(QThread):
    """Background worker for DE feature extraction."""
    log_signal = Signal(str)
    progress_signal = Signal(int)
    done_signal = Signal(object, list)
    error_signal = Signal(str)

    def __init__(self, raw, cfg):
        super().__init__()
        self.raw = raw.copy()
        self.cfg = cfg

    def run(self):
        try:
            sfreq = self.raw.info['sfreq']
            data = self.raw.get_data()
            n_channels, n_times = data.shape

            win_len = int(self.cfg['window_length_sec'] * sfreq)
            win_step = int(self.cfg['window_step_sec'] * sfreq)
            band_names = list(self.cfg['bands'].keys())
            n_bands = len(band_names)
            n_windows = (n_times - win_len) // win_step + 1

            if n_windows <= 0:
                raise ValueError("窗口长度大于信号长度，请检查数据时长或窗口参数")

            self.log_signal.emit(f"[DE] 通道={n_channels}, 窗口={n_windows}, 频段={n_bands}")
            self.progress_signal.emit(5)

            de_tensor = torch.zeros(n_channels, n_windows, n_bands)

            for band_idx, (band_name, (l_freq, h_freq)) in enumerate(self.cfg['bands'].items()):
                self.log_signal.emit(f"[DE] 处理频段 {band_name} ({l_freq}-{h_freq} Hz) ...")
                raw_band = self.raw.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
                band_data = raw_band.get_data()

                for w_idx in range(n_windows):
                    start = w_idx * win_step
                    end = start + win_len
                    window = band_data[:, start:end]
                    var = np.var(window, axis=1, ddof=1)
                    var = np.maximum(var, 1e-10)
                    de = np.log(2 * np.pi * np.e * var) / 2.0
                    de_tensor[:, w_idx, band_idx] = torch.from_numpy(de).float()

                progress = int(5 + (band_idx + 1) / n_bands * 90)
                self.progress_signal.emit(progress)

            self.progress_signal.emit(100)
            self.log_signal.emit(f"[DE] 特征提取完成，张量形状: {de_tensor.shape}")
            self.done_signal.emit(de_tensor, band_names)

        except Exception as e:
            self.error_signal.emit(str(e))
