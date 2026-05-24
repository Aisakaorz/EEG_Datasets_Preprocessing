import numpy as np
import mne
from PySide6.QtCore import QThread, Signal
from utils.compat import _drop_by_annotations_compat


class PreprocessStepWorker(QThread):
    """Background worker for a single preprocessing step."""
    log_signal = Signal(str)
    progress_signal = Signal(int)
    done_signal = Signal(object, object, object)   # raw_after, ica_obj, ica_exclude
    error_signal = Signal(str)

    def __init__(self, raw, step_name, cfg):
        super().__init__()
        self.raw = raw.copy()
        self.step_name = step_name
        self.cfg = cfg

    def _detect_bad_channels(self, raw, z_threshold=3.0):
        """Detect bad channels based on variance and correlation."""
        data = raw.get_data()
        n_channels = data.shape[0]
        bad_chs = []

        # Method 1: variance-based outlier detection
        variances = np.var(data, axis=1)
        if np.std(variances) > 0:
            z_var = np.abs((variances - np.mean(variances)) / np.std(variances))
            bad_var = [raw.ch_names[i] for i in range(n_channels) if z_var[i] > z_threshold]
            if bad_var:
                self.log_signal.emit(f"[坏导] 方差异常: {bad_var}")
                bad_chs.extend(bad_var)

        # Method 2: low correlation with other channels
        corrs = np.abs(np.corrcoef(data))
        np.fill_diagonal(corrs, 0)
        mean_corr = np.mean(corrs, axis=1)
        if np.std(mean_corr) > 0:
            z_corr = np.abs((mean_corr - np.mean(mean_corr)) / np.std(mean_corr))
            bad_corr = [raw.ch_names[i] for i in range(n_channels) if z_corr[i] > z_threshold]
            if bad_corr:
                self.log_signal.emit(f"[坏导] 相关性异常: {bad_corr}")
                bad_chs.extend(bad_corr)

        return sorted(set(bad_chs))

    def _reject_bad_segments(self, raw, threshold_uv=200.0, min_duration_sec=0.5):
        """Reject segments with amplitude exceeding threshold."""
        threshold = threshold_uv * 1e-6  # volts
        sfreq = raw.info['sfreq']
        data = raw.get_data()
        min_samples = max(1, int(min_duration_sec * sfreq))

        bad_mask = np.any(np.abs(data) > threshold, axis=0)
        bad_idx = np.where(bad_mask)[0]

        if len(bad_idx) == 0:
            self.log_signal.emit("[坏段] 未发现异常幅度段")
            return raw, []

        # Find contiguous segments
        segments = []
        seg_start = bad_idx[0]
        prev = bad_idx[0]
        for idx in bad_idx[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                if prev - seg_start + 1 >= min_samples:
                    segments.append((seg_start / sfreq, (prev + 1) / sfreq))
                seg_start = idx
                prev = idx
        if prev - seg_start + 1 >= min_samples:
            segments.append((seg_start / sfreq, (prev + 1) / sfreq))

        if not segments:
            self.log_signal.emit("[坏段] 异常幅度段太短，忽略")
            return raw, []

        # Create annotations and drop
        from mne import Annotations
        onset = [s[0] for s in segments]
        duration = [s[1] - s[0] for s in segments]
        raw.set_annotations(Annotations(onset=onset, duration=duration, description=['BAD'] * len(segments)))
        n_before = raw.n_times
        raw = _drop_by_annotations_compat(raw)
        n_dropped = n_before - raw.n_times
        self.log_signal.emit(f"[坏段] 剔除 {len(segments)} 段异常信号 ({n_dropped} 采样点)")
        return raw, segments

    def run(self):
        try:
            self.progress_signal.emit(10)
            ica_obj = None
            ica_exclude = []

            if self.step_name == 'bandpass':
                l_freq = self.cfg.get('l_freq', 0.1)
                h_freq = self.cfg.get('h_freq', 50.0)
                self.log_signal.emit(f"[带通滤波] {l_freq}-{h_freq} Hz")
                self.raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

            elif self.step_name == 'notch':
                freqs = self.cfg.get('notch_freqs', [50])
                if freqs:
                    self.log_signal.emit(f"[陷波滤波] {freqs} Hz")
                    self.raw.notch_filter(freqs=freqs, fir_design='firwin')
                else:
                    self.log_signal.emit("[陷波滤波] 已跳过")

            elif self.step_name == 'downsample':
                target = self.cfg.get('sfreq_target', 200)
                if target < self.raw.info['sfreq']:
                    self.log_signal.emit(f"[降采样] {self.raw.info['sfreq']:.0f} Hz → {target} Hz")
                    self.raw.resample(target, npad='auto')
                else:
                    self.log_signal.emit("[降采样] 目标采样率≥原始采样率，跳过")

            elif self.step_name == 'rereference':
                ref = self.cfg.get('reference', 'average')
                if ref == 'average':
                    self.log_signal.emit("[重参考] 平均参考")
                    self.raw.set_eeg_reference('average', projection=True)
                    self.raw.apply_proj()
                else:
                    self.log_signal.emit(f"[重参考] 参考电极: {ref}")
                    self.raw.set_eeg_reference([ref])

            elif self.step_name == 'badch':
                self.log_signal.emit("[坏导检测] 分析通道质量...")
                bad_chs = self._detect_bad_channels(self.raw)
                if bad_chs:
                    self.raw.info['bads'] = bad_chs
                    self.log_signal.emit(f"[坏导检测] 检测到 {len(bad_chs)} 个坏导: {bad_chs}")
                    self.log_signal.emit("[坏导检测] 正在进行插值...")
                    self.raw = self.raw.interpolate_bads()
                    self.log_signal.emit("[坏导检测] 插值完成")
                else:
                    self.log_signal.emit("[坏导检测] 未发现坏导")

            elif self.step_name == 'ica':
                n_comp = self.cfg.get('ica_n_components', 20)
                self.log_signal.emit(f"[ICA] 拟合 n_components={n_comp} ...")
                ica_obj = mne.preprocessing.ICA(
                    n_components=n_comp,
                    random_state=self.cfg.get('ica_random_state', 42),
                    max_iter='auto'
                )
                ica_obj.fit(self.raw)

                # EOG detection (robust: try auto-detect EOG channels first)
                try:
                    eog_indices = ica_obj.find_bads_eog(self.raw)[0]
                except Exception:
                    try:
                        eog_indices = ica_obj.find_bads_eog(self.raw, ch_name=self.raw.info['ch_names'][0])[0]
                    except Exception:
                        eog_indices = []
                # Muscle detection (robust: may raise ValueError on some data)
                try:
                    mus_indices = ica_obj.find_bads_muscle(self.raw)[0]
                except Exception:
                    mus_indices = []
                # Convert numpy int to plain Python int for clean display
                eog_indices = [int(x) for x in eog_indices]
                mus_indices = [int(x) for x in mus_indices]
                ica_exclude = sorted(set(eog_indices + mus_indices))
                # Safety: if muscle detection flags too many components, trust EOG only
                if len(ica_exclude) > ica_obj.n_components_ * 0.5:
                    self.log_signal.emit(f"[ICA] 警告: 检测到 {len(ica_exclude)} 个伪迹成分（超过50%），可能存在误报。仅保留眼动检测结果。")
                    ica_exclude = list(eog_indices)
                ica_obj.exclude = list(ica_exclude)

                if ica_exclude:
                    self.log_signal.emit(f"[ICA] 拟合完成，自动检测到伪迹成分: {ica_exclude} (EOG={eog_indices}, Muscle={mus_indices})")
                    self.log_signal.emit("[ICA] 请点击「审核ICA」确认或调整要排除的成分")
                else:
                    self.log_signal.emit("[ICA] 拟合完成，未检测到明显伪迹成分")

            elif self.step_name == 'badseg':
                self.raw, segments = self._reject_bad_segments(self.raw)
                if not segments:
                    self.log_signal.emit("[坏段剔除] 无坏段需要剔除")

            self.progress_signal.emit(100)
            self.log_signal.emit(f"✅ {self.step_name} 完成")
            self.done_signal.emit(self.raw, ica_obj, ica_exclude)

        except Exception as e:
            self.error_signal.emit(str(e))


class PreprocessAllWorker(QThread):
    """Run all preprocessing steps at once (auto mode, no manual ICA audit)."""
    log_signal = Signal(str)
    progress_signal = Signal(int)
    done_signal = Signal(object, object, object)
    error_signal = Signal(str)

    def __init__(self, raw, cfg):
        super().__init__()
        self.raw = raw.copy()
        self.cfg = cfg

    def _detect_bad_channels(self, raw, z_threshold=3.0):
        data = raw.get_data()
        n_channels = data.shape[0]
        bad_chs = []
        variances = np.var(data, axis=1)
        if np.std(variances) > 0:
            z_var = np.abs((variances - np.mean(variances)) / np.std(variances))
            bad_chs.extend([raw.ch_names[i] for i in range(n_channels) if z_var[i] > z_threshold])
        corrs = np.abs(np.corrcoef(data))
        np.fill_diagonal(corrs, 0)
        mean_corr = np.mean(corrs, axis=1)
        if np.std(mean_corr) > 0:
            z_corr = np.abs((mean_corr - np.mean(mean_corr)) / np.std(mean_corr))
            bad_chs.extend([raw.ch_names[i] for i in range(n_channels) if z_corr[i] > z_threshold])
        return sorted(set(bad_chs))

    def run(self):
        try:
            steps = [
                ('bandpass', 12),
                ('notch', 24),
                ('downsample', 36),
                ('rereference', 48),
                ('badch', 60),
                ('ica', 78),
                ('badseg', 95),
            ]
            ica_obj = None
            ica_exclude = []
            for step_name, prog in steps:
                self.progress_signal.emit(prog)
                if step_name == 'bandpass':
                    l, h = self.cfg.get('l_freq', 0.1), self.cfg.get('h_freq', 50.0)
                    self.log_signal.emit(f"[带通] {l}-{h} Hz")
                    self.raw.filter(l_freq=l, h_freq=h, fir_design='firwin')
                elif step_name == 'notch':
                    freqs = self.cfg.get('notch_freqs', [50])
                    if freqs:
                        self.log_signal.emit(f"[陷波] {freqs} Hz")
                        self.raw.notch_filter(freqs=freqs, fir_design='firwin')
                elif step_name == 'downsample':
                    target = self.cfg.get('sfreq_target', 200)
                    if target < self.raw.info['sfreq']:
                        self.log_signal.emit(f"[降采样] {self.raw.info['sfreq']:.0f} → {target} Hz")
                        self.raw.resample(target, npad='auto')
                elif step_name == 'rereference':
                    ref = self.cfg.get('reference', 'average')
                    if ref == 'average':
                        self.log_signal.emit("[重参考] 平均参考")
                        self.raw.set_eeg_reference('average', projection=True)
                        self.raw.apply_proj()
                    else:
                        self.raw.set_eeg_reference([ref])
                elif step_name == 'badch':
                    bad_chs = self._detect_bad_channels(self.raw)
                    if bad_chs:
                        self.raw.info['bads'] = bad_chs
                        self.log_signal.emit(f"[坏导] 检测: {bad_chs}")
                        self.raw = self.raw.interpolate_bads()
                    else:
                        self.log_signal.emit("[坏导] 未发现坏导")
                elif step_name == 'ica':
                    n_comp = self.cfg.get('ica_n_components', 20)
                    self.log_signal.emit(f"[ICA] n_components={n_comp}")
                    ica_obj = mne.preprocessing.ICA(n_components=n_comp,
                                                    random_state=self.cfg.get('ica_random_state', 42),
                                                    max_iter='auto')
                    ica_obj.fit(self.raw)
                    try:
                        eog = ica_obj.find_bads_eog(self.raw)[0]
                    except Exception:
                        try:
                            eog = ica_obj.find_bads_eog(self.raw, ch_name=self.raw.info['ch_names'][0])[0]
                        except Exception:
                            eog = []
                    try:
                        mus = ica_obj.find_bads_muscle(self.raw)[0]
                    except Exception:
                        mus = []
                    eog = [int(x) for x in eog]
                    mus = [int(x) for x in mus]
                    ica_exclude = sorted(set(eog + mus))
                    if len(ica_exclude) > ica_obj.n_components_ * 0.5:
                        self.log_signal.emit(f"[ICA] 警告: 检测到 {len(ica_exclude)} 个伪迹成分（超过50%），可能存在误报。仅保留眼动检测结果。")
                        ica_exclude = list(eog)
                    ica_obj.exclude = list(ica_exclude)
                    if ica_exclude:
                        self.log_signal.emit(f"[ICA] 排除: {ica_exclude}")
                        self.raw = ica_obj.apply(self.raw)
                    else:
                        self.log_signal.emit("[ICA] 无排除成分")
                elif step_name == 'badseg':
                    threshold = 200.0 * 1e-6
                    sfreq = self.raw.info['sfreq']
                    data = self.raw.get_data()
                    bad_mask = np.any(np.abs(data) > threshold, axis=0)
                    bad_idx = np.where(bad_mask)[0]
                    if len(bad_idx) > 0:
                        min_samples = max(1, int(0.5 * sfreq))
                        segments = []
                        seg_start = bad_idx[0]
                        prev = bad_idx[0]
                        for idx in bad_idx[1:]:
                            if idx == prev + 1:
                                prev = idx
                            else:
                                if prev - seg_start + 1 >= min_samples:
                                    segments.append((seg_start / sfreq, (prev + 1) / sfreq))
                                seg_start = idx
                                prev = idx
                        if prev - seg_start + 1 >= min_samples:
                            segments.append((seg_start / sfreq, (prev + 1) / sfreq))
                        if segments:
                            from mne import Annotations
                            self.raw.set_annotations(Annotations(
                                onset=[s[0] for s in segments],
                                duration=[s[1]-s[0] for s in segments],
                                description=['BAD'] * len(segments)))
                            self.raw = _drop_by_annotations_compat(self.raw)
                            self.log_signal.emit(f"[坏段] 剔除 {len(segments)} 段")

            self.progress_signal.emit(100)
            self.log_signal.emit("✅ 全部预处理步骤完成")
            self.done_signal.emit(self.raw, ica_obj, ica_exclude)

        except Exception as e:
            self.error_signal.emit(str(e))
