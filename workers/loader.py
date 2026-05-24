import mne
from PySide6.QtCore import QThread, Signal


class CNTLoaderWorker(QThread):
    """Background worker to load CNT file and extract events."""
    log_signal = Signal(str)
    progress_signal = Signal(int)
    done_signal = Signal(object, object)   # raw, events_info list
    error_signal = Signal(str)

    def __init__(self, cnt_path):
        super().__init__()
        self.cnt_path = cnt_path

    def run(self):
        try:
            self.log_signal.emit(f"正在加载 CNT 文件: {self.cnt_path}")
            self.progress_signal.emit(10)

            raw = mne.io.read_raw_cnt(self.cnt_path, preload=True, verbose=False)
            self.progress_signal.emit(50)

            # Extract events/annotations
            events_info = []
            try:
                if raw.annotations and len(raw.annotations) > 0:
                    for i in range(len(raw.annotations)):
                        onset = raw.annotations.onset[i]
                        desc = raw.annotations.description[i]
                        events_info.append((onset, desc, 'annotation'))
                else:
                    # Try stim channel
                    try:
                        stim_events = mne.find_events(raw, shortest_event=1, verbose=False)
                        if len(stim_events) > 0:
                            for ev in stim_events:
                                sample, _unused, val = ev
                                time = sample / raw.info['sfreq']
                                events_info.append((time, str(val), 'stim'))
                    except Exception:
                        pass
            except Exception as e:
                self.log_signal.emit(f"事件读取提示: {str(e)}")

            self.progress_signal.emit(100)
            self.log_signal.emit(f"加载完成: {len(raw.ch_names)} 通道, {raw.n_times} 采样点, {raw.info['sfreq']:.1f} Hz")
            self.log_signal.emit(f"发现 {len(events_info)} 个事件标记")
            self.done_signal.emit(raw, events_info)

        except Exception as e:
            self.error_signal.emit(str(e))
