import numpy as np
import mne


def _drop_by_annotations_compat(raw):
    """Drop BAD annotations from raw data. Compatible with older MNE versions."""
    if hasattr(raw, 'drop_by_annotations'):
        return raw.copy().drop_by_annotations()
    # Fallback for older MNE: manually filter data by keep mask
    if raw.annotations is None or len(raw.annotations) == 0:
        return raw.copy()
    sfreq = raw.info['sfreq']
    keep_mask = np.ones(raw.n_times, dtype=bool)
    for i in range(len(raw.annotations)):
        if raw.annotations.description[i] == 'BAD':
            start = int(raw.annotations.onset[i] * sfreq)
            end = min(int((raw.annotations.onset[i] + raw.annotations.duration[i]) * sfreq), raw.n_times)
            keep_mask[start:end] = False
    if not keep_mask.any():
        return raw.copy()  # everything is bad, return unchanged copy
    raw_copy = raw.copy()
    raw_copy._data = raw_copy._data[:, keep_mask]
    raw_copy._times = raw_copy._times[keep_mask]
    raw_copy.first_samp = 0
    raw_copy.set_annotations(None)
    return raw_copy
