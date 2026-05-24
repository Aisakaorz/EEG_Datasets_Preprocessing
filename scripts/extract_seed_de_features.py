import os
import scipy.io
import torch
import numpy as np
from tqdm import tqdm

SEED_subject_round = [
    [
        "dujingcheng_20131027.mat",
        "dujingcheng_20131030.mat",
        "dujingcheng_20131107.mat",
    ],
    [
        "jianglin_20140404.mat",
        "jianglin_20140413.mat",
        "jianglin_20140419.mat",
    ],
    [
        "jingjing_20140603.mat",
        "jingjing_20140611.mat",
        "jingjing_20140629.mat",
    ],
    [
        "liuqiujun_20140621.mat",
        "liuqiujun_20140702.mat",
        "liuqiujun_20140705.mat",
    ],
    [
        "liuye_20140411.mat",
        "liuye_20140418.mat",
        "liuye_20140506.mat",
    ],
    [
        "mahaiwei_20130712.mat",
        "mahaiwei_20131016.mat",
        "mahaiwei_20131113.mat",
    ],
    [
        "penghuiling_20131027.mat",
        "penghuiling_20131030.mat",
        "penghuiling_20131106.mat",
    ],
    [
        "sunxiangyu_20140511.mat",
        "sunxiangyu_20140514.mat",
        "sunxiangyu_20140521.mat",
    ],
    [
        "wangkui_20140620.mat",
        "wangkui_20140627.mat",
        "wangkui_20140704.mat",
    ],
    [
        "weiwei_20131130.mat",
        "weiwei_20131204.mat",
        "weiwei_20131211.mat",
    ],
    [
        "wusifan_20140618.mat",
        "wusifan_20140625.mat",
        "wusifan_20140630.mat",
    ],
    [
        "wuyangwei_20131127.mat",
        "wuyangwei_20131201.mat",
        "wuyangwei_20131207.mat",
    ],
    [
        "xiayulu_20140527.mat",
        "xiayulu_20140603.mat",
        "xiayulu_20140610.mat",
    ],
    [
        "yansheng_20140601.mat",
        "yansheng_20140615.mat",
        "yansheng_20140627.mat",
    ],
    [
        "zhujiayi_20130709.mat",
        "zhujiayi_20131016.mat",
        "zhujiayi_20131105.mat",
    ],
]

# ------------------------------------------------------------------
# SEED DE Feature Extraction & Save for PyTorch
# ------------------------------------------------------------------
# This script reads the ExtractedFeatures/ folder of the SEED dataset,
# collects the de_LDS features for all subjects/sessions/trials,
# pads them to a uniform length, and saves a single .pt file.
#
# Output file: features/SEED/seed_de_features.pt
#
# Data shape after loading:
#   data   -> torch.Tensor of shape (15, 3, 15, 62, 265, 5)
#             [subjects, sessions, trials, channels, time_steps, bands]
#   labels -> torch.Tensor of shape (15, 3, 15)
#             [subjects, sessions, trials]
#             Label mapping: 0 = negative, 1 = neutral, 2 = positive
#
# How to load in PyTorch:
#   checkpoint = torch.load('features/SEED/seed_de_features.pt')
#   data   = checkpoint['data']    # (15, 3, 15, 62, 265, 5)
#   labels = checkpoint['labels']  # (15, 3, 15)
#
#   # Example: flatten to (N, 62, 265, 5) for a standard Dataset/DataLoader
#   N = data.shape[0] * data.shape[1] * data.shape[2]
#   data_flat   = data.view(N, 62, 265, 5)
#   labels_flat = labels.view(N)
# ------------------------------------------------------------------


def main():
    out_dir = os.path.join('../data/features', 'data/raw/SEED')
    os.makedirs(out_dir, exist_ok=True)

    n_subjects = len(SEED_subject_round)   # 15
    n_sessions = 3
    n_trials = 15
    n_channels = 62
    n_bands = 5

    # First pass: determine the max time length across all de_LDS features
    max_time = 0
    for subject_idx, subject_files in enumerate(tqdm(SEED_subject_round, desc='[SEED] Scanning max time')):
        for session_idx in range(n_sessions):
            mat_path = os.path.join('../data/raw/SEED', 'data/raw/SEED', 'ExtractedFeatures', subject_files[session_idx])
            mat_data = scipy.io.loadmat(mat_path)
            for trial_idx in range(1, n_trials + 1):
                key = f'de_LDS{trial_idx}'
                trial_data = mat_data[key]
                max_time = max(max_time, trial_data.shape[1])
    print(f'[SEED] Max time length across all de_LDS features: {max_time}')

    # Allocate tensors
    data_tensor = torch.zeros(n_subjects, n_sessions, n_trials, n_channels, max_time, n_bands)
    labels_tensor = torch.zeros(n_subjects, n_sessions, n_trials, dtype=torch.long)

    # Read labels and remap: -1,0,1 -> 0,1,2
    label_mat = scipy.io.loadmat(os.path.join('../data/raw/SEED', 'data/raw/SEED', 'ExtractedFeatures', 'label_readme', 'label.mat'))
    raw_labels = label_mat['label'].flatten()          # shape (15,), values [-1, 0, 1]
    remapped_labels = raw_labels + 1                   # values [0, 1, 2]
    remapped_labels = torch.from_numpy(remapped_labels).long()

    for subject_idx, subject_files in enumerate(tqdm(SEED_subject_round, desc='[SEED] Processing subjects')):
        for session_idx in range(n_sessions):
            mat_path = os.path.join('../data/raw/SEED', 'data/raw/SEED', 'ExtractedFeatures', subject_files[session_idx])
            mat_data = scipy.io.loadmat(mat_path)

            for trial_idx in range(1, n_trials + 1):
                key = f'de_LDS{trial_idx}'
                trial_data = mat_data[key]              # numpy array, shape (62, time, 5)
                time_len = trial_data.shape[1]

                # Copy real data; the rest remains zero (padding)
                data_tensor[subject_idx, session_idx, trial_idx - 1, :, :time_len, :] = \
                    torch.from_numpy(trial_data).float()

                # Assign label
                labels_tensor[subject_idx, session_idx, trial_idx - 1] = remapped_labels[trial_idx - 1]

    save_path = os.path.join(out_dir, 'seed_de_features.pt')
    torch.save({
        'data': data_tensor,
        'labels': labels_tensor,
    }, save_path)
    print(f'[SEED] Saved to {save_path}')
    print(f'[SEED] data shape:   {data_tensor.shape}')
    print(f'[SEED] labels shape: {labels_tensor.shape}')


if __name__ == '__main__':
    main()
