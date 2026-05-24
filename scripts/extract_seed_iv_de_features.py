import os
import scipy.io
import torch
import numpy as np
from tqdm import tqdm

SEED_IV_subject_round = [
    [
        "1_20160518.mat",
        "1_20161125.mat",
        "1_20161126.mat",
    ],
    [
        "2_20150915.mat",
        "2_20150920.mat",
        "2_20151012.mat",
    ],
    [
        "3_20150919.mat",
        "3_20151018.mat",
        "3_20151101.mat",
    ],
    [
        "4_20151111.mat",
        "4_20151118.mat",
        "4_20151123.mat",
    ],
    [
        "5_20160406.mat",
        "5_20160413.mat",
        "5_20160420.mat",
    ],
    [
        "6_20150507.mat",
        "6_20150511.mat",
        "6_20150512.mat",
    ],
    [
        "7_20150715.mat",
        "7_20150717.mat",
        "7_20150721.mat",
    ],
    [
        "8_20151103.mat",
        "8_20151110.mat",
        "8_20151117.mat",
    ],
    [
        "9_20151028.mat",
        "9_20151119.mat",
        "9_20151209.mat",
    ],
    [
        "10_20151014.mat",
        "10_20151021.mat",
        "10_20151023.mat",
    ],
    [
        "11_20150916.mat",
        "11_20150921.mat",
        "11_20151011.mat",
    ],
    [
        "12_20150725.mat",
        "12_20150804.mat",
        "12_20150807.mat",
    ],
    [
        "13_20151115.mat",
        "13_20151125.mat",
        "13_20161130.mat",
    ],
    [
        "14_20151205.mat",
        "14_20151208.mat",
        "14_20151215.mat",
    ],
    [
        "15_20150508.mat",
        "15_20150514.mat",
        "15_20150527.mat",
    ],
]

# ------------------------------------------------------------------
# SEED-IV DE Feature Extraction & Save for PyTorch
# ------------------------------------------------------------------
# This script reads the eeg_feature_smooth/ folder of the SEED-IV dataset,
# collects the de_LDS features for all subjects/sessions/trials,
# pads them to a uniform length, and saves a single .pt file.
#
# Output file: features/SEED_IV/seed_iv_de_features.pt
#
# Data shape after loading:
#   data   -> torch.Tensor of shape (15, 3, 24, 62, 64, 5)
#             [subjects, sessions, trials, channels, time_steps, bands]
#   labels -> torch.Tensor of shape (15, 3, 24)
#             [subjects, sessions, trials]
#             Label mapping: 0 = neutral, 1 = sad, 2 = fear, 3 = happy
#
# How to load in PyTorch:
#   checkpoint = torch.load('features/SEED_IV/seed_iv_de_features.pt')
#   data   = checkpoint['data']    # (15, 3, 24, 62, 64, 5)
#   labels = checkpoint['labels']  # (15, 3, 24)
#
#   # Example: flatten to (N, 62, 64, 5) for a standard Dataset/DataLoader
#   N = data.shape[0] * data.shape[1] * data.shape[2]
#   data_flat   = data.view(N, 62, 64, 5)
#   labels_flat = labels.view(N)
# ------------------------------------------------------------------

# SEED-IV session labels (24 trials per session)
# 0=neutral, 1=sad, 2=fear, 3=happy
SESSION_LABELS = [
    [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
]


def main():
    out_dir = os.path.join('../data/features', 'data/raw/SEED_IV')
    os.makedirs(out_dir, exist_ok=True)

    n_subjects = len(SEED_IV_subject_round)   # 15
    n_sessions = 3
    n_trials = 24
    n_channels = 62
    n_bands = 5

    # First pass: determine the max time length across all de_LDS features
    max_time = 0
    for subject_idx, subject_files in enumerate(tqdm(SEED_IV_subject_round, desc='[SEED-IV] Scanning max time')):
        for session_idx in range(n_sessions):
            mat_path = os.path.join('../data/raw/SEED_IV', 'data/raw/SEED_IV', 'eeg_feature_smooth',
                                    str(session_idx + 1), subject_files[session_idx])
            mat_data = scipy.io.loadmat(mat_path)
            for trial_idx in range(1, n_trials + 1):
                key = f'de_LDS{trial_idx}'
                trial_data = mat_data[key]
                max_time = max(max_time, trial_data.shape[1])
    print(f'[SEED-IV] Max time length across all de_LDS features: {max_time}')

    # Allocate tensors
    data_tensor = torch.zeros(n_subjects, n_sessions, n_trials, n_channels, max_time, n_bands)
    labels_tensor = torch.zeros(n_subjects, n_sessions, n_trials, dtype=torch.long)

    # Convert session labels to tensor
    session_labels_tensor = torch.tensor(SESSION_LABELS, dtype=torch.long)  # (3, 24)

    for subject_idx, subject_files in enumerate(tqdm(SEED_IV_subject_round, desc='[SEED-IV] Processing subjects')):
        for session_idx in range(n_sessions):
            mat_path = os.path.join('../data/raw/SEED_IV', 'data/raw/SEED_IV', 'eeg_feature_smooth',
                                    str(session_idx + 1), subject_files[session_idx])
            mat_data = scipy.io.loadmat(mat_path)

            for trial_idx in range(1, n_trials + 1):
                key = f'de_LDS{trial_idx}'
                trial_data = mat_data[key]              # numpy array, shape (62, time, 5)
                time_len = trial_data.shape[1]

                # Copy real data; the rest remains zero (padding)
                data_tensor[subject_idx, session_idx, trial_idx - 1, :, :time_len, :] = \
                    torch.from_numpy(trial_data).float()

                # Assign label from the corresponding session
                labels_tensor[subject_idx, session_idx, trial_idx - 1] = session_labels_tensor[session_idx, trial_idx - 1]

    save_path = os.path.join(out_dir, 'seed_iv_de_features.pt')
    torch.save({
        'data': data_tensor,
        'labels': labels_tensor,
    }, save_path)
    print(f'[SEED-IV] Saved to {save_path}')
    print(f'[SEED-IV] data shape:   {data_tensor.shape}')
    print(f'[SEED-IV] labels shape: {labels_tensor.shape}')


if __name__ == '__main__':
    main()
