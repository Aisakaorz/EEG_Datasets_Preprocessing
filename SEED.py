import mne
import scipy.io

from SEED_subject_file_name import *

"""
# use np.save to get npy files
np.save('preprocessed_data.npy', preprocessed_data)
# use np.load to get npy datas
loaded_data = np.load('preprocessed_data.npy')
"""

"""
1. Preprocessed_EEG/: EEG raw datas
"""
for subject in SEED_subject_round:
    for rounds in range(3):  # 1 subject = 3 rounds
        raw_datas = scipy.io.loadmat(f'./SEED/Preprocessed_EEG/{subject[rounds]}')
        for key in raw_datas:
            if not key.startswith('__'):
                print(f"Key: {key}, Value shape: {raw_datas[key].shape}")   # (62 channels, data point)
                raw = raw_datas[key]
                # do the preprocessing (but SEED seem to require any more preprocessing on our part)
                ch_names = ['Ch' + str(i) for i in range(1, 63)]
                ch_types = ['eeg'] * 62
                sfreq = 200
                info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
                raw = mne.io.RawArray(raw, info)
                # 1. filter
                # raw = raw.notch_filter(freqs=50)
                # raw = raw.notch_filter(freqs=60)
                # raw = raw.filter(l_freq=0.1, h_freq=50)
                # 2. ICA
                # ica = mne.preprocessing.ICA(n_components=20)
                # ica.fit(raw, reject_by_annotation=False)
                # eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=raw.info['ch_names'][0],
                #                                             reject_by_annotation=False)
                # mus_indices, mus_scores = ica.find_bads_muscle(raw)
                # artifacts_indices = eog_indices + mus_indices
                # ica.exclude = artifacts_indices
                # raw = ica.apply(raw, exclude=ica.exclude)
                # 3. resample
                # raw.resample(200)
                # 4. ... other preprocessing or calculating DE / other features, and divided into five frequency bands
                # such as this: uses the formula for differential entropy: log(2πeσ^2)/2
                # clip is the point that you want to calculate DE / other features
                # variance = np.var(clip, ddof=1)  # calculate the variance of the signal
                # DE = math.log(2 * math.pi * math.e * variance) / 2
                # or you can calculate DE using scipy
                # DE = stats.differential_entropy(clip)
                data = raw  # This raw is similar to the original (62, 47001) becoming (62, 235).
                # 👆 data, you can save them with the key name

label = scipy.io.loadmat('./SEED/Preprocessed_EEG/label.mat')
for key in label:
    if not key.startswith('__'):
        print(f"Key: {key}, Value shape: {label[key].shape}")
        data = label["label"]  # data.shape = (1, 15), [1,  0, -1, -1,  0,  1, -1,  0,  1,  1,  0, -1,  0,  1, -1]
        # -1: negative, 0: neutral, 1: positive
        # If using pytorch it's best to add 1 to make the label 0 1 2
        # 👆 label["label"], you can save it

"""
2. ExtractedFeatures/: DE is calculated in units of 1 second (200Hz/s)
    DE_datas -> (675, 62, 265, 5)
        675 = 15 subjects * 15 trials * 3 rounds
        62 = 62 channels
        265 = 265s (1s, 200Hz/s)
        5 = delta, theta, alpha, beta, gamma band
"""
DE_datas = scipy.io.loadmat('./SEED/ExtractedFeatures/correct_data.mat')
for key in DE_datas:
    if not key.startswith('__'):
        print(f"Key: {key}, Value shape: {DE_datas[key].shape}")
        data = DE_datas["data"]     # data.shape = (675, 62, 265, 5)
        # 👆 DE_datas["data"], you can save it

for subject in SEED_subject_round:
    for rounds in range(3):  # 1 subject = 3 rounds
        DE_datas = scipy.io.loadmat(f'./SEED/ExtractedFeatures/{subject[rounds]}')
        # for key in DE_datas:
        #     if not key.startswith('__'):
        #         print(f"Key: {key}, Value shape: {DE_datas[key].shape}")
        for trial in range(1, 16):  # 1 round = 15 trials
            data = DE_datas[f"de_LDS{trial}"]   # data.shape = (62 channels, 1xx~2xx seconds, 5 bands)
            # 👆 DE_datas[f"de_LDS{trial}"], you can save it

label = scipy.io.loadmat('./SEED/ExtractedFeatures/label.mat')
for key in label:
    if not key.startswith('__'):
        print(f"Key: {key}, Value shape: {label[key].shape}")
        data = label["label"]  # data.shape = (1, 15), [1,  0, -1, -1,  0,  1, -1,  0,  1,  1,  0, -1,  0,  1, -1]
        # If using pytorch it's best to add 1 to make the label 0 1 2
        # 👆 label["label"], you can save it
