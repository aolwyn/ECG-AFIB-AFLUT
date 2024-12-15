# This file is for specifically beat classification.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter
import pandas as pd
import matplotlib.pyplot as plt 

import os
import wfdb
import pickle
import sys
import glob
from collections import Counter

import pprint

import dataloaders
import beat_utils
import beat_models

"""
Beat Annotation Mapping (MIT-BIH Arrhythmia Database):

Common Beat Annotations (Heartbeats):
- 'N': Normal beat
- 'L': Left bundle branch block beat
- 'R': Right bundle branch block beat
- 'A': Atrial premature contraction
- 'V': Premature ventricular contraction
- 'E': Ventricular escape beat
- 'F': Fusion of ventricular and normal beat
- 'J': Nodal (junctional) escape beat

Non-Beat Annotations (Events):
- '+': Rhythm change
- '/': Paced beat
- '~': Signal quality issue
- 'x': Non-conducted P-wave
- '|': Isolated QRS-like artifact

Miscellaneous Annotations:
- '[' and ']': Measurement markers
- '!': Ventricular flutter wave
- '"': Unknown rhythm change
- 'a', 'e': Aberrated beats
- 'f': Atrial flutter wave
- 'j': Nodal escape beat
- 'Q': Unclassifiable beat
"""


print("Device information:")
dataloaders.get_device_info()

# read in each file.

data_directory = 'D:/Datasets/mit-bih-arrhythmia-database-1.0.0/'

ecg_data = {}

print("-------------------------")

print("Reading in files.")

# Read and store each patient's ECG data
for filename in os.listdir(data_directory):
    if filename.endswith('.dat'):
        record_base = filename[:-4]
        record_path = os.path.join(data_directory, record_base)

        # Load ECG signal and annotations
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')

        # Extract data components
        ecg_signal = record.p_signal[:, 0]  # First lead
        labels = annotation.symbol          # Beat-level annotations
        peaks = annotation.sample           # R-peak locations

        # Save to dictionary
        ecg_data[record_base] = {
            'ecg_signal': ecg_signal,
            'labels': labels,
            'peaks': peaks
        }

        # Print confirmation
        print(f"Saved data for patient: {record_base}")

print("-------------------------")

for patient_id, data in ecg_data.items():
    print(f"Patient ID: {patient_id}")
    print(f"  ECG Signal Length: {len(data['ecg_signal'])}")
    print(f"  Number of Annotations: {len(data['labels'])}")
    print(f"  First 5 R-Peak Locations: {data['peaks'][:5]}")
    print(f"  First 5 Labels: {data['labels'][:5]}")
    print("-" * 50)

print("-------------------------")

print("Checking the baseline for normal, patient 100:")

beat_utils.visualize_ecg('100',ecg_data)

print("Checking the another patient for difference, patient 116:")

beat_utils.visualize_ecg('116',ecg_data)

print("-------------------------")

beat_utils.count_beats(ecg_data)

print("-------------------------")

beat_utils.print_heartbeat_annotations(ecg_data)

print("-------------------------")

print("The common beats that papers use in detection are typically N, L, R, A, V, E, F.") # specifically, we should maybe use N, V, /, A, L, R

beat_utils.plot_label_distribution(ecg_data)

print("-------------------------")

print("Preprocessing. ")
print("Steps: \n 1: Apply band pass filter \n 2: Apply notch filter \n 3: Z-score normalize")


# Apply bandpass filtering and overwrite original ECG signals
for patient_id, data in ecg_data.items():
    ecg_signal = data['ecg_signal']
    
    # Step 1
    ecg_signal  = beat_utils.bandpass_filter(ecg_signal)
    
    # Step 2
    ecg_signal  = beat_utils.notch_filter(ecg_signal)

    # Step 3
    ecg_signal = beat_utils.z_score_normalize(ecg_signal)

    # Overwrite original ECG signal
    ecg_data[patient_id]['ecg_signal'] = ecg_signal 

    print(f"Filtered ECG signal for patient: {patient_id}")

print("-------------------------")

print("Visualize after preprocessing.")

beat_utils.visualize_ecg('116',ecg_data)

# baseline wander significantly reduced, signal appears more centered. 
# amplitude of the peaks is consistent whch is good
# P-QRS-T is still intact :)
# might want to apply median still but meh.

print("-------------------------")

print("Before feature stuff, check the label --> peak lineup.")

beat_utils.check_annotation_sync(patient_id='116', ecg_data=ecg_data, window=(0, 720))

print("-------------------------")

print("we may have heart rate variation, so lets calculate the avg. ")

average_heart_rates = beat_utils.compute_average_heart_rate(ecg_data)

# below is just teh values in an array format.
bpm_values = [
    75.54, 62.27, 72.83, 69.47, 76.78, 89.44, 69.72, 71.11, 60.60, 84.22,
    70.88, 84.72, 59.65, 62.80, 65.17, 80.45, 51.12, 76.46, 69.58, 62.32,
    82.36, 50.47, 54.28, 92.77, 67.75, 71.31, 103.29, 88.78, 79.25, 101.00,
    101.42, 89.20, 91.81, 109.43, 76.31, 112.98, 75.78, 76.85, 68.73, 81.81,
    87.52, 87.83, 71.14, 81.94, 66.81, 60.38, 104.72, 91.86
]

# Summing all bpm values
total_bpm = sum(bpm_values)
print("Average BPM:",total_bpm / len(bpm_values))

# average is about 78 (60,000 ms / 78 bpm = 769ms between RR),
# so the R-R interval samples should be 769 x (360/1000) = 277 samples.
# so roughly 100 ms before the R peak and 200 ms after should be ok?

print("-------------------------")

print("Begin segmentation.")

segmented_data = {}

for patient_id, data in ecg_data.items():
    signal = data['ecg_signal']
    peaks = data['peaks']
    labels = data['labels']

    # Segment the ECG signal and get labels
    segments, segment_labels = beat_utils.segment_ecg(signal, peaks, labels, fs=360, window_before=100, window_after=200)
    segmented_data[patient_id] = {
        'segments': segments,
        'labels': segment_labels
    }

print("Visualize 1 beat to check. Check 1 patient within each 5 window for BPM (50, 55, 60, etc.)")

patient_id = '124'
beat_index = 1000

beat = segmented_data[patient_id]['segments'][beat_index]
label = segmented_data[patient_id]['labels'][beat_index]

print("Visualize 1 beat to check. Check 1 patient within each 5 window for BPM (50, 55, 60, etc.)")

# beat_utils.visualize_beat(beat, label=label)

print("Distribution of labels after segmentation.")

# label_counts = beat_utils.plot_beat_label_distribution(segmented_data)
# print("Label Counts:", label_counts)

print("-------------------------")

# As noted, like with rhythms, major class imbalance. let's select a subset.

print("After reviewing the labels. We need only a small subset. ")
print("Testing on the following: ['N', 'V', '/', 'A', 'L', 'R']")

target_classes = ['N', 'V', '/', 'A', 'L', 'R']
# min is A with 2,546 samples so lets take 2500

subset_segments, subset_labels = beat_utils.select_subset_by_classes(
    segmented_data,
    labels=target_classes,
    num_samples_per_class=2500
)


subset_data = {
    'segments': subset_segments,
    'labels': subset_labels
}

print(f"Total selected segments: {len(subset_segments)}")
print(f"Total selected labels: {len(subset_labels)}")

print("-------------------------")

print("70:15:15 T:V:T split.")
# Split the dataset
train_segments, test_segments, train_labels, test_labels = train_test_split(
    subset_data['segments'], subset_data['labels'], test_size=0.3, random_state=42
)


val_segments, test_segments, val_labels, test_labels = train_test_split(
    test_segments, test_labels, test_size=0.5, random_state=42
)

# see pytorch docs if you're confused about this @ Henry
train_dataset = beat_utils.ECGDataset(train_segments, train_labels)
val_dataset = beat_utils.ECGDataset(val_segments, val_labels)
test_dataset = beat_utils.ECGDataset(test_segments, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

print("-------------------------")

print("Train and Validate.")

models = ['Seq2Seq','CNN-LSTM','Res-ATT']

selected_model = ''

dataset = beat_utils.ECGDataset(subset_data['segments'], subset_data['labels'])


if selected_model == 'Seq2Seq':
    INPUT_SIZE = 108
    NUM_CLASSES = 6
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    EPOCHS = 100
    BATCH_SIZE = 32
    PATIENCE = 5

    
    beat_utils.train_k_fold_Seq(
    model_class=beat_models.Seq2Seq,
    dataset=dataset,
    num_classes=NUM_CLASSES,
    k_folds=5,
    num_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    patience=PATIENCE
    )

elif selected_model == 'CNN-LSTM':
    INPUT_SIZE = 300
    NUM_CLASSES = 6
    EPOCHS = 10
    LEARNING_RATE=0.001
    BATCH_SIZE=32
    PATIENCE=5

    beat_utils.train_k_fold(
    model_class=beat_models.CNNLSTM, 
    dataset=dataset, 
    num_classes=6, 
    k_folds=5, 
    num_epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    patience=PATIENCE
    )

elif selected_model == 'Res-ATT':
    INPUT_SIZE = 300
    NUM_CLASSES = 6
    EPOCHS = 25
    BATCH_SIZE = 32
    PATIENCE = 5

    beat_utils.train_k_fold_res_att(
        model_class=beat_models.MultiBranchResAttentionNetwork, 
        dataset=dataset, 
        num_classes=NUM_CLASSES, 
        k_folds=5, 
        num_epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        patience=PATIENCE
    )
else:
    print("No Model Selected.")

print("-------------------------")

print("See instructions in helper file to run the test section after validation + train.")
# print("Testing  model - Same as rhythm just modified")

# to test use the test model from beat utils, I took it out temporarily to fine tune.



# dataset = beat_utils.ECGDataset(subset_data['segments'], subset_data['labels'])

# model = beat_models.CNNLSTM(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
# model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), LEARNING_RATE)


# # standard training:
# for epoch in range(EPOCHS):  
#     model.train()
#     for segments, labels in train_loader:
#         segments, labels = segments.to('cuda'), labels.to('cuda')

#         # Forward pass
#         outputs = model(segments)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# # Evaluate on the validation set
# model.eval()
# y_true, y_pred = [], []

# with torch.no_grad():
#     for segments, labels in val_loader:
#         segments, labels = segments.to('cuda'), labels.to('cuda')
#         outputs = model(segments)
#         _, predicted = torch.max(outputs, 1)

#         y_true.extend(labels.cpu().numpy())
#         y_pred.extend(predicted.cpu().numpy())

# # Evaluate metrics
# target_classes = [0, 1, 2, 3, 4, 5]  # Update if class labels change
# beat_utils.evaluate_model(y_true, y_pred, target_classes)

# K FOLD VALIDATION VERS. - swap out to the held out section if needed.


# beat_utils.train_k_fold(
#     model_class=beat_models.CNNLSTM, 
#     dataset=dataset, 
#     num_classes=6, 
#     k_folds=5, 
#     num_epochs=EPOCHS, 
#     batch_size=BATCH_SIZE,
#     patience=PATIENCE
# )

# print("-------------------------")

# print("Testing  model - Same as rhythm just modified")

# to test use the test model from beat utils, I took it out temporarily to fine tune.

