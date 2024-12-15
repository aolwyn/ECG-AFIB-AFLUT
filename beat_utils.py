# utilities functions. no time to separate into sub files. we're in crunch hours.
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt 
import os
import wfdb
import pickle
import sys
import glob
from scipy.signal import butter, lfilter, filtfilt, iirnotch
import pprint
from collections import Counter
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold



def load_data(record_path):
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    ecg_signal = record.p_signal[:, 0] # only want top lead for beat. lower lead has missing.
    labels = annotation.symbol
    peaks = annotation.sample
    return ecg_signal, labels, peaks


def visualize_ecg(patient_id, ecg_data, sampling_rate=360):
    if patient_id not in ecg_data:
        print(f"Patient ID {patient_id} not found.")
        return
    
    ecg_signal = ecg_data[patient_id]['ecg_signal']
    duration = 10 * sampling_rate  

    if len(ecg_signal) < duration:
        print(f"Insufficient data for {patient_id}. Showing available samples.")
        duration = len(ecg_signal)

    plt.figure(figsize=(15, 5))
    plt.plot(ecg_signal[:duration], color='b')
    plt.title(f"ECG Signal for Patient {patient_id} - First 10 Seconds")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    plt.show()

def check_annotation_sync(patient_id, ecg_data, window=None):

    if patient_id not in ecg_data:
        print(f"Patient ID {patient_id} not found.")
        return
    
    signal = ecg_data[patient_id]['ecg_signal']
    peaks = ecg_data[patient_id]['peaks']
    
    if window:
        start, end = window
        signal = signal[start:end]
        peaks = [p for p in peaks if start <= p < end]
        peaks = [p - start for p in peaks]  
    
    plt.figure(figsize=(15, 5))
    plt.plot(signal, label="ECG Signal")
    plt.scatter(peaks, signal[peaks], color='red', label="R-Peaks", zorder=5)
    plt.title(f"ECG Signal with R-Peaks for Patient {patient_id}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.grid(True)
    plt.show()

def count_beats(ecg_data):
    beat_counts = Counter()

    for patient_id, data in ecg_data.items():
        labels = data['labels']
        beat_counts.update(labels)
    
    # Print results
    print("Beat Type Counts:")
    for beat, count in sorted(beat_counts.items()):
        print(f"  {beat}: {count}")

    return beat_counts

def print_heartbeat_annotations(ecg_data):
    heartbeats = {
        'N': "Normal beat",
        'L': "Left bundle branch block beat",
        'R': "Right bundle branch block beat",
        'A': "Atrial premature contraction",
        'V': "Premature ventricular contraction",
        'E': "Ventricular escape beat",
        'F': "Fusion of ventricular and normal beat",
        'J': "Nodal (junctional) escape beat",
        '/': "Paced beat"
    }

    # count beat types, data dist. 
    beat_counts = Counter()
    for patient_id, data in ecg_data.items():
        beat_counts.update(data['labels'])

    print("Heartbeat Annotations Count:")
    for beat, description in heartbeats.items():
        count = beat_counts.get(beat, 0)
        print(f"  {beat}: {count} ({description})")


def plot_label_distribution(ecg_data):
    # common heartbeat types, see website (the diana.edu one)
    common_beats = {
        'N': "Normal beat",
        'L': "Left bundle branch block",
        'R': "Right bundle branch block",
        'A': "Atrial premature contraction",
        'V': "Premature ventricular contraction",
        'E': "Ventricular escape beat",
        'F': "Fusion of ventricular and normal beat"
    }

    # Count beats
    beat_counts = Counter()
    for patient_id, data in ecg_data.items():
        beat_counts.update(data['labels'])

    # Extract counts for common beats
    labels = list(common_beats.keys())
    counts = [beat_counts[label] for label in labels]
    descriptions = [common_beats[label] for label in labels]

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.xticks(labels, descriptions, rotation=45, ha='right')
    plt.title("ECG Beat Label Distribution")
    plt.xlabel("Beat Type")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def visualize_beat(beat, label, fs=360):
    # NOTE for use post-segment
    time = np.linspace(0, len(beat) / fs, len(beat))  # Time vector in seconds, see paper for why this is needed

    plt.figure(figsize=(10, 4))
    plt.plot(time, beat, color='b', label='ECG Beat')
    plt.title(f"Segmented ECG Beat{' - Label: ' + label if label else ''}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_beat_label_distribution(segmented_data):
    all_labels = []

    for patient_data in segmented_data.values():
        all_labels.extend(patient_data['labels'])
    
    label_counts = Counter(all_labels)

    labels, counts = zip(*label_counts.items())
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.title("Distribution of Beat Labels")
    plt.xlabel("Beat Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    label_counts = dict(Counter(all_labels))

    return label_counts # for verifying the previous counts.


# -------------------------

# for the beat filters. we want to avoid changing the phase so always filtfilt at end

def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=360, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def notch_filter(signal, notch_freq=50, fs=360, quality_factor=30):
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist  # Normalized frequency, see scipy docs for the logic behidn this

    b, a = iirnotch(w0, quality_factor) # <-- b / a and the qf are noted things

    # Apply zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def lowpass_filter(signal, cutoff=50, fs=360, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    b, a = butter(order, normal_cutoff, btype='low')

    # Apply zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def z_score_normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)

    #edge case ZZZZ
    if std == 0:
        return signal
    
    normalized_signal = (signal - mean) / std
    return normalized_signal

# -------------------------

def compute_average_heart_rate(ecg_data, fs=360):
    # for this function i'm doing interval calculations against the peak indexes.
    heart_rates = {}

    for patient_id, data in ecg_data.items():
        peaks = data['peaks']  
        
        # Compute RR intervals in samples
        rr_intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        
        # Convert RR intervals to milliseconds
        rr_intervals_ms = [interval / fs * 1000 for interval in rr_intervals]
        
        # Compute average heart rate in bpm
        if rr_intervals_ms:  # Avoid division by zero
            avg_heart_rate = 60000 / (sum(rr_intervals_ms) / len(rr_intervals_ms))
            heart_rates[patient_id] = avg_heart_rate
            print(f"Patient {patient_id}: Average Heart Rate = {avg_heart_rate:.2f} bpm")
        else:
            heart_rates[patient_id] = None
            print(f"Patient {patient_id}: No RR intervals available.")

    return heart_rates

def segment_ecg(signal, peaks, labels, fs=360, window_before=100, window_after=200):
    # Convert window sizes from ms to samples, see note in beat_classification
    samples_before = int(window_before * fs / 1000)
    samples_after = int(window_after * fs / 1000)
    segment_length = samples_before + samples_after

    segments = []
    segment_labels = []

    for i, peak in enumerate(peaks):
        
        start = peak - samples_before
        end = peak + samples_after

        # segment extract + padding
        segment = signal[max(0, start):min(len(signal), end)]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), mode='constant')
        segment = segment[:segment_length]

        segments.append(segment)
        segment_labels.append(labels[i])

    return segments, segment_labels


def select_subset_by_classes(segmented_data, labels, num_samples_per_class):
    # Select a subset of beats belonging to specific classes.
    subset_segments = []
    subset_labels = []

    for label in labels:
        # Collect all beats and labels matching the current class
        all_segments = []
        for patient_data in segmented_data.values():
            for seg, lbl in zip(patient_data['segments'], patient_data['labels']):
                if lbl == label:
                    all_segments.append((seg, lbl))
        
        # Shuffle and select the desired number of samples for this class
        random.shuffle(all_segments)
        selected = all_segments[:num_samples_per_class]

        # Add to the global subset
        subset_segments.extend([seg for seg, lbl in selected])
        subset_labels.extend([lbl for seg, lbl in selected])

        print(f"Selected {len(selected)} segments for class '{label}'.")

    return subset_segments, subset_labels

# -------------------------

# model stuff? 

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, segments, labels, transform=None):
        # Encode labels as integers
        self.label_encoder = LabelEncoder()
        self.labels = torch.tensor(self.label_encoder.fit_transform(labels), dtype=torch.long)

        # Convert segments to a NumPy array for faster tensor conversion
        self.segments = torch.tensor(np.array(segments), dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]

        if self.transform:
            segment = self.transform(segment)

        return segment, label
    
def evaluate_model(y_true, y_pred, target_classes):
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix for sensitivity and specificity
    cm = confusion_matrix(y_true, y_pred, labels=target_classes)

    sensitivity_per_class = cm.diagonal() / cm.sum(axis=1)
    specificity_per_class = []

    for i in range(len(target_classes)):
        true_negatives = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        false_positives = cm[:, i].sum() - cm[i, i]
        specificity = true_negatives / (true_negatives + false_positives)
        specificity_per_class.append(specificity)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': dict(zip(target_classes, sensitivity_per_class)),
        'specificity': dict(zip(target_classes, specificity_per_class))
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClass-specific Metrics:")
    for i, cls in enumerate(target_classes):
        print(f"Class '{cls}': Sensitivity = {sensitivity_per_class[i]:.4f}, Specificity = {specificity_per_class[i]:.4f}")
    
    return metrics


# https://scikit-learn.org/1.5/modules/cross_validation.html#k-fold
# https://www.geeksforgeeks.org/cross-validation-using-k-fold-with-scikit-learn/
def train_k_fold(model_class, dataset, num_classes, k_folds=5, num_epochs=50, batch_size=32, patience=5):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    criterion = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Metrics Tracker
    avg_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'\nFold {fold+1}/{k_folds}')

        # Prepare DataLoaders
        train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        # Initialize Model and Optimizer
        model = model_class(input_size=300, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for segments, labels in train_loader:
                segments, labels = segments.to(device), labels.to(device)

                # Forward pass
                outputs = model(segments)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] | Fold {fold+1} | Loss: {avg_loss:.4f}")

            # Evaluate on Validation Set
            model.eval()
            y_true, y_pred = [], []
            val_loss = 0

            with torch.no_grad():
                for segments, labels in val_loader:
                    segments, labels = segments.to(device), labels.to(device)
                    outputs = model(segments)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early Stopping Triggered at Epoch {epoch+1}")
                    break

            metrics = evaluate_model(y_true, y_pred, list(range(num_classes)))

            # Save Metrics
            for key in avg_metrics:
                avg_metrics[key].append(metrics[key])

    # Print Average Metrics
    print("\nFinal Cross-Validation Metrics (Average):")
    for key, values in avg_metrics.items():
        avg_value = np.mean(values)
        print(f"{key.capitalize()}: {avg_value:.4f}")

def train_k_fold_Seq(model_class, dataset, num_classes, k_folds=5, num_epochs=50, batch_size=32, patience=5):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    criterion = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Metrics Tracker
    avg_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'\nFold {fold+1}/{k_folds}')

        # Prepare DataLoaders
        train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        # Initialize Model and Optimizer
        model = model_class(input_size=108, hidden_size=128, num_layers=2, num_classes=6).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for segments, labels in train_loader:
                segments, labels = segments.to(device), labels.to(device)

                # Forward pass
                outputs = model(segments)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] | Fold {fold+1} | Loss: {avg_loss:.4f}")

            # Evaluate on Validation Set
            model.eval()
            y_true, y_pred = [], []
            val_loss = 0

            with torch.no_grad():
                for segments, labels in val_loader:
                    segments, labels = segments.to(device), labels.to(device)
                    outputs = model(segments)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early Stopping Triggered at Epoch {epoch+1}")
                    break

            metrics = evaluate_model(y_true, y_pred, list(range(num_classes)))

            # Save Metrics
            for key in avg_metrics:
                avg_metrics[key].append(metrics[key])

    # Print Average Metrics
    print("\nFinal Cross-Validation Metrics (Average):")
    for key, values in avg_metrics.items():
        avg_value = np.mean(values)
        print(f"{key.capitalize()}: {avg_value:.4f}")



def train_k_fold_res_att(model_class, dataset, num_classes, k_folds=5, num_epochs=50, batch_size=32, patience=5):
    kfold = torch.utils.data.random_split(dataset, [len(dataset) // k_folds] * k_folds)
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Metrics Tracker
    avg_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

    for fold, val_dataset in enumerate(kfold):
        print(f'\nFold {fold+1}/{k_folds}')

        # Combine remaining datasets for training
        train_datasets = [d for i, d in enumerate(kfold) if i != fold]
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize Model and Optimizer
        model = model_class(input_size=300, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for segments, labels in train_loader:
                segments, labels = segments.to(device), labels.to(device)

                # Forward pass
                outputs = model(segments)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] | Fold {fold+1} | Loss: {avg_loss:.4f}")

            # Evaluate on Validation Set
            model.eval()
            y_true, y_pred = [], []
            val_loss = 0

            with torch.no_grad():
                for segments, labels in val_loader:
                    segments, labels = segments.to(device), labels.to(device)
                    outputs = model(segments)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early Stopping Triggered at Epoch {epoch+1}")
                    break

            metrics = evaluate_model(y_true, y_pred, list(range(num_classes)))

            # Save Metrics
            for key in avg_metrics:
                avg_metrics[key].append(metrics[key])

    # Print Average Metrics
    print("\nFinal Cross-Validation Metrics (Average):")
    for key, values in avg_metrics.items():
        avg_value = np.mean(values)
        print(f"{key.capitalize()}: {avg_value:.4f}")




def test_model(model, test_loader, num_classes):
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y_true, y_pred = [], []

    with torch.no_grad():
        for segments, labels in test_loader:
            segments, labels = segments.to(device), labels.to(device)
            outputs = model(segments)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    metrics = evaluate_model(y_true, y_pred, list(range(num_classes)))

    return metrics