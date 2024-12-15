import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE

# def segment_ecg_data(ecg_data, snippet_length=200, samples_before=90, samples_after=90):
#     """
#     Segment ECG data into fixed-length snippets around each beat.

#     Parameters:
#         ecg_data (dict): Dictionary with patient IDs as keys. Each value is a list of beat entries,
#                          where each entry contains 'label' and signal segments for each lead.
#         snippet_length (int): Total length of each snippet.
#         samples_before (int): Number of samples to include before each peak.
#         samples_after (int): Number of samples to include after each peak.

#     Returns:
#         dict: A dictionary with patient IDs as keys, where each value is a list
#               of segmented beat entries, with each beat entry containing label
#               and fixed-length signal segments.
#     """
#     segmented_data = {}

#     for patient_id, beats in ecg_data.items():
#         segmented_data[patient_id] = []
        
#         for beat in beats:
#             segmented_beat = {'label': beat['label']}
            
#             # Segment and pad each lead signal
#             for lead_key, signal in beat.items():
#                 if lead_key.startswith('signal_lead_'):
#                     peak_position = samples_before  # Center the peak
#                     start = max(0, peak_position - samples_before)
#                     end = min(len(signal), peak_position + samples_after)
                    
#                     segment = signal[start:end]
                    
#                     # Pad to the fixed snippet length if necessary
#                     if len(segment) < snippet_length:
#                         pad_width = snippet_length - len(segment)
#                         segment = np.pad(segment, (0, pad_width), 'constant')
                    
#                     segmented_beat[lead_key] = segment[:snippet_length]
            
#             segmented_data[patient_id].append(segmented_beat)
    
#     return segmented_data

# def visualize_segmented_data(segmented_data, patient_id, num_beats=5):
#     """
#     Visualize segmented ECG snippets for a specific patient in a single figure with subplots.

#     Parameters:
#         segmented_data (dict): Dictionary with patient IDs as keys and lists of segmented beats as values.
#         patient_id (str): ID of the patient to visualize.
#         num_beats (int): Number of beats to visualize in a single figure.
#     """
#     if patient_id not in segmented_data:
#         print(f"Patient ID {patient_id} not found in segmented data.")
#         return
    
#     beats = segmented_data[patient_id][:num_beats]
    
#     fig, axes = plt.subplots(num_beats, 1, figsize=(12, 4 * num_beats), sharex=True)
    
#     for i, beat in enumerate(beats):
#         ax = axes[i] if num_beats > 1 else axes  # Support for single subplot case
        
#         for lead_key, segment in beat.items():
#             if lead_key.startswith('signal_lead_'):
#                 ax.plot(segment, label=lead_key)
        
#         ax.set_title(f"Beat {i+1} - Label: {beat['label']}")
#         ax.set_ylabel("Amplitude")
#         ax.legend()
    
#     plt.xlabel("Sample")
#     plt.suptitle(f"Patient {patient_id} - First {num_beats} Segmented Beats", y=1.02, fontsize=14)
#     plt.tight_layout()
#     plt.show()

##

def split_data(signals, labels, test_size=0.2, val_size=0.1):
    """
    Split ECG data into training, validation, and test sets.

    Args:
        signals (array): Array of segmented ECG signals (shape: [num_segments, time_steps]).
        labels (array): Array of corresponding labels (shape: [num_segments]).
        test_size (float): Proportion of data to reserve for testing.
        val_size (float): Proportion of training data to reserve for validation.

    Returns:
        dict: Contains train, validation, and test splits for signals and labels.
    """
    # Split into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        signals, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Split training+validation into training and validation sets
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size relative to training+validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42, stratify=y_train_val
    )

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


##

def prepare_rnn_data(signal, labels, segment_length_sec, fs=250):
    """
    Segment ECG data for RNN training.

    Args:
        signal (array): 1D array of ECG signal values normalized to [0, 1].
        labels (list): 1D list of rhythm labels corresponding to each sample.
        segment_length_sec (int): Desired segment length in seconds.
        fs (int): Sampling frequency, default is 250 Hz.

    Returns:
        np.array: Segmented ECG data (shape: [num_segments, time_steps, features]).
        np.array: Corresponding labels for each segment.
    """
    segment_length_samples = segment_length_sec * fs
    segmented_signals = []
    segmented_labels = []
    # # Print TOTAL counts of labels across all patients
    # print("========== DEBUG: LABEL SUMMARY ==========")
    # total_label_counts = Counter(labels)
    # # total_labels_sum = sum(total_label_counts.values())
    # for label, count in total_label_counts.items():
    #         print(f"Label: '{label}' - Count: {count}")
    # print("========== DEBUG: LABEL SUMMARY END ==========")

    for start_idx in range(0, len(signal), segment_length_samples):
        end_idx = start_idx + segment_length_samples

        # Extract the segment
        segment = signal[start_idx:end_idx]

        # Skip incomplete segments
        if len(segment) < segment_length_samples:
            continue

        # Assign the majority label for the segment
        segment_labels = labels[start_idx:end_idx]
        majority_label = max(set(segment_labels), key=segment_labels.count)

        segmented_signals.append(segment)
        segmented_labels.append(majority_label)
    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # # Print label counts AFTER segmentation
    # print("\n========== DEBUG: LABEL SUMMARY AFTER SEGMENTATION ==========")
    # segmented_label_counts = Counter(segmented_labels)
    # for label, count in segmented_label_counts.items():
    #     print(f"Label: '{label}' - Count: {count}")
    # print("========== DEBUG: LABEL SUMMARY END ==========")

    # Convert to numpy arrays
    return np.array(segmented_signals), np.array(segmented_labels)

def apply_smote(data_by_label, target_counts, original_shape, random_state=42):
    """
    Augments data using SMOTE for specified target counts.

    Args:
        data_by_label (dict): Dictionary where keys are labels and values are lists of signals.
        target_counts (dict): Target counts for labels to be augmented using SMOTE.
        original_shape (tuple): Original shape of the signals (e.g., (length, features)).
        random_state (int): Random seed for reproducibility.

    Returns:
        np.ndarray, np.ndarray: Augmented signals and their corresponding labels.
    """
    # Prepare data for SMOTE
    X_minority = []
    y_minority = []
    for label, data in data_by_label.items():
        if label in target_counts:
            X_minority.extend(data)
            y_minority.extend([label] * len(data))
    X_minority = np.array(X_minority)
    y_minority = np.array(y_minority)
    
    # Flatten signals for SMOTE
    X_minority_flat = X_minority.reshape(X_minority.shape[0], -1)
    
    # Apply SMOTE
    smote = SMOTE(sampling_strategy=target_counts, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_minority_flat, y_minority)
    
    # Reshape signals back to original dimensions
    X_resampled = X_resampled.reshape(-1, *original_shape)
    
    return X_resampled, y_resampled

