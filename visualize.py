import numpy as np
import matplotlib.pyplot as plt
from dataloaders import load_record 
from scipy.signal import decimate, welch
from scipy.stats import entropy, pearsonr
from collections import Counter
import pandas as pd

def visualize_ecg_with_labels(signal, annotations, fs=250, duration=10):
    """
    Visualize ECG signal with arrhythmia labels. 
    NOTE OUTDATED. FOR USE WITH EDA
    Args:
        signal (np.array): ECG signal, typically 2D (samples, channels).
        annotations (np.array): Indices of arrhythmias.
        symbols (list or np.array): Corresponding symbols for each annotation.
        fs (int): Sampling frequency.
        duration (int): Duration (seconds) to visualize.
    """
    # Calculate the number of samples for the specified duration
    num_samples = duration * fs
    time_axis = np.arange(num_samples) / fs

    plt.figure(figsize=(15, 6))
    
    # Plot each channel, offsetting each for clarity
    for i in range(signal.shape[1]):
        plt.plot(time_axis, signal[:num_samples, i] + i * 2, label=f'Channel {i+1}')  # Offset channels for clarity
    
    # Mark annotations
    ann_indices = annotations.sample
    for j, ann in enumerate(ann_indices):
        if ann < num_samples:
            plt.axvline(x=ann / fs, color='r', linestyle='--')
            plt.text(ann / fs, signal[ann, 0] + 0.5, annotations.symbol[j], color='red', fontsize=8)
    
    plt.title(f"ECG Signal with Annotations (First {duration} seconds)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (arbitrary units)")
    plt.legend()
    plt.show()

##
def visualize_ecg_with_annotationsV2(patient_data, patient_id, num_beats=5, original_fs=360, target_fs=250):
    """
    Visualizes a sequence of downsampled ECG beats for a specific patient on a single plot,
    with annotations indicating each beat's type, adjusted for sample offset.

    NOTE OUTDATED. FOR USE WITH EDA

    Parameters:
        patient_data (dict): Dictionary containing ECG data for all patients.
        patient_id (str): The ID of the patient to visualize.
        num_beats (int): The number of beats to visualize.
        original_fs (int): Original sampling frequency in Hz, default is 360.
        target_fs (int): Target downsampled frequency in Hz, default is 250.
    """
    # Check if the patient_id exists in the data
    if patient_id not in patient_data:
        print(f"Patient ID {patient_id} not found in data.")
        return
    
    beats = patient_data[patient_id]
    
    # Limit the number of beats to visualize based on available data
    num_beats = min(num_beats, len(beats))
    
    # Downsampling factor
    downsample_factor = original_fs // target_fs
    
    # Initialize plot
    plt.figure(figsize=(15, 6))
    
    # Concatenate all beat segments for each lead
    lead_keys = [key for key in beats[0].keys() if key.startswith('signal_lead_')]
    for lead_key in lead_keys:
        all_segments = []
        annotations = []
        
        # Gather downsampled signal segments and annotations for each beat
        for i in range(num_beats):
            beat = beats[i]
            # Downsample the signal segment
            segment = decimate(beat[lead_key], downsample_factor)
            segment_length = len(segment)
            midpoint_offset = segment_length // 2  # Center of each beat
            
            # Append the downsampled segment to the continuous signal
            all_segments.extend(segment)
            
            # Calculate annotation position with offset adjustment
            annotation_position = len(all_segments) - midpoint_offset
            annotations.append((annotation_position, beat['label']))
        
        # Plot concatenated downsampled signals for this lead
        plt.plot(all_segments, label=lead_key)
        
        # Add annotations
        for pos, label in annotations:
            plt.axvline(x=pos, color='red', linestyle='--', linewidth=0.5)  # Mark beat position
            plt.text(pos, np.max(all_segments) * 0.9, label, color='red', rotation=45,
                     verticalalignment='bottom', horizontalalignment='right', fontsize=8)

    # Plot formatting
    plt.title(f"ECG Visualization for Patient {patient_id} with adjusted annotation positions")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

## 
def visualize_patient_data(processed_data, patient_id, display_seconds, fs=250):
    """
    Visualize the processed ECG signal for a given patient and duration.

    Args:
        processed_data (dict): Dictionary containing PROCESSED ECG data for all patients.
        patient_id (str): The patient ID to visualize.
        display_seconds (int): Number of seconds of data to display.
        fs (int): Sampling frequency of the ECG signals, default is 250 Hz.

    Returns:
        None
    """
    # Check if the patient ID exists in the processed data
    if patient_id not in processed_data:
        print(f"Patient ID {patient_id} not found in processed data.")
        return

    # Extract the patient's data
    patient_data = processed_data[patient_id]

    # Determine the number of samples to display
    samples_to_display = display_seconds * fs

    # Extract signals for the specified duration
    signals_lead_1 = patient_data['signals_lead_1'][:samples_to_display]
    signals_lead_2 = patient_data['signals_lead_2'][:samples_to_display]

    # Generate time axis
    time_axis = [i / fs for i in range(len(signals_lead_1))]

    # Plot the signals
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_axis, signals_lead_1, label='Lead 1')
    plt.title(f"Patient {patient_id} - Lead 1")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_axis, signals_lead_2, label='Lead 2', color='orange')
    plt.title(f"Patient {patient_id} - Lead 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

##
def visualize_patient_data_with_rhythm(processed_data, patient_id, display_seconds, fs=250):
    """
    Visualize the processed ECG signal for a given patient and duration, with rhythm annotations.

    Args:
        processed_data (dict): Dictionary containing processed ECG data for all patients.
        patient_id (str): The patient ID to visualize.
        display_seconds (int): Number of seconds of data to display.
        fs (int): Sampling frequency of the ECG signals, default is 250 Hz.

    Returns:
        None
    """
    # Check if the patient ID exists in the processed data
    if patient_id not in processed_data:
        print(f"Patient ID {patient_id} not found in processed data.")
        return

    # Extract the patient's data
    patient_data = processed_data[patient_id]

    # Determine the number of samples to display
    samples_to_display = display_seconds * fs

    # Extract signals for the specified duration
    signals_lead_1 = patient_data['signals_lead_1'][:samples_to_display]
    signals_lead_2 = patient_data['signals_lead_2'][:samples_to_display]
    labels = patient_data['labels'][:samples_to_display]

    # Generate time axis
    time_axis = [i / fs for i in range(len(signals_lead_1))]

    # Determine rhythm annotations for segments
    segment_length = fs  # 1-second segments for simplicity
    num_segments = len(signals_lead_1) // segment_length
    segment_annotations = []

    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment_labels = labels[start_idx:end_idx]

        # Find the majority label in the segment
        majority_label = Counter(segment_labels).most_common(1)[0][0]
        segment_annotations.append((start_idx + segment_length // 2, majority_label))

    # Plot the signals
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_axis, signals_lead_1, label='Lead 1')
    plt.title(f"Patient {patient_id} - Lead 1 with Rhythm Annotations")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Add rhythm annotations for Lead 1
    for midpoint, rhythm in segment_annotations:
        time = midpoint / fs
        amplitude = signals_lead_1[midpoint]
        plt.text(time, amplitude, rhythm, fontsize=10, color='red', ha='center')

    plt.subplot(2, 1, 2)
    plt.plot(time_axis, signals_lead_2, label='Lead 2', color='orange')
    plt.title(f"Patient {patient_id} - Lead 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


##
#NOISE AND ARTIFACTS
def compute_noise_metrics_for_patient(processed_data, patient_id, fs=250, weights=None):
    """
    Compute noise and artifact metrics for each lead of a patient's ECG data
    and determine which lead is better based on a scoring system.

    Args:
        processed_data (dict): Dictionary containing processed ECG data for all patients.
        patient_id (str): The patient ID to analyze.
        fs (int): Sampling frequency, default is 250 Hz.
        weights (list): Weights for each metric [SNR, Entropy, Std Dev, HighFreqPower].

    Returns:
        str: 'Lead 1' or 'Lead 2' based on the scoring system.
    """
    if patient_id not in processed_data:
        print(f"Patient ID {patient_id} not found in processed data.")
        return None

    # Default weights for scoring (prioritize SNR)
    if weights is None:
        weights = [0.4, 0.2, 0.2, 0.2]  # Default: prioritize SNR, almost every paper does this

    # Extract signals for the patient
    signals_lead_1 = processed_data[patient_id]['signals_lead_1']
    signals_lead_2 = processed_data[patient_id]['signals_lead_2']

    # Compute metrics for each lead
    metrics_lead_1 = compute_noise_metrics(signals_lead_1, fs)
    metrics_lead_2 = compute_noise_metrics(signals_lead_2, fs)

    # NOTE: COMMENT OUT THE BELOW IF DONT WANT ALL METRIC COMPUTATIONS PRINTED (SNR, ENTROPY, HF POWER, STD NORMALIZ)
    # print(metrics_lead_1)
    # print(metrics_lead_2)

    # Extract individual metrics
    snr_1, entropy_1, std_1, hf_power_1 = metrics_lead_1.values()
    snr_2, entropy_2, std_2, hf_power_2 = metrics_lead_2.values()

    # Normalize metrics
    snr_norm_1 = snr_1 / max(snr_1, snr_2)
    snr_norm_2 = snr_2 / max(snr_1, snr_2)

    entropy_norm_1 = entropy_1 / max(entropy_1, entropy_2)
    entropy_norm_2 = entropy_2 / max(entropy_1, entropy_2)

    std_norm_1 = std_1 / max(std_1, std_2)
    std_norm_2 = std_2 / max(std_1, std_2)

    hf_power_norm_1 = hf_power_1 / max(hf_power_1, hf_power_2)
    hf_power_norm_2 = hf_power_2 / max(hf_power_1, hf_power_2)

    # Compute scores for each lead
    score_lead_1 = (
        weights[0] * snr_norm_1 +
        weights[1] * (1 - entropy_norm_1) +
        weights[2] * (1 - std_norm_1) +
        weights[3] * (1 - hf_power_norm_1)
    )

    score_lead_2 = (
        weights[0] * snr_norm_2 +
        weights[1] * (1 - entropy_norm_2) +
        weights[2] * (1 - std_norm_2) +
        weights[3] * (1 - hf_power_norm_2)
    )

    # Determine the better lead
    # print(score_lead_1, score_lead_2)
    better_lead = 'Lead 1' if score_lead_1 > score_lead_2 else 'Lead 2'

    return better_lead



def compute_noise_metrics(signal, fs=250):
    """
    Compute noise and artifact metrics for a given ECG signal.

    Args:
        signal (array): ECG signal array (1D).
        fs (int): Sampling frequency, default is 250 Hz.

    Returns:
        dict: Dictionary of noise metrics.
    """
    # Signal-to-Noise Ratio (SNR)
    power_signal = np.mean(signal ** 2)
    noise_band = signal - np.mean(signal)  # Remove baseline
    power_noise = np.mean(noise_band ** 2)
    snr = 10 * np.log10(power_signal / power_noise)

    # Entropy
    signal_entropy = entropy(np.histogram(signal, bins=50, density=True)[0])

    # Standard Deviation
    std_dev = np.std(signal)

    # Power in Noise Bands (>50 Hz)
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    high_freq_power = np.sum(psd[freqs > 50])

    return {
        'SNR': snr,
        'Entropy': signal_entropy,
        'Standard Deviation': std_dev,
        'High Frequency Power': high_freq_power,
    }

##
