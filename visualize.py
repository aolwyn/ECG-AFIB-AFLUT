import numpy as np
import matplotlib.pyplot as plt
from dataloaders import load_record 

def visualize_ecg_with_labels(signal, annotations, fs=360, duration=10):
    """
    Visualize ECG signal with arrhythmia labels.
    
    Args:
        signal (np.array): ECG signal, typically 2D (samples, channels).
        annotations: WFDB annotations for arrhythmias.
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
