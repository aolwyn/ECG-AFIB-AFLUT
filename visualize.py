import numpy as np
import matplotlib.pyplot as plt
from dataloaders import load_record 

def visualize_ecg_with_labels(signal, annotations, fs=360, duration=10):
    """
    Visualize ECG signal with arrhythmia labels. 
    
    USE CASE IS FOR DIRECTLY USING THE WFDB OBJ.
    
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
def visualize_ecg_with_annotationsV2(patient_data, patient_id, num_beats=5, fs=360):
    """
    Visualizes a sequence of ECG beats for a specific patient on a single plot,
    with annotations indicating each beat's type, adjusted for sample offset.

    Parameters:
        patient_data (dict): Dictionary containing ECG data for all patients.
        patient_id (str): The ID of the patient to visualize.
        num_beats (int): The number of beats to visualize.
        fs (int): Sampling frequency in Hz, default is 360.
    """
    # Check if the patient_id exists in the data
    if patient_id not in patient_data:
        print(f"Patient ID {patient_id} not found in data.")
        return
    
    beats = patient_data[patient_id]
    
    # Limit the number of beats to visualize based on available data
    num_beats = min(num_beats, len(beats))
    
    # Initialize plot
    plt.figure(figsize=(15, 6))
    
    # Concatenate all beat segments for each lead
    lead_keys = [key for key in beats[0].keys() if key.startswith('signal_lead_')]
    for lead_key in lead_keys:
        all_segments = []
        annotations = []
        
        # Gather signal segments and annotations for each beat
        for i in range(num_beats):
            beat = beats[i]
            segment = beat[lead_key]
            segment_length = len(segment)
            midpoint_offset = segment_length // 2  # Center of each beat
            
            # Append the segment to the continuous signal
            all_segments.extend(segment)
            
            # Calculate annotation position with offset adjustment
            annotation_position = len(all_segments) - midpoint_offset
            annotations.append((annotation_position, beat['label']))
        
        # Plot concatenated signals for this lead
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
def visualize_ecg(patient_data, patient_id, num_beats=5):
    """
    Visualizes the ECG signal segments for a specific patient. Shows on DIFFERENT graphs.

    Parameters:
        patient_data (dict): Dictionary containing ECG data for all patients.
        patient_id (str): The ID of the patient to visualize.
        num_beats (int): The number of beats to visualize.
    """
    # Check if the patient_id exists in the data
    if patient_id not in patient_data:
        print(f"Patient ID {patient_id} not found in data.")
        return
    
    beats = patient_data[patient_id]
    
    # Limit the number of beats to visualize based on available data
    num_beats = min(num_beats, len(beats))
    
    # Plot each beat's leads as a subplot
    plt.figure(figsize=(10, 2 * num_beats))
    
    for i in range(num_beats):
        beat = beats[i]
        
        # Plot each lead in separate subplots
        for lead_key in [key for key in beat.keys() if key.startswith('signal_lead_')]:
            plt.subplot(num_beats, len([k for k in beat.keys() if k.startswith('signal_lead_')]), i * len([k for k in beat.keys() if k.startswith('signal_lead_')]) + int(lead_key[-1]))
            
            # Plot the signal segment for this lead
            plt.plot(beat[lead_key])
            plt.title(f'Beat {i+1} - {lead_key}')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()