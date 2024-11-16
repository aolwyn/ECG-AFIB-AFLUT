import numpy as np
import matplotlib.pyplot as plt

def segment_ecg_data(ecg_data, snippet_length=200, samples_before=90, samples_after=90):
    """
    Segment ECG data into fixed-length snippets around each beat.

    Parameters:
        ecg_data (dict): Dictionary with patient IDs as keys. Each value is a list of beat entries,
                         where each entry contains 'label' and signal segments for each lead.
        snippet_length (int): Total length of each snippet.
        samples_before (int): Number of samples to include before each peak.
        samples_after (int): Number of samples to include after each peak.

    Returns:
        dict: A dictionary with patient IDs as keys, where each value is a list
              of segmented beat entries, with each beat entry containing label
              and fixed-length signal segments.
    """
    segmented_data = {}

    for patient_id, beats in ecg_data.items():
        segmented_data[patient_id] = []
        
        for beat in beats:
            segmented_beat = {'label': beat['label']}
            
            # Segment and pad each lead signal
            for lead_key, signal in beat.items():
                if lead_key.startswith('signal_lead_'):
                    peak_position = samples_before  # Center the peak
                    start = max(0, peak_position - samples_before)
                    end = min(len(signal), peak_position + samples_after)
                    
                    segment = signal[start:end]
                    
                    # Pad to the fixed snippet length if necessary
                    if len(segment) < snippet_length:
                        pad_width = snippet_length - len(segment)
                        segment = np.pad(segment, (0, pad_width), 'constant')
                    
                    segmented_beat[lead_key] = segment[:snippet_length]
            
            segmented_data[patient_id].append(segmented_beat)
    
    return segmented_data

def visualize_segmented_data(segmented_data, patient_id, num_beats=5):
    """
    Visualize segmented ECG snippets for a specific patient in a single figure with subplots.

    Parameters:
        segmented_data (dict): Dictionary with patient IDs as keys and lists of segmented beats as values.
        patient_id (str): ID of the patient to visualize.
        num_beats (int): Number of beats to visualize in a single figure.
    """
    if patient_id not in segmented_data:
        print(f"Patient ID {patient_id} not found in segmented data.")
        return
    
    beats = segmented_data[patient_id][:num_beats]
    
    fig, axes = plt.subplots(num_beats, 1, figsize=(12, 4 * num_beats), sharex=True)
    
    for i, beat in enumerate(beats):
        ax = axes[i] if num_beats > 1 else axes  # Support for single subplot case
        
        for lead_key, segment in beat.items():
            if lead_key.startswith('signal_lead_'):
                ax.plot(segment, label=lead_key)
        
        ax.set_title(f"Beat {i+1} - Label: {beat['label']}")
        ax.set_ylabel("Amplitude")
        ax.legend()
    
    plt.xlabel("Sample")
    plt.suptitle(f"Patient {patient_id} - First {num_beats} Segmented Beats", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()