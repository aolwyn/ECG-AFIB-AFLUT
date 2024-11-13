import os
import wfdb 
from wfdb import processing
import torch
import numpy as np
from scipy.signal import decimate, find_peaks, resample 
import matplotlib.pyplot as plt

# Define a dictionary for mapping MIT-BIH annotations to classes
LABEL_MAPPING = {
    '(N': 'normal',  # Normal beat
    '(AFIB': 'atrial_fibrillation',
    '(AFL': 'atrial_flutter',
    # Add other relevant mappings if needed
}

##

def get_device_info():
    """
    Check and display device information, setting it to GPU if available, otherwise CPU.
    Prints details such as CUDA version, PyTorch version, memory usage, and device name.
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    
    # If GPU is available, display CUDA and memory information
    if device.type == 'cuda':
        print('CUDA Device Name:', torch.cuda.get_device_name(0))
        print('CUDA Version:', torch.version.cuda)
        print('PyTorch Version:', torch.__version__)
        
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')
        
        # Total memory and free memory calculations
        total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory = total_memory - torch.cuda.memory_reserved(0)
        
        print('Total Memory:', round(total_memory / 1024**3, 1), 'GB')
        print('Free Memory:', round(free_memory / 1024**3, 1), 'GB')
        
        # Test tensor creation on GPU to verify setup
        x = torch.rand(2, 3).to(device)
        print('Test Tensor on CUDA:', x)
    else:
        print('Using the CPU, no GPU found')

##

'''#NOTE:
Use load_record when you want to work directly with the ECG signal data (signal) and annotations, without needing additional metadata from the record object.

Use load_ecg_data if you need access to the entire wfdb.Record object, which may include metadata like sampling frequency, signal duration, or information about the recording channels.
'''

def load_record(record_path):
    """
    Load ECG signal and annotations from a given path.
    
    Args:
        record_path (str): Path to the record (without file extension).
    
    Returns:
        tuple: Tuple of (signal, annotations), where signal is a 2D numpy array
               of the ECG signal, and annotations are the WFDB annotation object.
    """
    # Load the ECG record and annotations
    record = wfdb.rdrecord(record_path)
    annotations = wfdb.rdann(record_path, 'atr')
    signal = record.p_signal  # Extract the ECG signal

    # Define original and target sampling rates as integers
    original_fs = int(360)
    target_fs = int(250)

    # Downsample signal and annotations together using wfdb.processing.resample_multichan
    downsampled_signal, annotations = processing.resample_multichan(
        signal,
        annotations,  # Pass annotation sample indices for downsampling
        original_fs,
        target_fs
    )

    return downsampled_signal, annotations



def load_ecg_data(record_path):
    """
    Load the ECG signal and annotations for a given record.
    
    Parameters:
        record_path (str): Path to the record (without file extension).
    
    Returns:
        tuple: A tuple (record, annotation) where record is the ECG signal data
               and annotation is the annotation object.
    """
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    
    return record, annotation

##

def get_labels(annotation):
    """
    Convert annotation symbols to class labels based on LABEL_MAPPING.
    
    Parameters:
        annotation (wfdb.Annotation): Annotation object containing beat labels.
    
    Returns:
        list: List of labels mapped from annotation symbols.
    """
    # labels = []
    # for symbol in annotation.symbol:
    #     if symbol in LABEL_MAPPING:
    #         labels.append(LABEL_MAPPING[symbol])
    #     else:
    #         #labels.append('other')  # Handle other labels as 'other'
    #         print(symbol)
    # return labels

    """
    Extract rhythm annotations from the 'aux_note' field in the annotation object, 
    removing any unwanted trailing characters.
    
    Parameters:
        annotation (wfdb.Annotation): Annotation object containing beat and rhythm annotations.
    
    Returns:
        list: List of cleaned rhythm types found in aux_note, or 'Unknown' if no rhythm information.
    """
    rhythms = []
    for aux in annotation.aux_note:
        # Remove the last character if it's a null or unwanted character
        cleaned_aux = aux.rstrip('\x00')  # Remove null character or any trailing whitespace
        
        if cleaned_aux in LABEL_MAPPING:
            rhythms.append(LABEL_MAPPING[cleaned_aux])
        elif cleaned_aux:  # If there is a cleaned aux_note and it's not mapped
            #print(f"Unmapped rhythm found: {cleaned_aux}")
            rhythms.append(cleaned_aux)  # Append the original, cleaned rhythm type
        else:
            rhythms.append('Unknown')  # Use 'Unknown' if no aux_note is available
    return rhythms

##

def load_all_records(data_dir):
    """
    Load ECG data and annotations for all records in the given directory and 
    compile them into a nested dictionary.
    
    Parameters:
        data_dir (str): Directory containing MIT-BIH ECG records (.dat files).
    
    Returns:
        dict: A dictionary with patient IDs as keys, and each value is a list
              of beat entries. Each beat entry is a dictionary containing label
              and signal segments for each lead.
    """
    all_data = {}

    for filename in os.listdir(data_dir):
        if filename.endswith('.dat'):
            record_path = os.path.join(data_dir, filename.replace('.dat', ''))
            
            # Load ECG signal and annotations
            record, annotation = load_ecg_data(record_path)
            signal = record.p_signal  # ECG signal (2D array: samples x channels)
            num_leads = signal.shape[1]  # Get the number of leads dynamically
            
            # Get labels for each beat
            labels = get_labels(annotation)
            
            # Get patient ID from filename
            patient_id = filename.split('.')[0]
            
            # Initialize patient data if not already present
            if patient_id not in all_data:
                all_data[patient_id] = []
            
            # Process each beat in the annotation. 
            #NOTE this line makes it so it only loads in what's in the LABEL_MAPPING dictionary
            for i, label in enumerate(labels):
                if label == 'other':  # Ignore irrelevant labels
                    continue
                
                # Extract segments around each beat for each lead
                start = max(0, annotation.sample[i] - 90)  # Adjust based on sampling rate
                end = min(len(signal), annotation.sample[i] + 90)
                
                # Prepare a dictionary for this beat entry
                beat_entry = {'label': label}
                
                # Add signal segment for each lead
                for lead_index in range(num_leads):
                    segment = signal[start:end, lead_index]  # Segment for current lead
                    beat_entry[f'signal_lead_{lead_index+1}'] = segment
                
                # Append this beat entry to the patient's data
                all_data[patient_id].append(beat_entry)
    
    return all_data