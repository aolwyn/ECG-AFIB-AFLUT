import os
import wfdb 
from wfdb import processing
import torch
import numpy as np
from scipy.signal import decimate, find_peaks, resample 
import matplotlib.pyplot as plt

# Define a dictionary for mapping MIT-BIH annotations to classes
LABEL_MAPPING = {
    '(N': 'normal',  
    'N': 'normal',
    '(AFIB': 'atrial_fibrillation',
    '(AFL': 'atrial_flutter',
    'AFIB': 'atrial_fibrillation',
    'AFL': 'atrial_flutter',
    'AFIB)': 'atrial_fibrillation',
    'AFL)': 'atrial_flutter',
    '(B' : 'ventricular_bigeminy',
    'B' : 'ventricular_bigeminy'

    # Add other relevant mappings if needed <-- added redundant cases incase it for some reason wasn't working
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
    # """
    # Convert annotation symbols to class labels based on LABEL_MAPPING.
    
    # Parameters:
    #     annotation (wfdb.Annotation): Annotation object containing beat labels.
    
    # Returns:
    #     list: List of labels mapped from annotation symbols.
    # """
    #NOTE: below is for beat level. 
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
        list: List of cleaned rhythm types found in aux_note, or 'Other' if no rhythm information.
    """
    #NOTE: BELOW RETURNS THE RHYTHM BASED LABELS. 
    labels = []
    current_label = "Other"  # Default label

    for aux in annotation.aux_note:
        # Remove any trailing null or unwanted characters
        cleaned_aux = aux.rstrip('\x00')

        # Check if the cleaned annotation is a valid rhythm label
        if cleaned_aux in LABEL_MAPPING:
            current_label = LABEL_MAPPING[cleaned_aux]  # Update the current label
        # Append the current label (propagate it forward)
        labels.append(current_label)

    return labels


##

from wfdb import processing

def load_all_records(data_dir):
    """
    Load ECG data and annotations for all records in the given directory,
    applying downsampling to the target sampling frequency.

    Parameters:
        data_dir (str): Directory containing MIT-BIH ECG records (.dat files).
    
    Returns:
        dict: A dictionary with patient IDs as keys, and each value is a list
              of entries where each entry has a 1:1 mapping to rhythm labels.
    """
    all_data = {}

    for filename in os.listdir(data_dir):
        if filename.endswith('.dat'):
            record_path = os.path.join(data_dir, filename.replace('.dat', ''))
            
            # Load ECG signal and annotations
            record, annotation = load_ecg_data(record_path)
            signal = record.p_signal  # ECG signal (2D array: samples x channels)
            original_fs = int(record.fs)  # Original sampling frequency
            num_leads = signal.shape[1]  # Get the number of leads dynamically
            
            # Apply downsampling to both signal and annotations
            downsampled_signal, downsampled_annotation = processing.resample_multichan(
                signal, 
                annotation, 
                fs = original_fs, 
                fs_target=250  
            )
            
            # Rhythm label propagation logic
            current_label = "Other"  # Default label
            labels = []  # Store 1:1 rhythm labels
            start_idx = 0

            for i, aux in enumerate(downsampled_annotation.aux_note):
                # Clean the auxiliary note
                cleaned_aux = aux.rstrip('\x00')

                # If a new rhythm label is detected, update the current label
                if cleaned_aux in LABEL_MAPPING:
                    current_label = LABEL_MAPPING[cleaned_aux]
                
                # Determine the range for this label
                end_idx = downsampled_annotation.sample[i]
                labels.extend([current_label] * (end_idx - start_idx))
                start_idx = end_idx
            
            # Handle remaining samples
            labels.extend([current_label] * (len(downsampled_signal) - start_idx))

            # Get patient ID from filename
            patient_id = filename.split('.')[0]
            
            # Initialize patient data if not already present
            if patient_id not in all_data:
                all_data[patient_id] = []

            # Create 1:1 mapping for every segment
            for idx, label in enumerate(labels):
                entry = {'label': label}

                # Add signal segments for each lead
                for lead_index in range(num_leads):
                    entry[f'signal_lead_{lead_index+1}'] = downsampled_signal[idx, lead_index]

                # Append the entry to the patient's data
                all_data[patient_id].append(entry)
            print("done patient: "+record_path)
    
    return all_data
