import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def check_for_missing_values(signal_data):
    """
    Check for missing or null values in ECG signal data for each patient.
    
    Args:
        signal_data (dict): A dictionary where each key is a patient ID, and each value is
                            a list of beat entries. Each beat entry is a dictionary with
                            labels and signal segments for each lead.
    
    Returns:
        dict: A dictionary containing patient IDs as keys and lists of indices of beats
              with missing values as values. If no missing values are found, it returns an empty dict.
    """
    missing_values_report = {}

    # Loop through each patient in the signal data
    for patient_id, beats in signal_data.items():
        missing_beats = []  # Track indices of beats with missing values for this patient
        
        # Loop through each beat in the patient's data
        for i, beat_entry in enumerate(beats):
            for lead, segment in beat_entry.items():
                if lead.startswith("signal_lead"):  # Only check signal leads
                    # Check if there are any NaN values in the segment
                    if np.isnan(segment).any():
                        missing_beats.append(i)
                        break  # Stop checking other leads for this beat if NaN is found

        # Add to report if any missing values are found for this patient
        if missing_beats:
            missing_values_report[patient_id] = missing_beats

    if missing_values_report:
        print("Missing values found in the following patients and beats:")
        for patient_id, beat_indices in missing_values_report.items():
            print(f"Patient {patient_id}: Beats with missing values at indices {beat_indices}")
    else:
        print("No missing values found in any patient records.")

    return missing_values_report

##

def check_segment_lengths(signal_data, target_length):
    """
    Check if each ECG segment for each patient has the specified target length.
    
    Args:
        signal_data (dict): A dictionary where each key is a patient ID, and each value is
                            a list of beat entries. Each beat entry is a dictionary with
                            labels and signal segments for each lead.
        target_length (int): The target segment length for each ECG beat segment.
    
    Returns:
        dict: A dictionary containing patient IDs as keys and lists of indices of beats
              that do not meet the target length. If all segments match the target length,
              it returns an empty dictionary.
    """
    inconsistent_lengths = {}

    # Loop through each patient in the signal data
    for patient_id, beats in signal_data.items():
        inconsistent_beats = []  # Track indices of beats with inconsistent length for this patient
        
        # Loop through each beat in the patient's data
        for i, beat_entry in enumerate(beats):
            for lead, segment in beat_entry.items():
                if lead.startswith("signal_lead"):  # Only check signal leads
                    if len(segment) != target_length:
                        inconsistent_beats.append(i)
                        break  # Stop checking other leads for this beat if length is inconsistent

        # Add to report if any inconsistent lengths are found for this patient
        if inconsistent_beats:
            inconsistent_lengths[patient_id] = inconsistent_beats

    if inconsistent_lengths:
        print("Inconsistent segment lengths found in the following patients and beats:")
        for patient_id, beat_indices in inconsistent_lengths.items():
            print(f"Patient {patient_id}: Beats with inconsistent length at indices {beat_indices}")
    else:
        print("All segments match the target length.")

    return inconsistent_lengths

##

def calculate_rr_intervals(signal_data, fs=360):
    """
    Calculate the R-R (peak-to-peak) intervals between consecutive beats for each patient.
    
    Args:
        signal_data (dict): A dictionary where each key is a patient ID, and each value is
                            a list of beat entries. Each beat entry is a dictionary with
                            labels and signal segments for each lead.
        fs (int): Sampling frequency of the ECG signals in Hz (default is 360).
    
    Returns:
        dict: A dictionary with patient IDs as keys and lists of R-R intervals (in seconds)
              for each consecutive beat.
    """
    rr_intervals = {}

    # Iterate over each patient in the signal data
    for patient_id, beats in signal_data.items():
        patient_rr_intervals = []
        
        # Loop through beats and calculate intervals between consecutive R-peaks
        for i in range(1, len(beats)):
            # Assuming each segment is centered on an R-peak, calculate interval in samples
            # Get the sample index for the R-peak in each consecutive segment
            # Here, you assume that each segment represents one R-peak
            r_peak_interval_samples = fs  # Time between segments if each segment represents an R-peak
            
            # Convert to time in seconds
            r_peak_interval_seconds = r_peak_interval_samples / fs
            
            # Append the interval in seconds
            patient_rr_intervals.append(r_peak_interval_seconds)
        
        # Store R-R intervals for this patient
        rr_intervals[patient_id] = patient_rr_intervals

    return rr_intervals

## FILTERS YAY

def apply_high_pass_filter(signal, cutoff=0.5, fs=360, order=4, padlen=10):
    """
    Apply a high-pass filter to remove baseline wander.
    
    Args:
        signal (np.array): ECG signal, expected to be 1D (samples) or 2D (samples, leads).
        cutoff (float): Cutoff frequency in Hz.
        fs (int): Sampling frequency.
        order (int): Order of the filter.
        padlen (int): Length of the padding for filtfilt function.
    
    Returns:
        np.array: Filtered signal with the same shape as input.
    """
    # Ensure signal is a NumPy array
    signal = np.array(signal)
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    if signal.ndim == 1:
        return filtfilt(b, a, signal, padlen=padlen)
    else:
        # Multi-lead case: apply filter independently to each lead
        filtered_signal = np.zeros_like(signal)
        for lead in range(signal.shape[1]):
            filtered_signal[:, lead] = filtfilt(b, a, signal[:, lead], padlen=padlen)
        return filtered_signal

##

def apply_notch_filter(signal, notch_freq=60, fs=360, quality_factor=30):
    """
    Apply a notch filter to remove powerline interference.
    
    Args:
        signal (np.array): ECG signal, expected to be 1D (samples) or 2D (samples, leads).
        notch_freq (float): Frequency to notch out (e.g., 50 or 60 Hz).
        fs (int): Sampling frequency.
        quality_factor (float): Quality factor of the notch filter.
    
    Returns:
        np.array: Filtered signal with the same shape as input.
    """
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        filtered_signal = np.zeros_like(signal)
        for lead in range(signal.shape[1]):
            filtered_signal[:, lead] = filtfilt(b, a, signal[:, lead])
        return filtered_signal

##

def apply_low_pass_filter(signal, cutoff=40, fs=360, order=4):
    """
    Apply a low-pass filter to remove high-frequency noise.
    
    Args:
        signal (np.array): ECG signal, expected to be 1D (samples) or 2D (samples, leads).
        cutoff (float): Cutoff frequency in Hz.
        fs (int): Sampling frequency.
        order (int): Order of the filter.
    
    Returns:
        np.array: Filtered signal with the same shape as input.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        filtered_signal = np.zeros_like(signal)
        for lead in range(signal.shape[1]):
            filtered_signal[:, lead] = filtfilt(b, a, signal[:, lead])
        return filtered_signal

##


def apply_moving_average_filter(signal, window_size=5):
    """
    Apply a moving average filter to smooth remaining noise for each lead in a multi-lead ECG signal.
    
    Args:
        signal (np.array): ECG signal, expected to be 1D (samples) or 2D (samples, leads).
        window_size (int): Number of samples for the moving average window.
    
    Returns:
        np.array: Smoothed signal with the same shape as input.
    """
    if signal.ndim == 1:
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    else:
        smoothed_signal = np.zeros_like(signal)
        for lead in range(signal.shape[1]):
            smoothed_signal[:, lead] = np.convolve(signal[:, lead], np.ones(window_size) / window_size, mode='same')
        return smoothed_signal


##

def normalize_signal(signal):
    """
    Normalize the ECG signal between 0 and 1 using min-max scaling.
    
    Args:
        signal (np.array): ECG signal, expected to be 1D (samples) or 2D (samples, leads).
    
    Returns:
        np.array: Normalized signal with the same shape as input.
    """
    if signal.ndim == 1:
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    else:
        normalized_signal = np.zeros_like(signal)
        for lead in range(signal.shape[1]):
            normalized_signal[:, lead] = (signal[:, lead] - np.min(signal[:, lead])) / (np.max(signal[:, lead]) - np.min(signal[:, lead]))
        return normalized_signal

##

def apply_filters_to_all_patients(all_data, fs=360):
    """
    Apply high-pass, notch, low-pass, moving average filters, and normalization to each patient's ECG data.
    
    Args:
        all_data (dict): Dictionary where each key is a patient ID and each value is the ECG signal.
        fs (int): Sampling frequency of the ECG signals.
    
    Returns:
        dict: Dictionary with the same keys as `all_data`, where each value is the fully preprocessed ECG signal.
    """
    processed_data = {}
    
    # Filter parameters
    high_pass_cutoff = 0.5
    notch_freq = 60
    low_pass_cutoff = 40
    window_size = 5  # For moving average filter

    for patient_id, signal in all_data.items():
        print(f"\nProcessing data for patient {patient_id}")
        
        # Ensure signal is a NumPy array and print initial shape and type
        signal = np.array(signal)
        print(f"Initial signal shape: {signal.shape}, type: {type(signal)}")
        
        # Step 1: Apply high-pass filter
        try:
            signal = apply_high_pass_filter(signal, cutoff=high_pass_cutoff, fs=fs)
            print(f"After high-pass filter: {signal.shape}, type: {type(signal)}")
        except ValueError as e:
            print(f"Skipping high-pass filter for patient {patient_id}: {e}")
            continue
        
        # Step 2: Apply notch filter
        signal = apply_notch_filter(signal, notch_freq=notch_freq, fs=fs)
        print(f"After notch filter: {signal.shape}, type: {type(signal)}")
        
        # Step 3: Apply low-pass filter
        signal = apply_low_pass_filter(signal, cutoff=low_pass_cutoff, fs=fs)
        print(f"After low-pass filter: {signal.shape}, type: {type(signal)}")
        
        # Step 4: Apply moving average filter
        signal = apply_moving_average_filter(signal, window_size=window_size)
        print(f"After moving average filter: {signal.shape}, type: {type(signal)}")
        
        # Step 5: Normalize signal between 0 and 1
        signal = normalize_signal(signal)
        print(f"After normalization: {signal.shape}, type: {type(signal)}")
        
        # Store the processed signal
        processed_data[patient_id] = signal

    return processed_data
##