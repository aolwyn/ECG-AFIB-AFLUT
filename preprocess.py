import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def check_for_missing_values(patient_data):
    """
    Check for missing or null values in ECG signal data for each patient.

    Parameters:
        patient_data (dict): Dictionary where each key is a patient ID, and each value is
                             a list of rhythm entries. Each rhythm entry contains 'label'
                             and signal segments for each lead.

    Returns:
        dict: A dictionary containing patient IDs as keys and lists of indices of entries
              with missing values. If no missing values are found, it returns an empty dict.
    """
    import numpy as np
    from collections import defaultdict

    missing_values_report = defaultdict(list)

    # Loop through each patient in the data
    for patient_id, entries in patient_data.items():
        # Check each entry for missing values
        for i, entry in enumerate(entries):
            for key, value in entry.items():
                if key.startswith("signal_lead_"):  # Check only signal leads
                    if np.isnan(value).any():  # Check for NaN values in the signal
                        missing_values_report[patient_id].append(i)
                        break  # No need to check other leads for this entry
        print("completed patient #"+patient_id)                    
    # Print a summary of the missing values
    if missing_values_report:
        print("Missing values found in the following patients and entries:")
        for patient_id, entry_indices in missing_values_report.items():
            print(f"Patient {patient_id}: Entries with missing values at indices {entry_indices}")
    else:
        print("No missing values found in any patient records.")

    return dict(missing_values_report)

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

def calculate_rr_intervals(patient_data, fs=250):
    """
    Calculate the R-R (peak-to-peak) intervals between consecutive rhythm segments for each patient.

    Parameters:
        patient_data (dict): Dictionary where each key is a patient ID, and each value is
                             a list of rhythm entries. Each rhythm entry contains a label
                             and signal segments for each lead.
        fs (int): Sampling frequency of the ECG signals in Hz (default is 250 Hz).

    Returns:
        dict: A dictionary with patient IDs as keys and lists of R-R intervals (in seconds)
              for consecutive rhythm segments.
    """
    rr_intervals = {}

    # Iterate over each patient's data
    for patient_id, entries in patient_data.items():
        patient_rr_intervals = []

        # Extract R-R intervals between consecutive rhythm segments
        for i in range(1, len(entries)):
            # Check if the signal_lead_1 value is an array
            signal_segment = entries[i - 1]["signal_lead_1"]
            if isinstance(signal_segment, (np.ndarray, list)):
                # Calculate the R-R interval in samples
                r_peak_interval_samples = len(signal_segment)  # Use the length of the previous segment
                # Convert interval from samples to seconds
                r_peak_interval_seconds = r_peak_interval_samples / fs
                # Append to the list of R-R intervals for this patient
                patient_rr_intervals.append(r_peak_interval_seconds)
            else:
                print(f"Warning: Unexpected format for signal_lead_1 in patient {patient_id}, entry {i-1}")
                continue

        # Store the patient's R-R intervals
        rr_intervals[patient_id] = patient_rr_intervals

    return rr_intervals



## Test Signal

def test_signal(signal):
    # test normalization between -1 and 1.
    min_val, max_val = signal.min(), signal.max()
    # print("lowest value:",min_val)
    
    if max_val - min_val != 0:
        signal = 2 * (signal - min_val) / (max_val - min_val) - 1
    return signal




## FILTERS YAY

def apply_high_pass_filter(signal, cutoff=0.5, fs=360, order=4, padlen=10):
    """
    Apply a high-pass filter to remove low-frequency noise such as baseline wander.

    Args:
        signal (np.array): ECG signal, expected to be 1D (samples) or 2D (samples, leads).
        cutoff (float): Cutoff frequency in Hz, default is 0.5 Hz.
        fs (int): Sampling frequency of the ECG signal in Hz, default is 360 Hz.
        order (int): The order of the filter, default is 4.
        padlen (int): The padding length used by `filtfilt` to reduce edge effects, default is 10.

    Returns:
        np.array: Filtered ECG signal with the same shape as the input.

    Note:
        If the signal is 2D, the filter is applied independently to each lead.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    if signal.ndim == 1:
        return filtfilt(b, a, signal, padlen=padlen)
    else:
        filtered_signal = np.zeros_like(signal)
        for lead in range(signal.shape[1]):
            filtered_signal[:, lead] = filtfilt(b, a, signal[:, lead], padlen=padlen)
        return filtered_signal

##

def apply_notch_filter(signal, notch_freq=60, fs=360, quality_factor=30):
    """
    Apply a notch filter to remove powerline interference at a specified frequency.

    Args:
        signal (np.array): ECG signal, expected to be 1D (samples) or 2D (samples, leads).
        notch_freq (float): Frequency to be notched out (e.g., 50 or 60 Hz), default is 60 Hz.
        fs (int): Sampling frequency of the ECG signal in Hz, default is 360 Hz.
        quality_factor (float): Quality factor of the notch filter, default is 30.
                               A higher quality factor gives a narrower notch.

    Returns:
        np.array: Filtered ECG signal with the same shape as the input.

    Note:
        If the signal is 2D, the filter is applied independently to each lead.
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
    Apply a low-pass filter to remove high-frequency noise from the ECG signal.

    Args:
        signal (np.array): ECG signal, expected to be 1D (samples) or 2D (samples, leads).
        cutoff (float): Cutoff frequency in Hz, default is 40 Hz.
        fs (int): Sampling frequency of the ECG signal in Hz, default is 360 Hz.
        order (int): The order of the filter, default is 4.

    Returns:
        np.array: Filtered ECG signal with the same shape as the input.

    Note:
        If the signal is 2D, the filter is applied independently to each lead.
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
    Apply a moving average filter to smooth the ECG signal by reducing high-frequency noise.

    Args:
        signal (np.array): ECG signal, expected to be 1D (samples) or 2D (samples, leads).
        window_size (int): Number of samples for the moving average window, default is 5.

    Returns:
        np.array: Smoothed ECG signal with the same shape as the input.

    Note:
        If the signal is 2D, the filter is applied independently to each lead.
        The window size should be chosen based on the sampling frequency and desired smoothing level.
    """
    # Ensure signal is a NumPy array
    signal = np.array(signal)
    
    if signal.ndim == 1:
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    else:
        smoothed_signal = np.zeros_like(signal)
        for lead in range(signal.shape[1]):
            smoothed_signal[:, lead] = np.convolve(signal[:, lead], np.ones(window_size) / window_size, mode='same')
        return smoothed_signal



##

def normalize_signal_0_to_1(signal):
    """
    Normalize the ECG signal values between 0 and 1 using min-max scaling.

    Args:
        signal (np.array): ECG signal, expected to be 1D (samples) or 2D (samples, leads).

    Returns:
        np.array: Normalized ECG signal with values between 0 and 1, with the same shape as the input.

    Note:
        If the signal is multi-lead (2D), each lead is normalized independently.
    """
    # Ensure signal is a NumPy array
    signal = np.array(signal)
    
    if signal.ndim == 1:
        # Min-max normalization for 1D signal
        min_val, max_val = signal.min(), signal.max()
        if max_val - min_val != 0:
            return (signal - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(signal)  # If the signal is constant, return zero array
    else:
        # Min-max normalization for each lead in multi-lead signals
        normalized_signal = np.zeros_like(signal)
        for lead in range(signal.shape[1]):
            min_val, max_val = signal[:, lead].min(), signal[:, lead].max()
            if max_val - min_val != 0:
                normalized_signal[:, lead] = (signal[:, lead] - min_val) / (max_val - min_val)
            else:
                normalized_signal[:, lead] = np.zeros(signal.shape[0])  # Zero array if constant
        return normalized_signal

##

def apply_filters_and_normalization(patient_data, fs=360):
    """
    Apply high-pass, notch, low-pass, moving average filters, and normalization to each lead
    in each beat for all patients in patient_data, with progress updates.

    Args:
        patient_data (dict): Dictionary containing ECG data for all patients.
        fs (int): Sampling frequency of the ECG signals, default is 360 Hz.

    Returns:
        dict: Processed patient_data with filtered and normalized signals.
    """
    # Define filter parameters
    high_pass_cutoff = 0.5   # Hz, for removing baseline wander
    notch_freq = 60          # Hz, for removing powerline interference
    low_pass_cutoff = 40     # Hz, for removing high-frequency noise
    moving_average_window = 5 # Window size for smoothing
    
    for patient_id, beats in patient_data.items():
        for beat in beats:
            for lead_key in beat.keys():
                if lead_key.startswith('signal_lead_'):
                    # Apply high-pass filter
                    beat[lead_key] = apply_high_pass_filter(
                        beat[lead_key], cutoff=high_pass_cutoff, fs=fs
                    )
                    
                    # Apply notch filter
                    beat[lead_key] = apply_notch_filter(
                        beat[lead_key], notch_freq=notch_freq, fs=fs
                    )
                    
                    # Apply low-pass filter
                    beat[lead_key] = apply_low_pass_filter(
                        beat[lead_key], cutoff=low_pass_cutoff, fs=fs
                    )
                    
                    # Apply moving average filter
                    beat[lead_key] = apply_moving_average_filter(
                        beat[lead_key], window_size=moving_average_window
                    )
                    
                    # Apply normalization to 0-1 range
                    beat[lead_key] = normalize_signal_0_to_1(beat[lead_key])
        
        # Print a progress update for each patient
        print(f"Processing complete for patient {patient_id}")
    
    return patient_data
##