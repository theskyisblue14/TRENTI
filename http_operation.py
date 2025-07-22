import os
import re
import time
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import traceback
import scipy.stats
from scipy.signal import butter, lfilter, find_peaks
import subprocess
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from lecroy3 import Lecroy

# Configuration parameters - These will be modified by TRENTI GUI
ROUTER_IP = '10.205.1.111'
OSCILLOSCOPE_IP = '10.205.1.95'
TARGET_CGI = 'hedwig.cgi'
CAPTURE_DELAY = 0.1  # Seconds to wait between triggering and capturing
PLOT_WAVEFORMS = True  # Display waveforms using matplotlib
SAVE_WAVEFORMS = False  # Option to save waveform plots (disabled)

# SCA Configuration - These will be modified by TRENTI GUI
SCA_CHANNEL = 'C3'
SCA_VOLTAGE_DIV = 0.01  # Voltage division in V/div
SCA_TIME_DIV = '500E-9'  # Time division
SCA_SAMPLE_RATE = 10000000000  # Sample rate in Hz (10 GS/s)
SCA_MEMORY_SIZE = '10K'  # Memory size
SCA_TRIGGER_LEVEL = 0.02  # Trigger level

# HTTP/Curl Configuration - These will be modified by TRENTI GUI
HTTP_METHOD = 'POST'
CONTENT_TYPE = 'application/x-www-form-urlencoded'
COOKIE_PATTERN = 'uid={VALUE}'
POST_DATA = 'dummy=1'
ADDITIONAL_HEADERS = ''
CURL_TIMEOUT = 10
CURL_ADVANCED_OPTIONS = ''

# Log configuration
# LOG_PATH = './fuzzing_logs/'

CRASH_DIRS = []  # Will be populated by load_configuration_from_environment()
EM_WAVEFORMS_PATH = './waveforms/'  # Default path
HTTP_RESPONSES_PATH = './HTTP_responses/'  # Solo para archivo 2
EXEC_TIMES_PATH = './exec_times/'  # Solo para archivo 2


# Create directories
# os.makedirs(LOG_PATH, exist_ok=True)

def load_curl_configuration():
    """Load curl configuration from environment variables"""
    global HTTP_METHOD, CONTENT_TYPE, COOKIE_PATTERN, POST_DATA
    global ADDITIONAL_HEADERS, CURL_TIMEOUT, CURL_ADVANCED_OPTIONS
    
    # Load HTTP configuration
    HTTP_METHOD = os.environ.get('TRENTI_HTTP_METHOD', HTTP_METHOD)
    CONTENT_TYPE = os.environ.get('TRENTI_CONTENT_TYPE', CONTENT_TYPE)
    COOKIE_PATTERN = os.environ.get('TRENTI_COOKIE_PATTERN', COOKIE_PATTERN)
    POST_DATA = os.environ.get('TRENTI_POST_DATA', POST_DATA)
    ADDITIONAL_HEADERS = os.environ.get('TRENTI_ADDITIONAL_HEADERS', ADDITIONAL_HEADERS)
    
    try:
        CURL_TIMEOUT = int(os.environ.get('TRENTI_CURL_TIMEOUT', CURL_TIMEOUT))
    except ValueError:
        print(f"[!] Warning: Invalid CURL_TIMEOUT value, using default: {CURL_TIMEOUT}")
    
    CURL_ADVANCED_OPTIONS = os.environ.get('TRENTI_CURL_ADVANCED_OPTIONS', CURL_ADVANCED_OPTIONS)
    
    print(f"[*] Curl configuration loaded:")
    print(f"    HTTP Method: {HTTP_METHOD}")
    print(f"    Content-Type: {CONTENT_TYPE}")
    print(f"    Cookie Pattern: {COOKIE_PATTERN}")
    print(f"    POST Data: {POST_DATA}")
    print(f"    Additional Headers: {ADDITIONAL_HEADERS}")
    print(f"    Curl Timeout: {CURL_TIMEOUT}s")
    print(f"    Advanced Options: {CURL_ADVANCED_OPTIONS}")

def build_curl_command(router_ip, target_cgi, uid_value):
    """Build curl command based on configuration"""
    
    # Base URL
    url = f'http://{router_ip}/{target_cgi}'
    
    # Start building command
    curl_cmd = ['curl', '-v']
    
    # Add timeout
    curl_cmd.extend(['--max-time', str(CURL_TIMEOUT)])
    
    # Add method-specific options
    if HTTP_METHOD.upper() == 'GET':
        # For GET, append uid as query parameter if needed
        if '{VALUE}' in POST_DATA:
            query_data = POST_DATA.replace('{VALUE}', uid_value)
            url += f'?{query_data}'
        curl_cmd.append(url)
    else:
        # For POST/PUT/DELETE
        curl_cmd.extend(['-X', HTTP_METHOD.upper()])
        curl_cmd.append(url)
        
        # Add Content-Type header
        curl_cmd.extend(['-H', f'Content-Type: {CONTENT_TYPE}'])
        
        # Add POST data
        post_data = POST_DATA.replace('{VALUE}', uid_value)
        curl_cmd.extend(['-d', post_data])
    
    # Add Cookie header
    if COOKIE_PATTERN and '{VALUE}' in COOKIE_PATTERN:
        cookie_value = COOKIE_PATTERN.replace('{VALUE}', uid_value)
        curl_cmd.extend(['-H', f'Cookie: {cookie_value}'])
    
    # Add additional headers
    if ADDITIONAL_HEADERS:
        # Split multiple headers by semicolon or newline
        headers = [h.strip() for h in ADDITIONAL_HEADERS.replace('\n', ';').split(';') if h.strip()]
        for header in headers:
            curl_cmd.extend(['-H', header])
    
    # Add advanced curl options
    if CURL_ADVANCED_OPTIONS:
        # Split advanced options and add them
        advanced_opts = CURL_ADVANCED_OPTIONS.split()
        curl_cmd.extend(advanced_opts)
    
    return curl_cmd

def extract_uid_from_file_content(content_str):
    """Extract UID value from file content with improved parsing"""
    
    # Try different extraction patterns
    uid_patterns = [
        r'HTTP_COOKIE=uid=([^&\s\n]+)',  # Original pattern
        r'uid=([^&\s\n]+)',              # Simple uid= pattern
        r'"uid":\s*"([^"]+)"',           # JSON format
        r'uid:\s*([^&\s\n]+)',           # Colon separator
    ]
    
    for pattern in uid_patterns:
        match = re.search(pattern, content_str)
        if match:
            uid_value = match.group(1).strip()
            return uid_value
    
    # If no pattern matches, use entire content (stripped)
    uid_value = content_str.strip()
    return uid_value

def save_http_response(file_id, execution_number, timestamp_str, http_response, full_response):
    """Save HTTP response to file"""
    try:
        response_filename = f"http_response_{file_id}_{execution_number}_{timestamp_str}.txt"
        response_filepath = os.path.join(HTTP_RESPONSES_PATH, response_filename)
        
        with open(response_filepath, 'w', encoding='utf-8') as f:
            f.write(f"HTTP Response for File ID: {file_id}\n")
            f.write(f"Execution Number: {execution_number}\n")
            f.write(f"Timestamp: {timestamp_str}\n")
            f.write(f"HTTP Status: {http_response}\n")
            f.write("=" * 50 + "\n")
            f.write("Full Response:\n")
            f.write(full_response)
        
        print(f"[*] HTTP response saved: {response_filepath}")
        return response_filepath
    except Exception as e:
        print(f"[!] Error saving HTTP response: {e}")
        return None

def save_execution_time(file_id, execution_number, timestamp_str, execution_time):
    """Save execution time to file"""
    try:
        time_filename = f"exec_time_{file_id}_{execution_number}_{timestamp_str}.txt"
        time_filepath = os.path.join(EXEC_TIMES_PATH, time_filename)
        
        with open(time_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Execution Time for File ID: {file_id}\n")
            f.write(f"Execution Number: {execution_number}\n")
            f.write(f"Timestamp: {timestamp_str}\n")
            f.write(f"Execution Time (seconds): {execution_time:.6f}\n")
            f.write(f"Execution Time (milliseconds): {execution_time * 1000:.3f}\n")
        
        print(f"[*] Execution time saved: {time_filepath}")
        return time_filepath
    except Exception as e:
        print(f"[!] Error saving execution time: {e}")
        return None

class EMCapture:
    """Class to manage EM signal capture using the LeCroy oscilloscope"""

    def __init__(self, oscilloscope_ip=None):
        print("[*] Initializing EM capture system for operation mode")
        self.oscilloscope_ip = oscilloscope_ip or OSCILLOSCOPE_IP
        self.lecroy = Lecroy()
        self._connected = False
        
        # SCA configuration from global variables
        self.channel = SCA_CHANNEL
        self.voltage_div = SCA_VOLTAGE_DIV
        self.time_div = SCA_TIME_DIV
        self.sample_rate = SCA_SAMPLE_RATE
        self.memory_size = SCA_MEMORY_SIZE
        self.trigger_level = SCA_TRIGGER_LEVEL

    def connect(self):
        """Connect to the oscilloscope and configure it for EM capture"""
        try:
            print(f"[*] Connecting to oscilloscope: {self.oscilloscope_ip}")
            self.lecroy.connect(self.oscilloscope_ip)
            self._connected = True
            self._configure_for_em_capture()
            return True
        except Exception as e:
            print(f"[!] Error connecting to oscilloscope: {str(e)}")
            return False

    def _configure_for_em_capture(self):
        """Configure oscilloscope for EM capture using GUI parameters"""
        print(f"[*] Configuring oscilloscope for EM capture on {self.channel}")
        print(f"[*] Using parameters: VDiv={self.voltage_div}V, TDiv={self.time_div}, SRate={self.sample_rate/1e9:.1f}GS/s")
        
        # Configure voltage division
        self.lecroy._scope.write(f"{self.channel}:VDIV {self.voltage_div:g}")
        
        # Configure trigger level
        self.lecroy._scope.write(f"{self.channel}:TRLV {self.trigger_level}")
        
        # Configure sample rate
        self.lecroy.setSampleRate(str(self.sample_rate))
        
        # Configure trigger slope
        self.lecroy._scope.write(f"{self.channel}:TRSL POS")
        
        # Enable channel
        self.lecroy._scope.write(f"{self.channel}:TRACE ON")
        
        # Configure time division
        self.lecroy._scope.write(f"TDIV {self.time_div}")
        
        # Configure trigger mode
        self.lecroy._scope.write("TRMD NORM")
        
        # Set trigger source
        self.lecroy.setTriggerSource(self.channel)
        
        print("[+] Oscilloscope configured successfully for operation mode")

    def arm_trigger(self):
        """Arm the trigger for the next capture"""
        if not self._connected:
            print("[!] Oscilloscope not connected")
            return False
        try:
            self.lecroy._scope.write("TRMD NORM")
            self.lecroy._scope.write("ARM")
            return True
        except Exception as e:
            print(f"[!] Error arming trigger: {str(e)}")
            return False

    def force_trigger(self):
        """Force the trigger to fire"""
        if not self._connected:
            print("[!] Oscilloscope not connected")
            return False
        try:
            self.lecroy._scope.write("FRTR")
            return True
        except Exception as e:
            print(f"[!] Error forcing trigger: {str(e)}")
            return False

    def capture_waveform(self, channel=None):
        """
        Capture raw signal from specified channel

        Args:
            channel: Channel to capture (default uses configured channel)

        Returns:
            Dictionary with raw signal data and parameters
        """
        if not self._connected:
            print("[!] Oscilloscope not connected")
            return None
            
        if channel is None:
            channel = self.channel
            
        try:
            self.lecroy._scope.write(f"{channel}:TRACE ON")
            self.lecroy._scope.write(f"MSIZ {self.memory_size}")
            self.lecroy._scope.write("TRMD NORM")
            self.arm_trigger()
            self.force_trigger()
            self.lecroy._scope.write("CFMT OFF,WORD,BIN")
            self.lecroy._scope.write("CORD HI")
            cmd = f"{channel}:WF? DAT1"
            self.lecroy._scope.write(cmd)
            received_buffer = self.lecroy._scope.read_raw()
            header_len = received_buffer.index(b'#') + 1
            len_digit = int(received_buffer[header_len:header_len+1].decode())
            data_len = int(received_buffer[header_len+1:header_len+1+len_digit].decode())
            data_start = header_len + 1 + len_digit
            data_end = data_start + data_len
            data = received_buffer[data_start:data_end]
            raw_signal = np.frombuffer(data, dtype='>i2')
            vertical_gain = 1.0
            vertical_offset = 0.0
            try:
                gain_query = self.lecroy._scope.query(f"{channel}:INSPECT? 'VERTICAL_GAIN'")
                gain_str = gain_query.strip().replace('"', '').split(':')[-1].strip()
                gain_match = re.search(r'[-+]?(?:\d*\.\d+|\d+)', gain_str)
                if gain_match:
                    vertical_gain = float(gain_match.group())
                offset_query = self.lecroy._scope.query(f"{channel}:INSPECT? 'VERTICAL_OFFSET'")
                offset_str = offset_query.strip().replace('"', '').split(':')[-1].strip()
                offset_match = re.search(r'[-+]?(?:\d*\.\d+|\d+)', offset_str)
                if offset_match:
                    vertical_offset = float(offset_match.group())
            except Exception as e:
                print(f"[!] Error retrieving vertical scaling: {e}")
            scaled_signal = raw_signal * vertical_gain - vertical_offset
            return {
                'signal': scaled_signal,
                'sample_rate': self.sample_rate,
                'vertical_gain': vertical_gain,
                'vertical_offset': vertical_offset
            }
        except Exception as e:
            print(f"[!] Error capturing waveform: {str(e)}")
            traceback.print_exc()
            return None

    def disconnect(self):
        """Disconnect from the oscilloscope"""
        if hasattr(self, 'lecroy') and hasattr(self.lecroy, '_scope'):
            try:
                self.lecroy.disconnect()
                print("[*] Disconnected from oscilloscope")
            except Exception as e:
                print(f"[!] Error disconnecting: {e}")
        else:
            print("[!] No connection established to disconnect")

# Global lists for storing results and signal colors
signal_results = []
signal_colors = []

# Table headers
TABLE_HEADERS = [
    "ID", "RMS", "Mean", "Std Dev", "Crest", "P2P", "Entropy", "Peaks (Time)",
    "Kurtosis", "Skewness", "ZCR",
    "E Low", "E Mid", "E High", "Peak Freq", "Peak Mag (dBm)", "Spec Entropy",
    "Spec Flatness", "Spec Rolloff (MHz)", "HNR", "THD", "Spec Crest"
]

# Flags for plotting and saving
PLOT_WAVEFORMS = True
SAVE_WAVEFORMS = False  # No PNG saving
if not os.path.exists(EM_WAVEFORMS_PATH):
    os.makedirs(EM_WAVEFORMS_PATH)

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, signal)

def shannon_entropy(signal):
    """
    Calculate the Shannon entropy of a signal with fixed binning.

    Args:
        signal: Input signal array.

    Returns:
        float: Shannon entropy value.
    """
    hist, _ = np.histogram(signal, bins=50, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0

def convert_to_dbm(magnitude, reference_impedance=50):
    epsilon = 1e-10
    power = (magnitude**2) / (2 * reference_impedance)
    dbm_values = 10 * np.log10(power / 0.001 + epsilon)
    return dbm_values

def analyze_raw_signal(signal, sample_rate, fft_freq=None, fft_magnitude=None):
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    processed_signal = np.clip(signal, -1e6, 1e6)
    rms = float(np.sqrt(np.mean(processed_signal**2)))
    mean_val = float(np.mean(processed_signal))
    std_dev = float(np.std(processed_signal))
    crest_factor = float(np.max(np.abs(processed_signal)) / rms if rms != 0 else 0)
    peak_to_peak = float(np.max(processed_signal) - np.min(processed_signal))
    entropy = float(shannon_entropy(processed_signal))
    peaks, _ = find_peaks(np.abs(processed_signal), height=np.max(np.abs(processed_signal))*0.05, distance=10)
    peak_count = int(len(peaks))
    kurtosis = float(scipy.stats.kurtosis(processed_signal))
    skewness = float(scipy.stats.skew(processed_signal))
    zero_crossings = np.sum(np.diff(np.signbit(processed_signal)) != 0)
    zcr = float(zero_crossings) / len(processed_signal) * sample_rate
    if fft_freq is None or fft_magnitude is None:
        fft_signal = np.fft.fft(processed_signal)
        fft_freq = np.fft.fftfreq(len(processed_signal), 1/sample_rate)
        fft_magnitude = np.abs(fft_signal)
    mask_low = (fft_freq >= 0) & (fft_freq <= 500e6)
    mask_mid = (fft_freq > 500e6) & (fft_freq <= 1000e6)
    mask_high = (fft_freq > 1000e6) & (fft_freq <= 2000e6)
    energy_low = float(np.sum(fft_magnitude[mask_low]**2))
    energy_mid = float(np.sum(fft_magnitude[mask_mid]**2))
    energy_high = float(np.sum(fft_magnitude[mask_high]**2))
    stats = {
        'rms': rms, 'mean': mean_val, 'std_dev': std_dev, 'crest_factor': crest_factor,
        'peak_to_peak': peak_to_peak, 'entropy': entropy, 'peak_count': peak_count,
        'kurtosis': kurtosis, 'skewness': skewness, 'zcr': zcr,
        'energy_low': energy_low, 'energy_mid': energy_mid, 'energy_high': energy_high,
        'processed_signal': processed_signal, 'sample_rate': sample_rate
    }
    return stats, fft_freq, fft_magnitude

def analyze_fft(fft_freq, fft_magnitude_dbm):
    fft_magnitude_dbm = np.nan_to_num(fft_magnitude_dbm, nan=-100.0, posinf=-100.0, neginf=-100.0)
    mask = (fft_freq >= 0) & (fft_freq <= 2e9)
    masked_freq = fft_freq[mask]
    masked_magnitude = fft_magnitude_dbm[mask]
    peaks, _ = find_peaks(masked_magnitude, height=np.max(masked_magnitude)*0.05, distance=10)
    peak_freqs = masked_freq[peaks] if len(peaks) > 0 else np.array([0])
    peak_mags = masked_magnitude[peaks] if len(peaks) > 0 else np.array([0])
    if len(peak_mags) > 0:
        max_idx = np.argmax(peak_mags)
        peak_freq = float(peak_freqs[max_idx])
        peak_magnitude = float(peak_mags[max_idx])
    else:
        peak_freq, peak_magnitude = 0.0, 0.0
    spectral_entropy = float(shannon_entropy(masked_magnitude))
    spectral_mean = float(np.mean(masked_magnitude))
    spectral_crest = float(np.max(masked_magnitude) / spectral_mean if spectral_mean != 0 else 0)
    noise_floor = np.percentile(masked_magnitude[masked_magnitude > 0], 25) if np.any(masked_magnitude > 0) else 1e-10
    harmonics = []
    if peak_freq > 0:
        for i in range(2, 6):
            harmonic_freq = peak_freq * i
            if harmonic_freq <= 2e9:
                harmonic_idx = np.argmin(np.abs(masked_freq - harmonic_freq))
                if masked_magnitude[harmonic_idx] > noise_floor * 1.41:
                    harmonics.append((float(harmonic_freq), float(masked_magnitude[harmonic_idx])))
    harmonic_distortion = float(np.sqrt(sum(h[1]**2 for h in harmonics)) / peak_magnitude if peak_magnitude > 0 else 0)
    power_spectrum = np.exp(masked_magnitude / 10)
    geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
    arithmetic_mean = np.mean(power_spectrum)
    spectral_flatness = float(geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0)
    cumulative_energy = np.cumsum(power_spectrum)
    total_energy = cumulative_energy[-1] if cumulative_energy[-1] > 0 else 1e-10
    rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
    spectral_rolloff = float(masked_freq[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0)
    harmonic_power = sum(h[1]**2 for h in harmonics)
    noise_power = np.sum(power_spectrum**2) - harmonic_power - peak_magnitude**2
    hnr = float(10 * np.log10(harmonic_power / (noise_power + 1e-10)) if noise_power > 0 and harmonic_power > 0 else 0)
    params = {
        'peak_freq': peak_freq, 'peak_magnitude': peak_magnitude,
        'spectral_entropy': spectral_entropy, 'harmonic_distortion': harmonic_distortion,
        'spectral_crest': spectral_crest, 'harmonics': harmonics,
        'spectral_flatness': spectral_flatness, 'spectral_rolloff': spectral_rolloff, 'hnr': hnr
    }
    for key in params:
        if key != 'harmonics':
            params[key] = np.nan_to_num(params[key], nan=0.0, posinf=0.0, neginf=0.0)
    return params

def update_table_figure(timestamp):
    """
    Update and display the table figure, saving it as CSV in operation folder.

    Args:
        timestamp: Timestamp for naming the operation folder.
    """
    operation_dir = os.path.join(EM_WAVEFORMS_PATH, f"operation_{timestamp}")
    os.makedirs(operation_dir, exist_ok=True)
    
    # Save table as CSV
    table_file = os.path.join(operation_dir, f"results_table_{timestamp}.csv")
    table_data = [result for result in signal_results]
    with open(table_file, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(TABLE_HEADERS)
        writer.writerows(table_data)
    print(f"[*] Table saved as CSV to {table_file}")
    
    # Display table if PLOT_WAVEFORMS is True
    if PLOT_WAVEFORMS and signal_results:
        plt.close('table_fig')
        fig = plt.figure(figsize=(12, len(signal_results) * 0.5 + 2), num='table_fig')
        ax = fig.add_subplot(111)
        ax.axis('off')
        cell_colors = [['white' for _ in range(len(TABLE_HEADERS))] for _ in range(len(signal_results))]
        for row_idx, color in enumerate(signal_colors):
            cell_colors[row_idx][0] = color
        table = ax.table(cellText=table_data, colLabels=TABLE_HEADERS, loc='center', cellLoc='center', cellColours=cell_colors)
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        plt.title("EM Signal Analysis Results - Operation Mode")
        plt.tight_layout()
        plt.show()

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    return obj

def select_median_signal(signals_data, file_id, operation_dir):
    """
    Select the signal closest to the median RMS from a list of signals.

    Args:
        signals_data: List of tuples (waveform_data, result_dict, timestamp_str).
        file_id: ID of the input file.
        operation_dir: Directory to save the selected signal.

    Returns:
        dict: Result dictionary of the selected signal, or None if no signals are valid.
    """
    if not signals_data:
        print("[!] No valid signals to select median from")
        return None

    # Calculate RMS for each signal
    rms_values = []
    valid_signals = []
    for waveform_data, result_dict, timestamp_str in signals_data:
        if waveform_data and 'signal' in waveform_data:
            signal = np.nan_to_num(waveform_data['signal'], nan=0.0, posinf=0.0, neginf=0.0)
            rms = float(np.sqrt(np.mean(signal**2)))
            rms_values.append(rms)
            valid_signals.append((waveform_data, result_dict, timestamp_str))
        else:
            print(f"[!] Skipping invalid waveform for file ID {file_id}")

    if not rms_values:
        print(f"[!] No valid RMS values for file ID {file_id}")
        return None

    # Find median RMS
    median_rms = np.median(rms_values)
    print(f"[*] Median RMS for file ID {file_id}: {median_rms:.4f}")

    # Find signal closest to median RMS
    closest_idx = np.argmin(np.abs(np.array(rms_values) - median_rms))
    selected_waveform, selected_result, selected_timestamp = valid_signals[closest_idx]
    
    # Save selected signal as .npz
    waveform_file = os.path.join(operation_dir, f"signal_{file_id}_{selected_timestamp}.npz")
    np.savez_compressed(
        waveform_file,
        signal=selected_waveform['signal'],
        sample_rate=selected_waveform['sample_rate'],
        vertical_gain=selected_waveform['vertical_gain'],
        vertical_offset=selected_waveform['vertical_offset']
    )
    print(f"[*] Selected signal saved to {waveform_file}")
    
    selected_result['em_waveform_file'] = waveform_file
    return selected_result

def process_crash_file_with_em_capture(router_ip, target_cgi, crash_file, em_capture, execution_number=1, log_file=None, calibration_dir=None):
    try:
        vuln_indicators = []
        file_basename = os.path.basename(crash_file)
        if file_basename == "README.txt":
            return None
        id_match = re.search(r'id:(\d+)', file_basename)
        file_id = id_match.group(0) if id_match else file_basename[:10]
        file_id = int(file_id[3:10])
        with open(crash_file, 'rb') as f:
            content = f.read()
        content_str = content.decode('utf-8', errors='replace').replace('\0', '')
        # Extract UID using improved parsing
        uid_value = extract_uid_from_file_content(content_str)
        
        # Build curl command using configuration
        curl_cmd = build_curl_command(router_ip, target_cgi, uid_value)
        
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # print(f"[*] Executing: {' '.join(curl_cmd[:5])}... (truncated)")
        em_capture.arm_trigger()
        start_time = time.time()
        result = subprocess.run(curl_cmd, capture_output=True, text=True)
        em_capture.force_trigger()
        waveform_data = em_capture.capture_waveform()
        end_time = time.time()
        duration = end_time - start_time
        
        # NEW: Save execution time
        save_execution_time(file_id, execution_number, timestamp_str, duration)
        
        serializable_raw_stats = {}
        serializable_fft_params = {}
        serializable_harmonics = []
        processed_signal = None
        processed_sample_rate = None
        fft_magnitude_dbm = None
        fft_freq = None
        fft_params = {}
        raw_stats = {}
        harmonics = []
        http_response = None
        waveform_file = None
        
        # Parse HTTP response
        full_response = result.stderr + "\n" + result.stdout
        for line in result.stderr.split('\n') + result.stdout.split('\n'):
            if line.startswith('< HTTP/1.1'):
                http_response = line[2:].strip()
                print(http_response)
                break
        
        if http_response is None:
            http_response = "No HTTP response detected"
        
        # NEW: Save HTTP response
        save_http_response(file_id, execution_number, timestamp_str, http_response, full_response)
        
        if http_response == "HTTP/1.1 200 OK":
            id_color = 'green'
        else:
            id_color = 'red'
        signal_colors.append(id_color)
        
        if waveform_data and 'signal' in waveform_data:
            waveform_data['signal'] = np.nan_to_num(waveform_data['signal'], nan=0.0, posinf=0.0, neginf=0.0)
            raw_stats, fft_freq, fft_magnitude = analyze_raw_signal(
                waveform_data['signal'], waveform_data['sample_rate']
            )
            processed_signal = raw_stats.pop('processed_signal')
            processed_sample_rate = raw_stats.pop('sample_rate')
            window = np.hanning(len(processed_signal))
            fft_signal = np.fft.fft(processed_signal * window)
            fft_magnitude = np.abs(fft_signal) / len(processed_signal)
            fft_magnitude_dbm = convert_to_dbm(fft_magnitude)
            fft_magnitude_dbm = np.nan_to_num(fft_magnitude_dbm, nan=-100.0, posinf=-100.0, neginf=-100.0)
            fft_params = analyze_fft(fft_freq, fft_magnitude_dbm)
            harmonics = fft_params.pop('harmonics')
            for key in fft_params:
                fft_params[key] = np.nan_to_num(fft_params[key], nan=0.0, posinf=0.0, neginf=0.0)
            for key in raw_stats:
                raw_stats[key] = np.nan_to_num(raw_stats[key], nan=0.0, posinf=0.0, neginf=0.0)
            serializable_raw_stats = convert_to_serializable(raw_stats)
            serializable_fft_params = convert_to_serializable(fft_params)
            serializable_harmonics = convert_to_serializable(harmonics)
            result_row = [
                str(file_id),
                f"{raw_stats.get('rms', 0):.4f}",
                f"{raw_stats.get('mean', 0):.4f}",
                f"{raw_stats.get('std_dev', 0):.4f}",
                f"{raw_stats.get('crest_factor', 0):.4f}",
                f"{raw_stats.get('peak_to_peak', 0):.4f}",
                f"{raw_stats.get('entropy', 0):.4f}",
                str(raw_stats.get('peak_count', 0)),
                f"{raw_stats.get('kurtosis', 0):.4f}",
                f"{raw_stats.get('skewness', 0):.4f}",
                f"{raw_stats.get('zcr', 0):.2f}",
                f"{raw_stats.get('energy_low', 0):.2e}",
                f"{raw_stats.get('energy_mid', 0):.2e}",
                f"{raw_stats.get('energy_high', 0):.2e}",
                f"{fft_params.get('peak_freq', 0)/1e6:.2f}",
                f"{fft_params.get('peak_magnitude', 0):.2f}",
                f"{fft_params.get('spectral_entropy', 0):.4f}",
                f"{fft_params.get('spectral_flatness', 0):.4f}",
                f"{fft_params.get('spectral_rolloff', 0)/1e6:.2f}",
                f"{fft_params.get('hnr', 0):.2f}",
                f"{fft_params.get('harmonic_distortion', 0):.4f}",
                f"{fft_params.get('spectral_crest', 0):.4f}"
            ]
            signal_results.append(result_row)

        return {
            'file_name': file_basename,
            'response_time': duration,
            'response': result.stdout,
            'http_response': http_response,
            'execution_number': execution_number,
            'vulnerable': bool(vuln_indicators),
            'vulnerability_indicators': vuln_indicators,
            'success': result.returncode == 0,
            'em_waveform_file': waveform_file,
            'em_data_summary': {
                'points': len(waveform_data['signal']) if waveform_data and 'signal' in waveform_data else 0,
                'raw_stats': serializable_raw_stats,
                'fft_params': serializable_fft_params,
                'harmonics': serializable_harmonics,
                'processed_signal': processed_signal,
                'sample_rate': processed_sample_rate
            }
        }, waveform_data, timestamp_str
    except Exception as e:
        print(f"Error processing file {crash_file}: {str(e)}")
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"Error processing file {crash_file}: {str(e)}\n")
        return {
            'file_name': os.path.basename(crash_file),
            'error': str(e),
            'success': False
        }, None, None

def process_crash_directory_with_em(router_ip, target_cgi, crash_dir, em_capture, max_files=None, log_file=None, timestamp=None):
    """
    Process all crash files in a directory with EM capture, sorted by numeric ID.

    Args:
        router_ip: Router IP.
        target_cgi: Target CGI script.
        crash_dir: Path to crash files directory.
        em_capture: EM capture object.
        max_files: Maximum number of files to process.
        log_file: Log file.
        timestamp: Timestamp for naming operation folder.

    Returns:
        list: Results for each file.
    """
    print(f"[+] Processing directory: {crash_dir}")
    if not os.path.isdir(crash_dir):
        print(f"Directory not found: {crash_dir}")
        return []
    crash_files = []
    for file in os.listdir(crash_dir):
        file_path = os.path.join(crash_dir, file)
        if os.path.isfile(file_path) and file != "README.txt":
            crash_files.append(file_path)
    def extract_id(file_path):
        file_basename = os.path.basename(file_path)
        id_match = re.search(r'id:(\d+)', file_basename)
        return int(id_match.group(1)) if id_match else float('inf')
    crash_files.sort(key=extract_id)
    if max_files is not None and max_files > 0 and len(crash_files) > max_files:
        print(f"Limiting to {max_files} files (out of {len(crash_files)} total)")
        crash_files = crash_files[:max_files]
    results = []
    operation_dir = os.path.join(EM_WAVEFORMS_PATH, f"operation_{timestamp}")
    os.makedirs(operation_dir, exist_ok=True)
    for crash_file in crash_files:
        print(f"Processing file: {os.path.basename(crash_file)}")
        id_match = re.search(r'id:(\d+)', os.path.basename(crash_file))
        file_id = int(id_match.group(1)) if id_match else 0
        # Execute 20 times
        signals_data = []
        n_rep = 20
        for i in range(n_rep):
            print(f"[*] Execution {i+1}/{n_rep} for file ID {file_id}")
            result, waveform_data, timestamp_str = process_crash_file_with_em_capture(
                router_ip, target_cgi, crash_file, em_capture, execution_number=i+1, log_file=log_file, calibration_dir=None
            )
            if result and waveform_data:
                signals_data.append((waveform_data, result, timestamp_str))
        # Select and save the median signal
        selected_result = select_median_signal(signals_data, file_id, operation_dir)
        if selected_result:
            results.append(selected_result)
    return results


def load_configuration_from_environment():
    """
    Load configuration from environment variables if available.
    This allows TRENTI GUI to pass configuration through environment.
    """
    global ROUTER_IP, OSCILLOSCOPE_IP, TARGET_CGI
    global SCA_CHANNEL, SCA_VOLTAGE_DIV, SCA_TIME_DIV, SCA_SAMPLE_RATE, SCA_MEMORY_SIZE, SCA_TRIGGER_LEVEL
    global CRASH_DIRS, EM_WAVEFORMS_PATH, HTTP_RESPONSES_PATH, EXEC_TIMES_PATH
    
    # Network configuration
    ROUTER_IP = os.environ.get('TRENTI_ROUTER_IP', ROUTER_IP)
    OSCILLOSCOPE_IP = os.environ.get('TRENTI_OSCILLOSCOPE_IP', OSCILLOSCOPE_IP)
    TARGET_CGI = os.environ.get('TRENTI_TARGET_CGI', TARGET_CGI)
    
    # Get firmadyne_id from environment for path configuration
    firmadyne_id = os.environ.get('TRENTI_FIRMADYNE_ID', '161160')
    host_path = os.environ.get('TRENTI_HOST_PATH', '/home/atenea/trenti/evaluations')
    campaign_number = os.environ.get('TRENTI_CAMPAIGN_NUMBER', '*')  # NUEVO
    
    # CORRECCIÓN: Construir la ruta correcta y expandirla inmediatamente
    if 'TRENTI_CRASH_DIRS' in os.environ:
        crash_dirs_str = os.environ['TRENTI_CRASH_DIRS']
        try:
            # Support both single directory and JSON array format
            if crash_dirs_str.startswith('['):
                import json
                provided_crash_dirs = json.loads(crash_dirs_str)
            else:
                provided_crash_dirs = [crash_dirs_str]
            
            # Expand any wildcards in provided directories
            import glob
            CRASH_DIRS = []
            for crash_dir in provided_crash_dirs:
                if '*' in crash_dir:
                    matched_dirs = glob.glob(crash_dir)
                    if matched_dirs:
                        CRASH_DIRS.extend(matched_dirs)
                        print(f"[*] Expanded crash dir pattern '{crash_dir}' to {len(matched_dirs)} directories:")
                        for matched_dir in matched_dirs:
                            print(f"    - {matched_dir}")
                    else:
                        print(f"[!] No directories matched pattern: {crash_dir}")
                else:
                    if os.path.exists(crash_dir):
                        CRASH_DIRS.append(crash_dir)
                        print(f"[*] Added crash dir: {crash_dir}")
                    else:
                        print(f"[!] Crash directory not found: {crash_dir}")
                        
        except (json.JSONDecodeError, Exception) as e:
            print(f"[!] Error parsing TRENTI_CRASH_DIRS: {e}")
            print(f"[!] Using fallback crash directory pattern")
            # Fallback to pattern-based approach
            crash_pattern = f"{host_path}/image_{firmadyne_id}/trenti_sca_outputs_{firmadyne_id}_*/queue"
            matched_dirs = glob.glob(crash_pattern)
            CRASH_DIRS = matched_dirs if matched_dirs else []
    else:
        # CORRECCIÓN: Si no se proporcionan CRASH_DIRS, buscar automáticamente
        print(f"[*] TRENTI_CRASH_DIRS not provided, searching for queue directories...")
        import glob
        
        # Buscar patrones de directorios de cola
        search_patterns = [
            f"{host_path}/image_{firmadyne_id}/trenti_sca_outputs_{firmadyne_id}_{campaign_number}/queue",
            f"{host_path}/image_{firmadyne_id}/trenti_sca_outputs_{firmadyne_id}_*/queue",
            f"{host_path}/image_{firmadyne_id}/outputs/queue",  # Fallback pattern
            f"{host_path}/image_{firmadyne_id}/*/queue"  # General pattern
        ]
        
        CRASH_DIRS = []
        for pattern in search_patterns:
            matched_dirs = glob.glob(pattern)
            if matched_dirs:
                CRASH_DIRS.extend(matched_dirs)
                print(f"[*] Found {len(matched_dirs)} queue directories with pattern: {pattern}")
                for matched_dir in matched_dirs:
                    print(f"    - {matched_dir}")
                break  # Use first successful pattern
        
        if not CRASH_DIRS:
            print(f"[!] No queue directories found with any pattern")
            print(f"[!] Searched patterns:")
            for pattern in search_patterns:
                print(f"    - {pattern}")
    
    # Update paths based on firmadyne_id
    EM_WAVEFORMS_PATH = f'{host_path}/image_{firmadyne_id}/waveforms/'
    HTTP_RESPONSES_PATH = f'{host_path}/image_{firmadyne_id}/HTTP_responses/'
    EXEC_TIMES_PATH = f'{host_path}/image_{firmadyne_id}/exec_times/'
    
    # Create directories
    os.makedirs(EM_WAVEFORMS_PATH, exist_ok=True)
    os.makedirs(HTTP_RESPONSES_PATH, exist_ok=True)
    os.makedirs(EXEC_TIMES_PATH, exist_ok=True)
    
    # SCA configuration
    SCA_CHANNEL = os.environ.get('TRENTI_SCA_CHANNEL', SCA_CHANNEL)
    
    if 'TRENTI_SCA_VOLTAGE_DIV' in os.environ:
        try:
            SCA_VOLTAGE_DIV = float(os.environ['TRENTI_SCA_VOLTAGE_DIV'])
        except ValueError:
            print(f"[!] Warning: Invalid TRENTI_SCA_VOLTAGE_DIV value, using default: {SCA_VOLTAGE_DIV}")
    
    SCA_TIME_DIV = os.environ.get('TRENTI_SCA_TIME_DIV', SCA_TIME_DIV)
    
    if 'TRENTI_SCA_SAMPLE_RATE' in os.environ:
        try:
            # Convert from GS/s to Hz
            sample_rate_gs = float(os.environ['TRENTI_SCA_SAMPLE_RATE'])
            SCA_SAMPLE_RATE = int(sample_rate_gs * 1e9)
        except ValueError:
            print(f"[!] Warning: Invalid TRENTI_SCA_SAMPLE_RATE value, using default: {SCA_SAMPLE_RATE/1e9:.1f} GS/s")
    
    SCA_MEMORY_SIZE = os.environ.get('TRENTI_SCA_MEMORY_SIZE', SCA_MEMORY_SIZE)
    
    if 'TRENTI_SCA_TRIGGER_LEVEL' in os.environ:
        try:
            SCA_TRIGGER_LEVEL = float(os.environ['TRENTI_SCA_TRIGGER_LEVEL'])
        except ValueError:
            print(f"[!] Warning: Invalid TRENTI_SCA_TRIGGER_LEVEL value, using default: {SCA_TRIGGER_LEVEL}")
    
    # Load curl configuration
    load_curl_configuration()
    
    # Print current configuration - MEJORADO
    print(f"[*] Operation mode configuration loaded:")
    print(f"    Router IP: {ROUTER_IP}")
    print(f"    Oscilloscope IP: {OSCILLOSCOPE_IP}")
    print(f"    Target CGI: {TARGET_CGI}")
    print(f"    Firmadyne ID: {firmadyne_id}")
    print(f"    Campaign Number: {campaign_number}")
    print(f"    SCA Channel: {SCA_CHANNEL}")
    print(f"    SCA Voltage Div: {SCA_VOLTAGE_DIV} V/div")
    print(f"    SCA Time Div: {SCA_TIME_DIV}")
    print(f"    SCA Sample Rate: {SCA_SAMPLE_RATE/1e9:.1f} GS/s")
    print(f"    SCA Memory Size: {SCA_MEMORY_SIZE}")
    print(f"    SCA Trigger Level: {SCA_TRIGGER_LEVEL}")
    print(f"    EM Waveforms Path: {EM_WAVEFORMS_PATH}")
    print(f"    HTTP Responses Path: {HTTP_RESPONSES_PATH}")
    print(f"    Execution Times Path: {EXEC_TIMES_PATH}")
    print(f"    Crash Directories Found: {len(CRASH_DIRS)}")
    
    # NUEVO: Mostrar detalles de los directorios encontrados
    if CRASH_DIRS:
        print(f"    Queue directories:")
        for i, crash_dir in enumerate(CRASH_DIRS, 1):
            if os.path.exists(crash_dir):
                try:
                    file_count = len([f for f in os.listdir(crash_dir) 
                                    if os.path.isfile(os.path.join(crash_dir, f)) and f != "README.txt"])
                    print(f"      {i}. {crash_dir} ({file_count} files)")
                except:
                    print(f"      {i}. {crash_dir} (cannot count files)")
            else:
                print(f"      {i}. {crash_dir} (NOT FOUND)")
    else:
        print(f"    ??  WARNING: No queue directories found!")


def main():
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%Ho%M")
    all_results = []
    total_vulnerable = 0
    
    # Load configuration from environment (set by TRENTI GUI)
    load_configuration_from_environment()
    
    # VERIFICACIÓN: Asegurar que tenemos directorios para procesar
    if not CRASH_DIRS:
        print(f"[!] ERROR: No crash/queue directories found to process!")
        print(f"[!] Check that fuzzing has run and queue directories exist.")
        return
    
    em_capture = EMCapture(OSCILLOSCOPE_IP)
    try:
        if not em_capture.connect():
            print("[!] Could not connect to oscilloscope. Aborting.")
            return
            
        print(f"\n{'*'*80}")
        print(f"ROUTER FUZZING TEST WITH EM CAPTURE (OPERATION MODE): {ROUTER_IP}/{TARGET_CGI}")
        print(f"OSCILLOSCOPE: {OSCILLOSCOPE_IP}, CHANNEL: {SCA_CHANNEL}")
        print(f"SCA CONFIG: {SCA_VOLTAGE_DIV}V/div, {SCA_TIME_DIV}, {SCA_SAMPLE_RATE/1e9:.1f}GS/s, {SCA_MEMORY_SIZE}")
        print(f"OPERATION MODE: 20 repetitions per file, median signal selection")
        print(f"HTTP RESPONSES: {HTTP_RESPONSES_PATH}")
        print(f"EXECUTION TIMES: {EXEC_TIMES_PATH}")
        print(f"PROCESSING {len(CRASH_DIRS)} QUEUE DIRECTORIES")
        print(f"{'*'*80}")
        
        # SIMPLIFICADO: Los directorios ya están expandidos
        for crash_dir in CRASH_DIRS:
            print(f"[+] Processing crash directory: {crash_dir}")
            
            # Verificar que el directorio existe antes de procesar
            if not os.path.exists(crash_dir):
                print(f"[!] Directory not found: {crash_dir}")
                continue
                
            results = process_crash_directory_with_em(
                ROUTER_IP,
                TARGET_CGI,
                crash_dir,
                em_capture,
                max_files=None,  # Process all files
                timestamp=timestamp
            )
            all_results.extend(results)
            vulnerable_files = [r for r in results if r.get('vulnerable', False)]
            total_vulnerable += len(vulnerable_files)
        
        print(f"\n{'='*80}")
        print("FUZZING TEST SUMMARY (OPERATION MODE)")
        print(f"{'='*80}")
        print(f"Total files tested: {len(all_results)}")
        print(f"Successful requests: {sum(1 for r in all_results if r.get('success', False))}")
        print(f"Failed requests: {sum(1 for r in all_results if not r.get('success', False))}")
        print(f"Potential vulnerabilities found: {total_vulnerable}")
        if total_vulnerable > 0:
            print("\nPOTENTIALLY VULNERABLE INPUTS:")
            for result in all_results:
                if result.get('vulnerable', False):
                    indicators = ', '.join(result.get('vulnerability_indicators', []))
                    print(f"- {result['file_name']}: {indicators}")
        
        print(f"- HTTP responses: {HTTP_RESPONSES_PATH}")
        print(f"- Execution times: {EXEC_TIMES_PATH}")
        
        # Save and optionally display the table
        update_table_figure(timestamp)
            
    finally:
        if em_capture:
            em_capture.disconnect()

if __name__ == "__main__":
    main()