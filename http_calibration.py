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
SAVE_WAVEFORMS = False  # Option to save waveform plots

# SCA Configuration - These will be modified by TRENTI GUI
SCA_CHANNEL = 'C3'
SCA_VOLTAGE_DIV = 0.010  # Voltage division in V/div
SCA_TIME_DIV = '500E-9'  # Time division
SCA_SAMPLE_RATE = 10e9  # Sample rate in Hz (10 GS/s)
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
            print(f"[*] Extracted UID using pattern '{pattern}': {uid_value[:20]}...")
            return uid_value
    
    # If no pattern matches, use entire content (stripped)
    uid_value = content_str.strip()
    # print(f"[*] Using entire file content as UID: {uid_value[:20]}...")
    return uid_value

class EMCapture:
    """Class to manage EM signal capture using the LeCroy oscilloscope"""

    def __init__(self, oscilloscope_ip=None):
        print("[*] Initializing EM capture system")
        self.oscilloscope_ip = oscilloscope_ip or OSCILLOSCOPE_IP
        self.lecroy = Lecroy()
        self._connected = False
        
        # SCA configuration from global variables
        self.channel = SCA_CHANNEL
        self.voltage_div = SCA_VOLTAGE_DIV
        self.time_div = SCA_TIME_DIV
        self.sample_rate = 10000000000  # 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s
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
        
        print("[+] Oscilloscope configured successfully")

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

# Table headers (reduced to match optimized features)
TABLE_HEADERS = [
    "ID", "RMS", "Mean", "Std Dev", "Crest", "P2P", "Entropy", "Peaks (Time)",
    "Kurtosis", "Skewness", "ZCR",  # New time-domain metrics
    "E Low", "E Mid", "E High", "Peak Freq", "Peak Mag (dBm)", "Spec Entropy",
    "Spec Flatness", "Spec Rolloff (MHz)", "HNR", "THD", "Spec Crest"  # New frequency-domain metrics
]

# Flags for plotting and saving
PLOT_WAVEFORMS = True
SAVE_WAVEFORMS = True

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
    
    # New parameters
    kurtosis = float(scipy.stats.kurtosis(processed_signal))  # Measures tailedness
    skewness = float(scipy.stats.skew(processed_signal))  # Measures asymmetry
    zero_crossings = np.sum(np.diff(np.signbit(processed_signal)) != 0)  # Zero-crossing count
    zcr = float(zero_crossings) / len(processed_signal) * sample_rate  # ZCR in Hz
    
    if fft_freq is None or fft_magnitude is None:
        fft_signal = np.fft.fft(processed_signal)
        fft_freq = np.fft.fftfreq(len(processed_signal), 1/sample_rate)
        fft_magnitude = np.abs(fft_signal)
    mask_low = (fft_freq >= 0) & (fft_freq <= 500e6)  # Adjusted to 500 MHz
    mask_mid = (fft_freq > 500e6) & (fft_freq <= 1000e6)
    mask_high = (fft_freq > 1000e6) & (fft_freq <= 2000e6)  # Extended to 2 GHz
    energy_low = float(np.sum(fft_magnitude[mask_low]**2))
    energy_mid = float(np.sum(fft_magnitude[mask_mid]**2))
    energy_high = float(np.sum(fft_magnitude[mask_high]**2))
    stats = {
        'rms': rms, 'mean': mean_val, 'std_dev': std_dev, 'crest_factor': crest_factor,
        'peak_to_peak': peak_to_peak, 'entropy': entropy, 'peak_count': peak_count,
        'kurtosis': kurtosis, 'skewness': skewness, 'zcr': zcr,  # Add new metrics
        'energy_low': energy_low, 'energy_mid': energy_mid, 'energy_high': energy_high,
        'processed_signal': processed_signal, 'sample_rate': sample_rate
    }
    return stats, fft_freq, fft_magnitude

def analyze_fft(fft_freq, fft_magnitude_dbm):
    fft_magnitude_dbm = np.nan_to_num(fft_magnitude_dbm, nan=-100.0, posinf=-100.0, neginf=-100.0)
    mask = (fft_freq >= 0) & (fft_freq <= 2e9)  # Extended to 2 GHz
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
        for i in range(2, 6):  # Up to 5th harmonic
            harmonic_freq = peak_freq * i
            if harmonic_freq <= 2e9:
                harmonic_idx = np.argmin(np.abs(masked_freq - harmonic_freq))
                if masked_magnitude[harmonic_idx] > noise_floor * 1.41:  # +3 dB threshold
                    harmonics.append((float(harmonic_freq), float(masked_magnitude[harmonic_idx])))
    harmonic_distortion = float(np.sqrt(sum(h[1]**2 for h in harmonics)) / peak_magnitude if peak_magnitude > 0 else 0)
    
    # New parameters
    # Spectral Flatness (Wiener Entropy)
    power_spectrum = np.exp(masked_magnitude / 10)  # Convert dBm to linear scale
    geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
    arithmetic_mean = np.mean(power_spectrum)
    spectral_flatness = float(geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0)
    
    # Spectral Rolloff (frequency where 85% of energy is below)
    cumulative_energy = np.cumsum(power_spectrum)
    total_energy = cumulative_energy[-1] if cumulative_energy[-1] > 0 else 1e-10
    rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
    spectral_rolloff = float(masked_freq[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0)
    
    # Harmonic-to-Noise Ratio (HNR)
    harmonic_power = sum(h[1]**2 for h in harmonics)
    noise_power = np.sum(power_spectrum**2) - harmonic_power - peak_magnitude**2
    hnr = float(10 * np.log10(harmonic_power / (noise_power + 1e-10)) if noise_power > 0 and harmonic_power > 0 else 0)
    
    params = {
        'peak_freq': peak_freq, 'peak_magnitude': peak_magnitude,
        'spectral_entropy': spectral_entropy, 'harmonic_distortion': harmonic_distortion,
        'spectral_crest': spectral_crest, 'harmonics': harmonics,
        'spectral_flatness': spectral_flatness, 'spectral_rolloff': spectral_rolloff, 'hnr': hnr  # Add new metrics
    }
    for key in params:
        if key != 'harmonics':
            params[key] = np.nan_to_num(params[key], nan=0.0, posinf=0.0, neginf=0.0)
    return params

def update_table_figure(timestamp):
    """
    Update and display the table figure, saving it as CSV in calibration folder.

    Args:
        timestamp: Timestamp for naming the calibration folder.
    """
    calibration_dir = os.path.join(EM_WAVEFORMS_PATH, f"calibration_{timestamp}")
    os.makedirs(calibration_dir, exist_ok=True)
    
    # Save table as CSV
    table_file = os.path.join(calibration_dir, f"results_table_{timestamp}.csv")
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
        plt.title("EM Signal Analysis Results")
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

def calculate_sample_size(feature_matrix, confidence_level=0.95, margin_of_error=0.05):
    """
    Calculate the required sample size based on normalized features.

    Args:
        feature_matrix: Matrix of signal features.
        confidence_level: Desired confidence level (e.g., 0.95).
        margin_of_error: Allowed margin of error.

    Returns:
        int: Required sample size, capped at 100.
    """
    if len(feature_matrix) < 2:
        return 100
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    selected_indices = [0, 1, 2, 4, 6, 11]  # rms, mean, std_dev, peak_to_peak, peak_count, peak_freq
    selected_features = normalized_features[:, selected_indices]
    std_dev_features = np.std(selected_features, axis=0)
    mean_std_dev = np.mean(std_dev_features[std_dev_features > 0]) if np.any(std_dev_features > 0) else 1.0
    z_score = scipy.stats.norm.ppf((1 + confidence_level) / 2)
    sample_size = (z_score * mean_std_dev / margin_of_error) ** 2
    sample_size = int(np.ceil(sample_size))
    return min(sample_size, 100)  # Cap at 100 as per original logic

def check_pattern_coverage(feature_matrix, eps=0.1, min_samples=8, window=25):
    """
    Check if new patterns have been discovered using DBSCAN clustering.

    Args:
        feature_matrix: Matrix of signal features.
        eps: DBSCAN max distance between points in a cluster.
        min_samples: Minimum points to form a cluster.
        window: Window of recent iterations.

    Returns:
        bool: True if no new clusters are detected in two consecutive windows, False otherwise.
    """
    if len(feature_matrix) < 2 * window:
        return False
    recent_features = np.array(feature_matrix[-window:])
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(recent_features)
    selected_indices = [0, 1, 2, 4, 6, 11]
    selected_features = normalized_features[:, selected_indices]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(selected_features)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}
    
    # Check previous window
    older_features = np.array(feature_matrix[-2*window:-window])
    normalized_older = scaler.fit_transform(older_features)
    selected_older = normalized_older[:, selected_indices]
    older_clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(selected_older)
    older_labels = set(older_clustering.labels_) - {-1}
    
    # Require no new clusters in both recent and previous windows
    current_stable = unique_labels.issubset(older_labels)
    return current_stable and len(unique_labels) <= 1

def check_entropy_stabilization(signal_matrix, window=20, threshold=0.02):
    """
    Check if signal entropy has stabilized.

    Args:
        signal_matrix: List of (signal, entropy) tuples.
        window: Window of recent iterations.
        threshold: Relative change threshold for entropy.

    Returns:
        bool: True if entropy is stabilized, False otherwise.
    """
    if len(signal_matrix) < window:
        return False
    entropies = np.array([entropy for _, entropy in signal_matrix[-window:]])
    
    # Remove outliers using IQR
    Q1, Q3 = np.percentile(entropies, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    valid_entropies = entropies[(entropies >= lower_bound) & (entropies <= upper_bound)]
    
    if len(valid_entropies) < 0.8 * window:  # Require at least 80% valid data
        return False
    
    mean_entropy = np.mean(valid_entropies)
    if mean_entropy == 0:
        return False
    rel_change = np.std(valid_entropies) / mean_entropy
    
    return rel_change < threshold

def execute_seeds_statistical(seeds, em_capture, confidence_level=0.99, margin_of_error=0.03, log_file=None, timestamp=None):
    """
    Perform calibration analysis for a correct input, collecting EM signals until convergence.

    Args:
        seeds: List of seed files.
        em_capture: EMCapture object.
        confidence_level: Confidence level for sample size.
        margin_of_error: Margin of error for sample size.
        log_file: File to log results.
        timestamp: Timestamp for naming calibration folder.

    Returns:
        list: Results of captured signals.
    """
    results = []
    global_feature_matrix = []
    global_signal_matrix = []  # Stores (signal, entropy) tuples
    feature_names = []

    # Create calibration folder
    calibration_dir = os.path.join(EM_WAVEFORMS_PATH, f"calibration_{timestamp}")
    os.makedirs(calibration_dir, exist_ok=True)

    feature_keys = [
        'rms', 'mean', 'std_dev', 'crest_factor', 'peak_to_peak', 'entropy',
        'peak_count', 'kurtosis', 'skewness', 'zcr',
        'energy_low', 'energy_mid', 'energy_high',
        'peak_freq', 'peak_magnitude', 'spectral_entropy', 'harmonic_distortion',
        'spectral_flatness', 'spectral_rolloff', 'hnr', 'spectral_crest'
    ]

    import random
    from itertools import cycle
    random.shuffle(seeds)
    seed_cycle = cycle(seeds)

    iteration = 0
    max_iterations = 1000  # Prevent infinite loops
    convergence_history = {
        'sample_size_sufficient': False,
        'patterns_stabilized': False,
        'entropy_stabilized': False
    }

    print("[*] Starting calibration to collect EM signals for normal behavior")

    while iteration < max_iterations:
        seed = next(seed_cycle)
        print(f"\n[*] Iteration {iteration + 1}: Processing seed {os.path.basename(seed)}")
        
        result = process_crash_file_with_em_capture(
            ROUTER_IP, TARGET_CGI, seed, em_capture, log_file, calibration_dir=calibration_dir
        )
        
        if result and 'em_data_summary' in result:
            results.append(result)

            feature_vector = []
            current_feature_names = []
            for key in feature_keys:
                if key in result['em_data_summary']['raw_stats']:
                    value = result['em_data_summary']['raw_stats'][key]
                    feature_vector.append(value)
                    current_feature_names.append(f"raw_{key}")
                elif key in result['em_data_summary']['fft_params']:
                    value = result['em_data_summary']['fft_params'][key]
                    feature_vector.append(value)
                    current_feature_names.append(f"fft_{key}")
                else:
                    feature_vector.append(0.0)
                    current_feature_names.append(f"missing_{key}")

            if len(feature_vector) > 0 and any(feature_vector):
                global_feature_matrix.append(feature_vector)
                if not feature_names:
                    feature_names = current_feature_names

                full_signal = result['em_data_summary'].get('processed_signal')
                if full_signal is not None and len(full_signal) > 0:
                    entropy = shannon_entropy(full_signal)
                    global_signal_matrix.append((full_signal, entropy))

                # print(f"[*] Features extracted (length: {len(feature_vector)}), total signals: {len(global_feature_matrix)}")
                # print(f"[*] Signal saved: {result.get('em_waveform_file', 'Not saved')}")

            # Check convergence
            try:
                feature_array = np.array(global_feature_matrix)

                # Log feature statistics
                feature_means = np.mean(feature_array, axis=0)
                feature_stds = np.std(feature_array, axis=0)
                # print(f"[*] Feature stats - Mean of first 5 features: {feature_means[:5]}")
                # print(f"[*] Feature stats - Std of first 5 features: {feature_stds[:5]}")

                # Check sample size
                required_sample_size = calculate_sample_size(
                    feature_matrix=feature_array,
                    confidence_level=confidence_level,
                    margin_of_error=margin_of_error
                )
                convergence_history['sample_size_sufficient'] = len(global_feature_matrix) >= required_sample_size
                # print(f"[*] Required sample size: {required_sample_size}, current: {len(global_feature_matrix)}")

                # Check pattern stabilization
                convergence_history['patterns_stabilized'] = check_pattern_coverage(
                    global_feature_matrix,
                    eps=0.1,
                    min_samples=8,
                    window=25
                )
                # print(f"[*] Patterns stabilized: {convergence_history['patterns_stabilized']}")

                # Check entropy stabilization
                convergence_history['entropy_stabilized'] = check_entropy_stabilization(
                    global_signal_matrix,
                    window=20,
                    threshold=0.02
                )
                # print(f"[*] Entropy stabilized: {convergence_history['entropy_stabilized']}")

                # Require ALL criteria to be met for convergence
                if all([
                    convergence_history['sample_size_sufficient'],
                    convergence_history['patterns_stabilized'],
                    convergence_history['entropy_stabilized']
                ]):
                    print("\n[+] Convergence reached:")
                    print("  - Sufficient sample size")
                    print("  - Patterns stabilized (no new clusters)")
                    print("  - Entropy stabilized")
                    print(f"[*] Total signals saved: {len(results)}")
                    break

            except Exception as e:
                print(f"[!] Error checking convergence: {e}")
                print(f"Current iteration: {iteration + 1}")
                traceback.print_exc()

        iteration += 1

        if iteration >= max_iterations:
            print("\n[!] Max iterations reached without convergence")
            print(f"[*] Total signals saved: {len(results)}")
            break

        # Periodic logging
        if (iteration + 1) % 50 == 0:
            print(f"\n[INFO] Progress after {iteration + 1} iterations:")
            print(f"  - Total signals collected: {len(global_feature_matrix)}")
            print(f"  - Sample size sufficient: {convergence_history['sample_size_sufficient']}")
            print(f"  - Patterns stabilized: {convergence_history['patterns_stabilized']}")
            print(f"  - Entropy stabilized: {convergence_history['entropy_stabilized']}")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"\nProgress after {iteration + 1} iterations:\n")
                    f.write(f"  - Total signals collected: {len(global_feature_matrix)}\n")
                    f.write(f"  - Sample size sufficient: {convergence_history['sample_size_sufficient']}\n")
                    f.write(f"  - Patterns stabilized: {convergence_history['patterns_stabilized']}\n")
                    f.write(f"  - Entropy stabilized: {convergence_history['entropy_stabilized']}\n")

    return results

def process_crash_file_with_em_capture(router_ip, target_cgi, crash_file, em_capture, log_file=None, calibration_dir=None):
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
        
        # print(f"[*] Executing: {' '.join(curl_cmd[:5])}... (truncated)")
        em_capture.arm_trigger()
        start_time = time.time()
        result = subprocess.run(curl_cmd, capture_output=True, text=True)
        em_capture.force_trigger()
        waveform_data = em_capture.capture_waveform()
        duration = time.time() - start_time
        serializable_raw_stats = {}
        serializable_fft_params = {}
        serializable_harmonics = []
        processed_signal = None
        processed_sample_rate = 10000000000  # 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s
        fft_magnitude_dbm = None
        fft_freq = None
        fft_params = {}
        raw_stats = {}
        harmonics = []
        http_response = None
        waveform_file = None
        for line in result.stderr.split('\n') + result.stdout.split('\n'):
            if line.startswith('< HTTP/1.1'):
                http_response = line[2:].strip()
                print(http_response)
                break
        if http_response == "HTTP/1.1 200 OK":
            id_color = 'green'
        else:
            id_color = 'red'
        signal_colors.append(id_color)
        if waveform_data and 'signal' in waveform_data:
            waveform_data['signal'] = np.nan_to_num(waveform_data['signal'], nan=0.0, posinf=0.0, neginf=0.0)
            # Save waveform to .npz in calibration_dir
            if calibration_dir:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                waveform_file = os.path.join(calibration_dir, f"signal_{file_id}_{timestamp}.npz")
                np.savez_compressed(
                    waveform_file,
                    signal=waveform_data['signal'],
                    sample_rate = 10000000000  # 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s
                    ,vertical_gain=waveform_data['vertical_gain'],
                    vertical_offset=waveform_data['vertical_offset']
                )
                print(f"[*] Waveform saved to {waveform_file}")
            
            raw_stats, fft_freq, fft_magnitude = analyze_raw_signal(
                waveform_data['signal'], waveform_data['sample_rate']
            )
            processed_signal = raw_stats.pop('processed_signal')
            processed_sample_rate = 10000000000  # 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s# 10.0 GS/s
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
            # Update signal_results for table with new parameters
            result_row = [
                str(file_id),
                f"{raw_stats.get('rms', 0):.4f}",
                f"{raw_stats.get('mean', 0):.4f}",
                f"{raw_stats.get('std_dev', 0):.4f}",
                f"{raw_stats.get('crest_factor', 0):.4f}",
                f"{raw_stats.get('peak_to_peak', 0):.4f}",
                f"{raw_stats.get('entropy', 0):.4f}",
                str(raw_stats.get('peak_count', 0)),
                f"{raw_stats.get('kurtosis', 0):.4f}",  # New
                f"{raw_stats.get('skewness', 0):.4f}",  # New
                f"{raw_stats.get('zcr', 0):.2f}",  # New
                f"{raw_stats.get('energy_low', 0):.2e}",
                f"{raw_stats.get('energy_mid', 0):.2e}",
                f"{raw_stats.get('energy_high', 0):.2e}",
                f"{fft_params.get('peak_freq', 0)/1e6:.2f}",
                f"{fft_params.get('peak_magnitude', 0):.2f}",
                f"{fft_params.get('spectral_entropy', 0):.4f}",
                f"{fft_params.get('spectral_flatness', 0):.4f}",  # New
                f"{fft_params.get('spectral_rolloff', 0)/1e6:.2f}",  # New
                f"{fft_params.get('hnr', 0):.2f}",  # New
                f"{fft_params.get('harmonic_distortion', 0):.4f}",
                f"{fft_params.get('spectral_crest', 0):.4f}"
            ]
            signal_results.append(result_row)

        return {
            'file_name': file_basename,
            'response_time': duration,
            'response': result.stdout,
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
        }
    except Exception as e:
        print(f"Error processing file {crash_file}: {str(e)}")
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"Error processing file {crash_file}: {str(e)}\n")
        return {
            'file_name': os.path.basename(crash_file),
            'error': str(e),
            'success': False
        }

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
        timestamp: Timestamp for naming calibration folder.

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
    calibration_dir = os.path.join(EM_WAVEFORMS_PATH, f"calibration_{timestamp}")
    os.makedirs(calibration_dir, exist_ok=True)
    for crash_file in crash_files:
        print(f"Processing file: {os.path.basename(crash_file)}")
        result = process_crash_file_with_em_capture(
            router_ip, target_cgi, crash_file, em_capture, log_file, calibration_dir=calibration_dir
        )
        if result:
            results.append(result)
    return results

def load_configuration_from_environment():
    """Load configuration from environment variables if available."""
    global ROUTER_IP, OSCILLOSCOPE_IP, TARGET_CGI
    global SCA_CHANNEL, SCA_VOLTAGE_DIV, SCA_TIME_DIV, SCA_SAMPLE_RATE, SCA_MEMORY_SIZE, SCA_TRIGGER_LEVEL
    global CRASH_DIRS, EM_WAVEFORMS_PATH, HTTP_RESPONSES_PATH, EXEC_TIMES_PATH
    
    # Get firmadyne_id y host_path
    firmadyne_id = os.environ.get('TRENTI_FIRMADYNE_ID', '9050')
    host_path = os.environ.get('TRENTI_HOST_PATH', '/home/atenea/trenti/evaluations')
    
    CRASH_DIRS = [f"{host_path}/image_{firmadyne_id}/trenti_sca_outputs_{firmadyne_id}_*/queue"]
    EM_WAVEFORMS_PATH = f'{host_path}/image_{firmadyne_id}/waveforms/'
    
    # Para archivo 2:
    HTTP_RESPONSES_PATH = f'{host_path}/image_{firmadyne_id}/HTTP_responses/'
    EXEC_TIMES_PATH = f'{host_path}/image_{firmadyne_id}/exec_times/'
    
    # Create directories after updating paths
    os.makedirs(EM_WAVEFORMS_PATH, exist_ok=True)
    os.makedirs(HTTP_RESPONSES_PATH, exist_ok=True)
    os.makedirs(EXEC_TIMES_PATH, exist_ok=True)
    
    # Network configuration
    ROUTER_IP = os.environ.get('TRENTI_ROUTER_IP', ROUTER_IP)
    OSCILLOSCOPE_IP = os.environ.get('TRENTI_OSCILLOSCOPE_IP', OSCILLOSCOPE_IP)
    TARGET_CGI = os.environ.get('TRENTI_TARGET_CGI', TARGET_CGI)
    
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
    
    # Crash directories configuration
    if 'TRENTI_CRASH_DIRS' in os.environ:
        crash_dirs_str = os.environ['TRENTI_CRASH_DIRS']
        try:
            # Support both single directory and JSON array format
            if crash_dirs_str.startswith('['):
                import json
                CRASH_DIRS = json.loads(crash_dirs_str)                
        except (json.JSONDecodeError, Exception):
            print(f"[!] Warning: Invalid TRENTI_CRASH_DIRS format, using default")
    
    load_curl_configuration()
    
    # Print current configuration
    print(f"[*] Configuration loaded:")
    print(f"    Router IP: {ROUTER_IP}")
    print(f"    Oscilloscope IP: {OSCILLOSCOPE_IP}")
    print(f"    Target CGI: {TARGET_CGI}")
    print(f"    SCA Channel: {SCA_CHANNEL}")
    print(f"    SCA Voltage Div: {SCA_VOLTAGE_DIV} V/div")
    print(f"    SCA Time Div: {SCA_TIME_DIV}")
    print(f"    SCA Sample Rate: {SCA_SAMPLE_RATE/1e9:.1f} GS/s")
    print(f"    SCA Memory Size: {SCA_MEMORY_SIZE}")
    print(f"    SCA Trigger Level: {SCA_TRIGGER_LEVEL}")
    print(f"    Crash Directories: {len(CRASH_DIRS)} configured")

def main():
    
    # Load configuration from environment (set by TRENTI GUI)
    load_configuration_from_environment()
    
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%Ho%M")
    # log_file = f"{LOG_PATH}curl_em_fuzzing_{timestamp}.log"
    all_results = []
    total_vulnerable = 0
    
    em_capture = EMCapture(OSCILLOSCOPE_IP)
    try:
        if not em_capture.connect():
            print("[!] Could not connect to oscilloscope. Aborting.")
            return
        # with open(log_file, 'w') as f:
        #     f.write(f"Router Fuzzing Test with EM Capture for {ROUTER_IP}/{TARGET_CGI}\n")
        #     f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        #     f.write(f"Oscilloscope: {OSCILLOSCOPE_IP}\n")
        #     f.write(f"Capturing on channel: {SCA_CHANNEL}\n")
        #     f.write(f"SCA Configuration:\n")
        #     f.write(f"  - Voltage Division: {SCA_VOLTAGE_DIV} V/div\n")
        #     f.write(f"  - Time Division: {SCA_TIME_DIV}\n")
        #     f.write(f"  - Sample Rate: {SCA_SAMPLE_RATE/1e9:.1f} GS/s\n")
        #     f.write(f"  - Memory Size: {SCA_MEMORY_SIZE}\n")
        #     f.write(f"  - Trigger Level: {SCA_TRIGGER_LEVEL}\n\n")
        print(f"\n{'*'*80}")
        print(f"ROUTER FUZZING TEST WITH EM CAPTURE: {ROUTER_IP}/{TARGET_CGI}")
        print(f"OSCILLOSCOPE: {OSCILLOSCOPE_IP}, CHANNEL: {SCA_CHANNEL}")
        print(f"SCA CONFIG: {SCA_VOLTAGE_DIV}V/div, {SCA_TIME_DIV}, {SCA_SAMPLE_RATE/1e9:.1f}GS/s, {SCA_MEMORY_SIZE}")
        print(f"{'*'*80}")
        
        # Expand crash directories with globbing support
        import glob
        expanded_crash_dirs = []
        for crash_dir in CRASH_DIRS:
            if '*' in crash_dir:
                # Use glob to expand wildcards
                matched_dirs = glob.glob(crash_dir)
                if matched_dirs:
                    expanded_crash_dirs.extend(matched_dirs)
                    print(f"[*] Expanded '{crash_dir}' to {len(matched_dirs)} directories")
                else:
                    print(f"[!] No directories matched pattern: {crash_dir}")
            else:
                expanded_crash_dirs.append(crash_dir)
        
        for crash_dir in expanded_crash_dirs:
            if 'queue' in crash_dir:
                seeds = [os.path.join(crash_dir, f) for f in os.listdir(crash_dir) if 'orig' in f]
                if not seeds:
                    print(f"[!] No seeds found in {crash_dir}")
                    continue
                print(f"[+] Found {len(seeds)} seeds in {crash_dir}")
                results = execute_seeds_statistical(
                    seeds,
                    em_capture,
                    confidence_level=0.99,
                    margin_of_error=0.03,
                    # log_file=log_file,
                    timestamp=timestamp
                )
            else:
                results = process_crash_directory_with_em(
                    ROUTER_IP,
                    TARGET_CGI,
                    crash_dir,
                    em_capture,
                    max_files=50,
                    # log_file=log_file,
                    timestamp=timestamp
                )
            all_results.extend(results)
            vulnerable_files = [r for r in results if r.get('vulnerable', False)]
            total_vulnerable += len(vulnerable_files)
            # with open(log_file, 'a') as f:
            #     f.write(f"\nSummary for {crash_dir}:\n")
            #     f.write(f"- Files processed: {len(results)}\n")
            #     f.write(f"- Potential vulnerabilities: {len(vulnerable_files)}\n")
        print(f"\n{'='*80}")
        print("FUZZING TEST SUMMARY")
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
        # json_log_file = f"{LOG_PATH}curl_em_fuzzing_results_{timestamp}.json"
        # with open(json_log_file, 'w') as f:
        #     json.dump({
        #         'router_ip': ROUTER_IP,
        #         'target_cgi': TARGET_CGI,
        #         'oscilloscope_ip': OSCILLOSCOPE_IP,
        #         'sca_configuration': {
        #             'channel': SCA_CHANNEL,
        #             'voltage_div': SCA_VOLTAGE_DIV,
        #             'time_div': SCA_TIME_DIV,
        #             'sample_rate_gs': SCA_SAMPLE_RATE / 1e9,
        #             'memory_size': SCA_MEMORY_SIZE,
        #             'trigger_level': SCA_TRIGGER_LEVEL
        #         },
        #         'timestamp': datetime.datetime.now().isoformat(),
        #         'total_tests': len(all_results),
        #         'successful_tests': sum(1 for r in all_results if r.get('success', False)),
        #         'vulnerable_tests': total_vulnerable,
        #         'results': convert_to_json_serializable(all_results)
        #     }, f, indent=2)
        # print(f"\nResults saved to:")
        # print(f"- Log file: {log_file}")
        # print(f"- JSON file: {json_log_file}")
        
        # Save and optionally display the table
        update_table_figure(timestamp)
            
    finally:
        if em_capture:
            em_capture.disconnect()

if __name__ == "__main__":
    main()