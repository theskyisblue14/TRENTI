import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import hdbscan
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from scipy.signal import find_peaks
import scipy.stats
from datetime import datetime
import glob

# Definimos mutation_strategies como en el código original
mutation_strategies = {
    'fft_mean': {
        'high': {
            'issue': 'Intensive computation/algorithmic loops',
            'mutations': ['arith32', 'havoc_blk_large', 'interest32'],
            'values': [0x7FFFFFFF, 0xFFFFFFFF, 0x80000000]
        },
        'low': {
            'issue': 'I/O or blocking operations',
            'mutations': ['flip8', 'havoc_blk_small', 'arith8'],
            'values': [0, 1, 0xFF]
        }
    },
    'fft_std': {
        'high': {
            'issue': 'Variable execution patterns/concurrent operations',
            'mutations': ['extras_ao', 'havoc_stack_8', 'interest16'],
            'values': [0x1000, 0x2000, 0x4000]
        },
        'low': {
            'issue': 'Single execution path/deterministic behavior',
            'mutations': ['flip1', 'flip2', 'arith8'],
            'values': [1, 2, 4, 8]
        }
    },
    'fft_max': {
        'high': {
            'issue': 'Peak resource usage/intensive operations',
            'mutations': ['havoc_blk_xl', 'arith32', 'interest32'],
            'values': [0x7FFFFFFF, 0xFFFFFFFF]
        },
        'low': {
            'issue': 'Resource underutilization',
            'mutations': ['flip4', 'arith8', 'interest8'],
            'values': [0x10, 0x20, 0x40]
        }
    },
    'fft_min': {
        'high': {
            'issue': 'Elevated baseline activity',
            'mutations': ['interest16', 'havoc_blk_medium'],
            'values': [0x100, 0x200, 0x400]
        },
        'low': {
            'issue': 'Minimal baseline operations',
            'mutations': ['flip1', 'flip2', 'arith8'],
            'values': [0, 1, 2]
        }
    },
    'spectral_entropy': {
        'high': {
            'issue': 'Complex control flow/many branches',
            'mutations': ['havoc_dict', 'extras_ao', 'havoc_stack_8'],
            'values': None
        },
        'low': {
            'issue': 'Simple linear execution',
            'mutations': ['flip8', 'arith8', 'interest8'],
            'values': [0, 1, 0xFF]
        }
    },
    'spectral_flatness': {
        'high': {
            'issue': 'Uniform resource usage/balanced execution',
            'mutations': ['havoc_blk_medium', 'arith16', 'interest16'],
            'values': [0x1000, 0x2000]
        },
        'low': {
            'issue': 'Sporadic resource usage',
            'mutations': ['flip4', 'havoc_blk_small'],
            'values': [16, 32, 64]
        }
    },
    'freq_centroid': {
        'high': {
            'issue': 'High-frequency operations dominance',
            'mutations': ['arith32', 'havoc_blk_large'],
            'values': [0x10000, 0x20000]
        },
        'low': {
            'issue': 'Low-frequency operations dominance',
            'mutations': ['flip8', 'arith8'],
            'values': [0x10, 0x20]
        }
    },
    'low_power': {
        'high': {
            'issue': 'Memory-intensive operations',
            'mutations': ['interest16', 'havoc_blk_xl', 'arith16'],
            'values': [0x100, 0x800, 0x1000]
        },
        'low': {
            'issue': 'Limited memory interaction',
            'mutations': ['flip4', 'havoc_blk_small'],
            'values': [16, 32, 64]
        }
    },
    'mid_power': {
        'high': {
            'issue': 'Mixed operation types/data processing',
            'mutations': ['havoc_blk_medium', 'arith16'],
            'values': [0x1000, 0x2000]
        },
        'low': {
            'issue': 'Limited data processing',
            'mutations': ['flip8', 'arith8'],
            'values': [0x80, 0x100]
        }
    },
    'high_power': {
        'high': {
            'issue': 'CPU-bound operations',
            'mutations': ['arith32', 'interest32'],
            'values': [0x7FFFFFFF, 0x80000000]
        },
        'low': {
            'issue': 'I/O-bound operations',
            'mutations': ['flip8', 'havoc_blk_small'],
            'values': [0xFF, 0x100]
        }
    },
    'low_to_high_ratio': {
        'high': {
            'issue': 'Memory-dominant execution',
            'mutations': ['havoc_blk_xl', 'interest16'],
            'values': [0x1000, 0x2000]
        },
        'low': {
            'issue': 'Computation-dominant execution',
            'mutations': ['arith32', 'interest32'],
            'values': [0x7FFFFFFF, 0x80000000]
        }
    },
    'mid_to_total_ratio': {
        'high': {
            'issue': 'Balanced processing distribution',
            'mutations': ['havoc_blk_medium', 'arith16'],
            'values': [0x1000, 0x2000]
        },
        'low': {
            'issue': 'Extreme processing bias',
            'mutations': ['flip8', 'arith8'],
            'values': [0x80, 0x100]
        }
    },
    'peak_count': {
        'high': {
            'issue': 'Multiple execution phases',
            'mutations': ['havoc_dict', 'extras_ao'],
            'values': None
        },
        'low': {
            'issue': 'Single-phase execution',
            'mutations': ['flip8', 'arith8'],
            'values': [0, 1, 0xFF]
        }
    },
    'peak_value': {
        'high': {
            'issue': 'Resource usage spikes',
            'mutations': ['havoc_blk_xl', 'arith32'],
            'values': [0x7FFFFFFF, 0x80000000]
        },
        'low': {
            'issue': 'Steady resource usage',
            'mutations': ['flip4', 'arith8'],
            'values': [16, 32, 64]
        }
    },
    'harmonic_ratio': {
        'high': {
            'issue': 'Regular cyclic patterns',
            'mutations': ['havoc_blk_medium', 'extras_ao'],
            'values': [0x100, 0x200]
        },
        'low': {
            'issue': 'Irregular execution',
            'mutations': ['flip8', 'havoc_blk_small'],
            'values': [0, 1, 0xFF]
        }
    },
    'harmonic1': {
        'high': {
            'issue': 'Primary frequency usage spikes',
            'mutations': ['havoc_blk_large', 'arith32', 'interest32'],
            'values': [0x7FFFFFFF, 0xFFFFFFFF, 0x80000000]
        },
        'low': {
            'issue': 'Reduced primary frequency activity',
            'mutations': ['flip8', 'arith8', 'havoc_blk_small'],
            'values': [0, 1, 0xFF]
        }
    },
    'harmonic2': {
        'high': {
            'issue': 'Secondary frequency component dominance',
            'mutations': ['havoc_blk_medium', 'arith16', 'extras_ao'],
            'values': [0x1000, 0x2000, 0x4000]
        },
        'low': {
            'issue': 'Limited secondary frequency patterns',
            'mutations': ['flip4', 'arith8', 'interest8'],
            'values': [16, 32, 64]
        }
    },
    'harmonic3': {
        'high': {
            'issue': 'Tertiary frequency patterns present',
            'mutations': ['havoc_blk_xl', 'extras_ao', 'interest16'],
            'values': [0x8000, 0x10000]
        },
        'low': {
            'issue': 'Missing tertiary frequency components',
            'mutations': ['flip2', 'arith8', 'havoc_blk_small'],
            'values': [4, 8, 16]
        }
    },
    'harmonic4': {
        'high': {
            'issue': 'Quaternary frequency activity',
            'mutations': ['havoc_dict', 'extras_ao', 'interest32'],
            'values': [0x20000, 0x40000]
        },
        'low': {
            'issue': 'Weak quaternary components',
            'mutations': ['flip1', 'arith8', 'havoc_blk_small'],
            'values': [2, 4, 8]
        }
    }
}

# Funciones auxiliares para recalcular características desde .npz
def shannon_entropy(signal):
    """Calculate Shannon entropy of signal"""
    try:
        hist, _ = np.histogram(signal, bins=50, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
    except Exception as e:
        print(f"Error calculating Shannon entropy: {e}")
        return 0

def convert_to_dbm(magnitude, reference_impedance=50):
    """Convert magnitude to dBm"""
    epsilon = 1e-10
    power = (magnitude**2) / (2 * reference_impedance)
    dbm_values = 10 * np.log10(power / 0.001 + epsilon)
    return dbm_values

def analyze_raw_signal(signal, sample_rate, fft_freq=None, fft_magnitude=None):
    """Analyze raw time-domain signal"""
    try:
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        processed_signal = np.clip(signal, -1e6, 1e6)
        
        # Basic statistics
        rms = float(np.sqrt(np.mean(processed_signal**2))) if len(processed_signal) > 0 else 0.0
        mean_val = float(np.mean(processed_signal))
        std_dev = float(np.std(processed_signal))
        crest_factor = float(np.max(np.abs(processed_signal)) / rms if rms != 0 else 0)
        peak_to_peak = float(np.max(processed_signal) - np.min(processed_signal))
        entropy = float(shannon_entropy(processed_signal))
        
        # Peak detection
        if len(processed_signal) > 10:
            peaks, _ = find_peaks(np.abs(processed_signal), 
                                height=np.max(np.abs(processed_signal))*0.05, 
                                distance=max(10, len(processed_signal)//100))
            peak_count = int(len(peaks))
        else:
            peak_count = 0
            
        # Statistical measures
        kurtosis = float(scipy.stats.kurtosis(processed_signal)) if len(processed_signal) > 3 else 0.0
        skewness = float(scipy.stats.skew(processed_signal)) if len(processed_signal) > 1 else 0.0
        
        # Zero crossing rate
        if len(processed_signal) > 1:
            zero_crossings = np.sum(np.diff(np.signbit(processed_signal)) != 0)
            zcr = float(zero_crossings) / len(processed_signal) * sample_rate
        else:
            zcr = 0.0
            
        # FFT analysis
        if fft_freq is None or fft_magnitude is None:
            fft_signal = np.fft.fft(processed_signal)
            fft_freq = np.fft.fftfreq(len(processed_signal), 1/sample_rate)
            fft_magnitude = np.abs(fft_signal)
            
        # Energy in frequency bands
        mask_low = (fft_freq >= 0) & (fft_freq <= 500e6)
        mask_mid = (fft_freq > 500e6) & (fft_freq <= 1000e6)
        mask_high = (fft_freq > 1000e6) & (fft_freq <= 2000e6)
        
        energy_low = float(np.sum(fft_magnitude[mask_low]**2)) if np.any(mask_low) else 0.0
        energy_mid = float(np.sum(fft_magnitude[mask_mid]**2)) if np.any(mask_mid) else 0.0
        energy_high = float(np.sum(fft_magnitude[mask_high]**2)) if np.any(mask_high) else 0.0
        
        stats = {
            'rms': rms, 'mean': mean_val, 'std_dev': std_dev, 'crest_factor': crest_factor,
            'peak_to_peak': peak_to_peak, 'entropy': entropy, 'peak_count': peak_count,
            'kurtosis': kurtosis, 'skewness': skewness, 'zcr': zcr,
            'energy_low': energy_low, 'energy_mid': energy_mid, 'energy_high': energy_high
        }
        return stats, fft_freq, fft_magnitude
        
    except Exception as e:
        print(f"Error analyzing raw signal: {e}")
        # Return default values on error
        default_stats = {
            'rms': 0.0, 'mean': 0.0, 'std_dev': 0.0, 'crest_factor': 0.0,
            'peak_to_peak': 0.0, 'entropy': 0.0, 'peak_count': 0,
            'kurtosis': 0.0, 'skewness': 0.0, 'zcr': 0.0,
            'energy_low': 0.0, 'energy_mid': 0.0, 'energy_high': 0.0
        }
        return default_stats, np.array([]), np.array([])

def analyze_fft(fft_freq, fft_magnitude_dbm):
    """Analyze frequency domain signal"""
    try:
        fft_magnitude_dbm = np.nan_to_num(fft_magnitude_dbm, nan=-100.0, posinf=-100.0, neginf=-100.0)
        mask = (fft_freq >= 0) & (fft_freq <= 2e9)
        masked_freq = fft_freq[mask]
        masked_magnitude = fft_magnitude_dbm[mask]
        
        if len(masked_magnitude) == 0:
            # Return default values if no valid frequency data
            return {
                'peak_freq': 0.0, 'peak_magnitude': 0.0,
                'spectral_entropy': 0.0, 'harmonic_distortion': 0.0,
                'spectral_crest': 0.0, 'spectral_flatness': 0.0,
                'spectral_rolloff': 0.0, 'hnr': 0.0
            }
            
        # Peak detection
        if len(masked_magnitude) > 10:
            peaks, _ = find_peaks(masked_magnitude, 
                                height=np.max(masked_magnitude)*0.05, 
                                distance=max(10, len(masked_magnitude)//100))
            peak_freqs = masked_freq[peaks] if len(peaks) > 0 else np.array([0])
            peak_mags = masked_magnitude[peaks] if len(peaks) > 0 else np.array([0])
        else:
            peak_freqs = np.array([0])
            peak_mags = np.array([0])
            
        if len(peak_mags) > 0:
            max_idx = np.argmax(peak_mags)
            peak_freq = float(peak_freqs[max_idx])
            peak_magnitude = float(peak_mags[max_idx])
        else:
            peak_freq, peak_magnitude = 0.0, 0.0
            
        # Spectral features
        spectral_entropy = float(shannon_entropy(masked_magnitude))
        spectral_mean = float(np.mean(masked_magnitude))
        spectral_crest = float(np.max(masked_magnitude) / spectral_mean if spectral_mean != 0 else 0)
        
        # Noise floor
        noise_floor = np.percentile(masked_magnitude[masked_magnitude > -np.inf], 25) if np.any(masked_magnitude > -np.inf) else -100.0
        
        # Harmonic analysis
        harmonics = []
        if peak_freq > 0:
            for i in range(2, 6):
                harmonic_freq = peak_freq * i
                if harmonic_freq <= 2e9:
                    harmonic_idx = np.argmin(np.abs(masked_freq - harmonic_freq))
                    if harmonic_idx < len(masked_magnitude) and masked_magnitude[harmonic_idx] > noise_floor + 3:  # 3 dB above noise
                        harmonics.append((float(harmonic_freq), float(masked_magnitude[harmonic_idx])))
                        
        harmonic_distortion = float(np.sqrt(sum(h[1]**2 for h in harmonics)) / peak_magnitude if peak_magnitude > 0 else 0)
        
        # Spectral flatness
        power_spectrum = np.power(10, masked_magnitude / 10)  # Convert from dBm to linear
        power_spectrum[power_spectrum <= 0] = 1e-10  # Avoid log(0)
        
        geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
        arithmetic_mean = np.mean(power_spectrum)
        spectral_flatness = float(geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0)
        
        # Spectral rolloff
        cumulative_energy = np.cumsum(power_spectrum)
        total_energy = cumulative_energy[-1] if cumulative_energy[-1] > 0 else 1e-10
        rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
        spectral_rolloff = float(masked_freq[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0)
        
        # Harmonic-to-noise ratio
        harmonic_power = sum(10**(h[1]/10) for h in harmonics)  # Convert dBm to linear power
        noise_power = np.sum(power_spectrum) - harmonic_power - 10**(peak_magnitude/10)
        hnr = float(10 * np.log10(harmonic_power / (noise_power + 1e-10)) if noise_power > 0 and harmonic_power > 0 else 0)
        
        params = {
            'peak_freq': peak_freq, 'peak_magnitude': peak_magnitude,
            'spectral_entropy': spectral_entropy, 'harmonic_distortion': harmonic_distortion,
            'spectral_crest': spectral_crest, 'spectral_flatness': spectral_flatness,
            'spectral_rolloff': spectral_rolloff, 'hnr': hnr
        }
        
        # Clean up any NaN/inf values
        for key in params:
            params[key] = np.nan_to_num(params[key], nan=0.0, posinf=0.0, neginf=0.0)
            
        return params
        
    except Exception as e:
        print(f"Error analyzing FFT: {e}")
        # Return default values on error
        return {
            'peak_freq': 0.0, 'peak_magnitude': 0.0,
            'spectral_entropy': 0.0, 'harmonic_distortion': 0.0,
            'spectral_crest': 0.0, 'spectral_flatness': 0.0,
            'spectral_rolloff': 0.0, 'hnr': 0.0
        }

def get_campaign_number_from_path(base_path):
    """Extract campaign number from the path structure"""
    try:
        # Look for trenti_sca_outputs_XXX_Y directories in base_path
        trenti_dirs = [d for d in os.listdir(base_path) 
                      if d.startswith('trenti_sca_outputs_') and os.path.isdir(os.path.join(base_path, d))]
        
        if not trenti_dirs:
            print("No trenti_sca_outputs directories found, using campaign number 1")
            return 1
            
        # Extract campaign numbers (Y from trenti_sca_outputs_XXX_Y)
        campaign_numbers = []
        for dir_name in trenti_dirs:
            match = re.search(r'trenti_sca_outputs_\d+_(\d+)', dir_name)
            if match:
                campaign_numbers.append(int(match.group(1)))
        
        if campaign_numbers:
            # Return the highest campaign number found
            max_campaign = max(campaign_numbers)
            print(f"Found campaign numbers: {campaign_numbers}, using: {max_campaign}")
            return max_campaign
        else:
            print("Could not extract campaign numbers, using campaign number 1")
            return 1
            
    except Exception as e:
        print(f"Error getting campaign number: {e}")
        return 1

def find_latest_directories(host_path=None):
    """Find the latest calibration and operation directories from host path structure"""
    try:
        # Use host_path from GUI if provided, otherwise use defaults
        if host_path:
            possible_host_paths = [host_path]
            print(f"Using host path from GUI: {host_path}")
        else:
            # Fallback to default paths if no host_path provided
            possible_host_paths = [
                '/home/atenea/trenti/evaluations',
                '/home/atenea/gaflerna/evaluations', 
                'evaluations',
                '.'
            ]
            print("No host path provided, using default search paths")
        
        base_path = None
        waveforms_path = None
        
        # Search for the evaluation structure with image_XXX/waveforms
        for host_path_candidate in possible_host_paths:
            if os.path.exists(host_path_candidate):
                print(f"Checking host path: {host_path_candidate}")
                
                # Look for image_XXX directories
                image_dirs = [d for d in os.listdir(host_path_candidate) 
                             if d.startswith('image_') and os.path.isdir(os.path.join(host_path_candidate, d))]
                
                for image_dir in image_dirs:
                    image_path = os.path.join(host_path_candidate, image_dir)
                    waveforms_candidate = os.path.join(image_path, 'waveforms')
                                       
                    if os.path.exists(waveforms_candidate):
                         base_path = image_path
                         waveforms_path = waveforms_candidate
                         print(f"Found waveforms directory: {waveforms_path}")
                         break
                
                if waveforms_path:
                    break
        
        if not waveforms_path:
            raise FileNotFoundError("Could not find waveforms directory in any image_XXX path")
            
        print(f"Using host base path: {base_path}")
        print(f"Using waveforms path: {waveforms_path}")
        
        # Find latest operation directory in waveforms
        operation_dirs = [d for d in os.listdir(waveforms_path) if d.startswith('operation_')]
        if not operation_dirs:
            raise FileNotFoundError(f"No operation directories found in {waveforms_path}")
            
        # Extract timestamps and find latest
        operation_timestamps = []
        for d in operation_dirs:
            # Match various timestamp formats
            match = re.search(r'_(20\d{2}_\d{2}_\d{2}_\d{2}o\d{2})', d)
            if not match:
                match = re.search(r'_(20\d{2}\d{2}\d{2}_\d{6})', d)
            if match:
                operation_timestamps.append((match.group(1), d))
                
        if not operation_timestamps:
            # If no timestamps found, use the last directory alphabetically
            latest_op_dir = sorted(operation_dirs)[-1]
            latest_op_timestamp = datetime.now().strftime("%Y_%m_%d_%Ho%M")
        else:
            latest_op_timestamp, latest_op_dir = max(operation_timestamps, key=lambda x: x[0])
            
        operation_path = os.path.join(waveforms_path, latest_op_dir)
        
        # Find corresponding calibration directory in waveforms
        calibration_dirs = [d for d in os.listdir(waveforms_path) if d.startswith('calibration_')]
        
        # Look for matching calibration first
        matching_calibration = f'calibration_{latest_op_timestamp}'
        matching_calibration_path = os.path.join(waveforms_path, matching_calibration)
        
        if os.path.exists(matching_calibration_path):
            calibration_path = matching_calibration_path
        else:
            # Find latest calibration directory
            calibration_timestamps = []
            for d in calibration_dirs:
                match = re.search(r'_(20\d{2}_\d{2}_\d{2}_\d{2}o\d{2})', d)
                if not match:
                    match = re.search(r'_(20\d{2}\d{2}\d{2}_\d{6})', d)
                if match:
                    calibration_timestamps.append((match.group(1), d))
                    
            if calibration_timestamps:
                _, latest_cal_dir = max(calibration_timestamps, key=lambda x: x[0])
                calibration_path = os.path.join(waveforms_path, latest_cal_dir)
            elif calibration_dirs:
                # Use the last calibration directory alphabetically
                latest_cal_dir = sorted(calibration_dirs)[-1]
                calibration_path = os.path.join(waveforms_path, latest_cal_dir)
            else:
                raise FileNotFoundError(f"No calibration directories found in {waveforms_path}")
                
        return calibration_path, operation_path, latest_op_timestamp, base_path
        
    except Exception as e:
        print(f"Error finding directories: {e}")
        raise

def save_in_sca_images_dir(filename, base_path):
    """Helper function to save files in the correct directory structure"""
    # Create sca_images directory in the base evaluation path
    sca_images_dir = os.path.join(base_path, 'sca_images')
    os.makedirs(sca_images_dir, exist_ok=True)
    
    # Full path for the file
    full_path = os.path.join(sca_images_dir, filename)
    return full_path

def save_in_anomaly_names_dir(filename, base_path):
    """Helper function to save anomaly names files in the correct directory structure"""
    # Create anomaly_names directory in the base evaluation path
    anomaly_names_dir = os.path.join(base_path, 'anomaly_names')
    os.makedirs(anomaly_names_dir, exist_ok=True)
    
    # Full path for the file
    full_path = os.path.join(anomaly_names_dir, filename)
    return full_path

class SCADetectorWithTraceClustering:
    def __init__(self, contamination=0.001, n_estimators=500, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.99)
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state
        )
        self.lof = LocalOutlierFactor(
            contamination=contamination,
            novelty=True,
            n_jobs=-1
        )

    def _calculate_adaptive_thresholds(self, pca_features):
        """Calculate adaptive thresholds for anomaly detection"""
        try:
            if_scores = self.isolation_forest.score_samples(pca_features)
            
            # Raw LOF scores
            raw_lof_scores = -self.lof.score_samples(pca_features)
            # Log-transform LOF scores to handle extreme values
            lof_scores = np.log1p(np.maximum(raw_lof_scores, 0))
            
            exec_centroid = np.mean(pca_features, axis=0)
            
            # Calculate raw distances
            raw_distances = [np.linalg.norm(x - exec_centroid) for x in pca_features]
            # Log-transform distances to handle extreme values
            distances = np.log1p(np.maximum(raw_distances, 0))
            
            # Calculate more robust thresholds with IQR-based outlier detection
            if_q1, if_q3 = np.percentile(if_scores, [25, 75])
            if_iqr = if_q3 - if_q1
            if_threshold = if_q1 - 1.5 * if_iqr  # Lower bound for IF
            
            lof_q1, lof_q3 = np.percentile(lof_scores, [25, 75])
            lof_iqr = lof_q3 - lof_q1
            lof_threshold = lof_q3 + 1.5 * lof_iqr  # Upper bound for LOF
            
            dist_q1, dist_q3 = np.percentile(distances, [25, 75])
            dist_iqr = dist_q3 - dist_q1
            dist_threshold = dist_q3 + 1.5 * dist_iqr  # Upper bound for distance
            
            return {
                'if_threshold': if_threshold,
                'lof_threshold': lof_threshold,
                'distance_threshold': dist_threshold,
                'exec_centroid': exec_centroid,
                'raw_lof_threshold': np.percentile(raw_lof_scores, 99),
                'raw_dist_threshold': np.percentile(raw_distances, 99)
            }
        except Exception as e:
            print(f"Error calculating adaptive thresholds: {e}")
            # Return conservative default thresholds
            return {
                'if_threshold': -0.5,
                'lof_threshold': 2.0,
                'distance_threshold': 3.0,
                'exec_centroid': np.zeros(pca_features.shape[1]),
                'raw_lof_threshold': 5.0,
                'raw_dist_threshold': 5.0
            }

    def fit(self, exec_signals):
        """Fit the detector with calibration signals"""
        try:
            print(f"Fitting detector with {len(exec_signals)} calibration signals")
            self.exec_signals = exec_signals
            self.features = exec_signals
            self.n_features = self.features.shape[1]

            scaled_features = self.scaler.fit_transform(self.features)
            pca_features = self.pca.fit_transform(scaled_features)
            
            self.feature_names = [
                'rms', 'mean', 'std_dev', 'crest_factor', 'peak_to_peak', 'entropy', 'peak_count',
                'kurtosis', 'skewness', 'zcr', 'energy_low', 'energy_mid', 'energy_high',
                'peak_freq', 'peak_magnitude', 'spectral_entropy', 'spectral_flatness',
                'spectral_rolloff', 'hnr', 'harmonic_distortion', 'spectral_crest'
            ]
            
            # Calculate feature importance
            feature_importance = np.abs(self.pca.components_).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            })
            
            top_features = importance_df.sort_values('importance', ascending=False)
            print("Feature Importance Analysis:")
            print("-" * 50)
            print("Top important features:")
            print(top_features.head(10))
            
            # Add Mahalanobis distance calculation
            self.exec_mean = np.mean(scaled_features, axis=0)
            self.exec_cov = np.cov(scaled_features, rowvar=False)
            self.exec_cov_inv = np.linalg.pinv(self.exec_cov + np.eye(scaled_features.shape[1]) * 1e-6)
            
            # Calculate Mahalanobis distances for calibration data
            mahalanobis_distances = []
            for i in range(scaled_features.shape[0]):
                x = scaled_features[i, :]
                dist = np.sqrt(np.dot(np.dot((x - self.exec_mean), self.exec_cov_inv), (x - self.exec_mean)))
                mahalanobis_distances.append(dist)
            
            # Set threshold as the 99th percentile of calibration distances
            self.mahalanobis_threshold = np.percentile(mahalanobis_distances, 99)
            
            # Train the anomaly detection models
            self.isolation_forest.fit(pca_features)
            self.lof.fit(pca_features)
            
            # Calculate adaptive thresholds
            self.thresholds = self._calculate_adaptive_thresholds(pca_features)
            
            # Store feature importance info
            self.feature_importance = feature_importance
            self.important_features = sorted(enumerate(self.feature_importance), 
                                          key=lambda x: x[1], reverse=True)
            
            print("Detector training completed successfully")
            
        except Exception as e:
            print(f"Error fitting detector: {e}")
            raise

    def check_signal_compatibility(self, features):
        """Check if the signal is compatible with calibration data"""
        # Enhanced compatibility check with better error handling
        try:
            # Basic shape check
            if features.shape[1] != self.n_features:
                return False, {"error": f"Feature count mismatch: expected {self.n_features}, got {features.shape[1]}"}
            
            # Check for extreme values that might indicate data corruption
            if np.any(np.isinf(features)) or np.any(np.isnan(features)):
                return False, {"error": "Signal contains NaN or infinite values"}
            
            # Check if signal is completely zero (dead signal)
            if np.all(features == 0):
                return False, {"error": "Signal appears to be completely zero (dead signal)"}
            
            # Check for reasonable value ranges compared to calibration
            feature_ranges = np.ptp(self.exec_signals, axis=0)  # Peak-to-peak range
            signal_values = features[0]
            
            # Allow signals to be within 100x the calibration range
            for i, (signal_val, cal_range) in enumerate(zip(signal_values, feature_ranges)):
                if cal_range > 0 and abs(signal_val) > 100 * cal_range:
                    print(f"Warning: Feature {self.feature_names[i]} has extreme value: {signal_val} (cal range: {cal_range})")
            
            return True, {}
            
        except Exception as e:
            return False, {"error": f"Compatibility check failed: {str(e)}"}

    def predict(self, features):
        """Predict if signal is anomalous"""
        try:
            # Check compatibility first
            is_compatible, compat_info = self.check_signal_compatibility(features)
            if not is_compatible:
                return "Incompatible", f"Signal incompatible: {compat_info.get('error', 'Unknown error')}", None, 0.0
            
            # Process features
            scaled_features = self.scaler.transform(features)
            pca_features = self.pca.transform(scaled_features)
            
            # 1. Isolation Forest score (negative is more anomalous)
            if_score = self.isolation_forest.score_samples(pca_features)[0]
            
            # 2. LOF score with log transformation (positive is more anomalous)
            raw_lof_score = -self.lof.score_samples(pca_features)[0]
            lof_score = np.log1p(max(raw_lof_score, 0))
            lof_threshold = np.log1p(max(self.thresholds['lof_threshold'], 0))
            
            # 3. Distance to centroid with log transformation
            raw_dist_exec = np.linalg.norm(pca_features[0] - self.thresholds['exec_centroid'])
            dist_exec = np.log1p(max(raw_dist_exec, 0))
            dist_threshold = np.log1p(max(self.thresholds['distance_threshold'], 0))
            
            # 4. Mahalanobis distance with regularization
            # Exclude peak_freq (index 13) from the Mahalanobis calculation to improve stability
            peak_freq_idx = 13
            if scaled_features.shape[1] > peak_freq_idx:
                features_for_mahalanobis = np.delete(scaled_features[0], peak_freq_idx)
                exec_mean_for_mahalanobis = np.delete(self.exec_mean, peak_freq_idx)
                cov_inv_for_mahalanobis = np.delete(np.delete(self.exec_cov_inv, peak_freq_idx, axis=0), peak_freq_idx, axis=1)
                
                # Calculate Mahalanobis distance with better numerical stability
                diff = features_for_mahalanobis - exec_mean_for_mahalanobis
                mahalanobis_dist = np.sqrt(np.maximum(0, np.dot(np.dot(diff, cov_inv_for_mahalanobis), diff)))
            else:
                mahalanobis_dist = 0.0
            
            # 5. Z-score analysis but EXCLUDE peak_freq (index 13)
            cal_std = np.std(self.scaler.transform(self.exec_signals), axis=0)
            z_scores = (scaled_features[0] - self.exec_mean) / (cal_std + 1e-10)
            
            if len(z_scores) > peak_freq_idx:
                z_scores_without_peak_freq = np.delete(z_scores, peak_freq_idx)
                feature_names_without_peak_freq = np.delete(np.array(self.feature_names), peak_freq_idx)
            else:
                z_scores_without_peak_freq = z_scores
                feature_names_without_peak_freq = np.array(self.feature_names)
            
            max_z_score = np.max(np.abs(z_scores_without_peak_freq)) if len(z_scores_without_peak_freq) > 0 else 0
            max_z_feature = feature_names_without_peak_freq[np.argmax(np.abs(z_scores_without_peak_freq))] if len(z_scores_without_peak_freq) > 0 else "unknown"
            
            # Modified confidence calculation
            confidence = np.clip(
                0.25 * ((self.thresholds['if_threshold'] - if_score) / (abs(self.thresholds['if_threshold']) + 1e-10)) + 
                0.25 * (lof_score / (lof_threshold + 1e-10)) +
                0.25 * (dist_exec / (dist_threshold + 1e-10)) +
                0.25 * (mahalanobis_dist / (self.mahalanobis_threshold + 1e-10)),
                0, 1
            )
            
            # More moderate anomaly conditions
            anomaly_conditions = [
                if_score < self.thresholds['if_threshold'] * 0.9,
                lof_score > lof_threshold * 1.1,
                dist_exec > dist_threshold * 1.2,
                mahalanobis_dist > self.mahalanobis_threshold * 1.3,
                max_z_score > 4.0
            ]
            
            anomaly_votes = sum(anomaly_conditions)
            
            # Require at least 3/5 votes to classify as anomaly
            is_anomaly = (anomaly_votes >= 3)
            
            # Create detailed explanation
            details = (
                f"[Votes: {anomaly_votes}/5, "
                f"IF: {if_score:.4f}({self.thresholds['if_threshold']:.4f}), "
                f"LOF: {lof_score:.4f}({lof_threshold:.4f}), "
                f"Dist: {dist_exec:.2f}({dist_threshold:.2f}), "
                f"Mahalanobis: {mahalanobis_dist:.2f}({self.mahalanobis_threshold:.2f}), "
                f"MaxZ({max_z_feature}): {max_z_score:.2f}, "
                f"Conf: {confidence:.2f}]"
            )
            
            return (
                "Anomaly" if is_anomaly else "Normal", 
                details, 
                pca_features[0], 
                confidence
            )
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Error", f"Prediction failed: {str(e)}", None, 0.0

    def visualize_calibration(self, base_path, latest_op_timestamp):
        """Visualize the calibration data in PCA space"""
        try:
            exec_pca = self.pca.transform(self.scaler.transform(self.exec_signals))
            
            plt.figure(figsize=(10, 8))
            plt.scatter(exec_pca[:, 0], exec_pca[:, 1], c='#2E86AB', label='Exec Calibration', alpha=0.6, s=50)
            
            plt.title('Calibration Signals in PCA Space', fontsize=12, fontweight='bold')
            plt.xlabel(f'PC1\n(Feature: {self.feature_names[self.important_features[0][0]]})', fontsize=10)
            plt.ylabel(f'PC2\n(Feature: {self.feature_names[self.important_features[1][0]]})', fontsize=10)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to sca_images directory
            file_path = save_in_sca_images_dir(f'calibration_{campaign_number}_{latest_op_timestamp}.pdf', base_path)
            plt.savefig(file_path, format='pdf', bbox_inches='tight')
            print(f"Calibration visualization saved to: {file_path}")
            plt.show()
            
        except Exception as e:
            print(f"Error visualizing calibration: {e}")
    
    def visualize_clusters_only(self, anomaly_features, hdbscan_labels, optics_labels, base_path, latest_op_timestamp):
        """Visualize clusters found in anomaly data"""
        try:
            if len(anomaly_features) < 2:
                print("Not enough anomalies to visualize clusters")
                return
                
            # Handle potential 3D arrays
            anom_features = anomaly_features
            if anom_features.ndim > 2:
                anom_features = anom_features.reshape(-1, anom_features.shape[-1])
                
            features_scaled = self.scaler.transform(anom_features)
            anomaly_pca = self.pca.transform(features_scaled)
            
            # Get the feature names for the first two principal components
            pc1_feature = self.feature_names[self.important_features[0][0]]
            pc2_feature = self.feature_names[self.important_features[1][0]]
            
            plt.figure(figsize=(4, 4), dpi=300)
            unique_labels = np.unique(hdbscan_labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, cluster in enumerate(unique_labels):
                mask = hdbscan_labels == cluster
                label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
                plt.scatter(anomaly_pca[mask, 0], anomaly_pca[mask, 1],
                          c=[colors[i]], label=label, alpha=0.8, s=25)
            
            plt.title('Anomaly Clusters', fontsize=11)
            plt.xlabel(f'PC1 ({pc1_feature})', fontsize=10)
            plt.ylabel(f'PC2 ({pc2_feature})', fontsize=10)
            plt.legend(fontsize=9, loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to sca_images directory
            cluster_file = save_in_sca_images_dir(f'em_anomaly_clusters_{campaign_number}_{latest_op_timestamp}.pdf', base_path)
            plt.savefig(cluster_file, format='pdf', bbox_inches='tight')
            print(f"Cluster visualization saved to: {cluster_file}")
            plt.show()
            
        except Exception as e:
            print(f"Error visualizing clusters: {e}")

    def visualize(self, anomaly_features, hdbscan_labels, operational_signals, operational_results, base_path, latest_op_timestamp):
        """Visualize operational data with anomalies highlighted"""
        try:
            # Handle potential 3D arrays
            op_signals = operational_signals
            if op_signals.ndim > 2:
                op_signals = op_signals.reshape(-1, op_signals.shape[-1])
            
            operational_scaled = self.scaler.transform(op_signals)
            operational_pca = self.pca.transform(operational_scaled)
            exec_pca = self.pca.transform(self.scaler.transform(self.exec_signals))
            
            # Get the feature names for the first two principal components
            pc1_feature = self.feature_names[self.important_features[0][0]]
            pc2_feature = self.feature_names[self.important_features[1][0]]
            
            # Figure for cluster view
            plt.figure(figsize=(4, 4), dpi=300)
            plt.scatter(exec_pca[:, 0], exec_pca[:, 1], c='gray', label='Calibration', alpha=0.4, s=15)
            
            normal_mask = np.array(operational_results) == 'Normal'
            if np.any(normal_mask):
                plt.scatter(operational_pca[normal_mask, 0], operational_pca[normal_mask, 1],
                           c='green', label='Normal', alpha=0.6, s=15)
            
            anomaly_mask = np.array(operational_results) == 'Anomaly'
            if np.any(anomaly_mask):
                anomaly_pca = operational_pca[anomaly_mask]
                
                if len(hdbscan_labels) > 0:
                    unique_labels = np.unique(hdbscan_labels)
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                    
                    for i, cluster in enumerate(unique_labels):
                        cluster_mask = hdbscan_labels == cluster
                        if np.any(cluster_mask) and len(cluster_mask) == len(anomaly_pca):
                            label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
                            try:
                                plt.scatter(anomaly_pca[cluster_mask, 0],
                                           anomaly_pca[cluster_mask, 1],
                                           c=[colors[i]], label=label, alpha=0.8, s=25, edgecolors='k')
                            except IndexError:
                                plt.scatter(anomaly_pca[:, 0], anomaly_pca[:, 1],
                                           c='red', label='Anomalies', alpha=0.8, s=25, edgecolors='k')
                        else:
                            plt.scatter(anomaly_pca[:, 0], anomaly_pca[:, 1],
                                      c='red', label='Anomalies', alpha=0.8, s=25, edgecolors='k')
                            break
                else:
                    plt.scatter(anomaly_pca[:, 0], anomaly_pca[:, 1],
                               c='red', label='Anomalies', alpha=0.8, s=25, edgecolors='k')
            
            plt.title('Cluster View of Anomalies', fontsize=11)
            plt.xlabel(f'PC1 ({pc1_feature})', fontsize=10)
            plt.ylabel(f'PC2 ({pc2_feature})', fontsize=10)
            plt.legend(fontsize=9, loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to sca_images directory
            cluster_file = save_in_sca_images_dir(f'em_cal_op_anomaly_clusters_{campaign_number}_{latest_op_timestamp}.pdf', base_path)
            plt.savefig(cluster_file, format='pdf', bbox_inches='tight')
            print(f"Cluster view saved to: {cluster_file}")
            plt.show()
            
            # Create a second figure for the binary classification view
            plt.figure(figsize=(4, 4), dpi=300)
            plt.scatter(exec_pca[:, 0], exec_pca[:, 1], c='gray', label='Calibration', alpha=0.4, s=15)
            
            if np.any(normal_mask):
                plt.scatter(operational_pca[normal_mask, 0], operational_pca[normal_mask, 1],
                           c='green', label='Normal', alpha=0.6, s=15)
            
            if np.any(anomaly_mask):
                plt.scatter(operational_pca[anomaly_mask, 0], operational_pca[anomaly_mask, 1],
                           c='red', label='Anomalies', alpha=0.8, s=25, edgecolors='k')
            
            plt.title('Binary Classification View', fontsize=11)
            plt.xlabel(f'PC1 ({pc1_feature})', fontsize=10)
            plt.ylabel(f'PC2 ({pc2_feature})', fontsize=10)
            plt.legend(fontsize=9, loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to sca_images directory
            binary_file = save_in_sca_images_dir(f'em_anomaly_binary_{campaign_number}_{latest_op_timestamp}.pdf', base_path)
            plt.savefig(binary_file, format='pdf', bbox_inches='tight')
            print(f"Binary view saved to: {binary_file}")
            plt.show()
            
        except Exception as e:
            print(f"Error in visualization: {e}")

    def visualize_3d(self, operational_signals=None, operational_results=None, base_path=None, latest_op_timestamp=None):
        """Visualize data in 3D PCA space"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            exec_pca = self.pca.transform(self.scaler.transform(self.exec_signals))
            
            # Check if operational_signals has more than 2 dimensions and reshape if needed
            if operational_signals is not None:
                op_signals = operational_signals
                if op_signals.ndim > 2:
                    op_signals = op_signals.reshape(-1, op_signals.shape[-1])
                
                operational_pca = self.pca.transform(self.scaler.transform(op_signals))
                normal_mask = np.array(operational_results) == 'Normal'
                anomaly_mask = np.array(operational_results) == 'Anomaly'
            
            # Get feature names for the first three principal components
            pc1_feature = self.feature_names[self.important_features[0][0]]
            pc2_feature = self.feature_names[self.important_features[1][0]]
            pc3_feature = self.feature_names[self.important_features[2][0]] if len(self.important_features) > 2 else "PC3"
            
            fig = plt.figure(figsize=(4, 4), dpi=300)
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(exec_pca[:, 0], exec_pca[:, 1], exec_pca[:, 2],
                      c='gray', label='Calibration', alpha=0.5, s=10)
            
            if operational_signals is not None:
                if np.any(normal_mask):
                    ax.scatter(operational_pca[normal_mask, 0], 
                              operational_pca[normal_mask, 1],
                              operational_pca[normal_mask, 2],
                              c='green', label='Normal', alpha=0.6, s=10)
                
                if np.any(anomaly_mask):
                    ax.scatter(operational_pca[anomaly_mask, 0],
                              operational_pca[anomaly_mask, 1],
                              operational_pca[anomaly_mask, 2],
                              c='red', label='Anomalies', alpha=0.8, s=25, edgecolors='k')
                            
            ax.set_xlabel(f'PC1\n({pc1_feature})', fontsize=10)
            ax.set_ylabel(f'PC2\n({pc2_feature})', fontsize=10)
            ax.set_zlabel(f'PC3\n({pc3_feature})', fontsize=10)
            ax.set_title('3D Projection of EM Signals', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.view_init(elev=25, azim=45)
            plt.tight_layout()
            
            if base_path and latest_op_timestamp:
                # Save to sca_images directory
                file_path = save_in_sca_images_dir(f'em_anomaly_3d_{campaign_number}_{latest_op_timestamp}.pdf', base_path)
                plt.savefig(file_path, format='pdf', bbox_inches='tight')
                print(f"3D visualization saved to: {file_path}")
            plt.show()
            
        except Exception as e:
            print(f"Error in 3D visualization: {e}")

    def cluster_anomalies(self, anomaly_features):
        """Cluster anomalies using HDBSCAN"""
        try:
            if len(anomaly_features) < 2:
                print("Not enough anomalies to cluster (need at least 2)")
                return np.array([]), np.array([]), np.array([])
            
            # Check if anomaly_features has more than 2 dimensions and reshape if needed
            anom_features = anomaly_features
            if anom_features.ndim > 2:
                anom_features = anom_features.reshape(-1, anom_features.shape[-1])
                
            features_scaled = self.scaler.transform(anom_features)
            features_pca = self.pca.transform(features_scaled)
            
            if len(anom_features) >= 5:  # Need reasonable number for clustering
                try:
                    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
                    hdbscan_labels = hdbscan_clusterer.fit_predict(features_pca)
                    
                    non_noise_mask = hdbscan_labels != -1
                    non_noise_features = features_pca[non_noise_mask] if np.any(non_noise_mask) else []
                    
                    optics_labels = np.full(len(features_pca), -1)
                    if len(non_noise_features) > 0:
                        from sklearn.cluster import OPTICS
                        optics_clusterer = OPTICS(
                            min_samples=2,
                            metric='euclidean',
                            n_jobs=-1,
                            cluster_method='xi'
                        )
                        try:
                            non_noise_optics_labels = optics_clusterer.fit_predict(non_noise_features)
                            optics_labels[non_noise_mask] = non_noise_optics_labels
                        except Exception as e:
                            print(f"OPTICS clustering failed: {e}")
                except Exception as e:
                    print(f"HDBSCAN clustering failed: {e}")
                    hdbscan_labels = np.zeros(len(features_pca))
                    optics_labels = np.zeros(len(features_pca))
            else:
                print(f"Only {len(anom_features)} anomalies found, not enough for meaningful clustering")
                hdbscan_labels = np.zeros(len(features_pca))
                optics_labels = np.zeros(len(features_pca))
                    
            return (hdbscan_labels, optics_labels, features_pca)
            
        except Exception as e:
            print(f"Error clustering anomalies: {e}")
            return np.array([]), np.array([]), np.array([])
    
def load_signals(file_path, is_calibration=False, sample_rate=2.5e9):
    """Load signals from calibration CSV or operational NPZ files"""
    try:
        if is_calibration:
            print(f"Loading calibration data from: {file_path}")
            df = pd.read_csv(file_path)
            
            # Enhanced feature mapping to handle different CSV formats
            feature_mapping = {
                'RMS': 'rms', 'rms': 'rms',
                'Mean': 'mean', 'mean': 'mean',
                'Std Dev': 'std_dev', 'std_dev': 'std_dev', 'Std_Dev': 'std_dev',
                'Crest': 'crest_factor', 'crest_factor': 'crest_factor', 'Crest_Factor': 'crest_factor',
                'P2P': 'peak_to_peak', 'peak_to_peak': 'peak_to_peak', 'Peak_to_Peak': 'peak_to_peak',
                'Entropy': 'entropy', 'entropy': 'entropy',
                'Peaks (Time)': 'peak_count', 'peak_count': 'peak_count', 'Peak_Count': 'peak_count',
                'Kurtosis': 'kurtosis', 'kurtosis': 'kurtosis',
                'Skewness': 'skewness', 'skewness': 'skewness',
                'ZCR': 'zcr', 'zcr': 'zcr',
                'E Low': 'energy_low', 'energy_low': 'energy_low', 'Energy_Low': 'energy_low',
                'E Mid': 'energy_mid', 'energy_mid': 'energy_mid', 'Energy_Mid': 'energy_mid',
                'E High': 'energy_high', 'energy_high': 'energy_high', 'Energy_High': 'energy_high',
                'Peak Freq': 'peak_freq', 'peak_freq': 'peak_freq', 'Peak_Freq': 'peak_freq',
                'Peak Mag (dBm)': 'peak_magnitude', 'peak_magnitude': 'peak_magnitude', 'Peak_Magnitude': 'peak_magnitude',
                'Spec Entropy': 'spectral_entropy', 'spectral_entropy': 'spectral_entropy', 'Spectral_Entropy': 'spectral_entropy',
                'Spec Flatness': 'spectral_flatness', 'spectral_flatness': 'spectral_flatness', 'Spectral_Flatness': 'spectral_flatness',
                'Spec Rolloff (MHz)': 'spectral_rolloff', 'spectral_rolloff': 'spectral_rolloff', 'Spectral_Rolloff': 'spectral_rolloff',
                'HNR': 'hnr', 'hnr': 'hnr',
                'THD': 'harmonic_distortion', 'harmonic_distortion': 'harmonic_distortion', 'Harmonic_Distortion': 'harmonic_distortion',
                'Spec Crest': 'spectral_crest', 'spectral_crest': 'spectral_crest', 'Spectral_Crest': 'spectral_crest'
            }
            
            # Convert all columns to numeric, handling errors gracefully
            for col in df.columns:
                if col not in ['ID', 'id', 'filename', 'Filename']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Extract features in the correct order
            feature_order = [
                'rms', 'mean', 'std_dev', 'crest_factor', 'peak_to_peak', 'entropy', 'peak_count',
                'kurtosis', 'skewness', 'zcr', 'energy_low', 'energy_mid', 'energy_high',
                'peak_freq', 'peak_magnitude', 'spectral_entropy', 'spectral_flatness',
                'spectral_rolloff', 'hnr', 'harmonic_distortion', 'spectral_crest'
            ]
            
            features = []
            for feature_name in feature_order:
                found = False
                for col in df.columns:
                    if col in feature_mapping and feature_mapping[col] == feature_name:
                        # Handle unit conversions
                        if feature_name == 'peak_freq' and 'MHz' in col:
                            # Convert from MHz to Hz
                            features.append(df[col].values * 1e6)
                        elif feature_name == 'spectral_rolloff' and 'MHz' in col:
                            # Convert from MHz to Hz
                            features.append(df[col].values * 1e6)
                        else:
                            features.append(df[col].values)
                        found = True
                        break
                
                if not found:
                    print(f"Warning: Feature {feature_name} not found in calibration data, using zeros")
                    features.append(np.zeros(len(df)))
            
            features = np.array(features).T
            print(f"Loaded {len(features)} calibration signals with {features.shape[1]} features")
            return features
            
        else:
            # Load operational signal from NPZ file
            print(f"Loading operational signal from: {file_path}")
            data = np.load(file_path)
            signal = data['signal']
            sample_rate = data.get('sample_rate', sample_rate)
            
            # Analyze the signal
            raw_stats, fft_freq, fft_magnitude = analyze_raw_signal(signal, sample_rate)
            
            # Apply windowing and compute FFT
            window = np.hanning(len(signal))
            fft_signal = np.fft.fft(signal * window)
            fft_magnitude = np.abs(fft_signal) / len(signal)
            fft_magnitude_dbm = convert_to_dbm(fft_magnitude)
            fft_params = analyze_fft(fft_freq, fft_magnitude_dbm)
            
            # Ensure peak_freq is in Hz (it should already be, but double-check)
            peak_freq = fft_params['peak_freq']
            
            # Extract features in the same order as calibration
            features = [
                raw_stats['rms'],
                raw_stats['mean'],
                raw_stats['std_dev'],
                raw_stats['crest_factor'],
                raw_stats['peak_to_peak'],
                raw_stats['entropy'],
                raw_stats['peak_count'],
                raw_stats['kurtosis'],
                raw_stats['skewness'],
                raw_stats['zcr'],
                raw_stats['energy_low'],
                raw_stats['energy_mid'],
                raw_stats['energy_high'],
                peak_freq,  # This should be in Hz
                fft_params['peak_magnitude'],
                fft_params['spectral_entropy'],
                fft_params['spectral_flatness'],
                fft_params['spectral_rolloff'],  # This should be in Hz
                fft_params['hnr'],
                fft_params['harmonic_distortion'],
                fft_params['spectral_crest']
            ]
            
            return np.array(features).reshape(1, -1)
            
    except Exception as e:
        print(f"Error loading signals from {file_path}: {e}")
        raise
    
def print_trace_analysis(detector, results_path, calibration_file):
    """Analyze all operational traces and detect anomalies"""
    try:
        operational_signals = []
        operational_results = []
        anomaly_signals = []
        anomaly_names = []
        anomaly_confidence = []
        total_count = 0
        
        print(f"\nAnalyzing operational signals from: {results_path}")
        print("=" * 80)

        # Get all signal files and sort by timestamp
        signal_files = []
        if not os.path.exists(results_path):
            print(f"Operation path does not exist: {results_path}")
            return None, [], None, [], [], None
            
        for name in os.listdir(results_path):
            if name.endswith('.npz'):
                # Extract timestamp from filename
                timestamp_match = re.search(r'(\d{8}_\d{6}_\d+)\.npz', name)
                if not timestamp_match:
                    # Try alternative timestamp formats
                    timestamp_match = re.search(r'(\d{14})\.npz', name)
                    if not timestamp_match:
                        timestamp_match = re.search(r'(\d+)\.npz', name)
                
                timestamp_str = timestamp_match.group(1) if timestamp_match else name
                signal_files.append((name, timestamp_str))
        
        if not signal_files:
            print(f"No .npz files found in {results_path}")
            return None, [], None, [], [], None
        
        # Sort by timestamp (second element of tuple)
        signal_files.sort(key=lambda x: x[1])
        
        print(f"Found {len(signal_files)} signal files to analyze")
        
        # Process files in timestamp order
        for name, _ in signal_files:
            signal_path = os.path.join(results_path, name)
            try:
                signal = load_signals(signal_path, is_calibration=False)
                result, reason, _, confidence = detector.predict(signal)
                
                operational_signals.append(signal)
                operational_results.append(result)
                
                print(f"\n{name}")
                print(f"{result} -> {reason}")
                        
                if result == "Anomaly":
                    anomaly_signals.append(signal)
                    anomaly_names.append(name)
                    anomaly_confidence.append(confidence)
                    
                total_count += 1
                
            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue
        
        print("\n" + "=" * 80)
        anomaly_count = len(anomaly_signals)
        print(f"\nResumen final:")
        print(f"- Amount of analyzed traces: {total_count}")
        print(f"- Detected anomalies: {anomaly_count} ({anomaly_count/total_count:.2%})")
        
        if anomaly_signals:
            print(f"\nConfidence stats:")
            print(f"- Mean: {np.mean(anomaly_confidence):.2f}")
            print(f"- Minimum: {min(anomaly_confidence):.2f}")
            print(f"- Maximum: {max(anomaly_confidence):.2f}")
        
        return (np.array(operational_signals) if operational_signals else None,
                operational_results,
                np.array(anomaly_signals) if anomaly_signals else None,
                anomaly_names,
                anomaly_confidence,
                None)  # exec_signals not needed here
                
    except Exception as e:
        print(f"Error in trace analysis: {e}")
        return None, [], None, [], [], None

def print_clustering_analysis(anomaly_names, hdbscan_labels, optics_labels, anomaly_signals, detector):
    """Analyze and print clustering results"""
    try:
        with open("real_anomalies.txt", "w") as file:
            print("\nHDBSCAN Clustering Results:")
            print("========================")
            for cluster in np.unique(hdbscan_labels):
                print(f"\nHDBSCAN CLUSTER {cluster}")
                print("=" * 50)
                
                cluster_mask = (hdbscan_labels == cluster)
                cluster_signals = anomaly_signals[cluster_mask]
                cluster_names = np.array(anomaly_names)[cluster_mask]

                print("\nCluster Members:")
                print("-" * 20)
                for name in cluster_names:
                    match = re.search(r'id:(\d+)', name)
                    anomaly_id = match.group(0) if match else name
                    print(f"  {anomaly_id}")
                    if cluster != -1:
                        file.write(f"{anomaly_id}\n")

                # Handle potential 3D arrays
                if cluster_signals.ndim > 2:
                    cluster_signals = cluster_signals.reshape(-1, cluster_signals.shape[-1])
                    
                # Now transform the signals
                try:
                    scaled_signals = detector.scaler.transform(cluster_signals)
                    pca_signals = detector.pca.transform(scaled_signals)
                    
                    distances_exec = [np.linalg.norm(signal - detector.thresholds['exec_centroid']) 
                                    for signal in pca_signals]
                    
                    print(f"\nCluster Statistics:")
                    print(f"- Number of traces: {len(cluster_signals)}")
                    print(f"- Mean distance to exec: {np.mean(distances_exec):.2f} ± {np.std(distances_exec):.2f}")

                    print("\nDominant Features:")
                    scaled_mean = np.mean(scaled_signals, axis=0)
                    top_features = sorted(enumerate(scaled_mean), key=lambda x: abs(x[1]), reverse=True)[:5]
                    for idx, value in top_features:
                        if idx < len(detector.feature_names):
                            print(f"  {detector.feature_names[idx]:20}: {value:.2f}")
                except Exception as e:
                    print(f"Error analyzing cluster {cluster}: {e}")
                    print("Unable to calculate statistics for this cluster.")
                    
    except Exception as e:
        print(f"Error in clustering analysis: {e}")

def save_anomaly_names(anomaly_names, base_path, campaign_number, timestamp, output_file=None):
    """Save anomaly names to file with campaign number and timestamp"""
    try:
        if output_file is None:
            output_file = f"anomaly_names_{campaign_number}_{timestamp}.txt"
        
        # Save to anomaly_names directory
        full_path = save_in_anomaly_names_dir(output_file, base_path)
        
        with open(full_path, 'w') as f:
            for name in anomaly_names:
                match = re.search(r'id:(\d+)', name)
                anomaly_id = match.group(0) if match else name
                f.write(f"{anomaly_id}\n")
        
        print(f"\nSaved {len(anomaly_names)} anomaly names to {full_path}")
        return full_path
        
    except Exception as e:
        print(f"Error saving anomaly names: {e}")
        return None

def main(firmadyne_id, host_path=None):
    """Main function for TRENTI anomaly detection"""
    try:
        global latest_op_timestamp
        global campaign_number
        
        print("TRENTI Side-Channel Analysis - Anomaly Detection")
        print("=" * 60)
                
        # Automatically find the latest directories from host path structure
        calibration_path, operation_path, latest_op_timestamp, base_path = find_latest_directories(host_path)
        
        print(f"Calibration path: {calibration_path}")
        print(f"Operation path: {operation_path}")
        print(f"Latest timestamp: {latest_op_timestamp}")
        print(f"Base path: {base_path}")
        
        # Get campaign number from the path structure
        campaign_number = get_campaign_number_from_path(base_path)
        print(f"Campaign number: {campaign_number}")
        
        # Find the calibration file
        csv_files = [f for f in os.listdir(calibration_path) 
                    if f.startswith('results_table_') and f.endswith('.csv')]
        if not csv_files:
            # Try alternative naming patterns
            csv_files = [f for f in os.listdir(calibration_path) if f.endswith('.csv')]
            
        if not csv_files:
            raise FileNotFoundError(f"No calibration CSV file found in {calibration_path}")
        
        calibration_file = os.path.join(calibration_path, csv_files[0])
        print(f"Using calibration file: {calibration_file}")
        
        # Initialize detector
        detector = SCADetectorWithTraceClustering(contamination=0.001, n_estimators=500)
        
        # Load calibration data and fit the model
        print("\nLoading calibration data...")
        exec_signals = load_signals(calibration_file, is_calibration=True)
        print("Fitting anomaly detection model...")
        detector.fit(exec_signals)
        
        # Analyze operational traces
        print("\nAnalyzing operational traces...")
        operational_signals, operational_results, anomaly_signals, anomaly_names, anomaly_confidence, _ = \
            print_trace_analysis(detector, operation_path, calibration_file)
        
        # Generate calibration visualization
        print("\nGenerating calibration visualization...")
        detector.visualize_calibration(base_path, latest_op_timestamp)
        
        # Count categories
        normal_count = operational_results.count('Normal')
        anomaly_count = operational_results.count('Anomaly')
        incompatible_count = operational_results.count('Incompatible')
        error_count = operational_results.count('Error')
        total_count = len(operational_results)
        
        print("\nSummary Statistics:")
        print(f"- Total traces analyzed: {total_count}")
        print(f"- Normal traces: {normal_count} ({normal_count/total_count:.2%} if total_count > 0 else 0)")
        print(f"- Anomalies: {anomaly_count} ({anomaly_count/total_count:.2%} if total_count > 0 else 0)")
        print(f"- Incompatible signals: {incompatible_count} ({incompatible_count/total_count:.2%} if total_count > 0 else 0)")
        print(f"- Errors: {error_count} ({error_count/total_count:.2%} if total_count > 0 else 0)")
        
        if anomaly_signals is not None and len(anomaly_signals) > 0:
            # Save anomaly names with campaign number and timestamp
            print("\nSaving anomaly names...")
            saved_file = save_anomaly_names(anomaly_names, base_path, campaign_number, latest_op_timestamp)
            
            # Generate visualizations and clustering analysis
            print("\nGenerating 3D visualization...")
            detector.visualize_3d(operational_signals=operational_signals, 
                                operational_results=operational_results,
                                base_path=base_path, 
                                latest_op_timestamp=latest_op_timestamp)
            
            print("\nPerforming clustering analysis...")
            hdbscan_labels, optics_labels, pca_signals = detector.cluster_anomalies(anomaly_signals)
            print_clustering_analysis(anomaly_names, hdbscan_labels, optics_labels, anomaly_signals, detector)
            
            print("\nGenerating cluster visualizations...")
            detector.visualize_clusters_only(anomaly_signals, hdbscan_labels, optics_labels, 
                                            base_path, latest_op_timestamp)
            detector.visualize(anomaly_signals, hdbscan_labels, 
                             operational_signals=operational_signals, 
                             operational_results=operational_results,
                             base_path=base_path, 
                             latest_op_timestamp=latest_op_timestamp)
        else:
            print("\nNo anomalies detected.")
            
        print("\nTRENTI analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("USAGE: python3 trenti.py <firmadyne_id> [<host_path>]")
        sys.exit(1)

    firmadyne_id = int(sys.argv[1])
    host_path    = sys.argv[2] if len(sys.argv) >= 3 else None

    main(firmadyne_id, host_path)
    # main(9050, '/home/atenea/trenti/evaluations/')