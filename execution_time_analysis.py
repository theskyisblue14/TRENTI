#!/usr/bin/env python3
"""
Execution Time Analysis for TRENTI Framework
Performs unsupervised clustering on execution times to detect anomalies
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse

def load_execution_times(exec_times_dir):
    """
    Load execution times from all files in the directory.
    
    Args:
        exec_times_dir: Path to execution times directory
        
    Returns:
        pandas.DataFrame: DataFrame with columns [file_id, execution_number, timestamp, exec_time_seconds]
    """
    print(f"[*] Loading execution times from: {exec_times_dir}")
    
    if not os.path.exists(exec_times_dir):
        print(f"[!] Directory not found: {exec_times_dir}")
        return pd.DataFrame()
    
    data = []
    files = glob.glob(os.path.join(exec_times_dir, "exec_time_*.txt"))
    
    if not files:
        print(f"[!] No execution time files found in {exec_times_dir}")
        return pd.DataFrame()
    
    print(f"[*] Found {len(files)} execution time files")
    
    for file_path in files:
        try:
            # Extract file_id, execution_number, timestamp from filename
            filename = os.path.basename(file_path)
            # Pattern: exec_time_[file_id]_[exec_num]_[timestamp].txt
            match = re.match(r'exec_time_(\d+)_(\d+)_(\d{8}_\d{6}_\d+)\.txt', filename)
            
            if not match:
                print(f"[!] Skipping file with unexpected format: {filename}")
                continue
                
            file_id = int(match.group(1))
            execution_number = int(match.group(2))
            timestamp_str = match.group(3)
            
            # Read execution time from file
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extract execution time in seconds
            time_match = re.search(r'Execution Time \(seconds\):\s*([\d.]+)', content)
            if time_match:
                exec_time = float(time_match.group(1))
                
                data.append({
                    'file_id': file_id,
                    'execution_number': execution_number,
                    'timestamp': timestamp_str,
                    'exec_time_seconds': exec_time,
                    'filename': filename,
                    'file_path': file_path
                })
            else:
                print(f"[!] Could not extract execution time from: {filename}")
                
        except Exception as e:
            print(f"[!] Error processing {file_path}: {e}")
            continue
    
    df = pd.DataFrame(data)
    print(f"[*] Loaded {len(df)} execution time records")
    
    return df

def perform_clustering_analysis(df, output_dir):
    """
    Perform multiple clustering analyses on execution times.
    
    Args:
        df: DataFrame with execution time data
        output_dir: Directory to save analysis results
        
    Returns:
        list: List of anomalous file_ids
    """
    print("[*] Performing clustering analysis on execution times...")
    
    if len(df) == 0:
        print("[!] No data to analyze")
        return []
    
    # Prepare features for clustering
    features = df[['exec_time_seconds']].copy()
    
    # Add statistical features
    file_stats = df.groupby('file_id')['exec_time_seconds'].agg([
        'mean', 'std', 'min', 'max', 'median'
    ]).reset_index()
    file_stats.columns = ['file_id', 'mean_time', 'std_time', 'min_time', 'max_time', 'median_time']
    file_stats['std_time'] = file_stats['std_time'].fillna(0)  # Handle single measurements
    
    print(f"[*] Analyzing {len(file_stats)} unique files")
    print(f"[*] Execution time statistics:")
    print(f"    Mean: {file_stats['mean_time'].mean():.4f}s")
    print(f"    Std:  {file_stats['mean_time'].std():.4f}s")
    print(f"    Min:  {file_stats['min_time'].min():.4f}s")
    print(f"    Max:  {file_stats['max_time'].max():.4f}s")
    
    anomalous_files = set()
    
    # Method 1: Statistical outlier detection (Z-score)
    print("\n[*] Method 1: Statistical outlier detection (Z-score > 3)")
    z_scores = np.abs(stats.zscore(file_stats['mean_time']))
    z_outliers = file_stats[z_scores > 3]['file_id'].tolist()
    anomalous_files.update(z_outliers)
    print(f"    Found {len(z_outliers)} Z-score outliers: {z_outliers}")
    
    # Method 2: Interquartile Range (IQR) method
    print("\n[*] Method 2: Interquartile Range (IQR) method")
    Q1 = file_stats['mean_time'].quantile(0.25)
    Q3 = file_stats['mean_time'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = file_stats[
        (file_stats['mean_time'] < lower_bound) | 
        (file_stats['mean_time'] > upper_bound)
    ]['file_id'].tolist()
    anomalous_files.update(iqr_outliers)
    print(f"    IQR bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"    Found {len(iqr_outliers)} IQR outliers: {iqr_outliers}")
    
    # Method 3: Isolation Forest
    print("\n[*] Method 3: Isolation Forest")
    if len(file_stats) >= 10:  # Need minimum samples for Isolation Forest
        features_for_isolation = file_stats[['mean_time', 'std_time', 'min_time', 'max_time']].copy()
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_for_isolation)
        
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        isolation_predictions = isolation_forest.fit_predict(features_scaled)
        
        isolation_outliers = file_stats[isolation_predictions == -1]['file_id'].tolist()
        anomalous_files.update(isolation_outliers)
        print(f"    Found {len(isolation_outliers)} Isolation Forest outliers: {isolation_outliers}")
    else:
        print("    Skipping Isolation Forest (insufficient data)")
    
    # Method 4: DBSCAN clustering
    print("\n[*] Method 4: DBSCAN clustering")
    if len(file_stats) >= 5:
        features_for_dbscan = file_stats[['mean_time']].copy()
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_for_dbscan)
        
        # Try different eps values
        eps_values = [0.3, 0.5, 0.8, 1.0]
        best_eps = 0.5
        best_outliers = []
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=2)
            clusters = dbscan.fit_predict(features_scaled)
            
            # Files in cluster -1 are considered noise/outliers
            outlier_mask = clusters == -1
            dbscan_outliers = file_stats[outlier_mask]['file_id'].tolist()
            
            if len(dbscan_outliers) > 0 and len(dbscan_outliers) < len(file_stats) * 0.5:
                best_outliers = dbscan_outliers
                best_eps = eps
                break
        
        anomalous_files.update(best_outliers)
        print(f"    Best eps: {best_eps}")
        print(f"    Found {len(best_outliers)} DBSCAN outliers: {best_outliers}")
    else:
        print("    Skipping DBSCAN (insufficient data)")
    
    # Create visualizations
    create_visualizations(file_stats, list(anomalous_files), output_dir)
    
    # Save detailed analysis
    save_analysis_results(file_stats, list(anomalous_files), output_dir)
    
    print(f"\n[+] Total unique anomalous files detected: {len(anomalous_files)}")
    print(f"[+] Anomalous file IDs: {sorted(list(anomalous_files))}")
    
    return sorted(list(anomalous_files))

def create_visualizations(file_stats, anomalous_files, output_dir):
    """Create visualization plots for the analysis."""
    print("[*] Creating visualizations...")
    
    try:
        # Set up the plot style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Execution Time Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Distribution of execution times
        ax1 = axes[0, 0]
        ax1.hist(file_stats['mean_time'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(file_stats['mean_time'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {file_stats["mean_time"].mean():.4f}s')
        ax1.axvline(file_stats['mean_time'].median(), color='green', linestyle='--', 
                   label=f'Median: {file_stats["mean_time"].median():.4f}s')
        ax1.set_xlabel('Mean Execution Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Mean Execution Times')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot with anomalies highlighted
        ax2 = axes[0, 1]
        normal_files = file_stats[~file_stats['file_id'].isin(anomalous_files)]
        anomaly_files = file_stats[file_stats['file_id'].isin(anomalous_files)]
        
        ax2.scatter(normal_files['file_id'], normal_files['mean_time'], 
                   alpha=0.6, color='blue', label='Normal', s=50)
        ax2.scatter(anomaly_files['file_id'], anomaly_files['mean_time'], 
                   alpha=0.8, color='red', label='Anomalous', s=80, marker='^')
        ax2.set_xlabel('File ID')
        ax2.set_ylabel('Mean Execution Time (seconds)')
        ax2.set_title('Execution Times by File ID')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Box plot
        ax3 = axes[1, 0]
        categories = ['Normal', 'Anomalous']
        normal_times = normal_files['mean_time'].tolist()
        anomaly_times = anomaly_files['mean_time'].tolist()
        
        data_to_plot = [normal_times, anomaly_times] if anomaly_times else [normal_times]
        labels_to_plot = categories[:len(data_to_plot)]
        
        bp = ax3.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax3.set_ylabel('Mean Execution Time (seconds)')
        ax3.set_title('Distribution Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Time series plot (if timestamps are available)
        ax4 = axes[1, 1]
        ax4.scatter(range(len(file_stats)), file_stats['mean_time'], 
                   c=['red' if fid in anomalous_files else 'blue' for fid in file_stats['file_id']], 
                   alpha=0.7)
        ax4.set_xlabel('File Order')
        ax4.set_ylabel('Mean Execution Time (seconds)')
        ax4.set_title('Execution Times Over File Order')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'execution_time_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[*] Visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"[!] Error creating visualizations: {e}")

def save_analysis_results(file_stats, anomalous_files, output_dir):
    """Save detailed analysis results."""
    try:
        # Save summary statistics
        summary_path = os.path.join(output_dir, 'execution_time_analysis_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("EXECUTION TIME ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("STATISTICS:\n")
            f.write(f"Total files analyzed: {len(file_stats)}\n")
            f.write(f"Anomalous files detected: {len(anomalous_files)}\n")
            f.write(f"Anomaly rate: {len(anomalous_files)/len(file_stats)*100:.2f}%\n\n")
            
            f.write("EXECUTION TIME STATISTICS:\n")
            f.write(f"Mean: {file_stats['mean_time'].mean():.6f} seconds\n")
            f.write(f"Median: {file_stats['mean_time'].median():.6f} seconds\n")
            f.write(f"Std Dev: {file_stats['mean_time'].std():.6f} seconds\n")
            f.write(f"Min: {file_stats['min_time'].min():.6f} seconds\n")
            f.write(f"Max: {file_stats['max_time'].max():.6f} seconds\n\n")
            
            f.write("ANOMALOUS FILES:\n")
            for file_id in sorted(anomalous_files):
                file_data = file_stats[file_stats['file_id'] == file_id].iloc[0]
                f.write(f"File ID {file_id}: {file_data['mean_time']:.6f}s "
                       f"(std: {file_data['std_time']:.6f}s)\n")
        
        # Save detailed results as CSV
        results_df = file_stats.copy()
        results_df['is_anomalous'] = results_df['file_id'].isin(anomalous_files)
        results_path = os.path.join(output_dir, 'execution_time_analysis_results.csv')
        results_df.to_csv(results_path, index=False)
        
        print(f"[*] Analysis summary saved to: {summary_path}")
        print(f"[*] Detailed results saved to: {results_path}")
        
    except Exception as e:
        print(f"[!] Error saving analysis results: {e}")

def append_to_anomaly_names(anomalous_files, anomaly_names_file):
    """
    Append execution time anomalies to the anomaly_names file.
    
    Args:
        anomalous_files: List of anomalous file IDs
        anomaly_names_file: Path to anomaly_names file
    """
    if not anomalous_files:
        print("[*] No execution time anomalies to append")
        return
    
    print(f"[*] Appending {len(anomalous_files)} execution time anomalies to {anomaly_names_file}")
    
    try:
        # Create backup
        if os.path.exists(anomaly_names_file):
            backup_file = anomaly_names_file + '.backup_exec_times'
            with open(anomaly_names_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())
            print(f"[*] Backup created: {backup_file}")
        
        # Append execution time anomalies
        with open(anomaly_names_file, 'a') as f:
            f.write("\n# Execution Time Anomalies (detected by execution_time_analysis.py)\n")
            for file_id in sorted(anomalous_files):
                # Format: signal_[file_id]_timestamp.npz (matching TRENTI format)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                f.write(f"signal_{file_id}_{timestamp}.npz  # EXEC_TIME_ANOMALY\n")
        
        print(f"[+] Successfully appended {len(anomalous_files)} execution time anomalies")
        
    except Exception as e:
        print(f"[!] Error appending to anomaly_names file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Execution Time Analysis for TRENTI Framework')
    parser.add_argument('firmadyne_id', help='Firmadyne ID (e.g., 9050)')
    parser.add_argument('--host-path', default='/home/atenea/trenti/evaluations', 
                       help='Base host path (default: /home/atenea/trenti/evaluations)')
    parser.add_argument('--output-dir', help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Set up paths
    image_dir = f"{args.host_path}/image_{args.firmadyne_id}"
    exec_times_dir = f"{image_dir}/exec_times"
    anomaly_names_dir = f"{image_dir}/anomaly_names"
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{image_dir}/execution_time_analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("EXECUTION TIME ANALYSIS FOR TRENTI FRAMEWORK")
    print("=" * 60)
    print(f"Firmadyne ID: {args.firmadyne_id}")
    print(f"Execution times directory: {exec_times_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load execution time data
    df = load_execution_times(exec_times_dir)
    
    if df.empty:
        print("[!] No execution time data found. Exiting.")
        return
    
    # Perform clustering analysis
    anomalous_files = perform_clustering_analysis(df, output_dir)
    
    # Find the most recent anomaly_names file
    anomaly_files = glob.glob(os.path.join(anomaly_names_dir, "anomaly_names_*.txt"))
    if anomaly_files:
        # Get the most recent file
        latest_anomaly_file = max(anomaly_files, key=os.path.getctime)
        print(f"[*] Found anomaly_names file: {latest_anomaly_file}")
        
        # Append execution time anomalies
        append_to_anomaly_names(anomalous_files, latest_anomaly_file)
    else:
        print("[!] No anomaly_names file found. Cannot append execution time anomalies.")
        print(f"[*] Please ensure TRENTI analysis has been run and anomaly_names files exist in: {anomaly_names_dir}")
    
    print("\n" + "=" * 60)
    print("EXECUTION TIME ANALYSIS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()