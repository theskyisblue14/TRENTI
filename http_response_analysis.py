#!/usr/bin/env python3
"""
HTTP Response Analysis for TRENTI Framework
Catalogs non-200 OK responses as anomalies and analyzes response patterns
"""

import os
import re
import glob
import pandas as pd
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_http_responses(http_responses_dir):
    """
    Load HTTP responses from all files in the directory.
    
    Args:
        http_responses_dir: Path to HTTP responses directory
        
    Returns:
        pandas.DataFrame: DataFrame with response data
    """
    print(f"[*] Loading HTTP responses from: {http_responses_dir}")
    
    if not os.path.exists(http_responses_dir):
        print(f"[!] Directory not found: {http_responses_dir}")
        return pd.DataFrame()
    
    data = []
    files = glob.glob(os.path.join(http_responses_dir, "http_response_*.txt"))
    
    if not files:
        print(f"[!] No HTTP response files found in {http_responses_dir}")
        return pd.DataFrame()
    
    print(f"[*] Found {len(files)} HTTP response files")
    
    for file_path in files:
        try:
            # Extract file_id, execution_number, timestamp from filename
            filename = os.path.basename(file_path)
            # Pattern: http_response_[file_id]_[exec_num]_[timestamp].txt
            match = re.match(r'http_response_(\d+)_(\d+)_(\d{8}_\d{6}_\d+)\.txt', filename)
            
            if not match:
                print(f"[!] Skipping file with unexpected format: {filename}")
                continue
                
            file_id = int(match.group(1))
            execution_number = int(match.group(2))
            timestamp_str = match.group(3)
            
            # Read response content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Extract HTTP status
            http_status = "Unknown"
            status_match = re.search(r'HTTP Status:\s*(.+)', content)
            if status_match:
                http_status = status_match.group(1).strip()
            
            # Extract response code
            response_code = None
            code_match = re.search(r'HTTP/1\.[01]\s+(\d+)', http_status)
            if code_match:
                response_code = int(code_match.group(1))
            
            # Check for common error patterns in full response
            full_response_start = content.find("Full Response:")
            full_response = content[full_response_start:] if full_response_start != -1 else content
            
            # Analyze response characteristics
            response_length = len(full_response)
            has_timeout = 'timeout' in full_response.lower() or 'timed out' in full_response.lower()
            has_connection_error = any(error in full_response.lower() for error in [
                'connection refused', 'connection reset', 'connection failed',
                'network unreachable', 'host unreachable'
            ])
            has_curl_error = 'curl:' in full_response.lower() and 'error' in full_response.lower()
            
            # Determine if response is anomalous
            is_anomalous = (
                response_code != 200 or 
                has_timeout or 
                has_connection_error or 
                has_curl_error or
                http_status == "No HTTP response detected"
            )
            
            # Categorize anomaly type
            anomaly_type = "Normal"
            if is_anomalous:
                if response_code and response_code != 200:
                    anomaly_type = f"HTTP_{response_code}"
                elif has_timeout:
                    anomaly_type = "Timeout"
                elif has_connection_error:
                    anomaly_type = "Connection_Error"
                elif has_curl_error:
                    anomaly_type = "Curl_Error"
                elif http_status == "No HTTP response detected":
                    anomaly_type = "No_Response"
                else:
                    anomaly_type = "Other_Anomaly"
            
            data.append({
                'file_id': file_id,
                'execution_number': execution_number,
                'timestamp': timestamp_str,
                'http_status': http_status,
                'response_code': response_code,
                'response_length': response_length,
                'has_timeout': has_timeout,
                'has_connection_error': has_connection_error,
                'has_curl_error': has_curl_error,
                'is_anomalous': is_anomalous,
                'anomaly_type': anomaly_type,
                'filename': filename,
                'file_path': file_path
            })
            
        except Exception as e:
            print(f"[!] Error processing {file_path}: {e}")
            continue
    
    df = pd.DataFrame(data)
    print(f"[*] Loaded {len(df)} HTTP response records")
    
    return df

def analyze_http_responses(df, output_dir):
    """
    Analyze HTTP responses to identify patterns and anomalies.
    
    Args:
        df: DataFrame with HTTP response data
        output_dir: Directory to save analysis results
        
    Returns:
        list: List of anomalous file_ids
    """
    print("[*] Analyzing HTTP responses for anomalies...")
    
    if len(df) == 0:
        print("[!] No data to analyze")
        return []
    
    # Basic statistics
    total_responses = len(df)
    anomalous_responses = df['is_anomalous'].sum()
    anomaly_rate = (anomalous_responses / total_responses) * 100
    
    print(f"[*] HTTP Response Analysis:")
    print(f"    Total responses: {total_responses}")
    print(f"    Anomalous responses: {anomalous_responses}")
    print(f"    Anomaly rate: {anomaly_rate:.2f}%")
    
    # Response code distribution
    print(f"\n[*] Response Code Distribution:")
    response_code_counts = df['response_code'].value_counts(dropna=False)
    for code, count in response_code_counts.items():
        percentage = (count / total_responses) * 100
        status = "NORMAL" if code == 200 else "ANOMALOUS"
        print(f"    {code}: {count} ({percentage:.1f}%) - {status}")
    
    # Anomaly type distribution
    print(f"\n[*] Anomaly Type Distribution:")
    anomaly_type_counts = df[df['is_anomalous']]['anomaly_type'].value_counts()
    for anom_type, count in anomaly_type_counts.items():
        percentage = (count / anomalous_responses) * 100 if anomalous_responses > 0 else 0
        print(f"    {anom_type}: {count} ({percentage:.1f}%)")
    
    # Identify files with anomalous responses
    anomalous_files = df[df['is_anomalous']]['file_id'].unique().tolist()
    
    # File-level analysis
    file_analysis = df.groupby('file_id').agg({
        'is_anomalous': ['count', 'sum'],
        'response_code': 'first',
        'anomaly_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
    }).round(2)
    
    file_analysis.columns = ['total_executions', 'anomalous_executions', 'primary_response_code', 'primary_anomaly_type']
    file_analysis['anomaly_rate'] = (file_analysis['anomalous_executions'] / file_analysis['total_executions'] * 100).round(1)
    file_analysis = file_analysis.reset_index()
    
    # Files with consistent anomalies (>50% anomalous)
    consistent_anomalous_files = file_analysis[file_analysis['anomaly_rate'] > 50]['file_id'].tolist()
    
    print(f"\n[*] File Analysis:")
    print(f"    Files with any anomalous responses: {len(anomalous_files)}")
    print(f"    Files with >50% anomalous responses: {len(consistent_anomalous_files)}")
    
    # Create visualizations
    create_http_visualizations(df, file_analysis, output_dir)
    
    # Save detailed analysis
    save_http_analysis_results(df, file_analysis, anomalous_files, output_dir)
    
    print(f"\n[+] Anomalous file IDs: {sorted(anomalous_files)}")
    
    return sorted(anomalous_files)

def create_http_visualizations(df, file_analysis, output_dir):
    """Create visualization plots for HTTP response analysis."""
    print("[*] Creating HTTP response visualizations...")
    
    try:
        # Set up the plot style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('HTTP Response Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Response code distribution (pie chart)
        ax1 = axes[0, 0]
        response_code_counts = df['response_code'].value_counts(dropna=False)
        colors = ['green' if code == 200 else 'red' for code in response_code_counts.index]
        
        wedges, texts, autotexts = ax1.pie(response_code_counts.values, 
                                          labels=[f'HTTP {code}' if pd.notna(code) else 'No Response' 
                                                 for code in response_code_counts.index],
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('HTTP Response Code Distribution')
        
        # Plot 2: Anomaly type distribution (bar chart)
        ax2 = axes[0, 1]
        anomaly_df = df[df['is_anomalous']]
        if len(anomaly_df) > 0:
            anomaly_counts = anomaly_df['anomaly_type'].value_counts()
            bars = ax2.bar(range(len(anomaly_counts)), anomaly_counts.values, 
                          color='red', alpha=0.7)
            ax2.set_xticks(range(len(anomaly_counts)))
            ax2.set_xticklabels(anomaly_counts.index, rotation=45, ha='right')
            ax2.set_ylabel('Count')
            ax2.set_title('Anomaly Type Distribution')
            
            # Add value labels on bars
            for bar, value in zip(bars, anomaly_counts.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(value), ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No Anomalies Detected', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Anomaly Type Distribution')
        
        # Plot 3: File anomaly rate distribution
        ax3 = axes[1, 0]
        ax3.hist(file_analysis['anomaly_rate'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(file_analysis['anomaly_rate'].mean(), color='red', linestyle='--',
                   label=f'Mean: {file_analysis["anomaly_rate"].mean():.1f}%')
        ax3.set_xlabel('Anomaly Rate (%)')
        ax3.set_ylabel('Number of Files')
        ax3.set_title('Distribution of File Anomaly Rates')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Scatter plot of files by anomaly rate
        ax4 = axes[1, 1]
        normal_files = file_analysis[file_analysis['anomaly_rate'] <= 50]
        anomalous_files = file_analysis[file_analysis['anomaly_rate'] > 50]
        
        ax4.scatter(normal_files['file_id'], normal_files['anomaly_rate'],
                   alpha=0.6, color='blue', label='Normal Files', s=50)
        ax4.scatter(anomalous_files['file_id'], anomalous_files['anomaly_rate'],
                   alpha=0.8, color='red', label='Anomalous Files', s=80, marker='^')
        ax4.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% Threshold')
        ax4.set_xlabel('File ID')
        ax4.set_ylabel('Anomaly Rate (%)')
        ax4.set_title('File Anomaly Rates')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'http_response_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[*] HTTP response visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"[!] Error creating HTTP visualizations: {e}")

def save_http_analysis_results(df, file_analysis, anomalous_files, output_dir):
    """Save detailed HTTP response analysis results."""
    try:
        # Save summary statistics
        summary_path = os.path.join(output_dir, 'http_response_analysis_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("HTTP RESPONSE ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_responses = len(df)
            anomalous_responses = df['is_anomalous'].sum()
            
            f.write("STATISTICS:\n")
            f.write(f"Total responses analyzed: {total_responses}\n")
            f.write(f"Anomalous responses detected: {anomalous_responses}\n")
            f.write(f"Anomaly rate: {anomalous_responses/total_responses*100:.2f}%\n")
            f.write(f"Unique files with anomalies: {len(anomalous_files)}\n\n")
            
            f.write("RESPONSE CODE DISTRIBUTION:\n")
            response_code_counts = df['response_code'].value_counts(dropna=False)
            for code, count in response_code_counts.items():
                percentage = (count / total_responses) * 100
                status = "NORMAL" if code == 200 else "ANOMALOUS"
                f.write(f"HTTP {code}: {count} ({percentage:.1f}%) - {status}\n")
            
            f.write("\nANOMALY TYPE DISTRIBUTION:\n")
            if anomalous_responses > 0:
                anomaly_type_counts = df[df['is_anomalous']]['anomaly_type'].value_counts()
                for anom_type, count in anomaly_type_counts.items():
                    percentage = (count / anomalous_responses) * 100
                    f.write(f"{anom_type}: {count} ({percentage:.1f}%)\n")
            else:
                f.write("No anomalies detected.\n")
            
            f.write("\nANOMALOUS FILES:\n")
            anomalous_file_analysis = file_analysis[file_analysis['file_id'].isin(anomalous_files)]
            for _, row in anomalous_file_analysis.iterrows():
                f.write(f"File ID {row['file_id']}: {row['anomalous_executions']}/{row['total_executions']} "
                       f"anomalous ({row['anomaly_rate']:.1f}%) - Primary type: {row['primary_anomaly_type']}\n")
        
        # Save detailed results as CSV
        results_path = os.path.join(output_dir, 'http_response_analysis_results.csv')
        df.to_csv(results_path, index=False)
        
        # Save file-level analysis
        file_results_path = os.path.join(output_dir, 'http_response_file_analysis.csv')
        file_analysis.to_csv(file_results_path, index=False)
        
        print(f"[*] HTTP analysis summary saved to: {summary_path}")
        print(f"[*] Detailed results saved to: {results_path}")
        print(f"[*] File analysis saved to: {file_results_path}")
        
    except Exception as e:
        print(f"[!] Error saving HTTP analysis results: {e}")

def append_to_anomaly_names(anomalous_files, anomaly_names_file):
    """
    Append HTTP response anomalies to the anomaly_names file.
    
    Args:
        anomalous_files: List of anomalous file IDs
        anomaly_names_file: Path to anomaly_names file
    """
    if not anomalous_files:
        print("[*] No HTTP response anomalies to append")
        return
    
    print(f"[*] Appending {len(anomalous_files)} HTTP response anomalies to {anomaly_names_file}")
    
    try:
        # Create backup
        if os.path.exists(anomaly_names_file):
            backup_file = anomaly_names_file + '.backup_http_responses'
            with open(anomaly_names_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())
            print(f"[*] Backup created: {backup_file}")
        
        # Append HTTP response anomalies
        with open(anomaly_names_file, 'a') as f:
            f.write("\n# HTTP Response Anomalies (detected by http_response_analysis.py)\n")
            for file_id in sorted(anomalous_files):
                # Format: signal_[file_id]_timestamp.npz (matching TRENTI format)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                f.write(f"signal_{file_id}_{timestamp}.npz  # HTTP_RESPONSE_ANOMALY\n")
        
        print(f"[+] Successfully appended {len(anomalous_files)} HTTP response anomalies")
        
    except Exception as e:
        print(f"[!] Error appending to anomaly_names file: {e}")

def main():
    parser = argparse.ArgumentParser(description='HTTP Response Analysis for TRENTI Framework')
    parser.add_argument('firmadyne_id', help='Firmadyne ID (e.g., 9050)')
    parser.add_argument('--host-path', default='/home/atenea/trenti/evaluations', 
                       help='Base host path (default: /home/atenea/trenti/evaluations)')
    parser.add_argument('--output-dir', help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Set up paths
    image_dir = f"{args.host_path}/image_{args.firmadyne_id}"
    http_responses_dir = f"{image_dir}/HTTP_responses"
    anomaly_names_dir = f"{image_dir}/anomaly_names"
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{image_dir}/http_response_analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("HTTP RESPONSE ANALYSIS FOR TRENTI FRAMEWORK")
    print("=" * 60)
    print(f"Firmadyne ID: {args.firmadyne_id}")
    print(f"HTTP responses directory: {http_responses_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load HTTP response data
    df = load_http_responses(http_responses_dir)
    
    if df.empty:
        print("[!] No HTTP response data found. Exiting.")
        return
    
    # Perform HTTP response analysis
    anomalous_files = analyze_http_responses(df, output_dir)
    
    # Find the most recent anomaly_names file
    anomaly_files = glob.glob(os.path.join(anomaly_names_dir, "anomaly_names_*.txt"))
    if anomaly_files:
        # Get the most recent file
        latest_anomaly_file = max(anomaly_files, key=os.path.getctime)
        print(f"[*] Found anomaly_names file: {latest_anomaly_file}")
        
        # Append HTTP response anomalies
        append_to_anomaly_names(anomalous_files, latest_anomaly_file)
    else:
        print("[!] No anomaly_names file found. Cannot append HTTP response anomalies.")
        print(f"[*] Please ensure TRENTI analysis has been run and anomaly_names files exist in: {anomaly_names_dir}")
    
    print("\n" + "=" * 60)
    print("HTTP RESPONSE ANALYSIS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()