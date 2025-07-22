#!/usr/bin/env python3
"""
Anomaly Names Deduplicator for TRENTI Framework
Removes duplicate entries from anomaly_names files after analysis
"""

import os
import re
import glob
import argparse
from datetime import datetime
from collections import OrderedDict

def extract_file_id_from_signal_name(signal_name):
    """
    Extract file ID from signal name.
    
    Args:
        signal_name: Signal filename (e.g., "signal_123_20241201_143022.npz")
        
    Returns:
        int or None: File ID if found, None otherwise
    """
    # Pattern: signal_[file_id]_[timestamp].npz
    match = re.search(r'signal_(\d+)_', signal_name)
    if match:
        return int(match.group(1))
    return None

def parse_anomaly_names_file(file_path):
    """
    Parse anomaly_names file and extract entries with metadata.
    
    Args:
        file_path: Path to anomaly_names file
        
    Returns:
        list: List of tuples (line_number, signal_name, file_id, comment, full_line)
    """
    entries = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            original_line = line
            line = line.strip()
            
            # Skip empty lines and pure comments
            if not line or line.startswith('#'):
                continue
            
            # Extract signal name and comment
            if '#' in line:
                signal_part, comment_part = line.split('#', 1)
                signal_name = signal_part.strip()
                comment = comment_part.strip()
            else:
                signal_name = line
                comment = ""
            
            # Extract file ID
            file_id = extract_file_id_from_signal_name(signal_name)
            
            if file_id is not None:
                entries.append((line_num, signal_name, file_id, comment, original_line))
            else:
                print(f"[!] Warning: Could not extract file ID from line {line_num}: {signal_name}")
    
    except Exception as e:
        print(f"[!] Error reading {file_path}: {e}")
        return []
    
    return entries

def deduplicate_anomaly_entries(entries):
    """
    Remove duplicate entries based on file_id, keeping the most comprehensive entry.
    
    Args:
        entries: List of tuples (line_number, signal_name, file_id, comment, full_line)
        
    Returns:
        tuple: (deduplicated_entries, duplicate_info)
    """
    print("[*] Deduplicating anomaly entries...")
    
    # Group entries by file_id
    file_groups = {}
    for entry in entries:
        line_num, signal_name, file_id, comment, full_line = entry
        if file_id not in file_groups:
            file_groups[file_id] = []
        file_groups[file_id].append(entry)
    
    deduplicated = []
    duplicate_info = []
    
    for file_id, file_entries in file_groups.items():
        if len(file_entries) == 1:
            # No duplicates for this file_id
            deduplicated.append(file_entries[0])
        else:
            # Multiple entries for same file_id - choose the best one
            print(f"[*] Found {len(file_entries)} entries for file_id {file_id}")
            
            # Priority order for selection:
            # 1. Entries with most specific comments (SCA anomalies first)
            # 2. Entries with latest timestamp
            # 3. First entry if tie
            
            def get_priority_score(entry):
                line_num, signal_name, file_id, comment, full_line = entry
                score = 0
                
                # Higher priority for specific anomaly types
                if 'SCA' in comment.upper() or 'TRENTI' in comment.upper():
                    score += 100
                elif 'EXEC_TIME' in comment.upper():
                    score += 50
                elif 'HTTP' in comment.upper():
                    score += 30
                
                # Extract timestamp for tie-breaking
                timestamp_match = re.search(r'(\d{8}_\d{6})', signal_name)
                if timestamp_match:
                    timestamp = timestamp_match.group(1)
                    # Convert to number for comparison (higher = more recent)
                    try:
                        timestamp_num = int(timestamp.replace('_', ''))
                        score += timestamp_num / 1e12  # Normalize to small value
                    except:
                        pass
                
                return score
            
            # Sort by priority score (descending)
            sorted_entries = sorted(file_entries, key=get_priority_score, reverse=True)
            
            # Keep the highest priority entry
            selected_entry = sorted_entries[0]
            deduplicated.append(selected_entry)
            
            # Track duplicates
            for dup_entry in sorted_entries[1:]:
                duplicate_info.append({
                    'file_id': file_id,
                    'removed_line': dup_entry[0],
                    'removed_signal': dup_entry[1],
                    'removed_comment': dup_entry[3],
                    'kept_line': selected_entry[0],
                    'kept_signal': selected_entry[1],
                    'kept_comment': selected_entry[3]
                })
            
            print(f"[*] File_id {file_id}: Kept '{selected_entry[1]}' (line {selected_entry[0]}), "
                  f"removed {len(sorted_entries)-1} duplicates")
    
    print(f"[*] Deduplication summary:")
    print(f"    Original entries: {len(entries)}")
    print(f"    Unique file_ids: {len(file_groups)}")
    print(f"    Deduplicated entries: {len(deduplicated)}")
    print(f"    Removed duplicates: {len(duplicate_info)}")
    
    return deduplicated, duplicate_info

def save_deduplicated_file(original_file, deduplicated_entries, duplicate_info, output_dir):
    """
    Save deduplicated anomaly_names file and create backup.
    
    Args:
        original_file: Path to original anomaly_names file
        deduplicated_entries: List of deduplicated entries
        duplicate_info: Information about removed duplicates
        output_dir: Output directory for results
    """
    try:
        # Create backup
        backup_file = original_file + '.backup_before_dedup'
        with open(original_file, 'r') as src, open(backup_file, 'w') as dst:
            dst.write(src.read())
        print(f"[*] Backup created: {backup_file}")
        
        # Read original file to preserve header comments
        with open(original_file, 'r') as f:
            original_lines = f.readlines()
        
        # Find header comments (lines before first signal entry)
        header_lines = []
        for line in original_lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#'):
                # Check if this looks like a signal line
                if 'signal_' in line_stripped:
                    break
            header_lines.append(line)
        
        # Write deduplicated file
        with open(original_file, 'w') as f:
            # Write header
            f.writelines(header_lines)
            
            # Add deduplication info
            f.write(f"\n# File deduplicated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Removed {len(duplicate_info)} duplicate entries\n")
            f.write(f"# Total unique anomalies: {len(deduplicated_entries)}\n\n")
            
            # Write deduplicated entries, sorted by file_id
            sorted_entries = sorted(deduplicated_entries, key=lambda x: x[2])  # Sort by file_id
            for entry in sorted_entries:
                line_num, signal_name, file_id, comment, full_line = entry
                if comment:
                    f.write(f"{signal_name}  # {comment}\n")
                else:
                    f.write(f"{signal_name}\n")
        
        print(f"[+] Deduplicated file saved: {original_file}")
        
        # Save deduplication report
        report_file = os.path.join(output_dir, f"deduplication_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_file, 'w') as f:
            f.write("ANOMALY NAMES DEDUPLICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original file: {original_file}\n")
            f.write(f"Backup file: {backup_file}\n\n")
            
            f.write("SUMMARY:\n")
            f.write(f"Original entries: {len(deduplicated_entries) + len(duplicate_info)}\n")
            f.write(f"Deduplicated entries: {len(deduplicated_entries)}\n")
            f.write(f"Removed duplicates: {len(duplicate_info)}\n\n")
            
            if duplicate_info:
                f.write("REMOVED DUPLICATES:\n")
                for dup in duplicate_info:
                    f.write(f"File ID {dup['file_id']}:\n")
                    f.write(f"  Removed: {dup['removed_signal']} (line {dup['removed_line']}) - {dup['removed_comment']}\n")
                    f.write(f"  Kept:    {dup['kept_signal']} (line {dup['kept_line']}) - {dup['kept_comment']}\n\n")
            
            f.write("FINAL ENTRIES (sorted by file_id):\n")
            sorted_entries = sorted(deduplicated_entries, key=lambda x: x[2])
            for entry in sorted_entries:
                line_num, signal_name, file_id, comment, full_line = entry
                f.write(f"File ID {file_id}: {signal_name}")
                if comment:
                    f.write(f" - {comment}")
                f.write("\n")
        
        print(f"[*] Deduplication report saved: {report_file}")
        
    except Exception as e:
        print(f"[!] Error saving deduplicated file: {e}")

def process_anomaly_names_file(file_path, output_dir):
    """
    Process a single anomaly_names file for deduplication.
    
    Args:
        file_path: Path to anomaly_names file
        output_dir: Output directory for analysis results
    """
    print(f"\n[*] Processing: {file_path}")
    
    # Parse the file
    entries = parse_anomaly_names_file(file_path)
    
    if not entries:
        print("[!] No valid entries found in file")
        return
    
    print(f"[*] Found {len(entries)} valid anomaly entries")
    
    # Deduplicate
    deduplicated_entries, duplicate_info = deduplicate_anomaly_entries(entries)
    
    # Save results
    if len(duplicate_info) > 0:
        save_deduplicated_file(file_path, deduplicated_entries, duplicate_info, output_dir)
        print(f"[+] Deduplication completed - removed {len(duplicate_info)} duplicates")
    else:
        print("[*] No duplicates found - file already clean")

def main():
    parser = argparse.ArgumentParser(description='Anomaly Names Deduplicator for TRENTI Framework')
    parser.add_argument('firmadyne_id', help='Firmadyne ID (e.g., 9050)')
    parser.add_argument('--host-path', default='/home/atenea/trenti/evaluations', 
                       help='Base host path (default: /home/atenea/trenti/evaluations)')
    parser.add_argument('--output-dir', help='Output directory for deduplication reports')
    parser.add_argument('--file', help='Specific anomaly_names file to process')
    
    args = parser.parse_args()
    
    # Set up paths
    image_dir = f"{args.host_path}/image_{args.firmadyne_id}"
    anomaly_names_dir = f"{image_dir}/anomaly_names"
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{image_dir}/deduplication_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ANOMALY NAMES DEDUPLICATOR FOR TRENTI FRAMEWORK")
    print("=" * 60)
    print(f"Firmadyne ID: {args.firmadyne_id}")
    print(f"Anomaly names directory: {anomaly_names_dir}")
    print(f"Output directory: {output_dir}")
    
    if args.file:
        # Process specific file
        if os.path.exists(args.file):
            process_anomaly_names_file(args.file, output_dir)
        else:
            print(f"[!] File not found: {args.file}")
            return
    else:
        # Find and process all anomaly_names files
        anomaly_files = glob.glob(os.path.join(anomaly_names_dir, "anomaly_names_*.txt"))
        
        if not anomaly_files:
            print(f"[!] No anomaly_names files found in: {anomaly_names_dir}")
            return
        
        print(f"[*] Found {len(anomaly_files)} anomaly_names files")
        
        # Sort by modification time (newest first)
        anomaly_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        for file_path in anomaly_files:
            process_anomaly_names_file(file_path, output_dir)
    
    print("\n" + "=" * 60)
    print("ANOMALY NAMES DEDUPLICATION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()