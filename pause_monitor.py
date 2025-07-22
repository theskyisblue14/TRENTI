import docker
import time
import re
import signal
import sys
import argparse
from datetime import datetime

DEFAULT_STATS_FILE = "test/image_9050/nuberu_outputs/fuzzer_stats"
DEFAULT_THRESHOLD = 4320  # seconds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-file", default=DEFAULT_STATS_FILE,
                      help="Path to AFL stats file")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                      help="Time threshold in seconds")
    return parser.parse_args()

def signal_handler(sig, frame):
    print("[+] Monitoring stopped")
    sys.exit(0)

def stop_fuzzing(container):
    """Stops AFL and QEMU cleanly"""
    print("\n[+] Threshold met! Stopping AFL and QEMU...")
    try:
        print("[+] Sending SIGUSR1 to afl-fuzz...")
        container.exec_run("pkill -USR1 -f afl-fuzz")
        time.sleep(5)  # Allow AFL to save progress
        print("[+] Killing all AFL and QEMU processes...")
        container.exec_run("pkill -9 -f afl-fuzz")
        container.exec_run("pkill -9 -f qemu-system-mipsel")
        print("[+] Fuzzing campaign stopped successfully.")
    except Exception as e:
        print(f"[!] Error stopping fuzzing: {e}")

def monitor_campaign(container, stats_file, threshold):
    print("[*] Monitoring fuzzing campaign...")
    
    # Variables for independent timer
    last_pending = float('inf')
    last_path_count = 0
    timer_start = time.time()  # Independent timer
    monitoring_start = time.time()
    
    # Variables for change detection
    last_stats_update = 0
    stats_file_stuck_time = None
    
    # Variables for synchronization with fuzzer_stats
    fuzzer_last_path_time = None
    is_synchronized = False
    
    while True:
        try:
            current_time = time.time()
            elapsed_monitoring = current_time - monitoring_start
            
            result = container.exec_run(f"cat {stats_file}")
            if result.exit_code != 0:
                print(f"[!] Error reading stats file")
                time.sleep(10)
                continue
                
            content = result.output.decode()
            stats = extract_stats(content)
            
            if stats is None:
                print("[!] Could not parse stats")
                time.sleep(10)
                continue
            
            # Detect if fuzzer_stats file is updating
            current_stats_update = stats['last_update']
            if current_stats_update != last_stats_update:
                # File was updated
                last_stats_update = current_stats_update
                stats_file_stuck_time = None
            else:
                # File hasn't been updated
                if stats_file_stuck_time is None:
                    stats_file_stuck_time = current_time
                
                stuck_duration = current_time - stats_file_stuck_time
                if stuck_duration > 300:  # 5 minutes without file update
                    print(f"[!] WARNING: fuzzer_stats hasn't updated for {stuck_duration:.0f}s")
            
            # Detect new paths and synchronize timer
            current_path_count = stats['paths_total']
            
            # Initial synchronization or when new paths are found
            if not is_synchronized and stats['last_path'] > 0:
                # Initial synchronization with fuzzer_stats
                fuzzer_last_path_time = stats['last_path']
                # Calculate elapsed time according to fuzzer_stats
                fuzzer_elapsed = stats['last_update'] - stats['last_path']
                # Adjust our timer to match fuzzer_stats
                timer_start = current_time - fuzzer_elapsed
                is_synchronized = True
                print(f"[+] SYNCHRONIZED with fuzzer_stats - last_path: {datetime.fromtimestamp(stats['last_path'])}")
                print(f"[+] Fuzzer shows {fuzzer_elapsed}s elapsed, adjusting independent timer to match")
            
            elif current_path_count > last_path_count:
                # New path found! Synchronize with fuzzer_stats
                print(f"[+] NEW PATH FOUND! Total paths: {current_path_count} (+{current_path_count - last_path_count})")
                
                if stats['last_path'] > 0:
                    # Use fuzzer_stats timestamp as reference
                    fuzzer_last_path_time = stats['last_path']
                    # Calculate elapsed time according to fuzzer_stats since last path
                    fuzzer_elapsed = stats['last_update'] - stats['last_path']
                    # Synchronize our timer with fuzzer_stats
                    timer_start = current_time - fuzzer_elapsed
                    
                    print(f"[+] Timer SYNCHRONIZED with fuzzer_stats")
                    print(f"[+] Fuzzer last_path: {datetime.fromtimestamp(stats['last_path'])}")
                    print(f"[+] Fuzzer shows {fuzzer_elapsed}s since last path")
                    print(f"[+] Independent timer adjusted to match fuzzer_stats value")
                else:
                    # Fallback if fuzzer_stats doesn't have valid timestamp
                    timer_start = current_time
                    print(f"[+] Timer reset to current time (fuzzer_stats timestamp not available)")
                
                last_path_count = current_path_count
            
            # Calculate time without new paths
            # Use independent timer but synchronized with fuzzer_stats
            time_without_paths = current_time - timer_start
            
            # Also calculate according to fuzzer_stats for comparison
            fuzzer_time_without_paths = 0
            if stats['last_path'] > 0:
                fuzzer_time_without_paths = stats['last_update'] - stats['last_path']
            
            # Display statistics with both timers
            display_stats_enhanced(stats, time_without_paths, fuzzer_time_without_paths, 
                                 elapsed_monitoring, stats_file_stuck_time, current_time, is_synchronized)
            
            # Check threshold using our synchronized independent timer
            if time_without_paths > threshold:
                print(f"\n[!] THRESHOLD MET - {time_without_paths/3600:.1f}h without new paths")
                print(f"[!] Independent timer (synchronized): {time_without_paths}s")
                print(f"[!] Fuzzer_stats timer: {fuzzer_time_without_paths}s")
                print(f"[!] Started monitoring at: {datetime.fromtimestamp(monitoring_start)}")
                if fuzzer_last_path_time:
                    print(f"[!] Last path found at: {datetime.fromtimestamp(fuzzer_last_path_time)} (from fuzzer_stats)")
                print(f"[!] Current time: {datetime.fromtimestamp(current_time)}")
                stop_fuzzing(container)
                return
            
            # Update pending for additional change detection
            last_pending = stats['pending_total']
            
        except Exception as e:
            print(f"[!] Error: {e}")
            
        time.sleep(60)

def extract_stats(content):
    stats = {}
    for key in ['paths_total', 'pending_total', 'last_path', 'last_update', 'unique_crashes', 'cycles_done']:
        match = re.search(rf'{key}\s*:\s*(\d+)', content)
        if not match:
            return None
        stats[key] = int(match.group(1))
    return stats

def display_stats_enhanced(stats, independent_time, fuzzer_time, elapsed_monitoring, stats_file_stuck_time, current_time, is_synchronized):
    """Display enhanced stats with synchronized independent timer"""
    
    # Time without paths (synchronized independent timer)
    days, rem = divmod(int(independent_time), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    independent_time_str = f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # Total monitoring time
    total_days, rem = divmod(int(elapsed_monitoring), 86400)
    total_hours, rem = divmod(rem, 3600)
    total_minutes, total_seconds = divmod(rem, 60)
    total_time_str = f"{total_days:02d}:{total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}"
    
    # Time according to fuzzer_stats (may be outdated)
    if stats['last_path'] == 0:
        fuzzer_time_str = "00:00:00:00 (no paths yet)"
    else:
        f_days, rem = divmod(int(fuzzer_time), 86400)
        f_hours, rem = divmod(rem, 3600)
        f_minutes, f_seconds = divmod(rem, 60)
        fuzzer_time_str = f"{f_days:02d}:{f_hours:02d}:{f_minutes:02d}:{f_seconds:02d}"
    
    sync_status = "SYNCHRONIZED" if is_synchronized else "NOT SYNC"
    
    print(f"\n{'='*80}")
    print(f"Stats [{datetime.fromtimestamp(stats['last_update'])}]:")
    print(f"{'='*80}")
    print(f"INDEPENDENT TIMER - Time without paths: {independent_time_str} [{sync_status}]")
    print(f"FUZZER STATS     - Time without paths: {fuzzer_time_str}")
    print(f"TOTAL MONITORING - Elapsed time:       {total_time_str}")
    print(f"-" * 80)
    print(f"Paths: {stats['paths_total']} | Pending: {stats['pending_total']}")
    print(f"Cycles: {stats['cycles_done']} | Crashes: {stats['unique_crashes']}")
    
    if stats_file_stuck_time:
        stuck_duration = current_time - stats_file_stuck_time
        print(f"WARNING: fuzzer_stats stuck for {stuck_duration:.0f}s")
    else:
        print("fuzzer_stats: UPDATING normally")
    
    print(f"{'='*80}")

def main():
    args = parse_args()
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"[*] Starting pause_monitor with synchronized independent timer")
    print(f"[*] Stats file: {args.stats_file}")
    print(f"[*] Threshold: {args.threshold}s ({args.threshold/3600:.1f}h)")
    print(f"[*] Monitor will synchronize with fuzzer_stats timestamps when paths are found")
    
    try:
        client = docker.from_env()
        container = client.containers.get('trenti')
        monitor_campaign(container, args.stats_file, args.threshold)
    except Exception as e:
        print(f"[!] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()