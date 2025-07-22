#!/usr/bin/env python3
"""
TRENTI - Fuzzing Automation Tool
Script to automate the complete fuzzing process with anomaly detection
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import subprocess
import time
import os, sys
import re
import json
from datetime import datetime, timedelta
import signal
import psutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk

import subprocess, tempfile, os, time, re, shutil, textwrap

import pandas as pd

def copy_httpd_specific_files():
    """Copy httpd-specific files from host image_161161 to container image_161161"""
    container_name = "trenti"
    host_httpd_path = "images/image_161161"
    container_target_path = "/test/image_161161"
    
    print("=" * 60)
    print("COPYING HTTPD-SPECIFIC FILES (inputs and user.sh)")
    print("=" * 60)
    
    # Check if host httpd directory exists
    if not os.path.exists(host_httpd_path):
        print(f"CRITICAL ERROR: Host httpd directory {host_httpd_path} does not exist")
        return False
    
    print(f" Host httpd directory {host_httpd_path} found")
    
    # Files and directories to copy
    items_to_copy = ['inputs', 'user.sh']
    
    success = True
    
    for item in items_to_copy:
        host_item_path = os.path.join(host_httpd_path, item)
        
        # Check if item exists on host
        if not os.path.exists(host_item_path):
            print(f"WARNING: {item} not found in host httpd directory: {host_item_path}")
            continue
        
        try:
            # Copy item from host to container
            copy_command = ['docker', 'cp', host_item_path, f'{container_name}:{container_target_path}/{item}']
            
            print(f"Copying {item} from host to container...")
            print(f"Command: {' '.join(copy_command)}")
            
            result = subprocess.run(copy_command, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f" Successfully copied {item}")
                
                # Verify the copy
                if item == 'inputs':
                    verify_cmd = ['docker', 'exec', container_name, 'test', '-d', f'{container_target_path}/{item}']
                else:  # user.sh
                    verify_cmd = ['docker', 'exec', container_name, 'test', '-f', f'{container_target_path}/{item}']
                
                verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=10)
                
                if verify_result.returncode == 0:
                    print(f" Verified {item} exists in container")
                    
                    # Show contents for verification
                    if item == 'inputs':
                        list_cmd = ['docker', 'exec', container_name, 'ls', '-la', f'{container_target_path}/{item}/']
                        list_result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=10)
                        if list_result.returncode == 0:
                            file_count = len([line for line in list_result.stdout.split('\n') 
                                            if line.strip() and not line.startswith('total')])
                            print(f"  inputs directory contains {file_count} items")
                    else:  # user.sh
                        size_cmd = ['docker', 'exec', container_name, 'ls', '-l', f'{container_target_path}/{item}']
                        size_result = subprocess.run(size_cmd, capture_output=True, text=True, timeout=10)
                        if size_result.returncode == 0:
                            print(f"  user.sh file details: {size_result.stdout.strip()}")
                else:
                    print(f" Copy verification failed for {item}")
                    success = False
            else:
                print(f" Failed to copy {item}: {result.stderr}")
                success = False
                
        except subprocess.TimeoutExpired:
            print(f" Timeout copying {item}")
            success = False
        except Exception as e:
            print(f" Error copying {item}: {e}")
            success = False
    
    if success:
        print("=" * 60)
        print("HTTPD-SPECIFIC FILES COPY COMPLETED SUCCESSFULLY")
        print("=" * 60)
    else:
        print("=" * 60)
        print("HTTPD-SPECIFIC FILES COPY COMPLETED WITH ERRORS")
        print("=" * 60)
    
    return success


# Docker helper functions for miniupnpd setup
def copy_image_to_docker(firmadyne_id="161160"):
    """Copy image directory from host to Docker container - FIXED for 161161"""
    
    container_name = "trenti"
    
    # FIXED: Para 161161, copiar 161160 primero (como en tu manual)
    if firmadyne_id == "161161":
        # EXACTAMENTE como tu manual: docker cp images/image_161160 trenti:/test/
        source_host_path = "images/image_161160"
        target_container_path = "/test/image_161160"
        print("SPECIAL 161161: Copying image_161160 as base (exactly like manual)")
    else:
        source_host_path = f"images/image_{firmadyne_id}"
        target_container_path = f"/test/image_{firmadyne_id}"
    
    print("=" * 60)
    print(f"COPYING: {source_host_path} -> {target_container_path}")
    print("=" * 60)
    
    # Check source exists
    if not os.path.exists(source_host_path):
        print(f"ERROR: Source {source_host_path} does not exist")
        return False
    
    # Check container is running
    try:
        check_cmd = ['docker', 'inspect', '-f', '{{.State.Running}}', container_name]
        result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0 or result.stdout.strip() != 'true':
            print(f"ERROR: Container {container_name} not running")
            return False
    except:
        print("ERROR: Cannot check container status")
        return False
    
    # Execute copy - EXACTLY like your manual command
    copy_command = ['docker', 'cp', source_host_path, f'{container_name}:/test/']
    
    print(f"EXECUTING: {' '.join(copy_command)}")
    
    try:
        result = subprocess.run(copy_command, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print(f" Copy completed successfully")
            
            # Quick verification
            verify_cmd = ['docker', 'exec', container_name, 'ls', '-la', target_container_path]
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30)
            
            if verify_result.returncode == 0:
                print(" Directory verified in container")
                return True
            else:
                print("ERROR: Directory not found after copy")
                return False
        else:
            print(f"ERROR: Copy failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: Copy timeout")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def copy_qemu_files_within_container(firmadyne_id="161160"):
    """Copy QEMU files from image_10853 to target image within the container - GENERIC VERSION"""
    container_name = "trenti"
    source_dir = "/test/image_10853"
    dest_dir = f"/test/image_{firmadyne_id}"
    
    # Files to copy
    qemu_files = [
        "qemu-system-mips",
        "qemu-system-mips-full", 
        "afl-qemu-trace"
    ]
    
    experiment_names = {
        "161160": "miniupnpd",
        "161161": "httpd"
    }
    experiment_name = experiment_names.get(firmadyne_id, f"image_{firmadyne_id}")
    
    print(f"Starting QEMU files copy within container for {experiment_name}...")
    
    # First verify source directory exists
    try:
        check_source_cmd = f"docker exec {container_name} ls -la {source_dir}"
        check_result = subprocess.run(check_source_cmd, shell=True, 
                                    capture_output=True, text=True)
        
        if check_result.returncode != 0:
            print(f"Error: Source directory {source_dir} not found in container")
            return False
            
        print(f"Source directory {source_dir} verified")
        
    except subprocess.CalledProcessError as e:
        print(f"Error checking source directory: {e}")
        return False
    
    success = True
    
    for file_name in qemu_files:
        source_path = f"{source_dir}/{file_name}"
        dest_path = f"{dest_dir}/{file_name}"
        
        try:
            # First check if source file exists
            check_cmd = f"docker exec {container_name} test -f {source_path}"
            check_result = subprocess.run(check_cmd, shell=True, capture_output=True)
            
            if check_result.returncode != 0:
                print(f"Warning: Source file {source_path} not found, skipping...")
                continue
            
            # Copy file within container using docker exec
            copy_command = f"docker exec {container_name} cp {source_path} {dest_path}"
            print(f"Copying {source_path} to {dest_path} within container...")
            
            result = subprocess.run(copy_command, shell=True, check=True, 
                                  capture_output=True, text=True)
            
            # Verify the copy
            verify_cmd = f"docker exec {container_name} test -f {dest_path}"
            verify_result = subprocess.run(verify_cmd, shell=True, capture_output=True)
            
            if verify_result.returncode == 0:
                print(f"Successfully copied and verified {file_name}")
            else:
                print(f"Copy failed for {file_name} - file not found after copy")
                success = False
            
        except subprocess.CalledProcessError as e:
            print(f"Error copying {file_name}: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
            success = False
    
    return success

def setup_special_environment(firmadyne_id):
    """Setup for 161161 EXACTLY like manual commands"""
    
    if firmadyne_id != "161161":
        # Para otros casos, usar la lógica original
        print(f"Using standard setup for {firmadyne_id}")
        return copy_image_to_docker(firmadyne_id) and copy_qemu_files_within_container(firmadyne_id)
    
    # SOLO para 161161: replicar comandos manuales EXACTOS
    print("=" * 70)
    print("161161 MANUAL REPLICATION")
    print("=" * 70)
    
    container_name = "trenti"
    
    # MANUAL STEP 1: docker cp images/image_161160 trenti:/test/
    print("MANUAL STEP 1: docker cp images/image_161160 trenti:/test/")
    if not copy_image_to_docker("161160"):  # Esto copia 161160 como base
        print("ERROR: Failed to copy base image_161160")
        return False
    
    # MANUAL STEP 2: docker cp images/image_161161/inputs trenti:/test/image_161160/
    print("MANUAL STEP 2: docker cp images/image_161161/inputs trenti:/test/image_161160/")
    inputs_cmd = ['docker', 'cp', 'images/image_161161/inputs', f'{container_name}:/test/image_161160/']
    inputs_result = subprocess.run(inputs_cmd, capture_output=True, text=True)
    
    if inputs_result.returncode != 0:
        print(f"ERROR copying inputs: {inputs_result.stderr}")
        return False
    print(" inputs copied")
    
    # MANUAL STEP 3: docker cp images/image_161161/user.sh trenti:/test/image_161160/
    print("MANUAL STEP 3: docker cp images/image_161161/user.sh trenti:/test/image_161160/")
    user_sh_cmd = ['docker', 'cp', 'images/image_161161/user.sh', f'{container_name}:/test/image_161160/']
    user_sh_result = subprocess.run(user_sh_cmd, capture_output=True, text=True)
    
    if user_sh_result.returncode != 0:
        print(f"ERROR copying user.sh: {user_sh_result.stderr}")
        return False
    print(" user.sh copied")
    
    # Copy QEMU files
    print("STEP 4: Copying QEMU files...")
    if not copy_qemu_files_within_container("161160"):
        print("WARNING: QEMU files copy had issues")
    
    # Final verification
    verify_cmd = ['docker', 'exec', container_name, 'ls', '-la', '/test/image_161160/']
    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
    
    if verify_result.returncode == 0:
        print(" Final verification passed")
        print("Files in container:")
        for line in verify_result.stdout.split('\n')[:10]:  # First 10 lines
            if line.strip():
                print(f"  {line}")
    else:
        print("ERROR: Final verification failed")
        return False
    
    print("=" * 70)
    print("161161 MANUAL REPLICATION COMPLETED")
    print("=" * 70)
    return True

class CopyResumeManager:
    """
    Copy anomalous queue entries from anomaly_names directory
    FIXED VERSION for firmadyne_id 161161
    """

    def __init__(self, container: str, firmadyne_id: str, host_base: str):
        self.container     = container               # e.g. trenti
        self.fid           = str(firmadyne_id)       # e.g. 161161
        self.host_base     = host_base.rstrip("/")   # evaluations folder on host
        
        # FIXED: Correct path handling for 161161
        if self.fid == "161161":
            # Para 161161, trabajar en image_161160 (como en el setup manual)
            self.image_root = "/test/image_161160"
            self.queue_dir = f"{self.image_root}/trenti_sca_outputs_161161_*/queue"
            self.anom_dir = f"{self.image_root}/anomalous_queue_entries"
            self.inputs_dir = f"{self.image_root}/inputs"
        else:
            # Para otros casos, usar la ruta normal
            self.image_root = f"/test/image_{self.fid}"
            self.queue_dir = f"{self.image_root}/trenti_sca_outputs_{self.fid}_*/queue"
            self.anom_dir = f"{self.image_root}/anomalous_queue_entries"
            self.inputs_dir = f"{self.image_root}/inputs"

    def _run(self, *cmd, check=True, capture=False):
        res = subprocess.run(cmd, text=True, capture_output=capture, check=check)
        return res.stdout.strip() if capture else None

    def _container_exec(self, *inner_cmd, check=True):
        self._run("docker", "exec", self.container, *inner_cmd, check=check)

    def _ensure_container_running(self):
        try:
            running = self._run("docker", "inspect", "-f",
                                "{{.State.Running}}", self.container,
                                capture=True, check=True)
            if running != "true":
                print(f"[CopyResume] Starting container {self.container}...")
                self._run("docker", "start", self.container)
        except subprocess.CalledProcessError:
            print(f"[CopyResume] Container {self.container} not found or cannot be started")
            raise

    def _detect_python_in_container(self):
        """Detect available Python interpreter in container"""
        python_options = ['python3', 'python', '/usr/bin/python3', '/usr/bin/python', 
                         '/usr/local/bin/python3', '/usr/local/bin/python']
        
        for python_cmd in python_options:
            try:
                test_cmd = ['docker', 'exec', self.container, python_cmd, '--version']
                result = subprocess.run(test_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"[CopyResume] Found Python: {python_cmd}")
                    return python_cmd
            except:
                continue
                
        print("[CopyResume] No Python interpreter found in container")
        return None

    def _find_actual_queue_directory(self):
        """Find the actual queue directory in container - FIXED VERSION for 161161"""
        try:
            print(f"[CopyResume] Searching for queue directories in container...")
            print(f"[CopyResume] Working in image root: {self.image_root}")
            
            # First, check what directories exist in the image root
            ls_cmd = ['docker', 'exec', self.container, 'ls', '-la', self.image_root]
            ls_result = subprocess.run(ls_cmd, capture_output=True, text=True)
            
            if ls_result.returncode == 0:
                print(f"[CopyResume] Contents of {self.image_root}:")
                for line in ls_result.stdout.split('\n'):
                    if line.strip():
                        print(f"  {line}")
            else:
                print(f"[CopyResume] Could not list {self.image_root}: {ls_result.stderr}")
            
            # FIXED: Search patterns adapted for correct firmadyne_id in output names
            if self.fid == "161161":
                # Para 161161, buscar outputs con 161161 en el nombre pero en directorio 161160
                search_patterns = [
                    f"{self.image_root}/trenti_sca_outputs_161161_*/queue",
                    f"{self.image_root}/trenti_sca_outputs_*/queue", 
                    f"{self.image_root}/*/queue"
                ]
            else:
                # Para otros casos, buscar normalmente
                search_patterns = [
                    f"{self.image_root}/trenti_sca_outputs_{self.fid}_*/queue",
                    f"{self.image_root}/trenti_sca_outputs_*/queue", 
                    f"{self.image_root}/aflpp_outputs/queue",
                    f"{self.image_root}/afl_outputs/queue",
                    f"{self.image_root}/*/queue",
                    f"{self.image_root}/queue"
                ]
            
            # Try each search pattern
            for pattern in search_patterns:
                print(f"[CopyResume] Trying pattern: {pattern}")
                
                # Use shell expansion to find directories matching pattern
                find_cmd = ['docker', 'exec', self.container, 'bash', '-c', 
                           f"ls -d {pattern} 2>/dev/null || true"]
                result = subprocess.run(find_cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    queue_dirs = [d.strip() for d in result.stdout.strip().split('\n') if d.strip()]
                    
                    for queue_dir in queue_dirs:
                        # Verify it's actually a directory with files
                        verify_cmd = ['docker', 'exec', self.container, 'test', '-d', queue_dir]
                        verify_result = subprocess.run(verify_cmd, capture_output=True)
                        
                        if verify_result.returncode == 0:
                            # Check if it contains files
                            count_cmd = ['docker', 'exec', self.container, 'bash', '-c',
                                       f"ls -1 {queue_dir} 2>/dev/null | wc -l"]
                            count_result = subprocess.run(count_cmd, capture_output=True, text=True)
                            
                            if count_result.returncode == 0:
                                file_count = int(count_result.stdout.strip() or 0)
                                print(f"[CopyResume] Found queue directory: {queue_dir} ({file_count} files)")
                                
                                if file_count > 0:
                                    return queue_dir
                                else:
                                    print(f"[CopyResume] Queue directory {queue_dir} is empty, continuing search...")
            
            # If no queue found, try to find any output directories
            print("[CopyResume] No existing queue directories found. Checking if fuzzing was run...")
            
            # Look for any output directories that might contain queue
            find_any_cmd = ['docker', 'exec', self.container, 'find', self.image_root, 
                           '-name', 'queue', '-type', 'd', '2>/dev/null']
            find_result = subprocess.run(find_any_cmd, capture_output=True, text=True)
            
            if find_result.returncode == 0 and find_result.stdout.strip():
                queue_dirs = find_result.stdout.strip().split('\n')
                for queue_dir in queue_dirs:
                    print(f"[CopyResume] Found queue with find command: {queue_dir}")
                    return queue_dir
            
            # Last resort: look for any fuzzing-related directories
            print("[CopyResume] Searching for any fuzzing-related directories...")
            search_fuzz_cmd = ['docker', 'exec', self.container, 'bash', '-c',
                              f"find {self.image_root} -type d -name '*queue*' -o -name '*afl*' -o -name '*fuzz*' -o -name '*trenti*' 2>/dev/null || true"]
            
            search_result = subprocess.run(search_fuzz_cmd, capture_output=True, text=True)
            if search_result.returncode == 0 and search_result.stdout.strip():
                print("[CopyResume] Found fuzzing-related directories:")
                for line in search_result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"  {line}")
            
            print("[CopyResume] No queue directories found. This could mean:")
            print("  1. Fuzzing hasn't been run yet")
            print("  2. Fuzzing failed to start properly") 
            print("  3. Different directory structure than expected")
            print(f"  4. Queue directory is in a different location in {self.image_root}")
            
            return None
            
        except Exception as e:
            print(f"[CopyResume] Error finding queue directory: {e}")
            import traceback
            traceback.print_exc()
            return None

    def copy_and_resume(self, anomaly_info):
        """
        Copy and resume using anomaly_info dict with timestamp, campaign, filename, full_path
        FIXED VERSION for 161161 paths
        """
        try:
            self._ensure_container_running()

            timestamp = anomaly_info['timestamp']
            campaign = anomaly_info['campaign'] 
            filename = anomaly_info['filename']
            anom_file_host = anomaly_info['full_path']

            print(f"[CopyResume] Processing {filename} for campaign {campaign}")
            print(f"[CopyResume] Working in image root: {self.image_root}")
            print(f"[CopyResume] Target firmadyne_id: {self.fid}")
            
            # 1. Verify anomaly file exists on host
            if not os.path.exists(anom_file_host):
                print(f"[CopyResume] ERROR: Anomaly file not found: {anom_file_host}")
                return False

            # Check file content to make sure it's not empty
            try:
                with open(anom_file_host, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        print(f"[CopyResume] WARNING: Anomaly file is empty: {anom_file_host}")
                        return True  # Not an error, just no anomalies
                    
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    print(f"[CopyResume] Anomaly file contains {len(lines)} entries")
                    
            except Exception as e:
                print(f"[CopyResume] ERROR reading anomaly file: {e}")
                return False

            # 2. Copy anomaly file to container
            anom_txt_cont = f"{self.image_root}/{filename}"
            print(f"[CopyResume] Copying {anom_file_host} to container:{anom_txt_cont}")
            
            copy_result = subprocess.run(['docker', 'cp', anom_file_host,
                                        f"{self.container}:{anom_txt_cont}"], 
                                       capture_output=True, text=True)
            
            if copy_result.returncode != 0:
                print(f"[CopyResume] ERROR copying file: {copy_result.stderr}")
                return False

            # 3. Find actual queue directory
            actual_queue_dir = self._find_actual_queue_directory()
            if not actual_queue_dir:
                print("[CopyResume] Cannot find queue directory.")
                print("[CopyResume] This might be expected if fuzzing hasn't run yet.")
                print("[CopyResume] Skipping copy operation but this is not considered an error.")
                
                # Clean up copied file
                try:
                    self._container_exec("rm", "-f", anom_txt_cont)
                except:
                    pass
                    
                return True  # Not a failure - just no queue to copy from yet

            # 4. Detect Python interpreter (with fallback to shell)
            python_cmd = self._detect_python_in_container()
            
            if python_cmd:
                success = self._use_python_approach(timestamp, anom_txt_cont, actual_queue_dir, python_cmd)
            else:
                print("[CopyResume] No Python found, using shell script approach...")
                success = self._use_shell_approach(timestamp, anom_txt_cont, actual_queue_dir)
                
            # Clean up anomaly file from container
            try:
                self._container_exec("rm", "-f", anom_txt_cont)
            except:
                pass
                
            return success

        except Exception as e:
            print(f"[CopyResume] FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _use_python_approach(self, timestamp, anom_txt_cont, actual_queue_dir, python_cmd):
        """Use Python script to copy anomalous queue entries - FIXED FOR SIGNAL FILES"""
        try:
            import textwrap
            
            helper_src = textwrap.dedent(f"""\
                #!/usr/bin/env python
                from __future__ import print_function
                import os, re, shutil, sys
            
                qdir  = "{actual_queue_dir}"
                dest  = "{self.anom_dir}"
                txt   = "{anom_txt_cont}"
            
                print("[CopyHelper] Starting copy operation...")
                print("[CopyHelper] Queue dir: " + qdir)
                print("[CopyHelper] Dest dir: " + dest)
                print("[CopyHelper] Anomaly file: " + txt)
                print("[CopyHelper] Target firmadyne_id: {self.fid}")
                print("[CopyHelper] Working in image_root: {self.image_root}")
            
                if not os.path.exists(dest):
                    print("[CopyHelper] Creating destination directory: " + dest)
                    os.makedirs(dest)
            
                if not os.path.exists(txt):
                    print("[CopyHelper] ERROR: Anomaly file not found: " + txt)
                    sys.exit(1)
                    
                if not os.path.exists(qdir):
                    print("[CopyHelper] ERROR: Queue directory not found: " + qdir)
                    sys.exit(1)
            
                indices = []
                signal_files = []
                print("[CopyHelper] Reading anomaly file...")
                try:
                    with open(txt, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line:
                                print("[CopyHelper] Processing line " + str(line_num) + ": " + repr(line))
                                
                                # Pattern for signal files: signal_N_timestamp.npz
                                signal_match = re.search(r'signal_(\\d+)_', line)
                                if signal_match:
                                    idx = int(signal_match.group(1))
                                    indices.append(idx)
                                    signal_files.append(line)
                                    print("[CopyHelper] Found signal index " + str(idx) + " from file: " + line)
                                else:
                                    # Try to extract any number that could be an ID
                                    id_patterns = [
                                        r'id:(\\d+)',     # id:123456
                                        r'id(\\d+)',      # id123456  
                                        r'^(\\d+)',       # number at start of line
                                        r'(\\d{{4,}})'    # any 4+ digit number
                                    ]
                                    
                                    found = False
                                    for pattern in id_patterns:
                                        m = re.search(pattern, line)
                                        if m:
                                            idx = int(m.group(1))
                                            indices.append(idx)
                                            print("[CopyHelper] Found index " + str(idx) + " using pattern " + pattern)
                                            found = True
                                            break
                                    
                                    if not found:
                                        print("[CopyHelper] No ID found in line: " + repr(line))
                                    
                except Exception as e:
                    print("[CopyHelper] ERROR reading file: " + str(e))
                    import traceback
                    traceback.print_exc()
                    sys.exit(1)
            
                print("[CopyHelper] Found " + str(len(indices)) + " anomalous indices: " + str(indices))
                
                if not indices:
                    print("[CopyHelper] No anomalous indices found")
                    sys.exit(0)
            
                copied = 0
                queue_files = os.listdir(qdir)
                print("[CopyHelper] Queue contains " + str(len(queue_files)) + " files")
                
                # Show sample queue files
                print("[CopyHelper] Sample queue files:")
                for i, f in enumerate(queue_files[:5]):
                    print("[CopyHelper]   " + f)
                
                for idx in indices:
                    # Try different prefix formats for queue files
                    prefixes = [
                        "id:%06d" % idx,    # id:000123
                        "id:%d" % idx,      # id:123
                        "id%06d" % idx,     # id000123
                        "id%d" % idx,       # id123
                        str(idx)            # just the number
                    ]
                    
                    found = False
                    for prefix in prefixes:
                        print("[CopyHelper] Looking for queue files with prefix: " + prefix)
                        for f in queue_files:
                            if f.startswith(prefix):
                                src_path = os.path.join(qdir, f)
                                dst_path = os.path.join(dest, f)
                                try:
                                    shutil.copy2(src_path, dst_path)
                                    copied += 1
                                    found = True
                                    print("[CopyHelper] Copied " + f + " (matched prefix: " + prefix + ")")
                                    break
                                except Exception as e:
                                    print("[CopyHelper] ERROR copying " + f + ": " + str(e))
                        if found:
                            break
                    
                    if not found:
                        print("[CopyHelper] WARNING: No queue file found for signal index " + str(idx))
                        print("[CopyHelper] This means signal_" + str(idx) + " doesn't have a corresponding queue entry")
            
                print("[CopyHelper] Copied " + str(copied) + " queue entries to " + dest)
                
                # Final verification
                if copied == 0:
                    print("[CopyHelper] WARNING: No files were copied!")
                    print("[CopyHelper] This could mean:")
                    print("[CopyHelper]   1. Signal indices don't match queue file naming")
                    print("[CopyHelper]   2. Queue files use different naming convention")
                    print("[CopyHelper]   3. The anomalous signals were not saved to queue")
            """)
    
            # Write and execute helper script
            import tempfile
            with tempfile.NamedTemporaryFile('w', suffix=".py", delete=False) as tf:
                tf.write(helper_src)
                tmp_helper = tf.name
    
            cont_helper = f"/tmp/copy_anom_{timestamp}.py"
            self._run("docker", "cp", tmp_helper, f"{self.container}:{cont_helper}")
            os.unlink(tmp_helper)
    
            # Execute helper inside container
            print(f"[CopyResume] Executing copy script with {python_cmd}...")
            exec_result = subprocess.run(['docker', 'exec', self.container, python_cmd, cont_helper],
                                       capture_output=True, text=True)
            
            # Print all output
            if exec_result.stdout:
                print("[CopyResume] Script output:")
                for line in exec_result.stdout.split('\n'):
                    if line.strip():
                        print(f"  {line}")
                        
            if exec_result.stderr:
                print("[CopyResume] Script errors:")
                for line in exec_result.stderr.split('\n'):
                    if line.strip():
                        print(f"  {line}")
                        
            if exec_result.returncode != 0:
                print(f"[CopyResume] Script failed with exit code: {exec_result.returncode}")
                return False
    
            # Merge into inputs/
            self._merge_to_inputs()
            
            # Clean up
            try:
                self._container_exec("rm", "-f", cont_helper)
            except:
                pass
    
            print("[CopyResume] Done - anomalous seeds now in inputs/")
            return True
    
        except Exception as e:
            print(f"[CopyResume] Python approach failed: {e}")
            return False

    def _use_shell_approach(self, timestamp, anom_txt_cont, actual_queue_dir):
        """Fallback shell script approach - FIXED FOR SIGNAL FILES"""
        try:
            shell_script = f"""#!/bin/bash
    echo "[CopyResume-Shell] Starting shell-based copy..."
    echo "[CopyResume-Shell] Working in image_root: {self.image_root}"
    echo "[CopyResume-Shell] Target firmadyne_id: {self.fid}"
    
    QDIR="{actual_queue_dir}"
    DEST="{self.anom_dir}"
    ANOM_FILE="{anom_txt_cont}"
    
    # Check if files exist
    if [ ! -f "$ANOM_FILE" ]; then
        echo "[CopyResume-Shell] ERROR: Anomaly file not found: $ANOM_FILE"
        exit 1
    fi
    
    if [ ! -d "$QDIR" ]; then
        echo "[CopyResume-Shell] ERROR: Queue directory not found: $QDIR"
        exit 1
    fi
    
    # Create destination directory
    mkdir -p "$DEST"
    
    # Extract indices and copy files
    echo "[CopyResume-Shell] Extracting indices from anomaly file..."
    COPIED=0
    
    while read -r line; do
        echo "[CopyResume-Shell] Processing: $line"
        
        # Extract signal index from signal_N_timestamp.npz format
        if [[ "$line" =~ signal_([0-9]+)_ ]]; then
            INDEX="${{BASH_REMATCH[1]}}"
            echo "[CopyResume-Shell] Found signal index: $INDEX"
            
            # Try different queue file prefixes
            PREFIXES=("id:$(printf "%06d" "$INDEX")" "id:$INDEX" "id$(printf "%06d" "$INDEX")" "id$INDEX" "$INDEX")
            
            FOUND=false
            for PREFIX in "${{PREFIXES[@]}}"; do
                echo "[CopyResume-Shell] Looking for prefix: $PREFIX"
                for file in "$QDIR"/"$PREFIX"*; do
                    if [ -f "$file" ]; then
                        cp "$file" "$DEST/"
                        echo "[CopyResume-Shell] Copied $(basename "$file")"
                        ((COPIED++))
                        FOUND=true
                        break 2
                    fi
                done
            done
            
            if [ "$FOUND" = false ]; then
                echo "[CopyResume-Shell] WARNING: No queue file found for signal index $INDEX"
            fi
            
        # Fallback: try to extract any ID pattern
        elif [[ "$line" =~ id:([0-9]+) ]]; then
            INDEX="${{BASH_REMATCH[1]}}"
            PREFIX=$(printf "id:%06d" "$INDEX")
            
            for file in "$QDIR"/"$PREFIX"*; do
                if [ -f "$file" ]; then
                    cp "$file" "$DEST/"
                    echo "[CopyResume-Shell] Copied $(basename "$file")"
                    ((COPIED++))
                    break
                fi
            done
        fi
    done < "$ANOM_FILE"
    
    echo "[CopyResume-Shell] Copied $COPIED files to $DEST"
    
    if [ "$COPIED" -eq 0 ]; then
        echo "[CopyResume-Shell] WARNING: No files were copied!"
        echo "[CopyResume-Shell] Signal indices might not correspond to queue entries"
    fi
    """
            
            # Write shell script to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile('w', suffix=".sh", delete=False) as tf:
                tf.write(shell_script)
                tmp_shell = tf.name
            
            # Copy to container and execute
            cont_shell = f"/tmp/copy_shell_{timestamp}.sh"
            self._run("docker", "cp", tmp_shell, f"{self.container}:{cont_shell}")
            os.unlink(tmp_shell)
            
            # Make executable and run
            self._container_exec("chmod", "+x", cont_shell)
            
            exec_result = subprocess.run(['docker', 'exec', self.container, 'bash', cont_shell],
                                       capture_output=True, text=True)
            
            # Print output
            if exec_result.stdout:
                for line in exec_result.stdout.split('\n'):
                    if line.strip():
                        print(f"  {line}")
                        
            if exec_result.stderr:
                for line in exec_result.stderr.split('\n'):
                    if line.strip():
                        print(f"  ERROR: {line}")
            
            # Merge into inputs/
            self._merge_to_inputs()
            
            # Clean up
            try:
                self._container_exec("rm", "-f", cont_shell)
            except:
                pass
                
            return exec_result.returncode == 0
            
        except Exception as e:
            print(f"[CopyResume] Shell approach failed: {e}")
            return False

    def _merge_to_inputs(self):
        """Merge anomalous entries into inputs directory - SAFE CREATION (only create if not exists)"""
        try:
            print("[CopyResume] Merging anomalous entries into inputs/ (safe creation, versioned copies)...")
            print(f"[CopyResume] Using inputs directory: {self.inputs_dir}")
            
            # SAFE CREATION: Only create inputs directory if it doesn't exist
            # This prevents any accidental data loss from mkdir operations
            check_inputs_result = subprocess.run(['docker', 'exec', self.container, 'test', '-d', self.inputs_dir],
                                               capture_output=True, text=True)
            
            if check_inputs_result.returncode != 0:
                # Directory doesn't exist, safe to create
                print(f"[CopyResume] inputs/ directory doesn't exist, creating: {self.inputs_dir}")
                mkdir_result = subprocess.run(['docker', 'exec', self.container, 'mkdir', '-p', self.inputs_dir],
                                            capture_output=True, text=True)
                
                if mkdir_result.returncode != 0:
                    print(f"[CopyResume] Error creating inputs directory: {mkdir_result.stderr}")
                    return
                else:
                    print("[CopyResume] inputs/ directory created successfully")
            else:
                print(f"[CopyResume] inputs/ directory already exists, preserving existing content")
            
            # Check if anomalous directory exists and has files
            check_anom_result = subprocess.run(['docker', 'exec', self.container, 'bash', '-c',
                f'test -d {self.anom_dir} && ls -1 {self.anom_dir} 2>/dev/null | wc -l || echo 0'],
                capture_output=True, text=True)
            
            if check_anom_result.returncode != 0:
                print(f"[CopyResume] No anomalous directory found: {self.anom_dir}")
                return
                
            anom_file_count = int(check_anom_result.stdout.strip() or 0)
            if anom_file_count == 0:
                print("[CopyResume] No anomalous files to merge")
                return
                
            print(f"[CopyResume] Found {anom_file_count} anomalous files to merge")
            
            # Count existing files in inputs before merge
            count_before_result = subprocess.run(['docker', 'exec', self.container, 'bash', '-c',
                f'ls -1 {self.inputs_dir} 2>/dev/null | wc -l || echo 0'],
                capture_output=True, text=True)
            files_before = int(count_before_result.stdout.strip() or 0)
            print(f"[CopyResume] Inputs directory has {files_before} files before merge")
            
            # VERSIONED MERGE: Create versioned copies (_1, _2, etc.) for existing files
            merge_command = f"""
            if [ ! -d {self.anom_dir} ]; then
                echo "No anomalous directory found"
                exit 0
            fi
            
            cd {self.anom_dir}
            copied_new=0
            copied_versioned=0
            errors=0
            
            for file in *; do
                # Skip if not a regular file
                if [ ! -f "$file" ]; then
                    continue
                fi
                
                # Check if file already exists in inputs/
                if [ -f "{self.inputs_dir}/$file" ]; then
                    # File exists, need to find next available version number
                    base_name="$file"
                    version=1
                    
                    # Find the next available version number
                    while [ -f "{self.inputs_dir}/${{base_name}}_$version" ]; do
                        version=$((version + 1))
                    done
                    
                    # Copy with version suffix
                    versioned_name="${{base_name}}_$version"
                    if cp "$file" "{self.inputs_dir}/$versioned_name"; then
                        echo "COPIED as $versioned_name (original $base_name exists)"
                        copied_versioned=$((copied_versioned + 1))
                    else
                        echo "ERROR copying $file as $versioned_name"
                        errors=$((errors + 1))
                    fi
                else
                    # File doesn't exist, copy normally
                    if cp "$file" "{self.inputs_dir}/"; then
                        echo "COPIED (new): $file"
                        copied_new=$((copied_new + 1))
                    else
                        echo "ERROR copying: $file"
                        errors=$((errors + 1))
                    fi
                fi
            done
            
            echo "=== MERGE SUMMARY ==="
            echo "Files copied (new): $copied_new"
            echo "Files copied (versioned): $copied_versioned" 
            echo "Total successful: $((copied_new + copied_versioned))"
            echo "Errors: $errors"
            """
            
            print("[CopyResume] Executing safe versioned merge...")
            merge_result = subprocess.run(['docker', 'exec', self.container, 'bash', '-c', merge_command],
                                        capture_output=True, text=True)
            
            # Show detailed merge output
            if merge_result.stdout:
                print("[CopyResume] Merge details:")
                for line in merge_result.stdout.split('\n'):
                    if line.strip():
                        print(f"[CopyResume]   {line}")
            
            if merge_result.stderr:
                print("[CopyResume] Merge warnings/errors:")
                for line in merge_result.stderr.split('\n'):
                    if line.strip():
                        print(f"[CopyResume]   ERROR: {line}")
            
            # Count files after merge for verification
            count_after_result = subprocess.run(['docker', 'exec', self.container, 'bash', '-c',
                f'ls -1 {self.inputs_dir} 2>/dev/null | wc -l || echo 0'],
                capture_output=True, text=True)
            files_after = int(count_after_result.stdout.strip() or 0)
            
            files_added = files_after - files_before
            
            print(f"[CopyResume] === FINAL SUMMARY ===")
            print(f"[CopyResume] Files in inputs/ before: {files_before}")
            print(f"[CopyResume] Files in inputs/ after:  {files_after}")
            print(f"[CopyResume] New files added:        {files_added}")
            print(f"[CopyResume] Original files preserved: {files_before}")
            
            if merge_result.returncode == 0:
                print("[CopyResume] ? Safe merge completed - inputs/ directory and all files preserved")
            else:
                print(f"[CopyResume] ? Merge completed with warnings")
                
        except Exception as e:
            print(f"[CopyResume] Error during safe merge: {e}")
            import traceback
            traceback.print_exc()
            
class TRENTIPlotViewer:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.plots = {}
        self.current_plot = None
        self.setup_plot_viewer()
        
    def setup_plot_viewer(self):
        """Setup the plot viewer interface"""
        # Main frame for plot viewer
        main_frame = tk.Frame(self.parent_frame, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Control frame
        control_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Plot selection
        tk.Label(control_frame, text="Select Plot:", bg='#34495e', fg='white', 
                font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
        self.plot_var = tk.StringVar()
        self.plot_dropdown = ttk.Combobox(control_frame, textvariable=self.plot_var, 
                                         state='readonly', width=40)
        self.plot_dropdown.pack(side='left', padx=5, pady=5)
        self.plot_dropdown.bind('<<ComboboxSelected>>', self.on_plot_selected)
        
        # Refresh button
        refresh_btn = tk.Button(control_frame, text="Refresh Plots", 
                               command=self.refresh_plots, bg='#3498db', fg='white',
                               font=('Arial', 9, 'bold'))
        refresh_btn.pack(side='left', padx=5)
        
        # Auto-refresh checkbox
        self.auto_refresh_var = tk.BooleanVar(value=True)
        auto_refresh_check = tk.Checkbutton(control_frame, text="Auto-refresh", 
                                          variable=self.auto_refresh_var,
                                          bg='#34495e', fg='white', selectcolor='#34495e',
                                          font=('Arial', 9))
        auto_refresh_check.pack(side='left', padx=5)
        
        # Export button
        export_btn = tk.Button(control_frame, text="Export Current Plot", 
                              command=self.export_current_plot, bg='#27ae60', fg='white',
                              font=('Arial', 9, 'bold'))
        export_btn.pack(side='right', padx=5)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="No plots available", 
                                    bg='#34495e', fg='#bdc3c7', font=('Arial', 9))
        self.status_label.pack(side='right', padx=10)
        
        # Separator
        separator = tk.Frame(main_frame, height=2, bg='#7f8c8d')
        separator.pack(fill='x', padx=5, pady=2)
        
        # Plot display frame
        self.plot_frame = tk.Frame(main_frame, bg='#2c3e50')
        self.plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6), dpi=100, facecolor='#2c3e50')
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Navigation toolbar
        toolbar_frame = tk.Frame(self.plot_frame)
        toolbar_frame.pack(fill='x')
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Initial empty plot
        self.show_empty_plot()
        
        # Start auto-refresh if enabled
        self.start_auto_refresh()
        
    def show_empty_plot(self):
        """Show empty plot with instructions"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, facecolor='#34495e')
        ax.text(0.5, 0.5, 'No plots available yet.\n\nPlots will appear here when TRENTI\nanalysis is executed.', 
                ha='center', va='center', fontsize=14, color='#bdc3c7',
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#7f8c8d')
        self.canvas.draw()
        
    def refresh_plots(self):
        """Refresh the list of available plots"""
        try:
            # Get the current configuration from the parent GUI
            parent = self.parent_frame.winfo_toplevel()
            if hasattr(parent, 'gui') and hasattr(parent.gui, 'get_config'):
                config = parent.gui.get_config()
                firmadyne_id = config.get('firmadyne_id', '')
                host_path = config.get('host_path', '')
                
                if host_path and firmadyne_id:
                    # Primary plot directory based on configuration
                    base_dir = f"{host_path}/image_{firmadyne_id}"
                    plot_dirs = [
                        f"{base_dir}/sca_images",  # Main plots directory
                        f"{base_dir}/trenti_sca_outputs_*/sca_images",  # Campaign-specific plots
                        "sca_images",  # Fallback local directory
                    ]
                else:
                    # Fallback directories if config not available
                    plot_dirs = [
                        'sca_images',
                        '/home/atenea/trenti/evaluations/*/sca_images',
                    ]
            else:
                # Fallback if can't access configuration
                plot_dirs = [
                    'sca_images',
                    '/home/atenea/trenti/evaluations/*/sca_images',
                ]
            
            found_plots = {}
            
            for pattern in plot_dirs:
                import glob
                for plot_dir in glob.glob(pattern):
                    if os.path.exists(plot_dir):
                        for file in os.listdir(plot_dir):
                            if file.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                                # Extract plot type and timestamp from filename
                                plot_type = self.extract_plot_type(file)
                                full_path = os.path.join(plot_dir, file)
                                found_plots[f"{plot_type} ({file})"] = full_path
            
            # Update dropdown
            self.plots = found_plots
            plot_names = list(self.plots.keys())
            self.plot_dropdown['values'] = plot_names
            
            if plot_names:
                if not self.plot_var.get() or self.plot_var.get() not in plot_names:
                    self.plot_var.set(plot_names[0])
                    self.on_plot_selected()
                self.status_label.config(text=f"{len(plot_names)} plots available")
                # self.plot_counter_label.config(text=f"Plots: {len(plot_names)}")
            else:
                self.plot_dropdown['values'] = []
                self.plot_var.set('')
                self.status_label.config(text="No plots found")
                # self.plot_counter_label.config(text="Plots: 0")
                self.show_empty_plot()
                
        except Exception as e:
            self.status_label.config(text=f"Error refreshing: {str(e)}")
            print(f"Error refreshing plots: {e}")
            
    def extract_plot_type(self, filename):
        """Extract plot type from filename"""
        if 'calibration' in filename.lower():
            return "Calibration View"
        elif 'cluster' in filename.lower() and 'binary' not in filename.lower():
            return "Anomaly Clusters"
        elif 'binary' in filename.lower():
            return "Binary Classification"
        elif '3d' in filename.lower():
            return "3D Projection"
        elif 'anomaly' in filename.lower():
            return "Anomaly Analysis"
        else:
            return "Analysis Plot"
            
    def on_plot_selected(self, event=None):
        """Handle plot selection"""
        selected = self.plot_var.get()
        if selected and selected in self.plots:
            self.load_plot(self.plots[selected])
            
    def load_plot(self, plot_path):
        """Load and display a plot"""
        try:
            self.figure.clear()
            
            if plot_path.endswith('.pdf'):
                # For PDF files, we need to convert them to image first
                self.load_pdf_plot(plot_path)
            else:
                # For image files
                self.load_image_plot(plot_path)
                
            self.canvas.draw()
            self.current_plot = plot_path
            
        except Exception as e:
            self.show_error_plot(f"Error loading plot: {str(e)}")
            print(f"Error loading plot {plot_path}: {e}")
            
    def load_pdf_plot(self, pdf_path):
        """Load PDF plot (requires pdf2image or similar)"""
        try:
            # Try to use pdf2image if available
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(pdf_path, first_page=1, last_page=1)
                if images:
                    img_array = np.array(images[0])
                    ax = self.figure.add_subplot(111)
                    ax.imshow(img_array)
                    ax.axis('off')
                    return
            except ImportError:
                pass
            
            # Fallback: try to use matplotlib to read PDF directly
            self.show_error_plot("PDF viewing requires pdf2image package.\nInstall with: pip install pdf2image")
            
        except Exception as e:
            self.show_error_plot(f"Cannot display PDF: {str(e)}")
            
    def load_image_plot(self, image_path):
        """Load image plot"""
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            ax = self.figure.add_subplot(111, facecolor='#2c3e50')
            ax.imshow(img_array)
            ax.axis('off')
            
        except Exception as e:
            self.show_error_plot(f"Cannot display image: {str(e)}")
            
    def show_error_plot(self, error_message):
        """Show error message in plot area"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, facecolor='#e74c3c')
        ax.text(0.5, 0.5, error_message, ha='center', va='center', 
                fontsize=12, color='white', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#c0392b')
        self.canvas.draw()
        
    def export_current_plot(self):
        """Export current plot to file"""
        if not self.current_plot:
            messagebox.showwarning("No Plot", "No plot is currently displayed")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Export plot"
        )
        
        if filename:
            try:
                if filename.endswith('.pdf'):
                    self.figure.savefig(filename, format='pdf', bbox_inches='tight')
                else:
                    self.figure.savefig(filename, format='png', bbox_inches='tight', dpi=300)
                messagebox.showinfo("Success", f"Plot exported to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export plot: {str(e)}")
                
    def start_auto_refresh(self):
        """Start auto-refresh timer"""
        def auto_refresh():
            if self.auto_refresh_var.get():
                self.refresh_plots()
            # Schedule next refresh
            self.parent_frame.after(5000, auto_refresh)  # Every 5 seconds
            
        # Start after 2 seconds
        self.parent_frame.after(2000, auto_refresh)

class TRENTIFuzzingGUI:
    
    def __init__(self, root):
        """Initialize TRENTI GUI with Docker optimization"""
        self.root = root
        self.root.title("TRENTI - Fuzzing Automation Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Docker optimization
        self.docker_client = None
        self.container = None
        
        # System state - MODIFICADO para incluir tiempo total
        self.automation_state = {
            'is_running': False,
            'is_paused': False,
            'current_campaign': 0,
            'start_time': None,
            'campaigns': [],
            'anomalies_count': 0,  # Solo SCA anomalies
            'container_id': None,
            'processes': {},
            'fuzzing_start_time': None,
            'total_start_time': None,
            'total_duration_seconds': 0,
            'elapsed_fuzzing_time': 0,
            # NUEVAS LÍNEAS:
            'exec_time_anomalies': 0,
            'http_response_anomalies': 0,
            'total_anomalies_found': 0
        }
        
        self.setup_ui()
        self.log("TRENTI initialized. Ready to start fuzzing automation...")
        
    def setup_docker_connection(self, config):
        """Setup direct Docker API connection like individual code"""
        try:
            import docker
            self.docker_client = docker.from_env()
            self.container = self.docker_client.containers.get(config['container_name'])
            
            if self.container.status != 'running':
                self.log("Starting container...", "INFO")
                self.container.start()
                time.sleep(5)
            
            self.log("Direct Docker API connection established", "SUCCESS")
            return True
            
        except ImportError:
            self.log("Docker module not available, using subprocess fallback", "WARNING")
            return False
        except Exception as e:
            self.log(f"Docker API connection failed: {e}, using subprocess fallback", "WARNING")
            return False
    
    def run_container_command_direct(self, cmd, detach=False, workdir=None, show_output=True):
        """Direct container command execution like individual code"""
        try:
            if not self.container:
                raise Exception("No direct Docker connection")
            
            full_cmd = f'bash -c "{cmd}"'
            if workdir:
                full_cmd = f'bash -c "cd {workdir} && {cmd}"'
            
            if show_output:
                self.raw_print(f"DOCKER DIRECT: {cmd}")
            
            if detach:
                # MODIFICADO: Para detach, también crear stream para monitoreo
                result = self.container.exec_run(full_cmd, detach=True, stream=True)
                if show_output:
                    self.raw_print("(Command started in background - monitoring output...)")
                
                # NUEVO: Monitorear salida en hilo separado
                if show_output:
                    def monitor_detached_output():
                        try:
                            exec_id = result.id
                            # Obtener logs del exec
                            for line in self.container.exec_run(f'tail -f /proc/{exec_id}/fd/1', stream=True)[1]:
                                if line:
                                    output = line.decode() if isinstance(line, bytes) else str(line)
                                    self.raw_print(f"AFL_OUTPUT: {output.strip()}")
                        except Exception as e:
                            self.raw_print(f"Error monitoring detached output: {e}")
                    
                    import threading
                    threading.Thread(target=monitor_detached_output, daemon=True).start()
                
                return result
            else:
                result = self.container.exec_run(full_cmd, stream=True)
                
                if show_output and result.output:
                    # MODIFICADO: Procesar stream en tiempo real
                    for line in result.output:
                        if line:
                            output = line.decode() if isinstance(line, bytes) else str(line)
                            if output.strip():
                                self.raw_print(f"OUTPUT: {output.strip()}")
                
                return result
                
        except Exception as e:
            if show_output:
                self.raw_print(f"Direct execution failed: {e}, using subprocess fallback")
            return None

    
    # --------------------------------------------------------------
    #  PRESETS
    # --------------------------------------------------------------
    def load_hedwig_preset(self):
        """
        Carga todos los campos necesarios para fuzzear hedwig.cgi
        (DIR-815  image_9050) con un solo clic.
        """
        # Valores de configuración
        self.config_vars['firmadyne_id'].set("9050")
        self.config_vars['router_ip'].set("10.205.1.111")
        self.config_vars['target_cgi'].set("hedwig.cgi")

        # Comando CURL (usa {VALUE} y sigue pudiendo editarse después)
        hedwig_cmd = (
            'curl -X POST \\ \n'
            '  -H "Content-Type: application/x-www-form-urlencoded" \\ \n'
            '  -H "Cookie: uid={VALUE}" \\ \n'
            '  -d "dummy=1" \\ \n'
            '  --connect-timeout 10 \\ \n'
            '  "http://10.205.1.111/hedwig.cgi"'
        )

        self.curl_command_text.delete('1.0', tk.END)
        self.curl_command_text.insert('1.0', hedwig_cmd)
        self.log("Preset hedwig.cgi loaded", "SUCCESS")

    def load_miniupnpd_preset(self):
        """
        Carga los parámetros y el comando para miniupnpd
        (image_161160  puerto 65535 vía netcat).
        """
        # Valores de configuración
        self.config_vars['firmadyne_id'].set("161160")
        self.config_vars['router_ip'].set("10.205.1.112")
        # miniupnpd no es un CGI, pero dejamos constancia
        self.config_vars['target_cgi'].set("miniupnpd")

        # Comando personalizado (usa {VALUE_FILE} y {router_ip})
        miniupnpd_cmd = (
            'timeout 10 bash -c '
            '"sed \'s/Host: 10.0.0.90/Host: 10.205.1.112:65535/\' {VALUE_FILE} '
            '| nc {router_ip} 65535"'
        )

        self.curl_command_text.delete('1.0', tk.END)
        self.curl_command_text.insert('1.0', miniupnpd_cmd)
        self.log("Preset miniupnpd loaded", "SUCCESS")

    def load_httpd_preset(self):
        """
        Load parameters and command template for fuzzing the embedded httpd
        (image_161161  port 80).
        """
        # --- Configuration -----------------------------------------------------
        self.config_vars['firmadyne_id'].set("161161")
        self.config_vars['router_ip'].set("10.205.1.112")   # adjust if needed
        self.config_vars['target_cgi'].set("httpd")
    
        # --- Command template --------------------------------------------------
        # 1. Rewrites the Host header in each seed from 10.0.0.90 -> <router_ip>.
        # 2. Pushes the (possibly mutated) request to port 80 with netcat.
        # 3. Uses a global 10-s timeout and tells nc to close immediately after EOF.
        httpd_cmd = (
            'timeout 10 bash -c '
            '"sed \'s/Host: 10.0.0.90/Host: {router_ip}/\' {VALUE_FILE} '
            '| nc -q 1 {router_ip} 80"'
        )
    
        self.curl_command_text.delete("1.0", tk.END)
        self.curl_command_text.insert("1.0", httpd_cmd)
        self.log("Preset httpd loaded", "SUCCESS")


    def cleanup_network_interfaces(self, config):
        """Clean network interfaces and routes between campaigns"""
        try:
            container_name = config['container_name']
            firmadyne_id = config['firmadyne_id']
            
            self.log("Cleaning network interfaces for next campaign...", "INFO")
            self.raw_print("Starting network cleanup...")
            
            # 1. Kill any remaining QEMU processes
            self.log("Killing QEMU processes...", "DEBUG")
            qemu_kill_commands = [
                ['docker', 'exec', container_name, 'pkill', '-9', '-f', 'qemu-system'],
                ['docker', 'exec', container_name, 'pkill', '-9', '-f', 'qemu-mips']
            ]
            
            for cmd in qemu_kill_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        self.raw_print(f"Killed QEMU processes: {' '.join(cmd[3:])}")
                except subprocess.TimeoutExpired:
                    self.raw_print(f"Timeout killing QEMU processes")
                except Exception as e:
                    self.raw_print(f"Error killing QEMU: {e}")
            
            # 2. Clean TAP devices
            tap_device = f"tap{firmadyne_id}_0"
            self.log(f"Cleaning TAP device: {tap_device}", "DEBUG")
            
            tap_cleanup_commands = [
                # Bring down the interface
                ['docker', 'exec', container_name, 'ip', 'link', 'set', tap_device, 'down'],
                # Delete the TAP device
                ['docker', 'exec', container_name, 'ip', 'link', 'delete', tap_device],
                # Alternative deletion method
                ['docker', 'exec', container_name, 'tunctl', '-d', tap_device]
            ]
            
            for cmd in tap_cleanup_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        self.raw_print(f"Success: {' '.join(cmd[3:])}")
                    else:
                        self.raw_print(f"Command failed (may be expected): {' '.join(cmd[3:])}")
                except subprocess.TimeoutExpired:
                    self.raw_print(f"Timeout: {' '.join(cmd[3:])}")
                except Exception as e:
                    self.raw_print(f"Error: {e}")
            
            # 3. Clean IP addresses and routes
            self.log("Cleaning IP addresses and routes...", "DEBUG")
            
            # Remove any existing IP addresses on br0
            ip_cleanup_commands = [
                ['docker', 'exec', container_name, 'ip', 'addr', 'flush', 'dev', 'br0'],
                ['docker', 'exec', container_name, 'ip', 'route', 'flush', 'dev', 'br0'],
                # Remove specific routes that might conflict
                ['docker', 'exec', container_name, 'ip', 'route', 'del', '192.168.10.0/24'],
                ['docker', 'exec', container_name, 'ip', 'route', 'del', '192.168.10.1']
            ]
            
            for cmd in ip_cleanup_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        self.raw_print(f"Success: {' '.join(cmd[3:])}")
                    else:
                        self.raw_print(f"Command failed (may be expected): {' '.join(cmd[3:])}")
                except subprocess.TimeoutExpired:
                    self.raw_print(f"Timeout: {' '.join(cmd[3:])}")
                except Exception as e:
                    self.raw_print(f"Error: {e}")
            
            # 4. Clean bridge interface
            bridge_cleanup_commands = [
                ['docker', 'exec', container_name, 'ip', 'link', 'set', 'br0', 'down'],
                ['docker', 'exec', container_name, 'brctl', 'delbr', 'br0']
            ]
            
            for cmd in bridge_cleanup_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        self.raw_print(f"Success: {' '.join(cmd[3:])}")
                    else:
                        self.raw_print(f"Command failed (may be expected): {' '.join(cmd[3:])}")
                except subprocess.TimeoutExpired:
                    self.raw_print(f"Timeout: {' '.join(cmd[3:])}")
                except Exception as e:
                    self.raw_print(f"Error: {e}")
            
            # 5. Wait for cleanup to complete
            self.raw_print("Waiting 5 seconds for network cleanup to complete...")
            time.sleep(5)
            
            # 6. Verify cleanup
            self.log("Verifying network cleanup...", "DEBUG")
            verify_commands = [
                ['docker', 'exec', container_name, 'ip', 'link', 'show', tap_device],
                ['docker', 'exec', container_name, 'pgrep', '-f', 'qemu']
            ]
            
            for cmd in verify_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode != 0:
                        self.raw_print(f"Good: {' '.join(cmd[3:])} not found (cleaned up)")
                    else:
                        self.raw_print(f"Warning: {' '.join(cmd[3:])} still exists")
                except Exception:
                    self.raw_print(f"Verification failed for: {' '.join(cmd[3:])}")
            
            self.log("Network cleanup completed", "SUCCESS")
            self.raw_print("Network cleanup completed - ready for next campaign")
            return True
            
        except Exception as e:
            self.log(f"Error in network cleanup: {str(e)}", "ERROR")
            self.raw_print(f"Network cleanup error: {str(e)}")
            return False
    
    def cleanup_between_campaigns(self, config):
        """Cleanup between campaigns - Kill run.sh, user.sh and afl-fuzz processes"""
        try:
            self.log("Starting cleanup between campaigns...", "INFO")
            self.raw_print("=== STARTING INTER-CAMPAIGN CLEANUP ===")
            
            container_name = config['container_name']
            
            # First, let's see what processes are actually running
            self.raw_print("Checking what processes are currently running...")
            try:
                ps_cmd = ['docker', 'exec', container_name, 'ps', 'auxww']
                ps_result = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=10)
                
                if ps_result.returncode == 0:
                    self.raw_print("Current processes in container:")
                    fuzzing_processes = []
                    for line in ps_result.stdout.split('\n'):
                        if line.strip() and any(keyword in line.lower() for keyword in 
                                              ['afl', 'qemu', 'fuzz', 'run.sh', 'user.sh']):
                            fuzzing_processes.append(line.strip())
                            self.raw_print(f"  FOUND: {line.strip()}")
                    
                    if not fuzzing_processes:
                        self.raw_print("  No fuzzing-related processes found")
                else:
                    self.raw_print(f"Could not list processes: {ps_result.stderr}")
            except Exception as e:
                self.raw_print(f"Error checking processes: {e}")
            
            # Define cleanup commands with better descriptions
            cleanup_commands = [
                (['docker', 'exec', container_name, 'pkill', '-9', '-f', 'run.sh'], 'run.sh scripts'),
                (['docker', 'exec', container_name, 'pkill', '-9', '-f', 'user.sh'], 'user.sh scripts'),
                (['docker', 'exec', container_name, 'pkill', '-9', 'afl-fuzz'], 'afl-fuzz'),
                (['docker', 'exec', container_name, 'pkill', '-9', 'afl-qemu-trace'], 'afl-qemu-trace'),
                (['docker', 'exec', container_name, 'pkill', '-9', '-f', 'qemu-system-mips'], 'qemu-system-mips'),
                (['docker', 'exec', container_name, 'pkill', '-9', '-f', 'qemu-mips'], 'qemu-mips processes'),
                (['docker', 'exec', container_name, 'killall', '-9', 'qemu-system-mips'], 'qemu-system-mips (killall)'),
            ]
            
            killed_something = False
            
            for cmd, description in cleanup_commands:
                try:
                    self.raw_print(f"Attempting to kill: {description}")
                    self.raw_print(f"Command: {' '.join(cmd)}")
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        self.raw_print(f" Successfully killed: {description}")
                        self.log(f"Successfully killed: {description}", "SUCCESS")
                        killed_something = True
                    else:
                        self.raw_print(f" No processes found for: {description}")
                        self.log(f"No processes found: {description}", "DEBUG")
                        
                        # Show any error output for debugging
                        if result.stderr.strip():
                            self.raw_print(f"  Error output: {result.stderr.strip()}")
                            
                except subprocess.TimeoutExpired:
                    self.raw_print(f" Timeout killing: {description}")
                    self.log(f"Timeout killing: {description}", "WARNING")
                except Exception as e:
                    self.raw_print(f" Error killing {description}: {e}")
                    self.log(f"Error killing {description}: {e}", "WARNING")
            
            # Alternative approach: Use pgrep to find processes first, then kill by PID
            self.raw_print("Trying alternative PID-based cleanup...")
            
            pid_patterns = ['afl-fuzz', 'qemu-system-mips', 'qemu-mips', 'afl-qemu-trace']
            
            for pattern in pid_patterns:
                try:
                    # Find PIDs
                    pgrep_cmd = ['docker', 'exec', container_name, 'pgrep', '-f', pattern]
                    pgrep_result = subprocess.run(pgrep_cmd, capture_output=True, text=True, timeout=5)
                    
                    if pgrep_result.returncode == 0 and pgrep_result.stdout.strip():
                        pids = pgrep_result.stdout.strip().split('\n')
                        self.raw_print(f"Found {len(pids)} {pattern} processes: {', '.join(pids)}")
                        
                        # Kill each PID
                        for pid in pids:
                            if pid.strip():
                                kill_cmd = ['docker', 'exec', container_name, 'kill', '-9', pid.strip()]
                                kill_result = subprocess.run(kill_cmd, capture_output=True, text=True, timeout=5)
                                
                                if kill_result.returncode == 0:
                                    self.raw_print(f" Killed {pattern} PID {pid.strip()}")
                                    killed_something = True
                                else:
                                    self.raw_print(f" Failed to kill {pattern} PID {pid.strip()}")
                    else:
                        self.raw_print(f"No {pattern} processes found via pgrep")
                        
                except Exception as e:
                    self.raw_print(f"Error in PID-based cleanup for {pattern}: {e}")
            
            # Clean up Python process references
            try:
                if 'run_sh' in self.automation_state['processes']:
                    proc = self.automation_state['processes']['run_sh']
                    if proc and proc.poll() is None:
                        proc.terminate()
                        self.log("Python run_sh process terminated", "DEBUG")
                        killed_something = True
                        
                if 'user_sh' in self.automation_state['processes']:
                    proc = self.automation_state['processes']['user_sh']
                    if proc and proc.poll() is None:
                        proc.terminate()
                        self.log("Python user_sh process terminated", "DEBUG")
                        killed_something = True
                        
            except Exception as e:
                self.log(f"Error cleaning Python processes: {e}", "WARNING")
            
            # Wait for processes to terminate
            self.raw_print("Waiting 5 seconds for processes to terminate...")
            time.sleep(5)
            
            # Final verification
            self.raw_print("Performing final verification...")
            try:
                ps_cmd = ['docker', 'exec', container_name, 'ps', 'auxww']
                ps_result = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=10)
                
                if ps_result.returncode == 0:
                    remaining_processes = []
                    for line in ps_result.stdout.split('\n'):
                        if line.strip() and any(keyword in line.lower() for keyword in 
                                              ['afl', 'qemu', 'fuzz', 'run.sh', 'user.sh']):
                            remaining_processes.append(line.strip())
                    
                    if remaining_processes:
                        self.raw_print(" WARNING: Some processes are still running:")
                        for proc in remaining_processes:
                            self.raw_print(f"  STILL RUNNING: {proc}")
                    else:
                        self.raw_print(" All fuzzing processes successfully terminated")
                
            except Exception as e:
                self.raw_print(f"Error in final verification: {e}")
            
            if killed_something:
                self.log("Inter-campaign cleanup completed - some processes were killed", "SUCCESS")
            else:
                self.log("Inter-campaign cleanup completed - no processes needed killing", "INFO")
                
            self.raw_print("=== INTER-CAMPAIGN CLEANUP COMPLETED ===")
            return True
            
        except Exception as e:
            self.log(f"Error in inter-campaign cleanup: {str(e)}", "ERROR")
            self.raw_print(f"Inter-campaign cleanup error: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return False
      
    def get_curl_command(self):
        """Get current curl command from text widget"""
        return self.curl_command_text.get('1.0', tk.END).strip()
    
    def setup_ui(self):
        """Setup user interface"""
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#2c3e50', foreground='white')
        style.configure('Config.TFrame', background='#34495e')
        
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="TRENTI - Fuzzing Automation Tool", 
                              font=('Arial', 20, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(title_frame, text="Automated system for fuzzing campaigns with side-channel analysis", 
                                 font=('Arial', 10), bg='#2c3e50', fg='#bdc3c7')
        subtitle_label.pack()
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Configuration tab
        self.setup_config_tab()
        
        
        # Logs tab
        self.setup_logs_tab()
                
        # SCA Plots tab
        self.setup_sca_plots_tab()
        
    def setup_config_tab(self):
        """Setup configuration tab - MODIFIED for time-based execution"""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="Configuration")
        
        # Main configuration frame
        main_config = ttk.LabelFrame(config_frame, text="TRENTI Configuration", padding="10")
        main_config.pack(fill='x', padx=10, pady=5)
        
        # Configuration grid
        config_grid = tk.Frame(main_config)
        config_grid.pack(fill='x')
        
        # Basic configurations - MODIFICADO: Cambiar "Max Campaigns" por "Total Duration"
        configs = [
            ("Firmadyne ID:", "firmadyne_id", "9050"),
            ("Container Name:", "container_name", "trenti"),
            ("Host Path:", "host_path", "/home/atenea/trenti/evaluations"),
            ("Router IP:", "router_ip", "10.205.1.111"),
            ("Oscilloscope IP:", "oscilloscope_ip", "10.205.1.95"),
            ("Target CGI:", "target_cgi", "hedwig.cgi"),
            ("Total Duration (hours):", "total_duration_hours", "24"),
            ("Monitoring Interval (sec):", "monitoring_interval", "30")
        ]
        
        self.config_vars = {}
        for i, (label, var_name, default) in enumerate(configs):
            row = i // 2
            col = (i % 2) * 2
            
            tk.Label(config_grid, text=label, font=('Arial', 10, 'bold')).grid(
                row=row, column=col, sticky='w', padx=5, pady=5)
            
            var = tk.StringVar(value=default)
            self.config_vars[var_name] = var
            entry = tk.Entry(config_grid, textvariable=var, width=30)
            entry.grid(row=row, column=col+1, padx=5, pady=5, sticky='ew')
        
        config_grid.columnconfigure(1, weight=1)
        config_grid.columnconfigure(3, weight=1)
        
        # Side-Channel Analysis Configuration
        sca_frame = ttk.LabelFrame(config_frame, text="Side-Channel Analysis Configuration", padding="10")
        sca_frame.pack(fill='x', padx=10, pady=5)
        
        # SCA options
        sca_grid = tk.Frame(sca_frame)
        sca_grid.pack(fill='x')
        
        # Enable SCA checkbox
        self.enable_sca_var = tk.BooleanVar(value=True)
        enable_sca_check = tk.Checkbutton(sca_grid, 
                                        text="Enable Side-Channel Analysis (EM Capture)",
                                        variable=self.enable_sca_var,
                                        font=('Arial', 10),
                                        command=self.on_sca_toggle)
        enable_sca_check.grid(row=0, column=0, columnspan=2, sticky='w', pady=5)
        
        # SCA parameters
        sca_configs = [
            ("Capture Channel:", "sca_channel", "C3"),
            ("Voltage Div (V):", "sca_voltage_div", "0.010"),
            ("Time Div (s):", "sca_time_div", "500E-9"),
            ("Sample Rate (GS/s):", "sca_sample_rate", "10"),
            ("Memory Size:", "sca_memory_size", "10K")
        ]
        
        for i, (label, var_name, default) in enumerate(sca_configs):
            row = i // 2 + 1
            col = (i % 2) * 2
            
            sca_label = tk.Label(sca_grid, text=label, font=('Arial', 9))
            sca_label.grid(row=row, column=col, sticky='w', padx=5, pady=2)
            
            var = tk.StringVar(value=default)
            self.config_vars[var_name] = var
            sca_entry = tk.Entry(sca_grid, textvariable=var, width=20)
            sca_entry.grid(row=row, column=col+1, padx=5, pady=2, sticky='ew')
        
        sca_grid.columnconfigure(1, weight=1)
        sca_grid.columnconfigure(3, weight=1)
        
        # NEW: Post-TRENTI Analysis Configuration
        post_analysis_frame = ttk.LabelFrame(config_frame, text="Side-Channel Analysis Configuration", padding="10")
        post_analysis_frame.pack(fill='x', padx=10, pady=5)
        
        post_analysis_grid = tk.Frame(post_analysis_frame)
        post_analysis_grid.pack(fill='x')
        
        # Enable post-analysis checkbox
        self.enable_post_analysis_var = tk.BooleanVar(value=True)
        enable_post_check = tk.Checkbutton(post_analysis_grid, 
                                         text="Enable Side-Channel Analysis (EM + Execution Time)",
                                         variable=self.enable_post_analysis_var,
                                         font=('Arial', 10))
        enable_post_check.grid(row=0, column=0, columnspan=2, sticky='w', pady=5)
        
        # Post-analysis options
        self.enable_exec_time_analysis_var = tk.BooleanVar(value=True)
        exec_time_check = tk.Checkbutton(post_analysis_grid, 
                                       text="  +- Execution Time Clustering Analysis",
                                       variable=self.enable_exec_time_analysis_var,
                                       font=('Arial', 9))
        exec_time_check.grid(row=1, column=0, columnspan=2, sticky='w', padx=20, pady=2)
        
        self.enable_http_analysis_var = tk.BooleanVar(value=True)
        http_check = tk.Checkbutton(post_analysis_grid, 
                                  text="  +- HTTP Response Analysis (Non-200 Detection)",
                                  variable=self.enable_http_analysis_var,
                                  font=('Arial', 9))
        http_check.grid(row=2, column=0, columnspan=2, sticky='w', padx=20, pady=2)
        
        self.enable_deduplication_var = tk.BooleanVar(value=True)
        dedup_check = tk.Checkbutton(post_analysis_grid, 
                                   text="  +- Anomaly Deduplication",
                                   variable=self.enable_deduplication_var,
                                   font=('Arial', 9))
        dedup_check.grid(row=3, column=0, columnspan=2, sticky='w', padx=20, pady=2)
        
        # Post-analysis parameters
        post_params_frame = tk.Frame(post_analysis_grid)
        post_params_frame.grid(row=4, column=0, columnspan=2, sticky='ew', pady=5)
        
        tk.Label(post_params_frame, text="Clustering Sensitivity:", font=('Arial', 9)).grid(
            row=0, column=0, sticky='w', padx=5)
        
        self.clustering_sensitivity_var = tk.StringVar(value="Medium")
        sensitivity_combo = ttk.Combobox(post_params_frame, textvariable=self.clustering_sensitivity_var,
                                       values=["Low", "Medium", "High"], state="readonly", width=15)
        sensitivity_combo.grid(row=0, column=1, padx=5)
                
        # HTTP/Curl Configuration - SIMPLIFIED for direct command input
        http_frame = ttk.LabelFrame(config_frame, text="HTTP/Curl Configuration", padding="10")
        http_frame.pack(fill='x', padx=10, pady=5)
        
        # Curl Command Configuration
        curl_command_frame = tk.Frame(http_frame)
        curl_command_frame.pack(fill='x')
        
        # Header with presets buttons
        header_frame = tk.Frame(curl_command_frame)
        header_frame.pack(fill='x', pady=(0, 5))
        
        hedwig_btn = tk.Button(
            header_frame, text="Load hedwig.cgi preset",
            command=self.load_hedwig_preset,
            bg='#28a745', fg='white', font=('Arial', 8)
        )
        hedwig_btn.pack(side='right', padx=3)
       
        miniupnpd_btn = tk.Button(
            header_frame, text="Load miniupnpd preset",
            command=self.load_miniupnpd_preset,
            bg='#28a745', fg='white', font=('Arial', 8)
        )
        miniupnpd_btn.pack(side='right', padx=3)
        
        httpd_btn = tk.Button(
            header_frame, text="Load httpd preset",
            command=self.load_httpd_preset,
            bg='#6f42c1', fg='white', font=('Arial', 8)  # Color púrpura
        )
        httpd_btn.pack(side='right', padx=3)

        
        # Curl command text area
        curl_text_frame = tk.Frame(curl_command_frame)
        curl_text_frame.pack(fill='both', expand=True)
        
        self.curl_command_text = tk.Text(curl_text_frame, height=8, font=('Courier New', 9),
                                         wrap='none', bg='#f8f9fa', fg='#212529')
        curl_scrollbar = tk.Scrollbar(curl_text_frame, orient='vertical', 
                                      command=self.curl_command_text.yview)
        self.curl_command_text.configure(yscrollcommand=curl_scrollbar.set)
        
        self.curl_command_text.pack(side='left', fill='both', expand=True)
        curl_scrollbar.pack(side='right', fill='y')
        
        # Set initial hedwig.cgi command
        hedwig_command = """curl -X POST \\
      -H "Content-Type: application/x-www-form-urlencoded" \\
      -H "Cookie: uid={VALUE}" \\
      -d "dummy=1" \\
      --connect-timeout 10 \\
      "http://10.205.1.111/hedwig.cgi\""""
        
        self.curl_command_text.insert('1.0', hedwig_command)
        
        #     # Help text
        #     help_text = """Available placeholders:
        #  {VALUE} - Content from test file (will replace uid cookie value)
        #  {VALUE_FILE} - Path to test file (for file uploads with -F)
        
        # Example: The uid={VALUE} will become uid=<content_of_test_file> for each fuzzing input.
        # You can modify this command completely to match your target's requirements."""
        
        # help_label = tk.Label(curl_command_frame, text=help_text, 
        #                      font=('TkDefaultFont', 8), foreground='#666666',
        #                      justify='left')
        # help_label.pack(anchor='w', pady=(5, 0))
        
        # Testing options frame
        test_frame = ttk.LabelFrame(config_frame, text="Testing Options", padding="10")
        test_frame.pack(fill='x', padx=10, pady=5)
        
        # Test threshold checkbox
        self.test_threshold_var = tk.BooleanVar()
        test_threshold_check = tk.Checkbutton(test_frame, 
                                            text="Enable Test Threshold (5 seconds instead of 1.2 hours)",
                                            variable=self.test_threshold_var,
                                            font=('Arial', 10),
                                            command=self.on_test_threshold_change)
        test_threshold_check.pack(anchor='w', pady=5)
        
        # Threshold value display
        self.threshold_info_label = tk.Label(test_frame, 
                                           text="Current threshold: 4320 seconds (1.2 hours)",
                                           font=('Arial', 9), fg='#666666')
        self.threshold_info_label.pack(anchor='w')
        
        # Warning label
        self.test_warning_label = tk.Label(test_frame, 
                                         text="Test mode will stop AFL after 5 seconds without new paths",
                                         font=('Arial', 9), fg='#e74c3c')
        self.test_warning_label.pack(anchor='w')
        self.test_warning_label.pack_forget()  # Initially hidden
        
        # Control buttons
        control_frame = ttk.LabelFrame(config_frame, text="Execution Control", padding="10")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        button_frame = tk.Frame(control_frame)
        button_frame.pack()
        
        self.start_btn = tk.Button(button_frame, text="Start TRENTI", 
                                  command=self.start_automation, bg='#27ae60', fg='white',
                                  font=('Arial', 12, 'bold'), padx=20, pady=5)
        self.start_btn.pack(side='left', padx=5)
        
        self.pause_btn = tk.Button(button_frame, text="Pause", 
                                  command=self.pause_automation, bg='#f39c12', fg='white',
                                  font=('Arial', 12, 'bold'), padx=20, pady=5, state='disabled')
        self.pause_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(button_frame, text="Stop", 
                                 command=self.stop_automation, bg='#e74c3c', fg='white',
                                 font=('Arial', 12, 'bold'), padx=20, pady=5, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Test SCA connection button
        self.test_sca_btn = tk.Button(button_frame, text="Test SCA Connection", 
                                     command=self.test_sca_connection, bg='#3498db', fg='white',
                                     font=('Arial', 10, 'bold'), padx=15, pady=5)
        self.test_sca_btn.pack(side='left', padx=5)
                
        # Current status
        status_frame = ttk.LabelFrame(config_frame, text="Current Status", padding="10")
        status_frame.pack(fill='x', padx=10, pady=5)
        
        self.status_label = tk.Label(status_frame, text="Status: Inactive", 
                                    font=('Arial', 12, 'bold'), fg='#7f8c8d')
        self.status_label.pack()
        
    def on_sca_toggle(self):
        """Handle SCA enable/disable toggle"""
        if self.enable_sca_var.get():
            self.log("Side-Channel Analysis ENABLED", "INFO")
            self.update_sca_status("Enabled")
        else:
            self.log("Side-Channel Analysis DISABLED", "WARNING")
            self.update_sca_status("Disabled")
            
    def update_sca_status(self, status):
        """Update SCA status indicator"""
        if hasattr(self, 'sca_status_var'):
            self.sca_status_var.set(status)
        if hasattr(self, 'metric_vars') and 'sca_status' in self.metric_vars:
            self.metric_vars['sca_status'].set(status)
        
        # Update color based on status
        if hasattr(self, 'sca_status_frame'):
            if status == "Running":
                self.sca_status_frame.config(bg='#27ae60')  # Green
                for child in self.sca_status_frame.winfo_children():
                    child.config(bg='#27ae60')
            elif status == "Enabled":
                self.sca_status_frame.config(bg='#3498db')  # Blue
                for child in self.sca_status_frame.winfo_children():
                    child.config(bg='#3498db')
            elif status == "Disabled":
                self.sca_status_frame.config(bg='#95a5a6')  # Gray
                for child in self.sca_status_frame.winfo_children():
                    child.config(bg='#95a5a6')
            else:  # Inactive, Error, etc.
                self.sca_status_frame.config(bg='#e74c3c')  # Red
                for child in self.sca_status_frame.winfo_children():
                    child.config(bg='#e74c3c')

    
    def open_plots_folder(self):
        """Open the plots folder in file manager"""
        try:
            import subprocess
            import platform
            
            # Look for sca_images directory
            possible_dirs = [
                '/home/atenea/gaflerna/evaluations/sca_images'
            ]
            
            plots_dir = None
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    plots_dir = dir_path
                    break
            
            if not plots_dir:
                # Create sca_images directory if it doesn't exist
                plots_dir = 'sca_images'
                os.makedirs(plots_dir, exist_ok=True)
            
            # Open directory based on platform
            system = platform.system()
            if system == "Windows":
                os.startfile(plots_dir)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", plots_dir])
            else:  # Linux
                subprocess.run(["xdg-open", plots_dir])
                
            self.log(f"Opened plots directory: {plots_dir}", "INFO")
            
        except Exception as e:
            self.log(f"Error opening plots folder: {str(e)}", "ERROR")

            
    def test_sca_connection(self):
        """Test connection to oscilloscope and router"""
        def test_connections():
            try:
                config = self.get_config()
                
                self.log("Testing SCA connections...", "INFO")
                self.raw_print("=== TESTING SCA CONNECTIONS ===")
                
                # Test router connection
                self.raw_print(f"Testing router connection: {config['router_ip']}")
                router_cmd = ['ping', '-c', '1', '-W', '3', config['router_ip']]
                router_result = subprocess.run(router_cmd, capture_output=True, text=True)
                
                if router_result.returncode == 0:
                    self.raw_print(f"Router {config['router_ip']} is reachable")
                    self.log(f"Router {config['router_ip']} is reachable", "SUCCESS")
                else:
                    self.raw_print(f"Router {config['router_ip']} is NOT reachable")
                    self.log(f"Router {config['router_ip']} is NOT reachable", "ERROR")
                
                # Test oscilloscope connection
                self.raw_print(f"Testing oscilloscope connection: {config['oscilloscope_ip']}")
                osc_cmd = ['ping', '-c', '1', '-W', '3', config['oscilloscope_ip']]
                osc_result = subprocess.run(osc_cmd, capture_output=True, text=True)
                
                if osc_result.returncode == 0:
                    self.raw_print(f"Oscilloscope {config['oscilloscope_ip']} is reachable")
                    self.log(f"Oscilloscope {config['oscilloscope_ip']} is reachable", "SUCCESS")
                else:
                    self.raw_print(f"Oscilloscope {config['oscilloscope_ip']} is NOT reachable")
                    self.log(f"Oscilloscope {config['oscilloscope_ip']} is NOT reachable", "ERROR")
                
                # Test target CGI
                self.raw_print(f"Testing target CGI: http://{config['router_ip']}/{config['target_cgi']}")
                cgi_cmd = ['curl', '-m', '5', '--connect-timeout', '3', 
                          f"http://{config['router_ip']}/{config['target_cgi']}"]
                cgi_result = subprocess.run(cgi_cmd, capture_output=True, text=True)
                
                if cgi_result.returncode == 0:
                    self.raw_print(f"Target CGI is accessible")
                    self.log("Target CGI is accessible", "SUCCESS")
                else:
                    self.raw_print(f"Target CGI is NOT accessible (this may be normal)")
                    self.log("Target CGI test failed (may be normal for protected endpoints)", "WARNING")
                
                self.raw_print("=== SCA CONNECTION TEST COMPLETED ===")
                
            except Exception as e:
                self.raw_print(f"Error testing connections: {str(e)}")
                self.log(f"Error testing SCA connections: {str(e)}", "ERROR")
        
        # Run test in separate thread
        threading.Thread(target=test_connections, daemon=True).start()

    def setup_logs_tab(self):
        """Setup logs tab"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="Logs")
        
        # Logs frame
        logs_main_frame = ttk.LabelFrame(logs_frame, text="Activity Log", padding="10")
        logs_main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Text widget for logs with scrollbar
        self.log_text = scrolledtext.ScrolledText(logs_main_frame, 
                                                 font=('Courier New', 10),
                                                 bg='#2c3e50', fg='#00ff00',
                                                 insertbackground='#00ff00',
                                                 wrap='word')
        self.log_text.pack(fill='both', expand=True)
        
        # Log buttons
        log_buttons_frame = tk.Frame(logs_frame, bg='#2c3e50')
        log_buttons_frame.pack(fill='x', padx=10, pady=5)
        
        clear_logs_btn = tk.Button(log_buttons_frame, text="Clear Logs", 
                                  command=self.clear_logs, bg='#6c757d', fg='white',
                                  font=('Arial', 10, 'bold'))
        clear_logs_btn.pack(side='left', padx=5)
        
        save_logs_btn = tk.Button(log_buttons_frame, text="Save Logs", 
                                 command=self.save_logs, bg='#17a2b8', fg='white',
                                 font=('Arial', 10, 'bold'))
        save_logs_btn.pack(side='left', padx=5)
        
    def setup_raw_terminal_tab(self):
        """Setup raw terminal output tab"""
        terminal_frame = ttk.Frame(self.notebook)
        self.notebook.add(terminal_frame, text="Raw Terminal")
        
        # Terminal frame
        terminal_main_frame = ttk.LabelFrame(terminal_frame, text="Raw System Output (Real Time)", padding="10")
        terminal_main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Text widget for raw terminal output
        self.raw_output = scrolledtext.ScrolledText(terminal_main_frame, 
                                                   font=('Courier New', 10),
                                                   bg='#000000', fg='#00ff00',
                                                   insertbackground='#00ff00',
                                                   wrap='word')
        self.raw_output.pack(fill='both', expand=True)
        
        # Command input area
        cmd_frame = tk.Frame(terminal_frame, bg='#2c3e50')
        cmd_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(cmd_frame, text="Manual Command:", bg='#2c3e50', fg='white', 
                font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
        self.cmd_entry = tk.Entry(cmd_frame, font=('Courier New', 10), width=50)
        self.cmd_entry.pack(side='left', padx=5, fill='x', expand=True)
        self.cmd_entry.bind('<Return>', self.execute_manual_command)
        
        execute_btn = tk.Button(cmd_frame, text="Execute", 
                               command=self.execute_manual_command, 
                               bg='#28a745', fg='white',
                               font=('Arial', 10, 'bold'))
        execute_btn.pack(side='left', padx=5)
        
        # Terminal buttons
        terminal_buttons_frame = tk.Frame(terminal_frame, bg='#2c3e50')
        terminal_buttons_frame.pack(fill='x', padx=10, pady=5)
        
        clear_terminal_btn = tk.Button(terminal_buttons_frame, text="Clear", 
                                      command=self.clear_raw_output, bg='#dc3545', fg='white',
                                      font=('Arial', 10, 'bold'))
        clear_terminal_btn.pack(side='left', padx=5)
        
        test_countdown_btn = tk.Button(terminal_buttons_frame, text="Test 10s Countdown", 
                                      command=self.test_countdown, bg='#ffc107', fg='black',
                                      font=('Arial', 10, 'bold'))
        test_countdown_btn.pack(side='left', padx=5)
        
    def setup_sca_plots_tab(self):
        """Setup SCA Plots visualization tab - MODIFIED to include HTTP response browser"""
        plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(plots_frame, text="SCA Plots")
        
        # Initialize plot viewer
        self.plot_viewer = TRENTIPlotViewer(plots_frame)
        
        # SCA control frame
        sca_control_frame = ttk.LabelFrame(plots_frame, text="SCA Analysis Control", padding="5")
        sca_control_frame.pack(fill='x', padx=5, pady=5)
        
        # SCA status and controls in a single row
        control_row = tk.Frame(sca_control_frame)
        control_row.pack(fill='x')
        
        # SCA status indicator
        self.sca_status_frame = tk.Frame(control_row, bg='#e74c3c', relief='raised', bd=2)
        self.sca_status_frame.pack(side='left', padx=5, pady=2)
        
        tk.Label(self.sca_status_frame, text="SCA Status:", bg='#e74c3c', fg='white', 
                font=('Arial', 9, 'bold')).pack(side='left', padx=5)
        
        self.sca_status_var = tk.StringVar(value="Inactive")
        tk.Label(self.sca_status_frame, textvariable=self.sca_status_var, bg='#e74c3c', fg='white', 
                font=('Arial', 9, 'bold')).pack(side='left', padx=5)
        
        # Refresh plots button
        refresh_plots_btn = tk.Button(control_row, text="Refresh", 
                                     command=self.plot_viewer.refresh_plots, 
                                     bg='#17a2b8', fg='white',
                                     font=('Arial', 9, 'bold'))
        refresh_plots_btn.pack(side='left', padx=5)
        
        # Open plots folder button
        open_folder_btn = tk.Button(control_row, text="Open Plots Folder", 
                                   command=self.open_plots_folder, 
                                   bg='#6c757d', fg='white',
                                   font=('Arial', 9, 'bold'))
        open_folder_btn.pack(side='left', padx=5)
        
        # NEW: Open HTTP responses folder button
        open_http_btn = tk.Button(control_row, text="Open HTTP Responses", 
                                 command=self.open_http_responses_folder, 
                                 bg='#28a745', fg='white',
                                 font=('Arial', 9, 'bold'))
        open_http_btn.pack(side='left', padx=5)
        
        # NEW: Open execution times folder button
        open_exec_btn = tk.Button(control_row, text="Open Exec Times", 
                                 command=self.open_exec_times_folder, 
                                 bg='#fd7e14', fg='white',
                                 font=('Arial', 9, 'bold'))
        open_exec_btn.pack(side='left', padx=5)
        
        # Auto-analysis checkbox
        self.auto_analysis_var = tk.BooleanVar(value=True)
        auto_analysis_check = tk.Checkbutton(control_row, text="Auto-run SCA after campaigns", 
                                           variable=self.auto_analysis_var,
                                           font=('Arial', 9))
        auto_analysis_check.pack(side='left', padx=10)
        
        # Plot counter
        self.plot_counter_label = tk.Label(control_row, text="Plots: 0", 
                                         font=('Arial', 9), fg='#666666')
        self.plot_counter_label.pack(side='right', padx=5)
        

    def open_http_responses_folder(self):
        """Open the HTTP responses folder in file manager"""
        try:
            import subprocess
            import platform
            
            config = self.get_config()
            firmadyne_id = config.get('firmadyne_id', '')
            host_path = config.get('host_path', '')
            
            if host_path and firmadyne_id:
                http_responses_path = f"{host_path}/image_{firmadyne_id}/HTTP_responses/"
            else:
                http_responses_path = f"/home/atenea/trenti/evaluations/image_161160/HTTP_responses/"
            
            # Create directory if it doesn't exist
            os.makedirs(http_responses_path, exist_ok=True)
            
            # Open directory based on platform
            system = platform.system()
            if system == "Windows":
                os.startfile(http_responses_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", http_responses_path])
            else:  # Linux
                subprocess.run(["xdg-open", http_responses_path])
                
            self.log(f"Opened HTTP responses directory: {http_responses_path}", "INFO")
            
        except Exception as e:
            self.log(f"Error opening HTTP responses folder: {str(e)}", "ERROR")
    
    def open_exec_times_folder(self):
        """Open the execution times folder in file manager"""
        try:
            import subprocess
            import platform
            
            config = self.get_config()
            firmadyne_id = config.get('firmadyne_id', '')
            host_path = config.get('host_path', '')
            
            if host_path and firmadyne_id:
                exec_times_path = f"{host_path}/image_{firmadyne_id}/exec_times/"
            else:
                exec_times_path = f"/home/atenea/trenti/evaluations/image_161160/exec_times/"
            
            # Create directory if it doesn't exist
            os.makedirs(exec_times_path, exist_ok=True)
            
            # Open directory based on platform
            system = platform.system()
            if system == "Windows":
                os.startfile(exec_times_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", exec_times_path])
            else:  # Linux
                subprocess.run(["xdg-open", exec_times_path])
                
            self.log(f"Opened execution times directory: {exec_times_path}", "INFO")
            
        except Exception as e:
            self.log(f"Error opening execution times folder: {str(e)}", "ERROR")

    def on_test_threshold_change(self):
        """Handle test threshold checkbox change"""
        if self.test_threshold_var.get():
            # Test mode enabled
            self.threshold_info_label.config(text="Current threshold: 5 seconds (TEST MODE)", fg='#e74c3c')
            self.test_warning_label.pack(anchor='w')
            self.log("Test threshold mode ENABLED - 5 second threshold", "WARNING")
        else:
            # Normal mode
            self.threshold_info_label.config(text="Current threshold: 4320 seconds (1.2 hours)", fg='#666666')
            self.test_warning_label.pack_forget()
            self.log("Test threshold mode DISABLED - normal 1.2 hour threshold", "INFO")

    def log(self, message, level="INFO"):
        """Add message to log - SAFER VERSION with colors"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # ALWAYS print to console first (safer)
        console_msg = f"[TRENTI-{level}] {message}"
        print(console_msg, flush=True)
        
        # Try to update GUI safely - catch all tkinter errors
        try:
            # Check if we're in main thread
            if threading.current_thread() == threading.main_thread():
                # Direct update in main thread with colors
                start_index = self.log_text.index(tk.END + "-1c linestart")
                self.log_text.insert(tk.END, log_entry)
                end_index = self.log_text.index(tk.END + "-1c")
                
                # Apply colors
                if level == "ERROR":
                    self.log_text.tag_add("error", start_index, end_index)
                    self.log_text.tag_config("error", foreground="#ff6b6b")
                elif level == "WARNING":
                    self.log_text.tag_add("warning", start_index, end_index)
                    self.log_text.tag_config("warning", foreground="#ffa726")
                elif level == "SUCCESS":
                    self.log_text.tag_add("success", start_index, end_index)
                    self.log_text.tag_config("success", foreground="#66bb6a")
                elif level == "DEBUG":
                    self.log_text.tag_add("debug", start_index, end_index)
                    self.log_text.tag_config("debug", foreground="#78909c")
                
                self.log_text.see(tk.END)
            else:
                # Schedule update in main thread
                self.root.after_idle(lambda: self._safe_log_update(log_entry, level))
                
        except Exception as e:
            # If GUI fails, just continue with console output
            print(f"[TRENTI-WARNING] GUI log failed: {str(e)}", flush=True)        
    
    def _safe_log_update(self, log_entry, level):
            """Safely update log in main thread with colors"""
            try:
                # Insert the log entry
                start_index = self.log_text.index(tk.END + "-1c linestart")
                self.log_text.insert(tk.END, log_entry)
                end_index = self.log_text.index(tk.END + "-1c")
                
                # Apply color based on level
                if level == "ERROR":
                    self.log_text.tag_add("error", start_index, end_index)
                    self.log_text.tag_config("error", foreground="#ff6b6b")  # Red
                elif level == "WARNING":
                    self.log_text.tag_add("warning", start_index, end_index)
                    self.log_text.tag_config("warning", foreground="#ffa726")  # Orange
                elif level == "SUCCESS":
                    self.log_text.tag_add("success", start_index, end_index)
                    self.log_text.tag_config("success", foreground="#66bb6a")  # Green
                elif level == "DEBUG":
                    self.log_text.tag_add("debug", start_index, end_index)
                    self.log_text.tag_config("debug", foreground="#78909c")  # Gray
                # INFO mantiene el color por defecto (verde)
                
                self.log_text.see(tk.END)
            except Exception:
                # Ignore GUI errors completely
                pass
            
    def clear_logs(self):
        """Clear logs"""
        self.log_text.delete(1.0, tk.END)
        self.log("Logs cleared", "INFO")
        
    def save_logs(self):
        """Save logs to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save logs"
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.log(f"Logs saved to: {filename}", "SUCCESS")
            except Exception as e:
                self.log(f"Error saving logs: {str(e)}", "ERROR")
                
    def raw_print(self, text):
        """Print to raw terminal window immediately"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        message = f"[{timestamp}] {text}"
        
        # Print to console IMMEDIATELY
        print(text)
        import sys
        sys.stdout.flush()
        
        # Add to raw terminal window
        try:
            self.raw_output.insert(tk.END, message + "\n")
            self.raw_output.see(tk.END)
            self.raw_output.update()  # Force immediate update
        except:
            pass
            
    def execute_manual_command(self, event=None):
        """Execute manual command in raw terminal"""
        cmd = self.cmd_entry.get().strip()
        if not cmd:
            return
            
        self.raw_print(f">>> EXECUTING: {cmd}")
        self.cmd_entry.delete(0, tk.END)
        
        # Execute command and show real output
        def run_cmd():
            try:
                if cmd.startswith('cd '):
                    self.raw_print("Note: cd commands don't work in subprocess. Use full paths.")
                    return
                    
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                                         stderr=subprocess.STDOUT, text=True, bufsize=1, 
                                         universal_newlines=True)
                
                for line in iter(process.stdout.readline, ''):
                    if line:
                        self.raw_print(line.rstrip())
                        
                process.wait()
                self.raw_print(f">>> COMMAND FINISHED (exit code: {process.returncode})")
                
            except Exception as e:
                self.raw_print(f">>> ERROR: {str(e)}")
                
        # Run in thread so GUI doesn't freeze
        threading.Thread(target=run_cmd, daemon=True).start()   
        
    def test_countdown(self):
        """Test countdown function"""
        self.raw_print("=== TESTING 10 SECOND COUNTDOWN ===")
        
        def countdown():
            for i in range(10, 0, -1):
                self.raw_print(f"COUNTDOWN: {i} seconds remaining")
                time.sleep(1)
            self.raw_print("COUNTDOWN COMPLETED")
            
        threading.Thread(target=countdown, daemon=True).start()
        
    def clear_raw_output(self):
        """Clear raw terminal output"""
        self.raw_output.delete(1.0, tk.END)
        self.raw_print("Raw terminal cleared")
                
    def get_config(self):
        """Get current configuration - MODIFIED to include custom curl command"""
        try:
            # Obtener duración total en segundos
            total_hours = float(self.config_vars['total_duration_hours'].get())
            total_duration_seconds = int(total_hours * 3600)
            
            # Get curl command from text widget
            custom_curl_command = self.get_curl_command()
                                    
            return {
                'firmadyne_id': self.config_vars['firmadyne_id'].get(),
                'container_name': self.config_vars['container_name'].get(),
                'host_path': self.config_vars['host_path'].get(),
                'router_ip': self.config_vars['router_ip'].get(),
                'oscilloscope_ip': self.config_vars['oscilloscope_ip'].get(),
                'target_cgi': self.config_vars['target_cgi'].get(),
                'total_duration_hours': total_hours,
                'total_duration_seconds': total_duration_seconds,
                'monitoring_interval': int(self.config_vars['monitoring_interval'].get()),
                'enable_sca': self.enable_sca_var.get(),
                'sca_channel': self.config_vars['sca_channel'].get(),
                'sca_voltage_div': self.config_vars['sca_voltage_div'].get(),
                'sca_time_div': self.config_vars['sca_time_div'].get(),
                'sca_sample_rate': self.config_vars['sca_sample_rate'].get(),
                'sca_memory_size': self.config_vars['sca_memory_size'].get(),
                # NEW: Custom curl command
                'custom_curl_command': custom_curl_command
            }
        except ValueError as e:
            raise ValueError(f"Invalid configuration: {str(e)}")
        

    def update_ui(self):
        """Update UI with enhanced time information"""
        try:
            config = self.get_config()
            state = self.automation_state
            
            # Update metrics
            self.metric_vars['current_campaign'].set(str(state['current_campaign']))
            self.metric_vars['sca_anomalies_count'].set(str(state.get('anomalies_count', 0)))
            self.metric_vars['exec_time_anomalies'].set(str(state.get('exec_time_anomalies', 0)))
            self.metric_vars['http_response_anomalies'].set(str(state.get('http_response_anomalies', 0)))
            self.metric_vars['total_anomalies_found'].set(str(state.get('total_anomalies_found', 0)))
            
            # Update elapsed time with ENHANCED duration info
            if state['total_start_time']:
                total_elapsed = time.time() - state['total_start_time']
                hours, remainder = divmod(total_elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                # Show total elapsed time
                total_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                self.metric_vars['elapsed_time'].set(total_time_str)
                
                # Calculate progress and remaining time
                if state['total_duration_seconds'] > 0:
                    progress_percent = min((total_elapsed / state['total_duration_seconds']) * 100, 100)
                    self.progress_var.set(progress_percent)
                    
                    remaining_seconds = max(0, state['total_duration_seconds'] - total_elapsed)
                    remaining_hours, remainder = divmod(remaining_seconds, 3600)
                    remaining_minutes, remaining_secs = divmod(remainder, 60)
                    
                    # Enhanced progress text with threshold warning
                    if remaining_seconds <= 0:
                        progress_text = f"100% - DURATION THRESHOLD REACHED"
                        self.progress_label.config(fg='#e74c3c')  # Red
                    elif remaining_seconds < 1800:  # Less than 30 minutes
                        progress_text = f"{progress_percent:.1f}% - ENDING SOON: {int(remaining_hours):02d}:{int(remaining_minutes):02d}:{int(remaining_secs):02d}"
                        self.progress_label.config(fg='#f39c12')  # Orange
                    else:
                        progress_text = f"{progress_percent:.1f}% - Remaining: {int(remaining_hours):02d}:{int(remaining_minutes):02d}:{int(remaining_secs):02d}"
                        self.progress_label.config(fg='#2c3e50')  # Normal
                        
                    self.progress_label.config(text=progress_text)
                else:
                    self.progress_var.set(0)
                    self.progress_label.config(text="0% - Waiting for start")
            
            # Update status
            if state['is_running']:
                status = "Paused" if state['is_paused'] else "Running"
                color = "#f39c12" if state['is_paused'] else "#27ae60"
            else:
                status = "Inactive"
                color = "#7f8c8d"
                
            self.status_label.config(text=f"Status: {status}", fg=color)
            self.metric_vars['container_status'].set(status)
                
        except Exception as e:
            # Silently handle UI update errors
            pass

    def validate_time_config(self):
        """Validate time-based configuration"""
        try:
            total_hours = float(self.config_vars['total_duration_hours'].get())
            
            if total_hours <= 0:
                raise ValueError("Total duration must be greater than 0 hours")
                
            total_seconds = total_hours * 3600
            
            self.log(f"Configuration validated:", "INFO")
            self.log(f"  Total duration: {total_hours} hours ({total_seconds:.0f} seconds)", "INFO")
            
            return True
            
        except ValueError as e:
            raise ValueError(f"Time configuration error: {str(e)}")
        
    def start_automation(self):
        """Start automation - MODIFIED for time-based execution"""
        if self.automation_state['is_running']:
            self.log("TRENTI is already running", "WARNING")
            return
            
        # Validate configuration
        try:
            config = self.get_config()
            if not all([config['firmadyne_id'], config['container_name'], config['host_path']]):
                raise ValueError("Required fields are missing")
        except Exception as e:
            self.log(f"Configuration error: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Invalid configuration: {str(e)}")
            return
            
        self.automation_state['is_running'] = True
        self.automation_state['is_paused'] = False
        self.automation_state['start_time'] = time.time()
        self.automation_state['total_start_time'] = time.time()  # NUEVO
        self.automation_state['total_duration_seconds'] = config['total_duration_seconds']  # NUEVO
        self.automation_state['elapsed_fuzzing_time'] = 0  # NUEVO
        self.automation_state['current_campaign'] = 0
        
        # Update buttons
        self.start_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        self.stop_btn.config(state='normal')
        
        self.log(f"Starting TRENTI - Total duration: {config['total_duration_hours']} hours ({config['total_duration_seconds']} seconds)", "SUCCESS")
        
        # Start automation thread
        self.automation_thread = threading.Thread(target=self.run_automation, daemon=True)
        self.automation_thread.start()
        
        # Start UI updater
        self.start_ui_updater()
        
    def pause_automation(self):
        """Pause/resume automation"""
        if not self.automation_state['is_running']:
            return
            
        self.automation_state['is_paused'] = not self.automation_state['is_paused']
        
        if self.automation_state['is_paused']:
            self.pause_btn.config(text="Resume")
            self.log("Automation paused", "WARNING")
        else:
            self.pause_btn.config(text="Pause")
            self.log("Automation resumed", "INFO")
            
    def stop_automation(self):
        """Stop automation"""
        if not self.automation_state['is_running']:
            return
            
        self.automation_state['is_running'] = False
        self.automation_state['is_paused'] = False
        
        # Clean up processes
        self.cleanup_processes()
                
        # Update buttons
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled', text="Pause")
        self.stop_btn.config(state='disabled')
        
        self.log("Automation stopped", "WARNING")
        
    def start_ui_updater(self):
        """Start UI updater"""
        def update():
            if self.automation_state['is_running']:
                self.update_ui()
                self.root.after(1000, update)  # Update every second
                
        update()
        
    def run_automation(self):
        """Run complete automation process - MODIFIED for duration-based control"""
        try:
            self.log("Starting TRENTI automation process...", "DEBUG")
            config = self.get_config()
            self.log(f"Configuration loaded: {config}", "DEBUG")
            
            # Step 1: Launch container
            self.log("Step 1: Launching Docker container...", "INFO")
            if not self.launch_docker_container(config):
                raise Exception("Error launching Docker container")
                
            # Main campaign loop with DURATION CONTROL
            campaign_num = 1
            total_start_time = self.automation_state['total_start_time']
            total_duration = self.automation_state['total_duration_seconds']
            
            self.log(f"Starting automation with {config['total_duration_hours']}h total duration", "INFO")
            self.raw_print(f"AUTOMATION START - Total duration: {config['total_duration_hours']} hours")
            
            while self.automation_state['is_running']:
                # CRITICAL: Check total duration BEFORE starting new campaign
                elapsed_total = time.time() - total_start_time
                remaining_time = total_duration - elapsed_total
                
                self.raw_print(f"=== CAMPAIGN {campaign_num} CHECK ===")
                self.raw_print(f"Total elapsed: {elapsed_total/3600:.2f} hours")
                self.raw_print(f"Remaining: {remaining_time/3600:.2f} hours")
                
                if remaining_time <= 0:
                    self.log("Total duration reached before starting new campaign - stopping automation", "SUCCESS")
                    self.raw_print("=== TOTAL DURATION REACHED ===")
                    self.raw_print("No time left for new campaigns")
                    break
                
                # Check if we have enough time for a meaningful campaign (at least 10 minutes)
                if remaining_time < 600:  # Less than 10 minutes
                    self.log(f"Insufficient time for new campaign ({remaining_time:.1f}s remaining) - stopping", "INFO")
                    self.raw_print(f"Insufficient time for new campaign: {remaining_time:.1f}s remaining")
                    break
                    
                # Simple cleanup between campaigns (except the first)
                if campaign_num > 1:
                    self.log(f"Performing simple cleanup before campaign {campaign_num}...", "INFO")
                    if not self.cleanup_between_campaigns(config):
                        self.log("Cleanup failed, but continuing...", "WARNING")
                                      
                self.automation_state['current_campaign'] = campaign_num
                self.log(f"Starting campaign {campaign_num} - Remaining total: {remaining_time:.1f}s", "INFO")
                
                # Add campaign to tracker
                self.root.after(0, lambda cn=campaign_num: self.add_campaign_to_tracker(cn, "Starting"))
                
                # Execute campaign with duration awareness
                campaign_start_time = time.time()
                campaign_success = self.run_single_campaign(config, campaign_num)
                
                if not campaign_success:
                    self.log(f"Error in campaign {campaign_num}", "ERROR")
                    self.root.after(0, lambda cn=campaign_num: self.update_campaign_status(cn, "Error"))
                    
                    # Check if we should continue or stop on error
                    if not self.automation_state['is_running']:
                        break
                        
                    campaign_num += 1
                    continue
                    
                # Update elapsed fuzzing time
                campaign_elapsed = time.time() - campaign_start_time
                self.automation_state['elapsed_fuzzing_time'] += campaign_elapsed
                
                self.root.after(0, lambda cn=campaign_num: self.update_campaign_status(cn, "Completed"))
                self.log(f"Campaign {campaign_num} completed in {campaign_elapsed:.1f}s", "SUCCESS")
                
                campaign_num += 1
                
            # Final information
            total_elapsed = time.time() - total_start_time
            self.log(f"Automation completed - Total time: {total_elapsed:.1f}s, Fuzzing time: {self.automation_state['elapsed_fuzzing_time']:.1f}s", "SUCCESS")
            self.log(f"Campaigns executed: {campaign_num - 1}", "INFO")
            
            # Final stats
            total_hours = total_elapsed / 3600
            fuzzing_hours = self.automation_state['elapsed_fuzzing_time'] / 3600
            self.raw_print(f"=== FINAL AUTOMATION STATS ===")
            self.raw_print(f"Total time: {total_hours:.2f} hours")
            self.raw_print(f"Fuzzing time: {fuzzing_hours:.2f} hours")
            self.raw_print(f"Campaigns: {campaign_num - 1}")
            self.raw_print(f"Total anomalies: {self.automation_state.get('total_anomalies_found', 0)}")
            
        except Exception as e:
            self.log(f"Critical error in automation: {str(e)}", "ERROR")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
        finally:
            # Final cleanup
            try:
                config = self.get_config()
                self.cleanup_between_campaigns(config)
            except:
                pass
            self.cleanup_processes()
            self.automation_state['is_running'] = False
            self.root.after(0, lambda: [
                self.start_btn.config(state='normal'),
                self.pause_btn.config(state='disabled', text="Pause"),
                self.stop_btn.config(state='disabled')
            ])


    def launch_docker_container(self, config):
        """Launch Docker container exactly like manual process"""
        try:
            self.log("Checking for existing container...", "DEBUG")
            
            # Clean existing container if exists
            cleanup_cmd = ['docker', 'rm', '-f', config['container_name']]
            cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True)
            if cleanup_result.returncode == 0:
                self.log(f"Existing container removed", "DEBUG")
            
            # Launch new container EXACTLY like manual
            cmd = [
                'docker', 'run', '-it', '--name', config['container_name'],
                '--env', 'USER=root', '--privileged', '--device=/dev/net/tun',
                '-d', 'zyw200/firmfuzzer', '/bin/bash'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log(f"Docker launch failed: {result.stderr}", "ERROR")
                return False
                
            self.automation_state['container_id'] = result.stdout.strip()
            self.log(f"Container {config['container_name']} launched successfully", "SUCCESS")
            
            # Setup Docker API connection
            if self.setup_docker_connection(config):
                self.log("Using optimized Docker API for commands", "SUCCESS")
            else:
                self.log("Using subprocess fallback for commands", "WARNING")
            
            # Setup special environments EXACTLY like manual
            if config['firmadyne_id'] in ['161160', '161161']:
                self.log("Setting up environment exactly like manual process...", "INFO")
                if not self.setup_environment_like_manual(config):
                    self.log("Environment setup failed", "ERROR")
                    return False
            
            return True
            
        except Exception as e:
            self.log(f"Error launching container: {e}", "ERROR")
            return False


    def setup_environment_like_manual(self, config):
        """Setup environment exactly like manual process"""
        try:
            container_name = config['container_name']
            firmadyne_id = config['firmadyne_id']
            
            self.log("Setting up environment EXACTLY like manual process...", "INFO")
            
            # PASO 1: Copiar image_161160 (siempre la base)
            self.log("Step 1: Copying image_161160 base (like manual)...", "INFO")
            base_copy_cmd = ['docker', 'cp', 'images/image_161160', f'{container_name}:/test/']
            
            self.raw_print(f"MANUAL STEP 1: {' '.join(base_copy_cmd)}")
            base_result = subprocess.run(base_copy_cmd, capture_output=True, text=True)
            
            if base_result.returncode != 0:
                self.log(f"Failed to copy base image: {base_result.stderr}", "ERROR")
                return False
            
            self.log(" Base image_161160 copied successfully", "SUCCESS")
            
            # PASO 2: Para 161161, copiar archivos específicos
            if firmadyne_id == "161161":
                self.log("Step 2: Copying 161161-specific files (like manual)...", "INFO")
                
                # Copiar inputs/ específico
                inputs_copy_cmd = ['docker', 'cp', 'images/image_161161/inputs', f'{container_name}:/test/image_161160/']
                self.raw_print(f"MANUAL STEP 2a: {' '.join(inputs_copy_cmd)}")
                inputs_result = subprocess.run(inputs_copy_cmd, capture_output=True, text=True)
                
                if inputs_result.returncode == 0:
                    self.log(" 161161 inputs copied successfully", "SUCCESS")
                else:
                    self.log(f"Failed to copy 161161 inputs: {inputs_result.stderr}", "ERROR")
                    return False
                
                # Copiar user.sh específico
                user_sh_copy_cmd = ['docker', 'cp', 'images/image_161161/user.sh', f'{container_name}:/test/image_161160/']
                self.raw_print(f"MANUAL STEP 2b: {' '.join(user_sh_copy_cmd)}")
                user_sh_result = subprocess.run(user_sh_copy_cmd, capture_output=True, text=True)
                
                if user_sh_result.returncode == 0:
                    self.log(" 161161 user.sh copied successfully", "SUCCESS")
                else:
                    self.log(f"Failed to copy 161161 user.sh: {user_sh_result.stderr}", "ERROR")
                    return False
            
            # PASO 3: Verificar que todo está en su lugar
            self.log("Step 3: Verifying setup like manual...", "INFO")
            
            # Verificar que image_161160 existe y tiene contenido
            verify_cmd = ['docker', 'exec', container_name, 'ls', '-la', '/test/image_161160/']
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
            
            if verify_result.returncode == 0:
                self.raw_print("Contents of /test/image_161160/ after manual setup:")
                self.raw_print(verify_result.stdout)
                
                # Verificar archivos críticos
                critical_files = ['run.sh', 'test.py', 'user.sh', 'inputs', 'afl-fuzz']
                missing_files = []
                
                for file in critical_files:
                    if file not in verify_result.stdout:
                        missing_files.append(file)
                
                if missing_files:
                    self.log(f"Missing critical files: {missing_files}", "ERROR")
                    return False
                else:
                    self.log(" All critical files present", "SUCCESS")
            else:
                self.log("Failed to verify setup", "ERROR")
                return False
            
            self.log("Environment setup completed like manual process", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Error in manual-like setup: {e}", "ERROR")
            return False


    def run_single_campaign(self, config, campaign_num):
        """Run campaign with CORRECTED directory handling for 161161"""
        try:
            container_name = config['container_name']
            firmadyne_id = config['firmadyne_id']
            
            # FIXED: Para 161161, SIEMPRE usar image_161160 (como en manual)
            if firmadyne_id == "161161":
                image_dir = "/test/image_161160"  # EXACTAMENTE como en manual
                output_dir = f"trenti_sca_outputs_161161_{campaign_num}"
            else:
                image_dir = f"/test/image_{firmadyne_id}"
                output_dir = f"trenti_sca_outputs_{firmadyne_id}_{campaign_num}"
            
            self.log(f"Starting campaign {campaign_num} - Working dir: {image_dir}", "INFO")
            self.automation_state['fuzzing_start_time'] = time.time()
            
            # MANUAL SEQUENCE: cd image_161160 && ./run.sh
            self.log("MANUAL: cd + ./run.sh", "INFO")
            self.raw_print(f"MANUAL SEQUENCE: cd {image_dir} && ./run.sh")
            
            # Execute run.sh in background (EXACTLY like manual)
            if self.container:
                self.container.exec_run(f'bash -c "cd {image_dir} && ./run.sh"', detach=True)
            else:
                cmd = ['docker', 'exec', container_name, 'bash', '-c', f'cd {image_dir} && ./run.sh &']
                subprocess.Popen(cmd)
            
            self.raw_print("run.sh started in background")
            
            # MANUAL: Wait 60 seconds
            self.log("MANUAL: Wait 60 seconds", "INFO")
            for i in range(60, 0, -1):
                if not self.automation_state['is_running']:
                    break
                if i % 10 == 0:
                    self.raw_print(f"Manual wait: {i} seconds remaining...")
                time.sleep(1)
            
            # MANUAL: python test.py
            self.log("MANUAL: python test.py", "INFO")
            self.raw_print("MANUAL: python test.py")
            
            if self.container:
                result = self.container.exec_run(f'bash -c "cd {image_dir} && python test.py"')
                success = (result.exit_code == 0)
            else:
                cmd = ['docker', 'exec', container_name, 'bash', '-c', f'cd {image_dir} && python test.py']
                result = subprocess.run(cmd, capture_output=True, text=True)
                success = (result.returncode == 0)
            
            if not success:
                self.log("test.py failed", "ERROR")
                return False
            
            self.log("test.py completed", "SUCCESS")
            
            # Configure user.sh for correct output
            if not self.configure_user_sh(config, output_dir, image_dir):
                return False
            
            # MANUAL: ./user.sh
            self.log("MANUAL: ./user.sh", "INFO")
            self.raw_print("MANUAL: ./user.sh")
            
            if self.container:
                self.container.exec_run(f'bash -c "cd {image_dir} && ./user.sh"', detach=True)
            else:
                cmd = ['docker', 'exec', container_name, 'bash', '-c', f'cd {image_dir} && ./user.sh &']
                subprocess.Popen(cmd)
            
            self.raw_print("user.sh started in background")
            
            # Wait for AFL to start
            self.log("Waiting for AFL to start...", "INFO")
            for i in range(120, 0, -1):
                if not self.automation_state['is_running']:
                    break
                
                if i % 15 == 0:
                    if self.check_afl_fuzz_running(container_name):
                        self.raw_print("AFL-FUZZ IS RUNNING!")
                        break
                    else:
                        self.raw_print(f"Waiting for AFL... {i}s remaining")
                
                time.sleep(1)
            
            # Continue with rest of campaign...
            if not self.configure_pause_monitor(config['host_path'], output_dir, image_dir):
                monitor_process = None
            else:
                monitor_process = self.start_pause_monitor()
            
            if not self.monitor_fuzzing_campaign_timed(config, campaign_num, output_dir, monitor_process, image_dir):
                return False
            
            if config['enable_sca']:
                if not self.run_anomaly_detection(config, campaign_num):
                    self.log("SCA failed, but campaign continues", "WARNING")
            
            return True
            
        except Exception as e:
            self.log(f"Error in campaign {campaign_num}: {e}", "ERROR")
            return False


    def check_afl_startup_status(self, container_name, image_dir):
        """Verificar estado de startup de AFL y mostrar información útil"""
        try:
            # Verificar procesos relacionados con AFL
            check_procs_cmd = ['docker', 'exec', container_name, 'ps', 'auxww']
            proc_result = subprocess.run(check_procs_cmd, capture_output=True, text=True)
            
            if proc_result.returncode == 0:
                afl_procs = []
                for line in proc_result.stdout.split('\n'):
                    if any(keyword in line.lower() for keyword in ['afl', 'user.sh', 'qemu']):
                        afl_procs.append(line.strip())
                
                if afl_procs:
                    self.raw_print("=== CURRENT AFL-RELATED PROCESSES ===")
                    for proc in afl_procs:
                        self.raw_print(f"PROC: {proc}")
                else:
                    self.raw_print("No AFL-related processes found")
            
            # Verificar si se ha creado el directorio de output
            check_output_cmd = ['docker', 'exec', container_name, 'ls', '-la', image_dir]
            output_result = subprocess.run(check_output_cmd, capture_output=True, text=True)
            
            if output_result.returncode == 0:
                output_lines = output_result.stdout.split('\n')
                trenti_dirs = [line for line in output_lines if 'trenti_sca_outputs' in line]
                if trenti_dirs:
                    self.raw_print(f"=== FOUND OUTPUT DIRECTORIES ===")
                    for dir_line in trenti_dirs:
                        self.raw_print(f"DIR: {dir_line}")
            
            # Verificar últimas líneas de logs
            log_cmd = ['docker', 'logs', '--tail', '5', container_name]
            log_result = subprocess.run(log_cmd, capture_output=True, text=True)
            
            if log_result.returncode == 0 and log_result.stdout.strip():
                self.raw_print("=== RECENT CONTAINER LOGS ===")
                for line in log_result.stdout.split('\n')[-3:]:
                    if line.strip():
                        self.raw_print(f"LOG: {line.strip()}")
            
        except Exception as e:
            self.raw_print(f"Error checking AFL startup status: {str(e)}")


    def start_user_sh_with_monitoring(self, container_name, image_dir):
        """Iniciar user.sh con monitoreo de salida en tiempo real"""
        try:
            def monitor_user_sh():
                try:
                    # MOSTRAR contenido de user.sh justo antes de ejecutar
                    self.raw_print("=== ABOUT TO EXECUTE user.sh ===")
                    show_cmd = ['docker', 'exec', container_name, 'cat', f'{image_dir}/user.sh']
                    show_result = subprocess.run(show_cmd, capture_output=True, text=True)
                    if show_result.returncode == 0:
                        for i, line in enumerate(show_result.stdout.split('\n'), 1):
                            self.raw_print(f"EXEC[{i:02d}]: {line}")
                    
                    # MOSTRAR directorio de trabajo
                    pwd_cmd = ['docker', 'exec', container_name, 'bash', '-c', f'cd {image_dir} && pwd && ls -la']
                    pwd_result = subprocess.run(pwd_cmd, capture_output=True, text=True)
                    if pwd_result.returncode == 0:
                        self.raw_print("=== WORKING DIRECTORY ===")
                        self.raw_print(pwd_result.stdout)
                    
                    # Ejecutar user.sh y capturar output
                    cmd = ['docker', 'exec', container_name, 'bash', '-c', 
                          f'cd {image_dir} && bash -x ./user.sh 2>&1']
                    
                    self.raw_print(f"=== EXECUTING: {' '.join(cmd)} ===")
                    
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                             stderr=subprocess.STDOUT, text=True,
                                             bufsize=1, universal_newlines=True)
                    
                    line_count = 0
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            line_count += 1
                            output_line = line.strip()
                            
                            # Mostrar TODAS las líneas
                            self.raw_print(f"USER_SH[{line_count:04d}]: {output_line}")
                            
                            # Detectar líneas de AFL específicas
                            if 'afl-fuzz' in output_line and '-o' in output_line:
                                self.log(f"AFL COMMAND: {output_line}", "INFO")
                            
                            # Detectar fork server específicamente
                            if 'fork server' in output_line.lower():
                                self.log(f"FORK SERVER: {output_line}", "WARNING")
                            
                            # Detectar handshake
                            if 'handshake' in output_line.lower():
                                self.log(f"HANDSHAKE: {output_line}", "ERROR")
                    
                    process.wait()
                    self.raw_print(f"*** USER.SH FINISHED - EXIT CODE: {process.returncode} - TOTAL LINES: {line_count} ***")
                    
                except Exception as e:
                    self.raw_print(f"Error monitoring user.sh: {str(e)}")
            
            # Iniciar hilo de monitoreo
            import threading
            monitor_thread = threading.Thread(target=monitor_user_sh, daemon=True)
            monitor_thread.start()
            
            self.raw_print("user.sh monitoring thread started")
            
        except Exception as e:
            self.raw_print(f"Error starting user.sh monitor: {str(e)}")



    def check_afl_startup_status(self, container_name, image_dir):
        """Verificar estado de startup de AFL y mostrar información útil"""
        try:
            # Verificar procesos relacionados con AFL
            check_procs_cmd = ['docker', 'exec', container_name, 'ps', 'auxww']
            proc_result = subprocess.run(check_procs_cmd, capture_output=True, text=True)
            
            if proc_result.returncode == 0:
                afl_procs = []
                for line in proc_result.stdout.split('\n'):
                    if any(keyword in line.lower() for keyword in ['afl', 'user.sh', 'qemu']):
                        afl_procs.append(line.strip())
                
                if afl_procs:
                    self.raw_print("=== CURRENT AFL-RELATED PROCESSES ===")
                    for proc in afl_procs:
                        self.raw_print(f"PROC: {proc}")
                else:
                    self.raw_print("No AFL-related processes found")
            
            # Verificar si se ha creado el directorio de output
            check_output_cmd = ['docker', 'exec', container_name, 'ls', '-la', image_dir]
            output_result = subprocess.run(check_output_cmd, capture_output=True, text=True)
            
            if output_result.returncode == 0:
                output_lines = output_result.stdout.split('\n')
                trenti_dirs = [line for line in output_lines if 'trenti_sca_outputs' in line]
                if trenti_dirs:
                    self.raw_print(f"=== FOUND OUTPUT DIRECTORIES ===")
                    for dir_line in trenti_dirs:
                        self.raw_print(f"DIR: {dir_line}")
            
            # Verificar últimas líneas de logs
            log_cmd = ['docker', 'logs', '--tail', '5', container_name]
            log_result = subprocess.run(log_cmd, capture_output=True, text=True)
            
            if log_result.returncode == 0 and log_result.stdout.strip():
                self.raw_print("=== RECENT CONTAINER LOGS ===")
                for line in log_result.stdout.split('\n')[-3:]:
                    if line.strip():
                        self.raw_print(f"LOG: {line.strip()}")
            
        except Exception as e:
            self.raw_print(f"Error checking AFL startup status: {str(e)}")

  
    def configure_user_sh_simple(self, config, output_dir, working_dir):
        """Configure user.sh with FIXED sed command"""
        try:
            container_name = config['container_name']
            
            self.log("Configuring user.sh with FIXED method...", "INFO")
            
            # Verify file exists
            check_cmd = ['docker', 'exec', container_name, 'test', '-f', f'{working_dir}/user.sh']
            if subprocess.run(check_cmd, capture_output=True).returncode != 0:
                self.log(f"user.sh not found at {working_dir}/user.sh", "ERROR")
                return False
            
            # Show original content (first few lines)
            show_cmd = ['docker', 'exec', container_name, 'head', '-10', f'{working_dir}/user.sh']
            show_result = subprocess.run(show_cmd, capture_output=True, text=True)
            if show_result.returncode == 0:
                self.raw_print("Original user.sh (first 10 lines):")
                for i, line in enumerate(show_result.stdout.split('\n'), 1):
                    if line.strip():
                        self.raw_print(f"  {i:02d}: {line}")
            
            # FIXED SED: More specific pattern for AFL output directory
            # Look specifically for "-o ./something" and replace with our output_dir
            sed_cmd = f'sed -i "s|-o \\./[^[:space:]]*|-o ./{output_dir}|g" {working_dir}/user.sh'
            
            self.raw_print(f"EXECUTING FIXED SED: {sed_cmd}")
            
            exec_sed_cmd = ['docker', 'exec', container_name, 'bash', '-c', sed_cmd]
            sed_result = subprocess.run(exec_sed_cmd, capture_output=True, text=True)
            
            if sed_result.returncode != 0:
                self.log(f"SED command failed: {sed_result.stderr}", "ERROR")
                return False
            
            # Verify the change
            verify_cmd = ['docker', 'exec', container_name, 'grep', output_dir, f'{working_dir}/user.sh']
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
            
            if verify_result.returncode == 0:
                self.log(f" user.sh updated with output directory: {output_dir}", "SUCCESS")
                self.raw_print(f"Updated AFL command: {verify_result.stdout.strip()}")
            else:
                self.log("WARNING: Could not verify user.sh update", "WARNING")
            
            return True
            
        except Exception as e:
            self.log(f"Error in FIXED user.sh config: {e}", "ERROR")
            return False


    def configure_user_sh(self, config, output_dir, image_dir=None):
        """Configure user.sh using sed method directly"""
        try:
            container_name = config['container_name']
            firmadyne_id = config['firmadyne_id']
            
            # FIXED: Usar image_dir si se proporciona, sino calcular basado en firmadyne_id
            if image_dir is None:
                if firmadyne_id == "161161":
                    image_dir = "/test/image_161160"
                else:
                    image_dir = f"/test/image_{firmadyne_id}"
            
            self.log("Configuring user.sh with sed method (direct)...", "INFO")
            self.log(f"Working in directory: {image_dir}", "DEBUG")
            
            # Go directly to sed method since it's working
            return self.configure_user_sh_simple(config, output_dir, image_dir)
            
        except Exception as e:
            self.log(f"Error in user.sh config: {e}", "ERROR")
            return False


    def configure_pause_monitor(self, host_path, output_dir, image_dir=None):
        """Configure pause_monitor.py to run from HOST"""
        
        try:
            container_name = self.get_config()['container_name']
            firmadyne_id = self.get_config()['firmadyne_id']
            
            # FIXED: Usar image_dir si se proporciona
            if image_dir is None:
                if firmadyne_id == "161161":
                    image_dir_name = "image_161160"
                else:
                    image_dir_name = f"image_{firmadyne_id}"
            else:
                # Extraer solo el nombre del directorio
                image_dir_name = image_dir.split('/')[-1]
            
            # Check if pause_monitor.py exists in current directory
            if os.path.exists('pause_monitor.py'):
                self.log("Found existing pause_monitor.py, configuring for HOST execution", "INFO")
                
                # Read the existing file
                with open('pause_monitor.py', 'r') as f:
                    content = f.read()
                
                # Determine threshold based on test mode setting
                if self.test_threshold_var.get():
                    # Test mode - 5 seconds
                    threshold_value = 5
                    threshold_line = 'DEFAULT_THRESHOLD = 5  # TEST MODE - 5 seconds'
                    self.log("Using TEST THRESHOLD: 5 seconds", "WARNING")
                    self.raw_print("CONFIGURED FOR TEST MODE: 5 second threshold")
                else:
                    # Normal mode - 1.2 hours  
                    threshold_value = 4320
                    threshold_line = 'DEFAULT_THRESHOLD = 4320  # Normal mode - 1.2 hours'
                    self.log("Using NORMAL THRESHOLD: 4320 seconds (1.2 hours)", "INFO")
                    self.raw_print("CONFIGURED FOR NORMAL MODE: 1.2 hour threshold")
                
                # Update the configuration for TRENTI
                updated_content = content.replace(
                    'DEFAULT_STATS_FILE = "test/image_9050/nuberu_outputs/fuzzer_stats"',
                    f'DEFAULT_STATS_FILE = "test/{image_dir_name}/{output_dir}/fuzzer_stats"'
                ).replace(
                    "client.containers.get('nuberu')",
                    f"client.containers.get('{container_name}')"
                ).replace(
                    'DEFAULT_THRESHOLD = 4320  # seconds',
                    threshold_line
                )
                
                # FIXED: Create the configured script path
                import tempfile
                
                # Create a temporary file for the configured pause_monitor
                temp_fd, temp_path = tempfile.mkstemp(suffix='_pause_monitor.py', prefix='trenti_')
                os.close(temp_fd)  # Close the file descriptor
                
                # Write the updated content
                with open(temp_path, 'w') as f:
                    f.write(updated_content)
                
                # Make it executable
                os.chmod(temp_path, 0o755)
                
                # Store the path
                self.pause_monitor_script_path = temp_path
                
                self.log(f"pause_monitor.py configured for HOST execution: {self.pause_monitor_script_path}", "SUCCESS")
                self.raw_print(f"pause_monitor.py configured with {threshold_value}s threshold - container '{container_name}' - stats: test/{image_dir_name}/{output_dir}/fuzzer_stats")
                return True
                
            else:
                self.log("pause_monitor.py not found in current directory", "WARNING")
                self.raw_print("No pause_monitor.py found - continuing without advanced monitoring")
                return False
                
        except Exception as e:
            self.log(f"Error configuring pause_monitor: {str(e)}", "ERROR")
            return False

 
    def start_pause_monitor(self):
        """Start pause_monitor from HOST using Docker API"""
        try:
            # Check if we have a configured pause_monitor script
            if not hasattr(self, 'pause_monitor_script_path') or not os.path.exists(self.pause_monitor_script_path):
                self.log("No pause_monitor script configured, skipping", "WARNING")
                self.raw_print("No pause_monitor available - continuing without advanced monitoring")
                return None
            
            # Check if docker module is available
            self.raw_print("Checking if 'docker' module is available on HOST...")
            try:
                import docker
                self.raw_print("Docker module found on HOST - using REAL pause_monitor")
            except ImportError:
                self.raw_print("Docker module NOT found on HOST - attempting to install...")
                
                # Try to install docker module
                install_cmd = ['pip3', 'install', 'docker']
                install_result = subprocess.run(install_cmd, capture_output=True, text=True)
                
                if install_result.returncode == 0:
                    self.raw_print("Docker module installed successfully!")
                    # Try importing again
                    try:
                        import docker
                        self.raw_print("Docker module now available - proceeding")
                    except ImportError:
                        self.raw_print("Docker module still not available after install - using fallback")
                        return self.start_simple_monitor_fallback()
                else:
                    self.raw_print(f"Failed to install docker module: {install_result.stderr}")
                    self.raw_print("Using simple fallback monitor")
                    return self.start_simple_monitor_fallback()
            
            self.raw_print("Starting REAL pause_monitor from HOST using Docker API")
            
            # Execute pause_monitor from HOST
            cmd = ['python3', self.pause_monitor_script_path]
            self.log(f"Starting pause_monitor from HOST: {' '.join(cmd)}", "DEBUG")
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, text=True, 
                                     bufsize=1, universal_newlines=True)
            
            # Monitor pause_monitor output in separate thread - SEND TO SAME TERMINAL
            def monitor_pause_monitor():
                try:
                    line_count = 0
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            line_count += 1
                            output_line = line.rstrip()
                            
                            # Send pause_monitor output to SAME raw terminal as AFL
                            self.raw_print(f"MONITOR[{line_count:03d}]: {output_line}")
                            
                            # Determine log level based on content
                            log_level = "INFO"  # Default level
                            if any(keyword in output_line.upper() for keyword in ['ERROR', 'FAILED', 'CRITICAL']):
                                log_level = "ERROR"
                            elif any(keyword in output_line.upper() for keyword in ['WARNING', 'WARN', 'STUCK']):
                                log_level = "WARNING"
                            elif any(keyword in output_line.upper() for keyword in ['SUCCESS', 'COMPLETED', 'FOUND']):
                                log_level = "SUCCESS"
                            
                            # Also send to main logs with appropriate level
                            self.root.after(0, lambda msg=output_line, level=log_level: self.log(f"PAUSE MONITOR: {msg}", level))
                    
                    process.wait()
                    self.raw_print(f"*** PAUSE_MONITOR FINISHED FROM HOST - EXIT CODE: {process.returncode} - TOTAL LINES: {line_count} ***")
                    
                except Exception as e:
                    self.raw_print(f"ERROR monitoring pause_monitor: {str(e)}")
                    
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_pause_monitor, daemon=True)
            monitor_thread.start()
            
            self.automation_state['processes']['pause_monitor'] = process
            self.log("pause_monitor started successfully from HOST", "SUCCESS")
            self.raw_print("*** PAUSE_MONITOR RUNNING FROM HOST - OUTPUT WILL APPEAR WITH [MONITOR] PREFIX ***")
            return process
            
        except Exception as e:
            self.log(f"Error starting pause_monitor: {str(e)}", "ERROR")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return None
            
    def start_simple_monitor_fallback(self):
        """Start simple monitoring as fallback when Docker API is not available"""
        try:
            config = self.get_config()
            container_name = config['container_name']
            firmadyne_id = config['firmadyne_id']
            output_dir = f"trenti_sca_outputs_{firmadyne_id}_{config.get('current_campaign', 1)}"
            
            self.raw_print("Starting SIMPLE fallback monitor from HOST...")
            
            # Create simple monitoring script
            simple_monitor = f"""#!/usr/bin/env python3
import subprocess
import time
import re
from datetime import datetime

container_name = '{container_name}'
stats_file = '/test/image_{firmadyne_id}/{output_dir}/fuzzer_stats'

def get_stats():
    try:
        cmd = ['docker', 'exec', container_name, 'cat', stats_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        
        stats = {{}}
        content = result.stdout
        for key in ['paths_total', 'pending_total', 'last_path', 'last_update', 'unique_crashes', 'cycles_done']:
            match = re.search(rf'{{key}}\\s*:\\s*(\\d+)', content)
            if match:
                stats[key] = int(match.group(1))
        
        return stats if len(stats) == 6 else None
    except:
        return None

def main():
    print("[SIMPLE_MONITOR] Starting simple monitoring...")
    last_pending = float('inf')
    no_change_cycles = 0
    
    while no_change_cycles < 20:  # More cycles for simple monitor
        stats = get_stats()
        if stats:
            print(f"[SIMPLE_MONITOR] Stats: Paths={{stats['paths_total']}}, Pending={{stats['pending_total']}}, Crashes={{stats['unique_crashes']}}")
            
            if stats['pending_total'] < last_pending:
                last_pending = stats['pending_total']
                no_change_cycles = 0
            else:
                no_change_cycles += 1
                print(f"[SIMPLE_MONITOR] No progress for {{no_change_cycles}} cycles")
        else:
            print("[SIMPLE_MONITOR] Could not read stats")
            no_change_cycles += 1
            
        time.sleep(30)
    
    print("[SIMPLE_MONITOR] No progress detected, monitoring complete")

if __name__ == '__main__':
    main()
"""
            
            # Write simple monitor script
            simple_script_path = '/tmp/simple_monitor_SCA_anomaly_detection.py'
            with open(simple_script_path, 'w') as f:
                f.write(simple_monitor)
            
            # Execute simple monitor
            cmd = ['python3', simple_script_path]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, text=True, 
                                     bufsize=1, universal_newlines=True)
            
            # Monitor simple monitor output
            def monitor_simple():
                try:
                    line_count = 0
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            line_count += 1
                            output_line = line.rstrip()
                            self.raw_print(f"SIMPLE[{line_count:03d}]: {output_line}")
                            self.root.after(0, lambda msg=output_line: self.log(f"simple_monitor: {msg}", "INFO"))
                    
                    process.wait()
                    self.raw_print(f"*** SIMPLE MONITOR FINISHED - EXIT CODE: {process.returncode} ***")
                    
                except Exception as e:
                    self.raw_print(f"ERROR in simple monitor: {str(e)}")
                    
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_simple, daemon=True)
            monitor_thread.start()
            
            self.automation_state['processes']['pause_monitor'] = process
            self.log("Simple fallback monitor started", "SUCCESS")
            return process
            
        except Exception as e:
            self.log(f"Error starting simple monitor: {str(e)}", "ERROR")
            return None
            
    def detect_python_in_container(self, container_name):
        """Detect which Python interpreter is available in container"""
        try:
            python_options = ['python3', 'python', '/usr/bin/python3', '/usr/bin/python', 
                            '/usr/local/bin/python3', '/usr/local/bin/python']
            
            for python_cmd in python_options:
                # Test if this Python interpreter works
                test_cmd = ['docker', 'exec', container_name, python_cmd, '--version']
                result = subprocess.run(test_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    version_info = result.stdout.strip() or result.stderr.strip()
                    self.raw_print(f"Found Python: {python_cmd} - {version_info}")
                    return python_cmd
                    
            # If no standard Python found, try to find it
            find_cmd = ['docker', 'exec', container_name, 'find', '/usr', '/opt', 
                       '-name', 'python*', '-type', 'f', '-executable', '2>/dev/null']
            find_result = subprocess.run(find_cmd, capture_output=True, text=True)
            
            if find_result.returncode == 0 and find_result.stdout:
                python_paths = find_result.stdout.strip().split('\n')
                for path in python_paths:
                    if 'python' in path and not path.endswith('.so'):
                        # Test this path
                        test_cmd = ['docker', 'exec', container_name, path, '--version']
                        test_result = subprocess.run(test_cmd, capture_output=True, text=True)
                        if test_result.returncode == 0:
                            self.raw_print(f"Found Python at: {path}")
                            return path
                            
            return None
            
        except Exception as e:
            self.raw_print(f"Error detecting Python: {str(e)}")
            return None

            
    def monitor_fuzzing_campaign(self, config, campaign_num, output_dir, monitor_process):
        """Monitor fuzzing campaign with enhanced debugging"""
        try:
            start_time = time.time()
            check_interval = config['monitoring_interval']
            
            # self.log(f"Monitoring campaign for maximum {max_duration} seconds...", "INFO")
            self.raw_print(f"Starting fuzzing monitoring")
            
            # Determinar image_dir basado en firmadyne_id
            firmadyne_id = config['firmadyne_id']
            if firmadyne_id == "161161":
                image_dir = "/test/image_161160"
            else:
                image_dir = f"/test/image_{firmadyne_id}"
    
            # Give user.sh some time to start afl-fuzz
            self.raw_print("Giving user.sh 60 seconds to start afl-fuzz...")
            startup_wait = 60
            startup_checks = 0
            
            while startup_checks < startup_wait and self.automation_state['is_running']:
                time.sleep(1)
                startup_checks += 1
                
                if startup_checks % 10 == 0:  # Check every 10 seconds during startup
                    if self.check_afl_fuzz_running(config['container_name']):
                        self.raw_print(f"AFL-FUZZ DETECTED after {startup_checks} seconds!")
                        break
                    else:
                        self.raw_print(f"Waiting for AFL-fuzz... ({startup_checks}/{startup_wait}s)")
            
            if not self.check_afl_fuzz_running(config['container_name']):
                self.raw_print("AFL-fuzz still not detected, but continuing monitoring...")
            
            while self.automation_state['is_running'] and not self.automation_state['is_paused']:
                elapsed = time.time() - start_time
                
                # Check if maximum time reached
                # if elapsed >= max_duration:
                #     self.log("Maximum time reached", "INFO")
                #     self.raw_print("Maximum fuzzing time reached - stopping")
                #     break
                    
                # Check if pause_monitor finished
                if monitor_process and monitor_process.poll() is not None:
                    self.log("pause_monitor finished", "INFO")
                    self.raw_print("pause_monitor finished - campaign may be complete")
                    break
                    
                # Check afl-fuzz status
                if not self.check_afl_fuzz_running(config['container_name']):
                    self.log("afl-fuzz not running (possible crash)", "WARNING")
                    self.raw_print("afl-fuzz process not found - checking for issues")
                    
                    # Try to get more info about what happened
                    self.check_container_logs(config['container_name'], config['firmadyne_id'])
                    
                    # Check if user.sh process is still active
                    if 'user_sh' in self.automation_state['processes']:
                        user_process = self.automation_state['processes']['user_sh']
                        if user_process and user_process.poll() is None:
                            self.raw_print("user.sh is still running, maybe AFL will start soon...")
                            # Give it more time
                            time.sleep(30)
                            continue
                    
                    break
                    
                # Check fuzzing stats
                self.check_fuzzing_stats_in_container(config, output_dir, image_dir)
                
                # Show progress
                # progress = (elapsed / max_duration) * 100
                # self.raw_print(f"Fuzzing progress: {progress:.1f}% ({elapsed:.0f}s / {max_duration}s)")
                
                time.sleep(check_interval)
                
            # Clean fuzzing processes
            self.cleanup_fuzzing_processes(config['container_name'])
            
            # Copy results to host
            self.copy_results_to_host(config, campaign_num, output_dir)
            
            return True
            
        except Exception as e:
            self.log(f"Error monitoring campaign: {str(e)}", "ERROR")
            return False
            

    def monitor_fuzzing_campaign_timed(self, config, campaign_num, output_dir, monitor_process, image_dir):
        """Monitor fuzzing campaign with time-based control and duration threshold"""
        try:
            start_time = time.time()
            check_interval = config['monitoring_interval']
            
            self.log(f"Monitoring campaign with duration control...", "INFO")
            self.raw_print(f"Starting fuzzing monitoring with {config['total_duration_hours']}h threshold")
            
            # Give user.sh some time to start afl-fuzz
            self.raw_print("Giving user.sh 60 seconds to start afl-fuzz...")
            startup_wait = 60
            startup_checks = 0
            
            while startup_checks < startup_wait and self.automation_state['is_running']:
                time.sleep(1)
                startup_checks += 1
                
                if startup_checks % 10 == 0:
                    if self.check_afl_fuzz_running(config['container_name']):
                        self.raw_print(f"AFL-FUZZ DETECTED after {startup_checks} seconds!")
                        break
                    else:
                        self.raw_print(f"Waiting for AFL-fuzz... ({startup_checks}/{startup_wait}s)")
            
            if not self.check_afl_fuzz_running(config['container_name']):
                self.raw_print("AFL-fuzz still not detected, but continuing monitoring...")
            
            # Main monitoring loop with DURATION CONTROL
            while self.automation_state['is_running'] and not self.automation_state['is_paused']:
                elapsed = time.time() - start_time
                
                # CRITICAL: Check total automation duration
                if self.automation_state['total_start_time']:
                    total_elapsed = time.time() - self.automation_state['total_start_time']
                    total_remaining = self.automation_state['total_duration_seconds'] - total_elapsed
                    
                    if total_remaining <= 0:
                        self.log("DURATION THRESHOLD REACHED - Initiating graceful shutdown", "WARNING")
                        self.raw_print("=== DURATION THRESHOLD REACHED ===")
                        self.raw_print("Copying results and stopping fuzzing...")
                        
                        # STEP 1: Copy current results to host BEFORE stopping
                        self.copy_current_fuzzing_results(config, campaign_num, output_dir)
                        
                        # STEP 2: Stop fuzzing processes gracefully
                        self.stop_fuzzing_processes_gracefully(config['container_name'])
                        
                        # STEP 3: Final results copy to ensure everything is saved
                        self.copy_results_to_host(config, campaign_num, output_dir)
                        
                        self.log("Duration-based shutdown completed", "SUCCESS")
                        self.automation_state['is_running'] = False
                        break
                    
                    # Log remaining time periodically
                    if int(total_elapsed) % 300 == 0:  # Every 5 minutes
                        remaining_hours = total_remaining / 3600
                        self.raw_print(f"Total time remaining: {remaining_hours:.2f} hours")
                        
                # Check if pause_monitor finished
                if monitor_process and monitor_process.poll() is not None:
                    self.log("pause_monitor finished", "INFO")
                    self.raw_print("pause_monitor finished - campaign may be complete")
                    break
                    
                # Check afl-fuzz status
                if not self.check_afl_fuzz_running(config['container_name']):
                    self.log("afl-fuzz not running (possible crash)", "WARNING")
                    self.raw_print("afl-fuzz process not found - checking for issues")
                    
                    # Check if user.sh process is still active
                    if 'user_sh' in self.automation_state['processes']:
                        user_process = self.automation_state['processes']['user_sh']
                        if user_process and user_process.poll() is None:
                            self.raw_print("user.sh is still running, maybe AFL will start soon...")
                            time.sleep(30)
                            continue
                    
                    break
                    
                # Check fuzzing stats
                self.check_fuzzing_stats_in_container(config, output_dir, image_dir)
                
                time.sleep(check_interval)
                
            # Normal campaign completion (not duration-based)
            if self.automation_state['is_running']:
                # Clean fuzzing processes normally
                self.cleanup_fuzzing_processes(config['container_name'])
                
                # Copy results to host
                self.copy_results_to_host(config, campaign_num, output_dir)
            
            actual_duration = time.time() - start_time
            self.log(f"Campaign monitoring completed - Actual duration: {actual_duration:.1f}s", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log(f"Error monitoring campaign: {str(e)}", "ERROR")
            return False


    # 2. Add new function to copy current fuzzing results while running
    def copy_current_fuzzing_results(self, config, campaign_num, output_dir):
        """Copy current fuzzing results while AFL is still running"""
        try:
            container_name = config['container_name']
            firmadyne_id = config['firmadyne_id']
            
            self.log("Copying current fuzzing results while running...", "INFO")
            self.raw_print("COPYING CURRENT RESULTS (AFL still running)")
            
            # Create host directory structure
            host_results_dir = f"{config['host_path']}/image_{firmadyne_id}"
            os.makedirs(host_results_dir, exist_ok=True)
            
            # Determine source path based on firmadyne_id
            if firmadyne_id == "161161":
                source_path = f"/test/image_161160/{output_dir}"
            else:
                source_path = f"/test/image_{firmadyne_id}/{output_dir}"
            
            # Check if source exists
            check_cmd = ['docker', 'exec', container_name, 'test', '-d', source_path]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if check_result.returncode != 0:
                self.log(f"Source directory {source_path} not found in container", "WARNING")
                return False
            
            # Copy with timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_output = f"{output_dir}_final_{timestamp}"
            
            container_path = f"{container_name}:{source_path}"
            host_path = f"{host_results_dir}/{timestamped_output}"
            
            copy_cmd = ['docker', 'cp', container_path, host_path]
            
            self.log(f"Executing: {' '.join(copy_cmd)}", "DEBUG")
            self.raw_print(f"COPY CMD: {' '.join(copy_cmd)}")
            
            result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.log(f"Current results copied to: {host_path}", "SUCCESS")
                self.raw_print(f"SUCCESS: Results copied to {host_path}")
                
                # Verify copy
                if os.path.exists(host_path):
                    self.log("Copy verification successful", "SUCCESS")
                    
                    # Count files in the copied directory
                    try:
                        file_count = sum([len(files) for r, d, files in os.walk(host_path)])
                        self.raw_print(f"Copied directory contains {file_count} files")
                    except:
                        pass
                        
                    return True
                else:
                    self.log("Copy verification failed - directory not found on host", "ERROR")
                    return False
            else:
                self.log(f"Copy failed: {result.stderr}", "ERROR")
                self.raw_print(f"COPY FAILED: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("Copy operation timed out", "ERROR")
            self.raw_print("COPY TIMEOUT - operation took too long")
            return False
        except Exception as e:
            self.log(f"Error copying current results: {str(e)}", "ERROR")
            self.raw_print(f"COPY ERROR: {str(e)}")
            return False
    
    # 3. Add new function to stop fuzzing processes gracefully
    def stop_fuzzing_processes_gracefully(self, container_name):
        """Stop fuzzing processes gracefully to preserve data"""
        try:
            self.log("Stopping fuzzing processes gracefully...", "INFO")
            self.raw_print("=== GRACEFUL FUZZING SHUTDOWN ===")
            
            # Step 1: Send SIGTERM to afl-fuzz (graceful shutdown)
            self.raw_print("Sending SIGTERM to afl-fuzz...")
            sigterm_cmd = ['docker', 'exec', container_name, 'pkill', '-TERM', 'afl-fuzz']
            sigterm_result = subprocess.run(sigterm_cmd, capture_output=True, text=True)
            
            if sigterm_result.returncode == 0:
                self.log("SIGTERM sent to afl-fuzz", "SUCCESS")
                self.raw_print("SIGTERM sent to afl-fuzz")
            else:
                self.log("No afl-fuzz processes found for SIGTERM", "INFO")
                self.raw_print("No afl-fuzz processes found")
            
            # Step 2: Wait for graceful shutdown
            self.raw_print("Waiting 10 seconds for graceful shutdown...")
            for i in range(10):
                time.sleep(1)
                if not self.check_afl_fuzz_running(container_name):
                    self.raw_print(f"afl-fuzz stopped gracefully after {i+1} seconds")
                    break
            else:
                self.raw_print("afl-fuzz did not stop gracefully, using SIGKILL...")
                
                # Step 3: Force kill if graceful shutdown failed
                kill_cmd = ['docker', 'exec', container_name, 'pkill', '-KILL', 'afl-fuzz']
                kill_result = subprocess.run(kill_cmd, capture_output=True, text=True)
                if kill_result.returncode == 0:
                    self.raw_print("afl-fuzz force killed")
                
            # Step 4: Stop QEMU processes
            self.raw_print("Stopping QEMU processes...")
            qemu_processes = ['afl-qemu-trace', 'qemu-system-mips', 'qemu-mips']
            
            for proc_name in qemu_processes:
                # Try graceful first
                sigterm_cmd = ['docker', 'exec', container_name, 'pkill', '-TERM', proc_name]
                subprocess.run(sigterm_cmd, capture_output=True, text=True)
                
                # Wait a bit
                time.sleep(2)
                
                # Force kill if still running
                kill_cmd = ['docker', 'exec', container_name, 'pkill', '-KILL', proc_name]
                kill_result = subprocess.run(kill_cmd, capture_output=True, text=True)
                
                if kill_result.returncode == 0:
                    self.raw_print(f"{proc_name} processes stopped")
            
            # Step 5: Stop user.sh and run.sh
            script_processes = ['user.sh', 'run.sh']
            for script in script_processes:
                kill_cmd = ['docker', 'exec', container_name, 'pkill', '-TERM', '-f', script]
                kill_result = subprocess.run(kill_cmd, capture_output=True, text=True)
                if kill_result.returncode == 0:
                    self.raw_print(f"{script} processes stopped")
            
            # Step 6: Final verification
            time.sleep(3)
            if self.check_afl_fuzz_running(container_name):
                self.log("Warning: Some fuzzing processes may still be running", "WARNING")
                self.raw_print("WARNING: Some processes may still be active")
            else:
                self.log("All fuzzing processes stopped successfully", "SUCCESS")
                self.raw_print("All fuzzing processes stopped")
            
            self.raw_print("=== GRACEFUL SHUTDOWN COMPLETED ===")
            return True
            
        except Exception as e:
            self.log(f"Error in graceful shutdown: {str(e)}", "ERROR")
            self.raw_print(f"GRACEFUL SHUTDOWN ERROR: {str(e)}")
            return False


    def check_afl_fuzz_running(self, container_name):
        """Check if afl-fuzz is running with enhanced detection"""
        try:
            # Check for afl-fuzz process
            cmd = ['docker', 'exec', container_name, 'pgrep', 'afl-fuzz']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                self.raw_print(f"afl-fuzz is running (PIDs: {', '.join(pids)})")
                return True
            else:
                # Enhanced process detection
                check_cmd = ['docker', 'exec', container_name, 'ps', 'auxww']
                ps_result = subprocess.run(check_cmd, capture_output=True, text=True)
                
                if ps_result.returncode == 0:
                    fuzzing_processes = []
                    for line in ps_result.stdout.split('\n'):
                        line_lower = line.lower()
                        if any(keyword in line_lower for keyword in ['afl', 'fuzz', 'qemu-trace']):
                            fuzzing_processes.append(line.strip())
                    
                    if fuzzing_processes:
                        self.raw_print("Found fuzzing-related processes:")
                        for proc in fuzzing_processes:
                            self.raw_print(f"  {proc}")
                        return True
                    else:
                        self.raw_print("No afl-fuzz or fuzzing processes found")
                        
                        # Check if user.sh is still running
                        user_sh_cmd = ['docker', 'exec', container_name, 'pgrep', '-f', 'user.sh']
                        user_sh_result = subprocess.run(user_sh_cmd, capture_output=True, text=True)
                        
                        if user_sh_result.returncode == 0:
                            self.raw_print(f"user.sh is still running (PID: {user_sh_result.stdout.strip()})")
                            return True
                        else:
                            self.raw_print("user.sh is not running either")
                        
                return False
                
        except Exception as e:
            self.log(f"Error checking afl-fuzz: {str(e)}", "WARNING")
            return False
            
    def check_fuzzing_stats_in_container(self, config, output_dir, image_dir=None):
        """Check fuzzing statistics inside container"""
        try:
            container_name = config['container_name']
            firmadyne_id = config['firmadyne_id']
            
            # FIXED: Usar image_dir si se proporciona
            if image_dir is None:
                if firmadyne_id == "161161":
                    working_dir = "/test/image_161160"
                else:
                    working_dir = f"/test/image_{firmadyne_id}"
            else:
                working_dir = image_dir
                
            stats_path = f"{working_dir}/{output_dir}/fuzzer_stats"
            
            # Check stats file inside container
            cmd = ['docker', 'exec', container_name, 'cat', stats_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                stats = result.stdout
                
                # Extract key stats
                execs_done = re.search(r'execs_done\s*:\s*(\d+)', stats)
                paths_total = re.search(r'paths_total\s*:\s*(\d+)', stats)
                pending_total = re.search(r'pending_total\s*:\s*(\d+)', stats)
                unique_crashes = re.search(r'unique_crashes\s*:\s*(\d+)', stats)
                
                if execs_done and paths_total:
                    stats_msg = f"AFL Stats - Execs: {execs_done.group(1)}, Paths: {paths_total.group(1)}"
                    if pending_total:
                        stats_msg += f", Pending: {pending_total.group(1)}"
                    if unique_crashes:
                        stats_msg += f", Crashes: {unique_crashes.group(1)}"
                        
                    self.raw_print(stats_msg)
                    self.log(stats_msg, "INFO")
                    
            else:
                # Stats file doesn't exist yet, check if fuzzing directory exists
                check_cmd = ['docker', 'exec', container_name, 'ls', '-la', f"/test/image_{firmadyne_id}/{output_dir}"]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True)
                
                if check_result.returncode == 0:
                    self.raw_print(f"Fuzzing directory exists but no stats yet:\n{check_result.stdout}")
                else:
                    self.raw_print(f"Fuzzing directory not found: {output_dir}")
                    
        except Exception as e:
            self.log(f"Error reading fuzzing stats: {str(e)}", "WARNING")
            
    def check_container_logs(self, container_name, firmadyne_id):
        """Check container logs for debugging"""
        try:
            # Get recent container logs
            logs_cmd = ['docker', 'logs', '--tail', '20', container_name]
            logs_result = subprocess.run(logs_cmd, capture_output=True, text=True)
            
            if logs_result.returncode == 0 and logs_result.stdout:
                self.raw_print("Recent container logs:")
                for line in logs_result.stdout.split('\n')[-10:]:
                    if line.strip():
                        self.raw_print(f"  {line}")
                        
            # Check if there are any error files in the image directory
            error_cmd = ['docker', 'exec', container_name, 'find', f'/test/image_{firmadyne_id}', 
                        '-name', '*.log', '-o', '-name', 'core*', '-o', '-name', '*.err']
            error_result = subprocess.run(error_cmd, capture_output=True, text=True)
            
            if error_result.returncode == 0 and error_result.stdout.strip():
                self.raw_print("Found potential error files:")
                for line in error_result.stdout.split('\n'):
                    if line.strip():
                        self.raw_print(f"  {line}")
                        
        except Exception as e:
            self.raw_print(f"Error checking container logs: {str(e)}")
            
    def cleanup_fuzzing_processes(self, container_name):
        """Clean fuzzing processes"""
        try:
            self.log("Starting process cleanup...", "DEBUG")
            
            # Kill afl-qemu-trace processes on host
            killed_count = 0
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if proc.info['name'] and 'afl-qemu-trace' in proc.info['name']:
                        os.kill(proc.info['pid'], signal.SIGKILL)
                        self.log(f"afl-qemu-trace process killed: {proc.info['pid']}", "DEBUG")
                        killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    pass
                    
            if killed_count > 0:
                self.log(f"Killed {killed_count} afl-qemu-trace processes", "SUCCESS")
            else:
                self.log("No afl-qemu-trace processes found", "DEBUG")
            
            # Kill processes in container
            for proc_name in ['afl-fuzz', 'afl-qemu-trace']:
                cmd = ['docker', 'exec', container_name, 'pkill', '-9', proc_name]
                self.log(f"Killing {proc_name} in container: {' '.join(cmd)}", "DEBUG")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    self.log(f"{proc_name} processes killed in container", "DEBUG")
                else:
                    self.log(f"No {proc_name} processes found in container", "DEBUG")
                          
        except Exception as e:
            self.log(f"Error cleaning processes: {str(e)}", "WARNING")

    def copy_results_to_host(self, config, campaign_num, output_dir):
        """Copy results from container to host with FIXED directory structure for 161161"""
        try:
            firmadyne_id = config['firmadyne_id']
            container_name = config['container_name']
            
            # Create directory structure: /home/atenea/trenti/evaluations/image_XXX/
            host_results_dir = f"{config['host_path']}/image_{firmadyne_id}"
            self.log(f"Creating host directory: {host_results_dir}", "DEBUG")
            os.makedirs(host_results_dir, exist_ok=True)
            
            # FIXED: Determine correct source path based on firmadyne_id
            if firmadyne_id == "161161":
                # Para 161161, usar image_161160 como directorio de trabajo
                source_image_dir = "image_161160"
                container_source_path = f"/test/{source_image_dir}/{output_dir}"
            else:
                # Para otros casos, usar el directorio normal
                source_image_dir = f"image_{firmadyne_id}"
                container_source_path = f"/test/{source_image_dir}/{output_dir}"
            
            self.log(f"Source path in container: {container_source_path}", "DEBUG")
            self.raw_print(f"COPY SOURCE: {container_source_path}")
            
            # Check if output directory exists in container first
            check_cmd = ['docker', 'exec', container_name, 'test', '-d', container_source_path]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if check_result.returncode != 0:
                self.log(f"Output directory {container_source_path} does not exist in container", "WARNING")
                
                # Try to find what directories actually exist
                find_cmd = ['docker', 'exec', container_name, 'bash', '-c',
                           f'find /test/{source_image_dir} -name "*trenti_sca_outputs*" -type d 2>/dev/null']
                find_result = subprocess.run(find_cmd, capture_output=True, text=True)
                
                if find_result.returncode == 0 and find_result.stdout.strip():
                    self.log(f"Found TRENTI output directories in {source_image_dir}:", "INFO")
                    available_dirs = find_result.stdout.strip().split('\n')
                    for dir_path in available_dirs:
                        self.raw_print(f"  AVAILABLE: {dir_path}")
                    
                    # Try to use the most recent one
                    latest_dir = available_dirs[-1]  # Last one (likely most recent)
                    container_source_path = latest_dir
                    self.log(f"Using latest directory: {container_source_path}", "INFO")
                    self.raw_print(f"USING LATEST: {container_source_path}")
                else:
                    # Fallback: List what's actually in the source image directory
                    ls_cmd = ['docker', 'exec', container_name, 'ls', '-la', f'/test/{source_image_dir}/']
                    ls_result = subprocess.run(ls_cmd, capture_output=True, text=True)
                    
                    if ls_result.returncode == 0:
                        self.log(f"Contents of /test/{source_image_dir}/:", "WARNING")
                        self.raw_print(f"CONTENTS OF /test/{source_image_dir}/:")
                        for line in ls_result.stdout.split('\n'):
                            if line.strip():
                                self.raw_print(f"  {line}")
                    else:
                        self.log(f"Cannot access /test/{source_image_dir}/ in container", "ERROR")
                    
                    self.log(f"No TRENTI output directories found in {source_image_dir}", "WARNING")
                    self.raw_print("No TRENTI fuzzing output directories found")
                    return False
            
            # Verify we can access the source directory
            verify_cmd = ['docker', 'exec', container_name, 'ls', '-la', container_source_path]
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
            
            if verify_result.returncode == 0:
                self.raw_print(f"VERIFIED SOURCE CONTENTS:")
                file_count = 0
                for line in verify_result.stdout.split('\n'):
                    if line.strip() and not line.startswith('total'):
                        file_count += 1
                        self.raw_print(f"  {line}")
                self.log(f"Source directory contains {file_count} items", "INFO")
            else:
                self.log(f"Cannot verify source directory: {verify_result.stderr}", "ERROR")
                return False
            
            # Copy from container to host: /home/atenea/trenti/evaluations/image_XXX/trenti_sca_outputs_XXX_YYY/
            container_path = f"{container_name}:{container_source_path}"
            host_path = f"{host_results_dir}/{output_dir}"
            cmd = ['docker', 'cp', container_path, host_path]
            
            self.log(f"Copy command: {' '.join(cmd)}", "DEBUG")
            self.raw_print(f"COPY CMD: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.log(f"Results copied successfully to: {host_path}", "SUCCESS")
                self.raw_print(f"SUCCESS: Results copied to {host_path}")
                
                # Verify the copy on host
                if os.path.exists(host_path):
                    try:
                        # Count files in copied directory
                        file_count = sum([len(files) for r, d, files in os.walk(host_path)])
                        dir_count = sum([len(dirs) for r, dirs, f in os.walk(host_path)])
                        self.log(f"Copied: {file_count} files in {dir_count} directories", "SUCCESS")
                        self.raw_print(f"VERIFICATION: {file_count} files, {dir_count} directories copied")
                        
                        # List main contents
                        main_contents = os.listdir(host_path)
                        self.raw_print(f"MAIN CONTENTS: {', '.join(main_contents[:10])}")  # First 10 items
                        
                    except Exception as e:
                        self.log(f"Error verifying copy: {e}", "WARNING")
                else:
                    self.log("ERROR: Copied directory not found on host after copy", "ERROR")
                    return False
            else:
                self.log(f"Error copying results: {result.stderr}", "ERROR")
                self.log(f"STDOUT: {result.stdout}", "DEBUG")
                self.raw_print(f"COPY FAILED: {result.stderr}")
                
                # Additional debugging: check if source exists one more time
                final_check_cmd = ['docker', 'exec', container_name, 'ls', '-la', container_source_path]
                final_check_result = subprocess.run(final_check_cmd, capture_output=True, text=True)
                
                if final_check_result.returncode == 0:
                    self.raw_print("SOURCE EXISTS but copy failed - possible permissions issue")
                else:
                    self.raw_print("SOURCE DISAPPEARED during copy operation")
                
                return False
            
            return True
                
        except subprocess.TimeoutExpired:
            self.log("Copy operation timed out", "ERROR")
            self.raw_print("COPY TIMEOUT - operation took too long")
            return False
        except Exception as e:
            self.log(f"Error in results copy: {str(e)}", "ERROR")
            self.raw_print(f"COPY ERROR: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return False    


    def run_anomaly_detection(self, config, campaign_num):
        """Run Side-Channel Analysis anomaly detection"""
        try:
            firmadyne_id = config['firmadyne_id']
            output_dir = f"trenti_sca_outputs_{firmadyne_id}_{campaign_num}"

            # Check if scripts exist before running
            scripts_to_check = ['http_calibration.py', 'http_operation.py', 'SCA_anomaly_detection.py']
            missing_scripts = []

            for script in scripts_to_check:
                if not os.path.exists(script):
                    missing_scripts.append(script)

            if missing_scripts:
                self.log(f"Missing SCA scripts: {missing_scripts}", "WARNING")
                self.log("Skipping Side-Channel Analysis phase", "INFO")
                return True  # Continue without SCA
            
            # Verify host path exists
            if not os.path.exists(config['host_path']):
                self.log(f"Host path does not exist: {config['host_path']}", "ERROR")
                return False

            if campaign_num == 1:
                # Step 1: HTTP Calibration Phase
                if not self.run_http_phase(config, "HTTP Calibration", "http_calibration.py", campaign_num):
                    self.log("HTTP calibration failed, but continuing...", "WARNING")
            
            # Step 2: HTTP Operation Phase  
            if not self.run_http_phase(config, "HTTP Operation", "http_operation.py", campaign_num):
                self.log("HTTP operation failed, but continuing...", "WARNING")

            # Step 3: Execute SCA_anomaly_detection.py with real-time output
            self.log("Executing SCA_anomaly_detection.py...", "INFO")
            cmd = ['python3', 'SCA_anomaly_detection.py', config['firmadyne_id'], config['host_path']]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True,
                                      bufsize=1, universal_newlines=True)
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_line = line.strip()
                    self.raw_print(f"SCA_TRENTI: {output_line}")
                    self.log(f"SCA_anomaly_detection.py: {output_line}", "INFO")
                    
                    # Refresh plots when visualizations are generated
                    if "visualization saved to:" in output_line.lower():
                        self.root.after(0, self.plot_viewer.refresh_plots)
            
            process.wait()
            if process.returncode != 0:
                self.log(f"SCA_anomaly_detection.py failed with exit code {process.returncode}", "ERROR")
                return False
            self.log("SCA_anomaly_detection.py completed successfully", "SUCCESS")
            
            # Update SCA anomaly count
            sca_anomalies = self.count_sca_anomalies(config, campaign_num)
            self.automation_state['anomalies_count'] = sca_anomalies
           
            # Step 4: Execution time and SW Analysis
            if self.enable_post_analysis_var.get():
               if not self.run_post_trenti_analysis(config, campaign_num):
                   self.log("Post-TRENTI analysis failed, but continuing...", "WARNING")
            else:
               self.log("Post-TRENTI analysis disabled, skipping...", "INFO")       
            
            # ------------------------------------------------------------------
            # STEP 5  copy anomalous queue entries & resume fuzzing
            # ------------------------------------------------------------------
            try:
                # Get the latest anomaly information
                anomaly_info = self.get_latest_anomaly_timestamp()
                if not anomaly_info:
                    self.log("No anomaly information found  skipping Copy-&-Resume step", "WARNING")
                else:
                    self.log(f"Found anomaly file: {anomaly_info['filename']}", "INFO")
                    
                    # Instantiate CopyResumeManager with correct parameters
                    crm = CopyResumeManager(
                        container    = config['container_name'],
                        firmadyne_id = config['firmadyne_id'],
                        host_base    = config['host_path']
                    )
                    
                    # Call copy_and_resume with anomaly_info dict
                    success = crm.copy_and_resume(anomaly_info)
                    
                    if success:
                        self.log("Copy & Resume step finished successfully", "SUCCESS")
                    else:
                        self.log("Copy & Resume step failed", "ERROR")
            
            except Exception as e:
                self.log(f"Copy & Resume step failed  {e}", "ERROR")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")

            # Increment anomaly counter
            self.automation_state['anomalies_count'] += 1

            return True

        except Exception as e:
            self.log(f"Error in Side-Channel Analysis: {str(e)}", "ERROR")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return False


    def run_post_trenti_analysis(self, config, campaign_num):
        """Run Post-TRENTI Analysis (execution time + HTTP response + deduplication)"""
        try:
            self.log("Starting Post-TRENTI Analysis...", "INFO")
            self.metric_vars['post_analysis_status'].set("Running")
            
            firmadyne_id = config['firmadyne_id']
            host_path = config['host_path']
            
            # Check if post-analysis scripts exist
            post_scripts = ['execution_time_analysis.py', 'http_response_analysis.py', 'anomaly_deduplicator.py']
            missing_post_scripts = []
            
            for script in post_scripts:
                if not os.path.exists(script):
                    missing_post_scripts.append(script)
            
            if missing_post_scripts:
                self.log(f"Missing post-analysis scripts: {missing_post_scripts}", "WARNING")
                self.metric_vars['post_analysis_status'].set("Scripts Missing")
                return False
            
            # Step 1: Execution Time Analysis
            if self.enable_exec_time_analysis_var.get():
                self.log("Running execution time clustering analysis...", "INFO")
                
                cmd = ['python3', 'execution_time_analysis.py', firmadyne_id, 
                       '--host-path', host_path]
                
                exec_process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, text=True,
                                              bufsize=1, universal_newlines=True)
                
                for line in iter(exec_process.stdout.readline, ''):
                    if line:
                        output_line = line.strip()
                        self.raw_print(f"EXEC_TIME_ANALYSIS: {output_line}")
                        self.log(f"exec_time_analysis: {output_line}", "INFO")
                        
                        # Extract anomaly count if found
                        if "anomalous file ids:" in output_line.lower():
                            try:
                                import re
                                match = re.search(r'\[([^\]]*)\]', output_line)
                                if match:
                                    ids_str = match.group(1)
                                    if ids_str.strip():
                                        anomaly_count = len([x.strip() for x in ids_str.split(',') if x.strip()])
                                        self.automation_state['exec_time_anomalies'] += anomaly_count
                                        self.metric_vars['exec_time_anomalies'].set(str(self.automation_state['exec_time_anomalies']))
                            except:
                                pass
                
                exec_process.wait()
                if exec_process.returncode != 0:
                    self.log("Execution time analysis failed", "ERROR")
                else:
                    self.log("Execution time analysis completed", "SUCCESS")
            
            # Step 2: HTTP Response Analysis
            if self.enable_http_analysis_var.get():
                self.log("Running HTTP response analysis...", "INFO")
                
                cmd = ['python3', 'http_response_analysis.py', firmadyne_id, 
                       '--host-path', host_path]
                
                http_process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, text=True,
                                              bufsize=1, universal_newlines=True)
                
                for line in iter(http_process.stdout.readline, ''):
                    if line:
                        output_line = line.strip()
                        self.raw_print(f"HTTP_ANALYSIS: {output_line}")
                        self.log(f"http_response_analysis: {output_line}", "INFO")
                        
                        # Extract anomaly count
                        if "anomalous file ids:" in output_line.lower():
                            try:
                                import re
                                match = re.search(r'\[([^\]]*)\]', output_line)
                                if match:
                                    ids_str = match.group(1)
                                    if ids_str.strip():
                                        anomaly_count = len([x.strip() for x in ids_str.split(',') if x.strip()])
                                        self.automation_state['http_response_anomalies'] += anomaly_count
                                        self.metric_vars['http_response_anomalies'].set(str(self.automation_state['http_response_anomalies']))
                            except:
                                pass
                
                http_process.wait()
                if http_process.returncode != 0:
                    self.log("HTTP response analysis failed", "ERROR")
                else:
                    self.log("HTTP response analysis completed", "SUCCESS")
            
            # Step 3: Anomaly Deduplication
            if self.enable_deduplication_var.get():
                self.log("Running anomaly deduplication...", "INFO")
                
                cmd = ['python3', 'anomaly_deduplicator.py', firmadyne_id, 
                       '--host-path', host_path]
                
                dedup_process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True,
                                               bufsize=1, universal_newlines=True)
                
                for line in iter(dedup_process.stdout.readline, ''):
                    if line:
                        output_line = line.strip()
                        self.raw_print(f"DEDUPLICATION: {output_line}")
                        self.log(f"ANOMALY DEDUPLICATOR: {output_line}", "INFO")
                
                dedup_process.wait()
                if dedup_process.returncode != 0:
                    self.log("Anomaly deduplication failed", "ERROR")
                else:
                    self.log("Anomaly deduplication completed", "SUCCESS")
            
            # Update total anomaly count
            total_anomalies = (self.automation_state['anomalies_count'] + 
                              self.automation_state['exec_time_anomalies'] + 
                              self.automation_state['http_response_anomalies'])
            self.automation_state['total_anomalies_found'] = total_anomalies
            self.metric_vars['total_anomalies_found'].set(str(total_anomalies))
            self.metric_vars['sca_anomalies_count'].set(str(self.automation_state['anomalies_count']))
            
            self.log(f"Iteration Analysis completed - Total anomalies: {total_anomalies}", "SUCCESS")
            self.metric_vars['post_analysis_status'].set("Completed")
            
            return True
            
        except Exception as e:
            self.log(f"Error in Post-TRENTI Analysis: {str(e)}", "ERROR")
            self.metric_vars['post_analysis_status'].set("Error")
            return False
        
    def count_sca_anomalies(self, config, campaign_num):
        """Count SCA anomalies from anomaly_names files"""
        try:
            anomaly_names_dir = f"{config['host_path']}/image_{config['firmadyne_id']}/anomaly_names"
            
            if not os.path.exists(anomaly_names_dir):
                return 0
            
            import glob
            anomaly_files = glob.glob(os.path.join(anomaly_names_dir, "anomaly_names_*.txt"))
            
            if not anomaly_files:
                return 0
            
            # Get the most recent file
            latest_file = max(anomaly_files, key=os.path.getctime)
            
            sca_count = 0
            with open(latest_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and 'signal_' in line:
                        # Count only SCA anomalies (not exec_time or http_response)
                        if 'EXEC_TIME_ANOMALY' not in line and 'HTTP_RESPONSE_ANOMALY' not in line:
                            sca_count += 1
            
            return sca_count
            
        except Exception as e:
            self.log(f"Error counting SCA anomalies: {str(e)}", "DEBUG")
            return 0


    def run_http_phase(self, config, phase_name, script_name, campaign_num):
        """Run HTTP calibration or operation phase with custom curl command - FIXED"""
        try:
            self.log(f"Starting {phase_name}...", "INFO")
            
            # DEBUGGING: Mostrar configuración antes de establecer variables de entorno
            self.log(f"[DEBUG] Config values for {phase_name}:", "DEBUG")
            self.log(f"[DEBUG]   firmadyne_id: '{config['firmadyne_id']}'", "DEBUG")
            self.log(f"[DEBUG]   host_path: '{config['host_path']}'", "DEBUG")
            self.log(f"[DEBUG]   campaign_num: {campaign_num}", "DEBUG")
            self.log(f"[DEBUG]   router_ip: '{config['router_ip']}'", "DEBUG")
            self.log(f"[DEBUG]   container_name: '{config['container_name']}'", "DEBUG")
            
            # Construir CRASH_DIRS para esta campaña específica
            crash_dirs = [f"{config['host_path']}/image_{config['firmadyne_id']}/trenti_sca_outputs_{config['firmadyne_id']}_{campaign_num}/queue"]
            
            self.log(f"[DEBUG] Constructed crash_dirs: {crash_dirs}", "DEBUG")
            
            # Set environment variables for configuration - FIXED: Eliminar duplicación
            env = os.environ.copy()
            env_updates = {
                'TRENTI_ROUTER_IP': config['router_ip'],
                'TRENTI_OSCILLOSCOPE_IP': config['oscilloscope_ip'],
                'TRENTI_TARGET_CGI': config['target_cgi'],
                'TRENTI_SCA_CHANNEL': config['sca_channel'],
                'TRENTI_SCA_VOLTAGE_DIV': str(config['sca_voltage_div']),
                'TRENTI_SCA_TIME_DIV': config['sca_time_div'],
                'TRENTI_SCA_SAMPLE_RATE': str(config['sca_sample_rate']),
                'TRENTI_SCA_MEMORY_SIZE': config['sca_memory_size'],
                'TRENTI_FIRMADYNE_ID': str(config['firmadyne_id']),  # FIXED: Asegurar que es string
                'TRENTI_HOST_PATH': config['host_path'],
                'TRENTI_CUSTOM_CURL_COMMAND': config['custom_curl_command'],
                'TRENTI_CAMPAIGN_NUMBER': str(campaign_num),
                'TRENTI_CRASH_DIRS': json.dumps(crash_dirs),  # FIXED: Solo una vez
            }
            
            # DEBUGGING: Mostrar todas las variables que se van a establecer
            self.log(f"[DEBUG] Environment variables to set:", "DEBUG")
            for key, value in env_updates.items():
                # Truncar valores largos para el log
                display_value = value[:100] + "..." if len(value) > 100 else value
                self.log(f"[DEBUG]   {key} = '{display_value}'", "DEBUG")
            
            # CRITICAL: Aplicar las variables de entorno
            env.update(env_updates)
            
            # DEBUGGING: Verificar que las variables se establecieron correctamente
            self.log(f"[DEBUG] Verifying environment variables after update:", "DEBUG")
            for key in env_updates.keys():
                if key in env:
                    actual_value = env[key]
                    display_value = actual_value[:100] + "..." if len(actual_value) > 100 else actual_value
                    self.log(f"[DEBUG]    {key} = '{display_value}'", "DEBUG")
                else:
                    self.log(f"[DEBUG]    {key} NOT FOUND in environment", "ERROR")
            
            # DEBUGGING: Verificar que los directorios de crash existen
            self.log(f"[DEBUG] Checking crash directories existence:", "DEBUG")
            for crash_dir in crash_dirs:
                exists = os.path.exists(crash_dir)
                self.log(f"[DEBUG]   {crash_dir} exists: {exists}", "DEBUG")
                if exists:
                    try:
                        file_count = len([f for f in os.listdir(crash_dir) 
                                        if os.path.isfile(os.path.join(crash_dir, f)) and f != "README.txt"])
                        self.log(f"[DEBUG]     Contains {file_count} files", "DEBUG")
                    except Exception as e:
                        self.log(f"[DEBUG]     Error counting files: {e}", "DEBUG")
            
            # CRITICAL: Verificar que el firmadyne_id se está pasando correctamente
            self.log(f"[DEBUG] Final verification - TRENTI_FIRMADYNE_ID in env: {env.get('TRENTI_FIRMADYNE_ID', 'NOT_FOUND')}", "DEBUG")
            
            # Execute the script
            cmd = ['python3', script_name]
            self.log(f"[DEBUG] Executing command: {' '.join(cmd)}", "DEBUG")
            self.log(f"[DEBUG] Working directory: {os.getcwd()}", "DEBUG")
            
            # CRITICAL: Pasar el environment al proceso
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True,
                                      bufsize=1, universal_newlines=True, env=env)
            
            line_count = 0
            for line in iter(process.stdout.readline, ''):
                if line:
                    line_count += 1
                    output_line = line.strip()
                    self.raw_print(f"{phase_name.upper()}[{line_count:04d}]: {output_line}")
                    self.log(f"{script_name}: {output_line}", "INFO")
            
            process.wait()
            
            self.log(f"[DEBUG] Process finished. Return code: {process.returncode}", "DEBUG")
            self.log(f"[DEBUG] Total output lines: {line_count}", "DEBUG")
            
            if process.returncode != 0:
                self.log(f"{script_name} failed with exit code {process.returncode}", "ERROR")
                return False
            
            self.log(f"{phase_name} completed successfully", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Error in {phase_name}: {str(e)}", "ERROR")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return False

    def configure_http_calibration(self, config):
        """Configure http_calibration.py with parameters from GUI"""
        try:
            if not os.path.exists('http_calibration.py'):
                self.log("http_calibration.py not found, skipping configuration", "WARNING")
                return True

            # Read current file
            with open('http_calibration.py', 'r') as f:
                content = f.read()

            # Replace configuration variables with GUI values
            # ROUTER_IP configuration
            content = re.sub(
                r"ROUTER_IP\s*=\s*['\"][^'\"]*['\"]",
                f"ROUTER_IP = '{config['router_ip']}'",
                content
            )

            # OSCILLOSCOPE_IP configuration
            content = re.sub(
                r"OSCILLOSCOPE_IP\s*=\s*['\"][^'\"]*['\"]",
                f"OSCILLOSCOPE_IP = '{config['oscilloscope_ip']}'",
                content
            )

            # TARGET_CGI configuration
            content = re.sub(
                r"TARGET_CGI\s*=\s*['\"][^'\"]*['\"]",
                f"TARGET_CGI = '{config['target_cgi']}'",
                content
            )

            # Update CRASH_DIRS to point to current queue
            firmadyne_id = config['firmadyne_id']
            host_path = config['host_path']
            queue_path = f"{host_path}/image_{firmadyne_id}/trenti_sca_outputs_{firmadyne_id}_*/queue"
        
            # Replace CRASH_DIRS array
            crash_dirs_replacement = f'''CRASH_DIRS = [
    "{queue_path}"
]'''
            content = re.sub(
                r'CRASH_DIRS\s*=\s*\[[^\]]*\]',
                crash_dirs_replacement,
                content,
                flags=re.DOTALL
            )

            # Configure SCA parameters if enabled
            if config['enable_sca']:
                # Update sample rate (convert from GS/s to Hz)
                try:
                    sample_rate_gs = float(config['sca_sample_rate'])
                    sample_rate_hz = int(sample_rate_gs * 1e9)
                    content = re.sub(
                        r'sample_rate\s*=\s*[^#\n]*',
                        f'sample_rate = {sample_rate_hz}  # {sample_rate_gs} GS/s',
                        content
                    )
                except ValueError:
                    self.log("Invalid sample rate, using default", "WARNING")

                # Update voltage division
                try:
                    voltage_div = float(config['sca_voltage_div'])
                    content = re.sub(
                        r'self\.lecroy\._scope\.write\("C3:VDIV [^"]*"\)',
                        f'self.lecroy._scope.write("C3:VDIV {voltage_div:g}")',
                        content
                    )
                except ValueError:
                    self.log("Invalid voltage division, using default", "WARNING")

                # Update time division
                try:
                    time_div = config['sca_time_div']
                    content = re.sub(
                        r'self\.lecroy\._scope\.write\("TDIV [^"]*"\)',
                        f'self.lecroy._scope.write("TDIV {time_div}")',
                        content
                    )
                except:
                    self.log("Invalid time division, using default", "WARNING")

                # Update memory size
                try:
                    memory_size = config['sca_memory_size']
                    content = re.sub(
                        r'self\.lecroy\._scope\.write\("MSIZ [^"]*"\)',
                        f'self.lecroy._scope.write("MSIZ {memory_size}")',
                        content
                    )
                except:
                    self.log("Invalid memory size, using default", "WARNING")

                # Update capture channel
                channel = config['sca_channel']
                content = re.sub(r'C3', channel, content)

            waveforms_path = f'{host_path}/image_{firmadyne_id}/waveforms/'
            content = re.sub(
                r"EM_WAVEFORMS_PATH\s*=\s*['\"][^'\"]*['\"]",
                f"EM_WAVEFORMS_PATH = '{waveforms_path}'",
                content
            )

            # Write modified file
            with open('http_calibration.py', 'w') as f:
                f.write(content)

            self.log("http_calibration.py configured with GUI parameters", "SUCCESS")
            self.log(f"Router IP: {config['router_ip']}", "DEBUG")
            self.log(f"Oscilloscope IP: {config['oscilloscope_ip']}", "DEBUG")
            self.log(f"Target CGI: {config['target_cgi']}", "DEBUG")
            self.log(f"SCA Enabled: {config['enable_sca']}", "DEBUG")
            return True

        except Exception as e:
            self.log(f"Error configuring http_calibration.py: {str(e)}", "ERROR")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return False
            
    def configure_http_operation(self, config, output_dir):
            """Configure http_operation.py with parameters from GUI"""
            try:
                if not os.path.exists('http_operation.py'):
                    self.log("http_operation.py not found, skipping configuration", "WARNING")
                    return True
    
                # Read current file
                with open('http_operation.py', 'r') as f:
                    content = f.read()
    
                # Replace configuration variables with GUI values
                # ROUTER_IP configuration
                content = re.sub(
                    r"ROUTER_IP\s*=\s*['\"][^'\"]*['\"]",
                    f"ROUTER_IP = '{config['router_ip']}'",
                    content
                )
    
                # OSCILLOSCOPE_IP configuration
                content = re.sub(
                    r"OSCILLOSCOPE_IP\s*=\s*['\"][^'\"]*['\"]",
                    f"OSCILLOSCOPE_IP = '{config['oscilloscope_ip']}'",
                    content
                )
    
                # TARGET_CGI configuration
                content = re.sub(
                    r"TARGET_CGI\s*=\s*['\"][^'\"]*['\"]",
                    f"TARGET_CGI = '{config['target_cgi']}'",
                    content
                )
    
                # Update CRASH_DIRS to point to current campaign queue
                firmadyne_id = config['firmadyne_id']
                queue_path = f"{config['host_path']}/image_{firmadyne_id}/{output_dir}/queue"
                
                # Replace CRASH_DIRS array
                crash_dirs_replacement = f'''CRASH_DIRS = [
        "{queue_path}"
    ]'''
                content = re.sub(
                    r'CRASH_DIRS\s*=\s*\[[^\]]*\]',
                    crash_dirs_replacement,
                    content,
                    flags=re.DOTALL
                )
    
                # Configure SCA parameters if enabled
                if config['enable_sca']:
                    # Update SCA_CHANNEL
                    content = re.sub(
                        r"SCA_CHANNEL\s*=\s*['\"][^'\"]*['\"]",
                        f"SCA_CHANNEL = '{config['sca_channel']}'",
                        content
                    )
    
                    # Update sample rate (convert from GS/s to Hz)
                    try:
                        sample_rate_gs = float(config['sca_sample_rate'])
                        sample_rate_hz = int(sample_rate_gs * 1e9)
                        content = re.sub(
                            r'SCA_SAMPLE_RATE\s*=\s*[^#\n]*',
                            f'SCA_SAMPLE_RATE = {sample_rate_hz}  # {sample_rate_gs} GS/s',
                            content
                        )
                    except ValueError:
                        self.log("Invalid sample rate, using default", "WARNING")
    
                    # Update voltage division
                    try:
                        voltage_div = float(config['sca_voltage_div'])
                        content = re.sub(
                            r'SCA_VOLTAGE_DIV\s*=\s*[^#\n]*',
                            f'SCA_VOLTAGE_DIV = {voltage_div:g}  # Voltage division in V/div',
                            content
                        )
                    except ValueError:
                        self.log("Invalid voltage division, using default", "WARNING")
    
                    # Update time division
                    try:
                        time_div = config['sca_time_div']
                        content = re.sub(
                            r"SCA_TIME_DIV\s*=\s*['\"][^'\"]*['\"]",
                            f"SCA_TIME_DIV = '{time_div}'",
                            content
                        )
                    except:
                        self.log("Invalid time division, using default", "WARNING")
    
                    # Update memory size
                    try:
                        memory_size = config['sca_memory_size']
                        content = re.sub(
                            r"SCA_MEMORY_SIZE\s*=\s*['\"][^'\"]*['\"]",
                            f"SCA_MEMORY_SIZE = '{memory_size}'",
                            content
                        )
                    except:
                        self.log("Invalid memory size, using default", "WARNING")
    
                    # Update trigger level
                    try:
                        trigger_level = float(config['sca_trigger_level']) if 'sca_trigger_level' in config else 0.02
                        content = re.sub(
                            r'SCA_TRIGGER_LEVEL\s*=\s*[^#\n]*',
                            f'SCA_TRIGGER_LEVEL = {trigger_level:g}  # Trigger level',
                            content
                        )
                    except ValueError:
                        self.log("Invalid trigger level, using default", "WARNING")
    
                # Write modified file
                with open('http_operation.py', 'w') as f:
                    f.write(content)
    
                self.log("http_operation.py configured with GUI parameters", "SUCCESS")
                self.log(f"Router IP: {config['router_ip']}", "DEBUG")
                self.log(f"Oscilloscope IP: {config['oscilloscope_ip']}", "DEBUG")
                self.log(f"Target CGI: {config['target_cgi']}", "DEBUG")
                self.log(f"Queue path: {queue_path}", "DEBUG")
                self.log(f"SCA Enabled: {config['enable_sca']}", "DEBUG")
                return True
    
            except Exception as e:
                self.log(f"Error configuring http_operation.py: {str(e)}", "ERROR")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
                return False
    
    def get_latest_anomaly_timestamp(self):
        """Get latest timestamp from anomaly_names files - FIXED VERSION"""
        try:
            config = self.get_config()
            firmadyne_id = config['firmadyne_id']
            host_path = config['host_path']
            
            # Look in the anomaly_names directory
            anomaly_names_dir = f"{host_path}/image_{firmadyne_id}/anomaly_names"
            
            print(f"[CopyResume] Looking for anomaly files in: {anomaly_names_dir}")
            
            if not os.path.exists(anomaly_names_dir):
                print(f"[CopyResume] Anomaly names directory not found: {anomaly_names_dir}")
                return None
            
            anomaly_files = []
            for file in os.listdir(anomaly_names_dir):
                if file.startswith('anomaly_names') and file.endswith('.txt'):
                    print(f"[CopyResume] Found anomaly file: {file}")
                    # Extract campaign number and timestamp from filename: anomaly_names_X_Y.txt
                    match = re.search(r'anomaly_names_(\d+)_(\d{4}_\d{2}_\d{2}_\d{2}o\d{2})\.txt', file)
                    if match:
                        campaign_num = int(match.group(1))
                        timestamp = match.group(2)
                        anomaly_files.append((file, campaign_num, timestamp))
                        
            if not anomaly_files:
                print("[CopyResume] No anomaly files found with correct format")
                return None
                
            # Sort by campaign number (descending) then by timestamp (descending) and take most recent
            anomaly_files.sort(key=lambda x: (x[1], x[2]), reverse=True)
            latest_file, latest_campaign, latest_timestamp = anomaly_files[0]
            
            print(f"[CopyResume] Latest anomaly file: {latest_file}")
            print(f"[CopyResume] Campaign: {latest_campaign}, Timestamp: {latest_timestamp}")
            
            # Return both campaign and timestamp for the CopyResumeManager
            return {
                'timestamp': latest_timestamp,
                'campaign': latest_campaign,
                'filename': latest_file,
                'full_path': os.path.join(anomaly_names_dir, latest_file)
            }
            
        except Exception as e:
            print(f"[CopyResume] Error getting timestamp: {str(e)}")
            return None
            
    def configure_copy_and_resume(self, config, timestamp):
        """Configure copy_and_resume_fuzzing.py"""
        try:
            if not os.path.exists('copy_and_resume_fuzzing.py'):
                self.log("copy_and_resume_fuzzing.py not found, skipping configuration", "WARNING")
                return True
                
            # Read current file
            with open('copy_and_resume_fuzzing.py', 'r') as f:
                content = f.read()
                
            # Modify variables
            content = re.sub(r'timestamp\s*=.*', f'timestamp = "{timestamp}"', content)
            content = re.sub(r'container_name\s*=.*', f'container_name = "{config["container_name"]}"', content)
            
            # Write modified file
            with open('copy_and_resume_fuzzing.py', 'w') as f:
                f.write(content)
                
            self.log(f"copy_and_resume_fuzzing.py configured", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Error configuring copy_and_resume_fuzzing.py: {str(e)}", "ERROR")
            return False
            
    def wait_with_pause_check(self, seconds):
        """Wait specified time respecting pauses - IMPROVED VERSION with better threading"""
        self.raw_print(f"STARTING {seconds} SECOND WAIT")
        self.log(f"Starting {seconds} second countdown", "INFO")
        
        # Use a more robust approach with threading Event
        import threading
        
        stop_event = threading.Event()
        
        def countdown_worker():
            for i in range(seconds):
                if stop_event.is_set() or not self.automation_state['is_running']:
                    self.raw_print("COUNTDOWN STOPPED - automation halted")
                    return
                    
                remaining = seconds - i
                
                # Direct console output
                msg = f"COUNTDOWN: {remaining-1} seconds left"
                print(msg, flush=True)
                
                # Update GUI safely
                try:
                    self.root.after_idle(lambda m=msg: self.raw_print(m))
                    
                    # Log every 10 seconds or last 5 seconds
                    if remaining % 10 == 0 or remaining <= 5:
                        level = "WARNING" if remaining <= 5 else "INFO"
                        self.root.after_idle(
                            lambda r=remaining-1, l=level: 
                            self.log(f"COUNTDOWN: {r} seconds remaining", l)
                        )
                except:
                    pass  # GUI might not be available
                
                # Wait 1 second or until stop event
                if stop_event.wait(timeout=1.0):
                    self.raw_print("COUNTDOWN INTERRUPTED")
                    return
                    
            print("COUNTDOWN FINISHED", flush=True)
            try:
                self.root.after_idle(lambda: self.raw_print("COUNTDOWN FINISHED"))
                self.root.after_idle(lambda: self.log("Countdown completed", "SUCCESS"))
            except:
                pass
        
        # Start countdown in daemon thread
        countdown_thread = threading.Thread(target=countdown_worker, daemon=True)
        countdown_thread.start()
        
        # Wait for countdown to complete or stop
        countdown_thread.join()
        
        # Clean up
        stop_event.set()
        
        if self.automation_state['is_running']:
            self.log(f"WAIT COMPLETED - {seconds} seconds", "SUCCESS")
        else:
            self.log("WAIT INTERRUPTED - automation stopped", "WARNING")

    def emergency_container_reset(self, config):
        """Emergency reset of container when everything else fails"""
        try:
            self.log("EMERGENCY: Performing container reset...", "ERROR")
            self.raw_print("=== EMERGENCY CONTAINER RESET ===")
            
            container_name = config['container_name']
            
            # Force kill container
            kill_cmd = ['docker', 'kill', container_name]
            subprocess.run(kill_cmd, capture_output=True, text=True, timeout=30)
            
            # Force remove container
            remove_cmd = ['docker', 'rm', '-f', container_name]
            subprocess.run(remove_cmd, capture_output=True, text=True, timeout=30)
            
            # Wait a bit
            time.sleep(5)
            
            # Relaunch container
            if self.launch_docker_container(config):
                self.log("Emergency container reset successful", "SUCCESS")
                self.raw_print("=== EMERGENCY RESET COMPLETED ===")
                return True
            else:
                self.log("Emergency container reset failed", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error in emergency reset: {e}", "ERROR")
            return False


    def cleanup_processes(self):
        """Clean all processes"""
        try:
            self.log("Starting comprehensive cleanup...", "INFO")
            config = self.get_config()
            
            # Clean run_sh process
            if 'run_sh' in self.automation_state['processes']:
                proc = self.automation_state['processes']['run_sh']
                if proc and proc.poll() is None:
                    proc.terminate()
                    self.log("run_sh process terminated", "DEBUG")
            
            # Clean monitoring processes
            if 'pause_monitor' in self.automation_state['processes']:
                proc = self.automation_state['processes']['pause_monitor']
                if proc and proc.poll() is None:
                    proc.terminate()
                    self.log("pause_monitor terminated", "DEBUG")
                    
            # Clean user_sh process
            if 'user_sh' in self.automation_state['processes']:
                proc = self.automation_state['processes']['user_sh']
                if proc and proc.poll() is None:
                    proc.terminate()
                    self.log("user_sh terminated", "DEBUG")
            
            # FIXED: Clean up temporary pause_monitor script
            if hasattr(self, 'pause_monitor_script_path') and os.path.exists(self.pause_monitor_script_path):
                try:
                    os.unlink(self.pause_monitor_script_path)
                    self.log("Temporary pause_monitor script removed", "DEBUG")
                except Exception as e:
                    self.log(f"Error removing temp script: {e}", "DEBUG")
                    
            # Clean fuzzing processes
            self.cleanup_fuzzing_processes(config['container_name'])
            
            # Clean Docker container
            self.log("Removing Docker container...", "DEBUG")
            result = subprocess.run(['docker', 'rm', '-f', config['container_name']], 
                          capture_output=True, text=True)
            if result.returncode == 0:
                self.log("Docker container removed", "SUCCESS")
            else:
                self.log(f"Container removal result: {result.stderr}", "DEBUG")
                          
            self.log("Cleanup completed", "SUCCESS")
            
        except Exception as e:
            self.log(f"Error in cleanup: {str(e)}", "WARNING")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")

 

           
def main():
    """Main function"""
       
    # Force unbuffered output
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Try to reconfigure stdout if available, otherwise skip
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        # Not available in this environment, use alternative
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # Line buffered
    except:
        # If all fails, continue without buffering changes
        pass
    
    # Create main window
    root = tk.Tk()
    
    # Configure icon and style
    try:
        root.iconify()  # Minimize temporarily
        root.deiconify()  # Restore
    except:
        pass
        
    # Create application
    app = TRENTIFuzzingGUI(root)
    
    # Configure window closing
    def on_closing():
        if app.automation_state['is_running']:
            if messagebox.askokcancel("Close TRENTI", 
                                    "Are you sure you want to close TRENTI\n"
                                    "This will stop all automation in progress."):
                app.stop_automation()
                root.destroy()
        else:
            root.destroy()
            
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Configure signal handling
    def signal_handler(signum, frame):
        print(f"\n[TRENTI] Signal {signum} received. Closing TRENTI...", flush=True)
        app.stop_automation()
        root.quit()
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Show startup information with immediate output
    print("=" * 50, flush=True)
    print("TRENTI - Fuzzing Automation Tool with Side-Channel Analysis", flush=True)
    print("=" * 50, flush=True)
    print("Starting graphical interface...", flush=True)
    print("Press Ctrl+C to exit", flush=True)
    print("=" * 50, flush=True)
    
    try:
        # Start main loop
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[TRENTI] Interrupted by user", flush=True)
        app.stop_automation()
    except Exception as e:
        print(f"[TRENTI] Fatal error: {e}", flush=True)
        app.stop_automation()
    finally:
        print("[TRENTI] TRENTI closed", flush=True)


if __name__ == "__main__":
    main()