import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

# Complete mapping of experiment IDs to their details
EXPERIMENT_MAPPING = {
    9925: {
        "firmadyne_id": 9925,
        "arch": "mips",
        "vendor": "DLink",
        "model": "DAP-2695",
        "program": "httpd",
        "firmware": "DAP-2695_REVA_FIRMWARE_1.11.RC044.ZIP"
    },
    9050: {
        "firmadyne_id": 9050,
        "arch": "mipsel",
        "vendor": "DLink",
        "model": "DIR-815",
        "program": "hedwig.cgi",
        "firmware": "DIR-815_FIRMWARE_1.01.ZIP"
    },
    9054: {
        "firmadyne_id": 9054,
        "arch": "mips",
        "vendor": "DLink",
        "model": "DIR-817LW",
        "program": "hnap",
        "firmware": "DIR-817LW_REVA_FIRMWARE_1.00B05.ZIP"
    },
    10566: {
        "firmadyne_id": 10566,
        "arch": "mips",
        "vendor": "DLink",
        "model": "DIR-850L",
        "program": "hnap",
        "firmware": "DIR-850L_FIRMWARE_1.03.ZIP"
    },
    10853: {
        "firmadyne_id": 10853,
        "arch": "mips",
        "vendor": "DLink",
        "model": "DIR-825",
        "program": "httpd",
        "firmware": "DIR-825_REVB_FIRMWARE_2.02.ZIP"
    },
    161161: {
        "firmadyne_id": 16116,
        "arch": "mips",
        "vendor": "Trendnet",
        "model": "TEW-632BRP",
        "program": "httpd",
        "firmware": "tew-632brpa1_(fw1.10b32).zip"
    },
    161160: {
        "firmadyne_id": 16116,
        "arch": "mips",
        "vendor": "Trendnet",
        "model": "TEW-632BRP",
        "program": "miniupnpd",
        "firmware": "tew-632brpa1_(fw1.10b32).zip"
    },
    12978: {
        "firmadyne_id": 12978,
        "arch": "mips",
        "vendor": "Trendnet",
        "model": "TV-IP110WN",
        "program": "video.cgi",
        "firmware": "fw_tv-ip110wn_v2(1.2.2.68).zip"
    },
    12981: {
        "firmadyne_id": 12978,
        "arch": "mips",
        "vendor": "Trendnet",
        "model": "TV-IP110WN",
        "program": "network.cgi",
        "firmware": "fw_tv-ip110wn_v2(1.2.2.68).zip"
    }
}

# Set the experiment ID here - CHANGE THIS TO YOUR DESIRED EXPERIMENT ID
EXPERIMENT_ID = 161161

# Base paths for different methods
BASE_PATH_SOA = "/home/atenea/trenti/SoA_results"  # For FirmAFL and FSE
BASE_PATH_TRENTI = "/home/atenea/trenti/evaluations"  # For TRENTI

# Define consistent colors and styles for better distinction
PLOT_STYLES = {
    'FIRM-AFL': {
        'color': 'black',
        'linestyle': 'solid',
        'linewidth': 1,
        'marker': 'o',
        'markersize': 4,
        'markevery': 28
    },
    'Full-System Emulation': {
        'color': 'Blue',
        'linestyle': 'dashed',
        'linewidth': 1,
        'marker': 's',
        'markersize': 4,
        'markevery': 28
    },
    'TRENTI': {
        'color': 'Green',
        'linestyle': 'dashdot',
        'linewidth': 1,
        'marker': 'None',
        'markersize': 4,
        'markevery': 28
    }
}

def get_plot_title():
    """Generate plot title based on experiment ID"""
    if EXPERIMENT_ID in EXPERIMENT_MAPPING:
        exp_info = EXPERIMENT_MAPPING[EXPERIMENT_ID]
        return f"{exp_info['program']} ({exp_info['vendor']} {exp_info['model']})"
    return f"Experiment {EXPERIMENT_ID}"

def read_and_prepare_data(file_path, label, extend_to_24=False, global_start_time=None):
    column_names = ["unix_time", "cycles_done", "cur_path", "paths_total", "pending_total", 
        "pending_favs", "map_size", "unique_crashes", "unique_hangs", 
        "max_depth", "execs_per_sec"]
    
    try:
        df = pd.read_csv(file_path, names=column_names, delimiter=",", comment='#', skipinitialspace=True)
        
        if df.empty or len(df) == 0:
            print(f"Warning: Empty dataframe from {file_path}")
            return None
            
        print(f"Loaded {len(df)} rows from {file_path}")
        
        # Parse map_size correctly
        def parse_map_size(value):
            if isinstance(value, str):
                if value.endswith('%'):
                    return float(value.rstrip('%'))
                else:
                    try:
                        return float(value)
                    except ValueError:
                        return 0.0
            return float(value) if pd.notna(value) else 0.0
        
        df["map_size"] = df["map_size"].apply(parse_map_size)
        df["timestamp"] = pd.to_datetime(df["unix_time"], unit='s')
        
        # Calculate hours from global start time if provided, otherwise from first timestamp
        if global_start_time is not None:
            df["hours"] = (df["timestamp"] - global_start_time).dt.total_seconds() / 3600
        else:
            df["hours"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 3600
            
        df["label"] = label
        
        if extend_to_24 and df["hours"].max() < 24:
            last_values = df.iloc[-1].copy()
            new_row = pd.DataFrame([last_values])
            new_row["hours"] = 24
            df = pd.concat([df, new_row])
        
        return df
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_numbers_from_string(s):
    """Extract all numbers from a string and return as list of integers"""
    numbers = re.findall(r'\d+', s)
    return [int(n) for n in numbers]

def find_firmafl_datasets():
    """Find all FirmAFL output directories with flexible pattern matching"""
    base_path = os.path.join(BASE_PATH_SOA, f"image_{EXPERIMENT_ID}", "firmafl")
    
    if not os.path.exists(base_path):
        print(f"FirmAFL directory not found: {base_path}")
        return []
    
    # Try multiple patterns
    patterns = [
        f"FirmAFL_*_outputs_{EXPERIMENT_ID}_*",  # Pattern for 161160
        "AFL_c*"  # Pattern for 161161
    ]
    
    directories = []
    for pattern in patterns:
        full_pattern = os.path.join(base_path, pattern)
        found_dirs = glob.glob(full_pattern)
        directories.extend(found_dirs)
    
    # Improved sorting function
    def sort_key(path):
        dir_name = os.path.basename(path)
        
        # For FirmAFL_*_outputs_ID_# pattern
        match = re.search(rf'FirmAFL_.*_outputs_{EXPERIMENT_ID}_(\d+)', dir_name)
        if match:
            return int(match.group(1))
        
        # For AFL_c# pattern - extract the number after 'c'
        match = re.search(r'AFL_c(\d+)', dir_name)
        if match:
            return int(match.group(1))
        
        return 999  # Fallback
    
    directories.sort(key=sort_key)
    print(f"FirmAFL directories found and sorted: {[os.path.basename(d) for d in directories]}")
    return directories

def find_fse_datasets():
    """Find all FSE output directories with improved sorting"""
    base_path = os.path.join(BASE_PATH_SOA, f"image_{EXPERIMENT_ID}", "fse")
    
    if not os.path.exists(base_path):
        print(f"FSE directory not found: {base_path}")
        return []
    
    # Pattern for FSE directories
    pattern = os.path.join(base_path, "FSE_*")
    directories = glob.glob(pattern)
    
    # Improved sorting function
    def sort_key(path):
        dir_name = os.path.basename(path)
        
        # Remove FSE_ prefix
        suffix = dir_name.replace('FSE_', '')
        
        # Extract all numbers from the suffix
        numbers = extract_numbers_from_string(suffix)
        
        if numbers:
            # Sort by the first number, then by total count of numbers (complexity)
            return (numbers[0], len(numbers), numbers)
        
        return (999, 0, [])  # Fallback for directories without numbers
    
    directories.sort(key=sort_key)
    print(f"FSE directories found and sorted: {[os.path.basename(d) for d in directories]}")
    return directories

def find_trenti_datasets():
    """Find all TRENTI output directories"""
    # TRENTI has its own base path, different from SoA_results
    base_path = os.path.join(BASE_PATH_TRENTI, f"image_{EXPERIMENT_ID}")
    
    if not os.path.exists(base_path):
        print(f"TRENTI directory not found: {base_path}")
        return []
    
    print(f"Searching for TRENTI in: {base_path}")
    
    # Find all TRENTI directories matching trenti_sca_outputs_[EXPERIMENT_ID]_*
    pattern = os.path.join(base_path, f"trenti_sca_outputs_{EXPERIMENT_ID}_*")
    directories = glob.glob(pattern)
    
    if directories:
        print(f"Found TRENTI directories: {[os.path.basename(d) for d in directories]}")
    else:
        print(f"No TRENTI directories found with pattern: {pattern}")
    
    # Improved sorting function
    def sort_key(path):
        dir_name = os.path.basename(path)
        # Extract number from trenti_sca_outputs_[EXPERIMENT_ID]_#
        match = re.search(rf'trenti_sca_outputs_{EXPERIMENT_ID}_(\d+)', dir_name)
        if match:
            return int(match.group(1))
        return 999
    
    directories.sort(key=sort_key)
    print(f"TRENTI directories sorted: {[os.path.basename(d) for d in directories]}")
    return directories

def combine_datasets_temporal(df1, df2):
    """Combine two datasets respecting real temporal continuity with gap filling"""
    last_time_df1 = df1["hours"].max()
    first_time_df2 = df2["hours"].min()
    
    print(f"Dataset 1 ends at {last_time_df1:.3f}h, Dataset 2 starts at {first_time_df2:.3f}h")
    
    # If there's a gap, fill it with horizontal line
    if first_time_df2 > last_time_df1:
        gap_duration = first_time_df2 - last_time_df1
        print(f"Found temporal gap of {gap_duration:.3f} hours, filling with horizontal line")
        
        gap_filler = df1.iloc[-1].copy()
        gap_filler["hours"] = first_time_df2 - 0.001
        gap_filler_df = pd.DataFrame([gap_filler])
        df1_with_gap = pd.concat([df1, gap_filler_df], ignore_index=True)
    else:
        df1_with_gap = df1
    
    # For truly cumulative metrics (unique_crashes, unique_hangs)
    last_values_df1 = df1.iloc[-1]
    df2_adjusted = df2.copy()
    
    cumulative_columns = ["unique_crashes", "unique_hangs"]
    for col in cumulative_columns:
        if col in df2_adjusted.columns:
            df2_adjusted[col] = df2_adjusted[col] + last_values_df1[col]
    
    combined_df = pd.concat([df1_with_gap, df2_adjusted], ignore_index=True)
    return combined_df

def combine_datasets_cumulative_paths_temporal(df1, df2):
    """Combine two datasets with cumulative paths calculation"""
    last_time_df1 = df1["hours"].max()
    first_time_df2 = df2["hours"].min()
    
    # If there's a gap, fill it with horizontal line
    if first_time_df2 > last_time_df1:
        gap_filler = df1.iloc[-1].copy()
        gap_filler["hours"] = first_time_df2 - 0.001
        gap_filler_df = pd.DataFrame([gap_filler])
        df1_with_gap = pd.concat([df1, gap_filler_df], ignore_index=True)
    else:
        df1_with_gap = df1
    
    last_values_df1 = df1.iloc[-1]
    df2_adjusted = df2.copy()
    
    # Add cumulative paths
    df2_adjusted["cumulative_paths_total"] = df2_adjusted["paths_total"] + last_values_df1.get("cumulative_paths_total", last_values_df1["paths_total"])
    
    # For other cumulative metrics
    cumulative_columns = ["unique_crashes", "unique_hangs"]
    for col in cumulative_columns:
        if col in df2_adjusted.columns:
            df2_adjusted[col] = df2_adjusted[col] + last_values_df1[col]
    
    combined_df = pd.concat([df1_with_gap, df2_adjusted], ignore_index=True)
    return combined_df

def load_and_combine_data(method_name, cumulative_paths=False):
    """Generic function to load and combine data for any method"""
    if method_name == "FIRM-AFL":
        dirs = find_firmafl_datasets()
    elif method_name == "Full-System Emulation":
        dirs = find_fse_datasets()
    elif method_name == "TRENTI":
        dirs = find_trenti_datasets()
    else:
        print(f"Unknown method: {method_name}")
        return None, []
    
    if not dirs:
        print(f"No {method_name} output directories found for experiment {EXPERIMENT_ID}!")
        return None, []
    
    print(f"Processing {method_name} directories in order: {[os.path.basename(d) for d in dirs]}")
    
    combined_df = None
    concatenation_points = []
    global_start_time = None
    
    for i, data_dir in enumerate(dirs):
        plot_data_path = os.path.join(data_dir, "plot_data")
        
        if not os.path.exists(plot_data_path):
            print(f"Warning: plot_data not found in {data_dir}")
            continue
            
        print(f"Loading data from: {plot_data_path}")
        current_df = read_and_prepare_data(plot_data_path, method_name, 
                                         extend_to_24=False, global_start_time=global_start_time)
        
        if current_df is None:
            print(f"Skipping {data_dir} due to empty or invalid data")
            continue
        
        if combined_df is None:
            # First dataset
            global_start_time = current_df["timestamp"].iloc[0]
            if cumulative_paths:
                current_df["cumulative_paths_total"] = current_df["paths_total"]
            combined_df = current_df
            print(f"First dataset: {os.path.basename(data_dir)} (start time: {global_start_time})")
        else:
            # Subsequent datasets
            last_hour_before_concat = combined_df["hours"].max()
            concatenation_points.append(last_hour_before_concat)
            
            print(f"Concatenating dataset: {os.path.basename(data_dir)}")
            
            if cumulative_paths:
                combined_df = combine_datasets_cumulative_paths_temporal(combined_df, current_df)
            else:
                combined_df = combine_datasets_temporal(combined_df, current_df)
    
    return combined_df, concatenation_points

def extend_to_target_hours(df, target_hours=24):
    """Extend dataframe to target hours with horizontal line (last values)"""
    if df is None or df.empty:
        return df
        
    current_max_hours = df["hours"].max()
    
    if current_max_hours < target_hours:
        print(f"Extending data from {current_max_hours:.2f} hours to {target_hours} hours with horizontal line")
        
        last_values = df.iloc[-1].copy()
        new_row = pd.DataFrame([last_values])
        new_row["hours"] = target_hours
        df_extended = pd.concat([df, new_row], ignore_index=True)
        return df_extended
    
    return df

def plot_with_style(ax, hours, values, method_name):
    """Plot data with consistent styling"""
    style = PLOT_STYLES[method_name]
    ax.plot(hours, values,
           color=style['color'],
           linestyle=style['linestyle'],
           linewidth=style['linewidth'],
           marker=style['marker'],
           markersize=style['markersize'],
           markevery=style['markevery'],
           label=method_name)

def create_separate_legend(output_dir, available_methods):
    """Create a separate legend figure with the same styling as the main plots"""
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.axis('off')
    
    from matplotlib.lines import Line2D
    legend_elements = []
    for method_name in available_methods:
        if method_name in PLOT_STYLES:
            style = PLOT_STYLES[method_name]
            legend_elements.append(
                Line2D([0], [0], 
                      color=style['color'], 
                      linestyle=style['linestyle'], 
                      linewidth=style['linewidth'],
                      marker=style['marker'],
                      markersize=style['markersize'],
                      label=method_name)
            )
    
    legend = ax.legend(handles=legend_elements, 
                      loc='center', 
                      ncol=len(legend_elements), 
                      frameon=False,
                      fontsize=10)
    
    plt.savefig(os.path.join(output_dir, 'legend.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()

def create_plot(data_dict, column, ylabel, filename, output_dir, plot_title):
    """Generic function to create plots"""
    fig, ax = plt.subplots()
    
    for method_name, df in data_dict.items():
        if df is not None:
            plot_with_style(ax, df["hours"], df[column], method_name)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel(ylabel)
    ax.set_title(plot_title)
    ax.grid(True, linestyle='-', alpha=0.3, color='lightgray')
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(0, 24)
    
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    print(f"Processing experiment {EXPERIMENT_ID}")
    print(f"SoA results base path: {BASE_PATH_SOA}")
    print(f"TRENTI base path: {BASE_PATH_TRENTI}")
    
    # Load data for all methods
    methods = ['FIRM-AFL', 'Full-System Emulation', 'TRENTI']
    
    combined_data = {}
    combined_data_cumulative = {}
    available_methods = []
    
    for method in methods:
        print(f"\n--- Processing {method} ---")
        
        # Regular data
        df, _ = load_and_combine_data(method, cumulative_paths=False)
        if df is not None:
            combined_data[method] = extend_to_target_hours(df, 24)
            available_methods.append(method)
            print(f"Successfully loaded {method} data")
        else:
            print(f"No data available for {method}")
        
        # Cumulative paths data
        df_cum, _ = load_and_combine_data(method, cumulative_paths=True)
        if df_cum is not None:
            combined_data_cumulative[method] = extend_to_target_hours(df_cum, 24)
    
    # Check if we have data for at least one method
    if not combined_data:
        print("Error: Could not load data for any method")
        return
    
    print(f"\nAvailable methods: {available_methods}")
    
    # Setup plot parameters
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [3, 2]
    plt.rcParams['font.size'] = 10
    
    # Create output directory
    output_dir = f"fw_comparison_all_methods_{EXPERIMENT_ID}"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_title = get_plot_title()
    
    # Create separate legend
    create_separate_legend(output_dir, available_methods)
    
    # Create all plots
    create_plot(combined_data, 'paths_total', 'Paths', 'paths_total.pdf', output_dir, plot_title)
    
    if combined_data_cumulative:
        create_plot(combined_data_cumulative, 'cumulative_paths_total', 'Paths', 'cumulative_paths_total.pdf', output_dir, plot_title)
    
    create_plot(combined_data, 'map_size', 'Map size', 'map_size.pdf', output_dir, plot_title)
    create_plot(combined_data, 'pending_total', 'Pending paths', 'pending_paths.pdf', output_dir, plot_title)
    create_plot(combined_data, 'execs_per_sec', 'Executions per second', 'execs_per_sec.pdf', output_dir, plot_title)
    create_plot(combined_data, 'unique_crashes', 'Unique crashes found', 'crashes.pdf', output_dir, plot_title)
    
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    main()