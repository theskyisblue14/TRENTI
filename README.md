# TRENTI Firmware Fuzzing GUI

This repository contains the graphical user interface (GUI) for **TRENTI**, a side‑channel‑guided grey‑box fuzzing framework for IoT firmware and binaries. The GUI enables you to configure, launch and monitor fuzzing campaigns that integrate electromagnetic (EM) feedback in real time.

## Features

- **Live Dashboard**  
  View execution throughput, crash statistics and side‑channel anomaly scores as they happen.

- **Flexible Fuzzer Support**  
  Compatible with AFL++, libFuzzer and FIRM‑AFL back‑ends.

- **Device Management**  
  Connect to multiple hardware targets, calibrate EM probes and manage sessions seamlessly.

- **Result Export**  
  Save logs, crash dumps and coverage bitmaps for later analysis.

## Prerequisites

- **Operating System**  
  Linux (Ubuntu 18.04 or later recommended)

- **Python**  
  Version 3.8 or newer

- **Dependencies**  
  ```bash
  pip install pyqt5 numpy scipy scikit-learn hdbscan
  ```
  Moreover, you will need:  
  - A high‑bandwidth oscilloscope (≥ 100 MHz) with EM probe  
  - Firmware images or binaries for your target device  
  - Docker (optional, for containerised back‑end)

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/trenti-gui.git
   cd trenti-gui
   ```

2. **Install Python packages**  
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Start Docker containers**  
   ```bash
   docker-compose up -d
   ```

## Usage

<img width="1787" height="973" alt="image" src="https://github.com/user-attachments/assets/6f804df8-4773-42f3-bc40-712f05d184db" />


Launch the GUI:

```bash
python3 TRENTI_GUI.py
```

1. **Probe Calibration**  
   Select your EM channel and run calibration to establish baseline noise patterns.

2. **Fuzzing Session**  
   Choose a fuzzer back‑end, load your firmware image and specify your input corpus.

3. **Monitoring**  
   Observe real‑time charts for bit‑coverage, anomaly rate and crash count.

4. **Export**  
   Use the **Export** menu to save session data; logs and bitmaps will be stored in `./outputs/`.

## Artifact Links

- **SoA Results** (≈ 244 MB):  
  https://mega.nz/file/uxB2GLzR#UdOjjWp7bGFHdTfj49SR_MR1PYBRC4R-sAjAPXTikB0

- **Evaluation Images** (≈ 1.1 GB):  
  https://mega.nz/folder/XkAFgbpb#LpX2YqgYuHj28kYpqdcqAA

## Directory Structure

```text
.
├── TRENTI_GUI.py           # Main GUI application
├── anomaly_deduplicator.py # Post‑processing of side‑channel anomalies
├── http_calibration.py     # EM‑based calibration routines
├── http_operation.py       # Live fuzzing loop with side‑channel feedback
├── figures/                # Sample charts for publication
├── evaluations/            # Binary images for evaluation (download separately)
├── SoA_results.7z          # State‑of‑the‑art result set (download separately)
└── requirements.txt        # Python dependencies
```

## Citation

If you use this GUI in your research, please cite:

> A. Author et al., “TRENTI: Side‑Channel Guided Grey‑Box Fuzzing for IoT Firmware and Binaries,” *USENIX Security 2025*, July 2025

---

For questions or feedback, please open an issue or contact the maintainers.
