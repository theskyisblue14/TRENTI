# TRENTI Firmware Fuzzing GUI

This repository contains the graphical user interface (GUI) for **TRENTI**, a side‑channel‑guided grey‑box fuzzing framework for IoT firmware and binaries. The GUI enables you to configure, launch and monitor fuzzing campaigns that integrate electromagnetic (EM) feedback in real time.

## Features

* **Live Dashboard** View execution throughput, crash statistics and side‑channel anomaly scores as they happen.
* **Flexible Fuzzer Support** Compatible with AFL++ and FIRM‑AFL back‑ends.
* **Device Management** Connect to multiple hardware targets and manage sessions seamlessly.
* **Side-Channel Integration** Real-time electromagnetic emission analysis with configurable oscilloscope parameters.
* **Result Export** Save logs, crash dumps and coverage bitmaps for later analysis.

## Prerequisites

* **Operating System** Linux (Ubuntu 18.04 or later recommended)
* **Python** Version 3.8 or newer
* **Hardware Requirements**
  * High‑bandwidth oscilloscope (≥ 100 MHz) with EM probe
  * Target IoT device for real-world testing
  * Stable network connection between host and device under test (DUT)
* **Software Dependencies** All Python dependencies are listed in `requirements.txt`
* **Optional** Docker (for containerised back‑end)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/trenti-gui.git
cd trenti-gui
```

2. **Install Python packages**
```bash
# Using conda (recommended)
conda create --name trenti-env --file requirements.txt
conda activate trenti-env

# Or using pip
pip install pyqt5 numpy scipy scikit-learn hdbscan matplotlib pandas docker paramiko
```

3. **Download firmware images**
   
   Download the Firmadyne images from https://mega.nz/file/nspQSaTY#HgUOuav18EOxWGheSrF6MlESZq4sgMmtfslUG058XQ0 and extract to the `images/` folder.

## Usage

<img width="1787" height="973" alt="TRENTI GUI Interface" src="https://github.com/user-attachments/assets/6f804df8-4773-42f3-bc40-712f05d184db" />

### Launch the GUI

```bash
python3 TRENTI_GUI.py
```

### Configuration Workflow

The GUI is organized into several tabs for comprehensive fuzzing configuration:

#### 1. Configuration Tab
This tab handles TRENTI's firmware fuzzing setup and consists of four main sections:

**TRENTI Configuration**
* Specify the Firmadyne ID for evaluation
* Configure connection parameters to the real device under test (DUT)
* Select from tested configurations:

| Experiment ID | Firmadyne ID | Arch    | Vendor   | Model      | Program      |
|---------------|--------------|---------|----------|------------|--------------|
| 9050          | 9050         | mipsel  | D-Link   | DIR-815    | hedwig.cgi   |
| 161160        | 16116        | mips    | TRENDnet | TEW-632BRP | miniupnpd    |
| 161161        | 16116        | mips    | TRENDnet | TEW-632BRP | httpd        |

**Side-Channel Analysis Configuration**
* Configure oscilloscope parameters remotely
* Set measurement channel and axis scaling
* Adjust sample rate and acquisition settings
* Select measurement parameters (execution time, EM emission, etc.)

**HTTP/Curl Configuration**
* Define curl commands for DUT communication
* Configure seed generation and queue execution
* Includes presets for the three experiments from the paper
* Set parameters for side-channel anomaly detection model

**Testing Options**
* Adjust test thresholds for early termination
* Configure debugging parameters
* Set conventional fuzzing duration limits

**Execution Control**
* **Start TRENTI** button to begin fuzzing campaign
* Pause/resume functionality for long-running tests
* Stop button for emergency termination
* Test connection to DUT before starting

#### 2. Monitoring and Analysis
* **Real-time Charts** Observe bit-coverage, anomaly rate, and crash count
* **Live Feedback** Monitor EM anomaly scores as they occur

#### 3. Results and Export
* Use the **Export** menu to save session data
* Logs and coverage bitmaps stored in `./outputs/`
* Side-channel anomaly reports and crash dumps available for analysis

## Artifact Links

* **Firmware SoA Results** (≈ 244 MB): https://mega.nz/file/uxB2GLzR#UdOjjWp7bGFHdTfj49SR_MR1PYBRC4R-sAjAPXTikB0
* **Firmadyne Evaluation Images** (≈ 1.1 GB): https://mega.nz/folder/XkAFgbpb#LpX2YqgYuHj28kYpqdcqAA
* **Binary TRENTI Results** (≈ 17.5 GB):
  1. html2xhtml (9.25GB): https://mega.nz/folder/rxAQBYjS#jeewkwzg2r3RSrJiF8DCEQ
  2. picoc-CVE-2022-34556 (8.29GB): https://mega.nz/folder/Xsh0zZhQ#wv2gBjE0EV2TMv4w6jGt3Q

## Directory Structure

```text
.
├── TRENTI_GUI.py           # Main GUI application
├── anomaly_deduplicator.py # Post‑processing of side‑channel anomalies
├── http_calibration.py     # EM‑based calibration routines
├── http_operation.py       # Live fuzzing loop with side‑channel feedback
├── figures/                # Sample charts for publication
├── images/                 # Firmadyne evaluation images (download separately)
├── evaluations/            # Results from firmware-fuzzing evaluations
├── SoA_results/            # State‑of‑the‑art result set (download separately)
├── outputs/                # Generated logs, crash dumps, and coverage data
└── requirements.txt        # Conda environment export (complete dependencies)
```

## Research Context

This GUI implements the methodology described in our USENIX Security 2026 paper. The tool enables researchers to reproduce the experimental results and extend the work to new IoT targets. The three evaluated programs (hedwig.cgi, miniupnpd, httpd) demonstrate TRENTI's effectiveness across different firmware architectures and vendors.

## Troubleshooting

* **Connection Issues**: Verify network connectivity to DUT using the "Test Connection" feature
* **Oscilloscope Problems**: Check remote control settings and ensure proper EM probe placement
* **Performance**: For better results, ensure stable power supply to DUT and minimize electromagnetic interference

## Citation

If you use this GUI in your research, please cite:

```bibtex
@inproceedings{trenti2026,
  title={TRENTI: Embedded Fuzzing with Multimodal Side-Channel Feedback},
  author={A. Author and B. Author and C. Author},
  booktitle={Proceedings of the 11th IEEE European Symposium on Security and Privacy},
  year={2026},
  month={July}
}
```

## Contact

For questions, bug reports, or collaboration inquiries, please:
* Open an issue on GitHub
* Contact the maintainers via the repository
* Refer to the paper for detailed methodology and experimental setup

---

**Note**: This is research software intended for academic use. Please ensure compliance with local laws and ethical guidelines when testing on IoT devices.
