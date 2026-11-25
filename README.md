# Network Traffic Anomaly Detector (NDAT)

A flow-based network anomaly detector in two levels: a rule-based detector (Level 1) and a machine-learning classifier evaluated for deployment on a compute-in-memory (CIM) hardware macro (Level 2).

## Level 1: Rule-Based Detector (Complete)

Python + Scapy + Streamlit. Captures packets, aggregates them into flows, and applies detection rules.

| File | Role |
| :--- | :--- |
| `capture.py` | Scapy packet capture and flow aggregation |
| `rules.py` | Rules: `PORT_SCAN`, `TRAFFIC_FLOOD`, `KNOWN_BAD_PORT`, `BEACONING` |
| `main.py` | Threaded orchestration; writes live state to JSON and flow logs |
| `dashboard.py` | Streamlit dashboard, deliberately run as a separate non-root process reading shared state |

**Testing:** `nmap` scans and ICMP ping floods against an isolated Debian VM (UTM) and a home router. A 3.5-hour soak test (3,220 flows) produced 13 false positives, all `TRAFFIC_FLOOD`, which were addressed by requiring both a volume gate (500 packets minimum) and a rate check with an epsilon floor on duration.

**Run** (capture requires raw-socket access):

```bash
source venv/bin/activate
sudo venv/bin/python3 main.py
streamlit run dashboard.py     # in a second terminal, no sudo
```

## Level 2: ML Classifier and CIM-Macro Error Study (In Progress)

**Question:** Can a small flow-based classifier, quantized to FP8 (`E4M3`), stay accurate when run through the error of a hybrid digital/analog CIM macro?

* **Data:** CIC-IDS2017 packet captures are replayed through this project's flow pipeline (`cic_replay.py`, `capture_cic.py`), using time-based splits.
* **Baseline (3-feature logistic regression, corrected pipeline):**
  * Recall **76.4%** at a **1%** false-positive rate (`PR-AUC ≈ 0.46`, `144,083` test flows with `195` attacks).
  * FP8 (`E4M3`) quantization and single-row readout error evaluation show **75.9%** recall at **1%** FPR (within ~±6 percentage points given 195 test attacks), with simulated error sweeps remaining stable up to ~5% before degrading at higher error levels.
  * An earlier version of the pipeline and its metrics were found to be invalid and withdrawn.

### CIM Macro

The macro is a 4x4 mantissa-precision design from Prof. Weidong Cao's lab, simulated at the SPICE level (`Xschem` + `ngspice`, `sky130`); full simulation files are kept outside this repository.

* Published lab error figures (~1.5% MAC error / ~30 dB SINAD) come from the lab's published work and have not yet been measured on my netlist.
* My earlier netlist exhibited shared-line contention under multi-row activation rather than clean multi-row summation; topology/branch isolation and readout/ADC details are being confirmed and updated with the lab.
* Applying the single-row readout error model to FP8 weights yields **75.9%** vs. **76.4%** recall at 1% FPR (`195` test attacks, ~±6 percentage points uncertainty). Activation-side error and full lab macro integration remain ongoing work.

## Development Note

Developed with AI coding assistance. I chose the experiments, ran the simulations and evaluations, and verified the results; limitations are documented below so claims can be audited.

## Limitations

* Simulation only; no fabricated hardware.
* Evaluated on one dataset (CIC-IDS2017) and one model family so far.
* CIC-IDS2017 is replayed through a custom flow pipeline, so results are not directly comparable to published numbers using CICFlowMeter features.

## Repository Map

Scripts are run from the repo root (reading and writing files by relative path).

### Active Level 1 & Level 2 Scripts
* `capture.py`: Scapy capture and flow aggregation (reused by replay scripts)
* `rules.py`: Detection rules
* `main.py`: Runs capture and rules, writes `dashboard_state.json` and `flow_log.jsonl`
* `dashboard.py`: Streamlit dashboard reading `dashboard_state.json`
* `capture_cic.py`, `cic_replay.py`: Replay CIC-IDS2017 pcaps through the flow pipeline with labels from dataset CSVs
* `train_model_v3.py`: Trains the classifier, saving `.joblib` and `_weights.json`
* `apply_model.py`: Applies a trained model to a flows file
* `quantize_eval.py`: FP8 quantization and noise-injection evaluation

### Folders
* `baseline_v1/`: Frozen baseline model (`model_v3_lean_seen`) used by `quantize_eval.py`
* `results/`: Quantization result CSVs and evaluation plot
* `archive/`: Legacy scripts (`archive/scripts/`), earlier models (`archive/models/`), earlier dataset CSVs (`archive/data/`), and run logs (`archive/logs/`)

Large datasets (CIC-IDS2017 pcaps and replayed flow files) are not tracked in this repository; keep CIC-IDS2017 locally under `cic_data/`.
