# Network Traffic Anomaly Detector (NDAT)

A flow-based network anomaly detector in two levels: a rule-based detector (Level 1) and a machine-learning classifier evaluated for deployment on a compute-in-memory (CIM) hardware macro (Level 2).

## Level 1: rule-based detector (complete)

Python + Scapy + Streamlit. Captures packets, aggregates them into flows, and applies detection rules.

| File | Role |
|---|---|
| `capture.py` | Scapy packet capture and flow aggregation |
| `rules.py` | Rules: PORT_SCAN, TRAFFIC_FLOOD, KNOWN_BAD_PORT, BEACONING |
| `main.py` | Threaded orchestration; writes live state to JSON |
| `dashboard.py` | Streamlit dashboard, deliberately run as a separate non-root process that reads the shared JSON state |

Testing: nmap scans and ICMP ping floods against an isolated Debian VM (UTM) and a home router. A 3.5-hour soak test (3,220 flows) produced 13 false positives, all TRAFFIC_FLOOD. They were fixed by requiring both a volume gate (500 packets minimum) and a rate check, with an epsilon floor on duration.

Run (capture needs raw-socket access):

    source venv/bin/activate
    sudo venv/bin/python3 main.py
    streamlit run dashboard.py     # in a second terminal, no sudo

## Level 2: ML classifier and CIM-macro error study (in progress)

Question: can a small flow-based classifier, quantized to FP8 (E4M3), stay accurate when run through the error of a hybrid digital/analog CIM macro?

**Data.** CIC-IDS2017 packet captures are replayed through this project's own flow pipeline (`replay_pcap.py`, `build_training_csv.py`), so the features come from the same code as Level 1. Splits are time-based.

**Baseline (logistic regression, corrected pipeline).**
- Recall **76.4% at 1% false-positive rate**, PR-AUC ≈ 0.46.
- FP8 (E4M3) quantization: **no measurable change** versus the float model (`quantize_weights.py`, `quantize_eval.py`).
- An earlier version of the pipeline was found to be invalid and its results were withdrawn.

**CIM macro.** The macro is a 4x4 mantissa-precision design from Prof. Weidong Cao's lab. It is simulated at SPICE level (Xschem + ngspice, sky130); the simulation files are kept outside this repo. What has and has not been done:
- The published error figure (~1.5% MAC error) belongs to the lab's own design. It has **not** been measured on my netlist.
- My current netlist uses a simplified capacitor array. Under multi-row activation it returns the bitwise AND of stored codes rather than a sum, so it does not yet match the lab's intended topology. I am confirming the intended design with the lab and will rebuild that stage.
- Applying the readout error of the single-row path of my netlist to the FP8 weights costs no detectable recall (75.9% vs 76.4% at 1% FPR; only 195 test attacks, so about ±6 points of uncertainty). Activation-side error and the lab macro itself are **not yet evaluated**.

## Limitations
- Simulation only; no fabricated hardware.
- One dataset (CIC-IDS2017) and one model family so far.
- CIC-IDS2017 is replayed through a custom flow pipeline, so results are not directly comparable to published numbers using CICFlowMeter features.

## Repository map

Scripts are run from the repo root (they read and write files by relative path).

**Level 1 (live detection)**
- `capture.py`: Scapy capture and flow aggregation (also reused by the replay scripts)
- `rules.py`: detection rules
- `main.py`: runs capture and rules, writes `dashboard_state.json` and `flow_log.jsonl`
- `dashboard.py`: Streamlit dashboard reading `dashboard_state.json`
- `run_attacks.sh`: runs attacks against the VM and logs start/end times to `attack_log.csv`
- `label_flows.py`: joins `flow_log.jsonl` with `attack_log.csv` into `training_data.csv`

**Level 2 (ML on CIC-IDS2017)**
- `replay_pcap.py`, `capture_cic.py`, `cic_replay.py`: replay CIC-IDS2017 pcaps through the flow pipeline, with labels from the dataset CSVs
- `build_training_csv.py`: flow records to `training_data.csv`
- `train_model_v3.py`: trains the classifier, saving `.joblib` and `_weights.json`
- `apply_model.py`: applies a trained model to a flows file
- `quantize_eval.py`: FP8 quantization and noise-injection evaluation (`--noise-model product|fullscale`)
- `quantize_weights.py`, `evaluate_curves.py`: FP8 check and PR/ROC curves for the original VM-dataset model (`model.joblib`, `scaler.joblib`)

**Folders**
- `baseline_v1/`: frozen baseline model (`model_v3_lean_seen`) used by `quantize_eval.py`
- `results/`: quantization result CSVs and evaluation plot
- `archive/`: superseded model versions and early training scripts

Large data (CIC-IDS2017 pcaps and replayed flow files) is not in this repo; download CIC-IDS2017 separately into `cic_data/`.
