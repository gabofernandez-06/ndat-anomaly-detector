"""
label_flows.py — joins flow_log.jsonl (full flow features, from main.py)
against attack_log.csv (attack time windows, from run_attacks.sh) to produce
a labeled training dataset.

Labeling logic: a flow is labeled with whichever attack window it overlaps
most (by time), or NORMAL if it doesn't overlap any window at all.
"""
import json
import csv
from collections import Counter

FLOW_LOG = "flow_log.jsonl"
ATTACK_LOG = "attack_log.csv"
OUTPUT = "training_data.csv"


def load_attack_windows():
    windows = []
    with open(ATTACK_LOG) as f:
        reader = csv.DictReader(f)
        for row in reader:
            windows.append({
                "label": row["attack_type"],
                "start": float(row["start_time"]),
                "end": float(row["end_time"]),
            })
    return windows


def label_for_flow(first_seen, last_seen, windows):
    """Returns the attack label with the largest time overlap, or NORMAL."""
    best_label = "NORMAL"
    best_overlap = 0
    for w in windows:
        overlap_start = max(first_seen, w["start"])
        overlap_end = min(last_seen, w["end"])
        overlap = overlap_end - overlap_start
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = w["label"]
    return best_label


def compute_features(flow):
    """Derives the training features from a flow's raw accumulators,
    same math as the variance formulas built earlier in this project."""
    n = flow["packet_count"]
    total_bytes = flow["total_bytes"]
    sum_sq = flow["sum_of_squares"]

    avg_size = total_bytes / n
    size_variance = max(0.0, (sum_sq / n) - (avg_size ** 2))

    if n > 1:
        n_gaps = n - 1
        sum_gaps = flow["sum_of_gaps"]
        sum_sq_gaps = flow["sum_of_squared_gaps"]
        mean_gap = sum_gaps / n_gaps
        timing_variance = max(0.0, (sum_sq_gaps / n_gaps) - (mean_gap ** 2))
    else:
        timing_variance = 0.0  # single-packet flow — no gap to measure

    port_count = len(flow.get("ports", []))

    return avg_size, size_variance, timing_variance, port_count


def main():
    windows = load_attack_windows()
    rows_out = []

    with open(FLOW_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            flow = json.loads(line)

            if flow.get("packet_count", 0) < 1:
                continue

            avg_size, size_variance, timing_variance, port_count = compute_features(flow)
            label = label_for_flow(flow["first_seen"], flow["last_seen"], windows)

            rows_out.append({
                "src": flow["src"],
                "dst": flow["dst"],
                "packet_count": flow["packet_count"],
                "avg_size": round(avg_size, 2),
                "size_variance": round(size_variance, 2),
                "timing_variance": round(timing_variance, 6),
                "port_count": port_count,
                "label": label,
            })

    with open(OUTPUT, "w", newline="") as f:
        fieldnames = ["src", "dst", "packet_count", "avg_size",
                      "size_variance", "timing_variance", "port_count", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} labeled rows to {OUTPUT}")
    label_counts = Counter(r["label"] for r in rows_out)
    print("Label distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
