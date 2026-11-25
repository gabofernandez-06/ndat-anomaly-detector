import json
import pandas as pd
import numpy as np

def compute_features(flow):
    pkt_count = flow.get("packet_count", 1)
    if pkt_count < 1:
        pkt_count = 1
        
    total_bytes = flow.get("total_bytes", 0)
    sum_sq_bytes = flow.get("sum_of_squares", 0)
    
    # 1. Average packet size
    avg_size = total_bytes / pkt_count
    
    # 2. Packet size variance: E[X^2] - (E[X])^2
    mean_sq_bytes = sum_sq_bytes / pkt_count
    size_variance = max(0.0, mean_sq_bytes - (avg_size ** 2))
    
    # 3. Timing variance (inter-arrival gap variance)
    sum_gaps = flow.get("sum_of_gaps", 0)
    sum_sq_gaps = flow.get("sum_of_squared_gaps", 0)
    if pkt_count > 1:
        avg_gap = sum_gaps / (pkt_count - 1)
        mean_sq_gap = sum_sq_gaps / (pkt_count - 1)
        timing_variance = max(0.0, mean_sq_gap - (avg_gap ** 2))
    else:
        timing_variance = 0.0

    # 4. Port count
    ports = flow.get("ports", [])
    port_count = len(ports) if isinstance(ports, list) else len(set(ports))

    return {
        "avg_size": avg_size,
        "size_variance": size_variance,
        "timing_variance": timing_variance,
        "port_count": port_count
    }

def label_flow(flow):
    # Check if any detection rules already fired on this flow
    if flow.get("fired_rules"):
        return "ATTACK_CANDIDATE"
    
    # Behavioral heuristic for PortScan / anomaly detection:
    # High unique port count from a single flow indicates scanning behavior
    ports = flow.get("ports", [])
    port_count = len(ports) if isinstance(ports, list) else len(set(ports))
    if port_count > 15:
        return "ATTACK_CANDIDATE"
        
    return "NORMAL"

def main():
    input_jsonl = "test_replayed_flows.jsonl"
    output_csv = "training_data.csv"
    
    rows = []
    print(f"Reading flows from {input_jsonl}...")
    with open(input_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            flow = json.loads(line)
            feats = compute_features(flow)
            feats["label"] = label_flow(flow)
            rows.append(feats)
            
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Successfully generated {output_csv} with {len(df)} rows!")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()

