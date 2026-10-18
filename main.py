"""
main.py — ties capture.py (sniffing + flow aggregation) to rules.py (detection).
Runs sniff() in a background thread so the main thread can periodically
pull finalized flows and run them through the rule engine. Also writes a
snapshot of current state to STATE_FILE on each cycle so dashboard.py
(a separate, non-root Streamlit process) can read and display it live.

Also appends every finalized, evaluated flow's FULL raw feature data
(including timestamps, ports, flag counts) to FLOW_LOG_FILE, one JSON
record per line — this is the actual training-data source, since
dashboard_state.json only keeps a trimmed, capped display summary.
"""
import json
import threading
import time
from collections import Counter, deque
from scapy.all import sniff
import capture
import rules

CHECK_INTERVAL = 5       # seconds between rule-evaluation passes
STATE_FILE = "dashboard_state.json"
FLOW_LOG_FILE = "flow_log.jsonl"
HISTORY_LENGTH = 200     # number of check-cycles to keep for the time-series chart
RECENT_FLOWS_LIMIT = 100  # number of recent flows to show in the dashboard table

history = deque(maxlen=HISTORY_LENGTH)  # each entry: {time, packets, flows, flagged}
rule_counts = Counter()                 # cumulative count of how many times each rule fired
recent_flows = deque(maxlen=RECENT_FLOWS_LIMIT)


def sniffer_thread():
    sniff(prn=capture.handle_packet, store=False, iface="bridge100")


def flow_to_dict(flow):
    """Turns a flow dict (with a python set for ports) into something JSON-safe."""
    return {
        "src": flow.get("src"),
        "dst": flow.get("dst"),
        "packet_count": flow.get("packet_count"),
        "total_bytes": flow.get("total_bytes"),
        "duration": round(flow.get("duration", 0), 2),
        "port_count": len(flow.get("ports", [])),
        "fired_rules": flow.get("fired_rules", []),
    }


def flow_to_full_dict(flow):
    """Full raw feature dict for training data — includes everything flow_to_dict
    drops (timestamps, actual ports, flag counts, variance-source sums)."""
    return {
        "src": flow.get("src"),
        "dst": flow.get("dst"),
        "packet_count": flow.get("packet_count"),
        "total_bytes": flow.get("total_bytes"),
        "sum_of_squares": flow.get("sum_of_squares"),
        "first_seen": flow.get("first_seen"),
        "last_seen": flow.get("last_seen"),
        "duration": flow.get("duration"),
        "sum_of_gaps": flow.get("sum_of_gaps"),
        "sum_of_squared_gaps": flow.get("sum_of_squared_gaps"),
        "ports": sorted(flow.get("ports", [])),
        "flag_counts": flow.get("flag_counts", {}),
        "fired_rules": flow.get("fired_rules", []),
    }


def log_flow(flow):
    """Append one flow's full feature data as a JSON line to FLOW_LOG_FILE."""
    with open(FLOW_LOG_FILE, "a") as f:
        f.write(json.dumps(flow_to_full_dict(flow)) + "\n")


def write_state():
    state = {
        "last_updated": time.strftime("%H:%M:%S"),
        "total_flows": len(capture.finalized_flows),
        "total_flagged": sum(1 for f in capture.finalized_flows if f.get("fired_rules")),
        "rule_counts": dict(rule_counts),
        "history": list(history),
        "recent_flows": list(recent_flows),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def main():
    print("Starting capture thread... Ctrl+C to stop.")
    t = threading.Thread(target=sniffer_thread, daemon=True)
    t.start()

    seen_count = 0
    try:
        while True:
            time.sleep(CHECK_INTERVAL)

            # force a flush pass so recently-quiet flows get finalized even
            # between packet arrivals
            capture.flush_stale_flows(time.time())

            new_flows = capture.finalized_flows[seen_count:]
            seen_count = len(capture.finalized_flows)
            flagged = rules.evaluate_flows(new_flows) if new_flows else []

            total_packets = sum(f["packet_count"] for f in new_flows)
            history.append({
                "time": time.strftime("%H:%M:%S"),
                "packets": total_packets,
                "flows": len(new_flows),
                "flagged": len(flagged),
            })

            for flow in new_flows:
                for rule_name in flow.get("fired_rules", []):
                    rule_counts[rule_name] += 1
                recent_flows.appendleft(flow_to_dict(flow))
                log_flow(flow)

            write_state()

            if new_flows:
                print(f"\n[{time.strftime('%H:%M:%S')}] Evaluated {len(new_flows)} flow(s), "
                      f"{len(flagged)} flagged.")
                for flow in flagged:
                    print(
                        f"  ALERT {flow['src']} -> {flow['dst']} | "
                        f"rules={flow['fired_rules']} | "
                        f"pkts={flow['packet_count']} bytes={flow['total_bytes']} "
                        f"duration={flow['duration']:.1f}s ports={sorted(flow['ports'])[:10]}"
                    )

    except KeyboardInterrupt:
        # final flush + evaluation on exit
        capture.flush_stale_flows(time.time() + capture.FLOW_TIMEOUT + 1)
        remaining = capture.finalized_flows[seen_count:]
        flagged = rules.evaluate_flows(remaining)
        for flow in remaining:
            for rule_name in flow.get("fired_rules", []):
                rule_counts[rule_name] += 1
            recent_flows.appendleft(flow_to_dict(flow))
            log_flow(flow)
        write_state()
        print(f"\nShutting down. {len(capture.finalized_flows)} total flows, "
              f"{sum(1 for f in capture.finalized_flows if f.get('fired_rules'))} flagged overall.")


if __name__ == "__main__":
    main()

