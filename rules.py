"""
Rule-based detection engine.
Each rule takes a finalized flow dict (see capture.py) and returns True/False.
Every rule is one sentence, on purpose — that's what goes in your README/demo.
"""

# ---- Tunable thresholds — tune these against YOUR normal traffic on Day 3/4 ----
PORT_SCAN_UNIQUE_PORTS = 15       # >N unique dst ports in the flow window = scan
PORT_SCAN_MAX_PORT = 1024         # only count well-known/system ports as scan targets —
                                   # ephemeral response ports (>1024) from normal traffic
                                   # shouldn't count against this rule

FLOOD_PACKETS_PER_SEC = 300       # packets/sec from one src->dst pair = flood
                                   # (raised from 100 — normal downloads/streams easily hit 150-250pps)
FLOOD_MIN_PACKETS = 500            # legitimate bursts (page loads, video, CDN) topped out
                                    # around 450 packets in testing — real floods should
                                    # dwarf that, so this is the primary flood signal

KNOWN_BAD_PORTS = {23, 2323, 3389, 4444, 31337}  # telnet, RDP, common C2/backdoor ports

BEACON_MIN_DURATION = 60          # seconds — must be long-lived to count as beaconing
BEACON_MAX_TOTAL_BYTES = 5000     # low total data despite long duration
BEACON_MAX_JITTER_STDDEV = 2.0    # low variance in gap timing = suspiciously regular


def rule_port_scan(flow):
    """Flags a flow if the src contacted more unique WELL-KNOWN dst ports than expected for normal traffic."""
    well_known_ports = {p for p in flow["ports"] if p <= PORT_SCAN_MAX_PORT}
    return len(well_known_ports) > PORT_SCAN_UNIQUE_PORTS


def rule_traffic_flood(flow):
    """Flags a flow if it has both high packet volume AND a high sustained packet rate —
    volume alone filters out brief-but-legitimate bursts (page loads, video chunks)."""
    n = flow["packet_count"]
    if n < FLOOD_MIN_PACKETS:
        return False
    duration = max(flow.get("duration", 0), 0.01)  # tiny epsilon, just to avoid /0
    pps = n / duration
    return pps > FLOOD_PACKETS_PER_SEC


def rule_known_bad_port(flow):
    """Flags a flow if any contacted port matches a small hardcoded list of suspicious ports."""
    return bool(flow["ports"] & KNOWN_BAD_PORTS)


def rule_beaconing(flow):
    """Flags a flow if it's long-lived, low-data, and has unusually regular timing gaps (possible C2)."""
    duration = flow.get("duration", 0)
    n = flow["packet_count"]
    if duration < BEACON_MIN_DURATION or n < 2:
        return False
    if flow["total_bytes"] > BEACON_MAX_TOTAL_BYTES:
        return False

    mean_gap = flow["sum_of_gaps"] / (n - 1)
    # variance of gaps = E[gap^2] - (E[gap])^2
    variance = (flow["sum_of_squared_gaps"] / (n - 1)) - (mean_gap ** 2)
    stddev = variance ** 0.5 if variance > 0 else 0
    return stddev < BEACON_MAX_JITTER_STDDEV


RULES = {
    "PORT_SCAN": rule_port_scan,
    "TRAFFIC_FLOOD": rule_traffic_flood,
    "KNOWN_BAD_PORT": rule_known_bad_port,
    "BEACONING": rule_beaconing,
}


def evaluate_flow(flow):
    """Runs all rules against a flow, returns a list of rule names that fired."""
    return [name for name, rule_fn in RULES.items() if rule_fn(flow)]


def evaluate_flows(flow_list):
    """Tags each flow in-place with which rules fired; returns only the flagged ones."""
    flagged = []
    for flow in flow_list:
        fired = evaluate_flow(flow)
        flow["fired_rules"] = fired
        if fired:
            flagged.append(flow)
    return flagged


if __name__ == "__main__":
    # Quick manual sanity check with a fake flow
    test_flow = {
        "packet_count": 200, "total_bytes": 12000, "sum_of_squares": 0,
        "first_seen": 0, "last_seen": 1.5, "duration": 1.5,
        "sum_of_gaps": 1.5, "sum_of_squared_gaps": 0.01,
        "ports": set(range(1000, 1025)), "flag_counts": {"S": 200},
    }
    print(evaluate_flow(test_flow))

