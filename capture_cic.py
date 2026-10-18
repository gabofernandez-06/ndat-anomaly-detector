"""
capture_cic.py - flow aggregation that takes the clock as an ARGUMENT.

Same flow logic as capture.py, but every function receives `now` instead of
calling time.time(). For live capture, pass time.time(). For pcap replay, pass
the packet's own recorded timestamp. That one change is the whole fix: flow
duration, inter-arrival gaps, timeouts and the 60 s cap then reflect the
ORIGINAL traffic, not how fast your laptop reads the file.
"""
from collections import defaultdict
from scapy.all import IP, TCP, UDP

FLOW_TIMEOUT = 10        # seconds of silence before a flow is closed
MAX_FLOW_DURATION = 60   # active timeout: long flows are cut into 60 s pieces
FLUSH_INTERVAL = 1.0     # how often (in packet-clock seconds) we scan for stale flows


def _new_flow():
    return {
        "packet_count": 0, "total_bytes": 0, "sum_of_squares": 0,
        "first_seen": None, "last_seen": None,
        "sum_of_gaps": 0.0, "sum_of_squared_gaps": 0.0,
        "ports": set(), "flag_counts": defaultdict(int),
    }


flows = defaultdict(_new_flow)
finalized_flows = []
_last_flush = [0.0]


def _finalize(key, flow):
    flow["src"], flow["dst"] = key
    flow["duration"] = flow["last_seen"] - flow["first_seen"]
    flow["flag_counts"] = dict(flow["flag_counts"])
    finalized_flows.append(flow)


def process_packet(packet, now):
    if IP not in packet:
        return
    key = (packet[IP].src, packet[IP].dst)

    flow = flows[key]
    # active timeout: a flow that has run >= 60 s is closed and restarted
    if flow["first_seen"] is not None and now - flow["first_seen"] >= MAX_FLOW_DURATION:
        _finalize(key, flows.pop(key))
        flow = flows[key]

    size = len(packet)
    if flow["first_seen"] is None:
        flow["first_seen"] = now
    else:
        gap = now - flow["last_seen"]
        flow["sum_of_gaps"] += gap
        flow["sum_of_squared_gaps"] += gap * gap
    flow["last_seen"] = now
    flow["packet_count"] += 1
    flow["total_bytes"] += size
    flow["sum_of_squares"] += size * size

    if TCP in packet:
        flow["ports"].add(packet[TCP].dport)
        flow["flag_counts"][str(packet[TCP].flags)] += 1
    elif UDP in packet:
        flow["ports"].add(packet[UDP].dport)


def handle_packet(packet):
    """Live-capture entry point: the clock is the wall clock."""
    import time
    process_packet(packet, time.time())


def flush_stale_flows(now, force=False):
    if not force and now - _last_flush[0] < FLUSH_INTERVAL:
        return
    _last_flush[0] = now
    stale = [k for k, f in flows.items()
             if force or now - f["last_seen"] > FLOW_TIMEOUT]
    for k in stale:
        _finalize(k, flows.pop(k))	
