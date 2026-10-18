from scapy.all import sniff, IP, IPv6, TCP, UDP
from collections import defaultdict
import time

FLOW_TIMEOUT = 10  # seconds — rolling window before a flow is considered "closed"

flows = defaultdict(lambda: {
    "packet_count": 0,
    "total_bytes": 0,
    "sum_of_squares": 0,      # for packet-size variance
    "first_seen": None,
    "last_seen": None,
    "sum_of_gaps": 0,         # for inter-arrival timing (jitter / beaconing)
    "sum_of_squared_gaps": 0,
    "ports": set(),           # unique dst ports touched by this src->dst pair
    "flag_counts": {}         # TCP flag -> count, e.g. {"S": 4, "A": 12}
})

finalized_flows = []  # closed flows get moved here for the rule engine / dashboard

def flush_stale_flows(current_time):
    # Flush logic based on flow timeouts
    global flows, finalized_flows
    stale_keys = []
    for key, flow in flows.items():
        if flow["last_seen"] and (current_time - flow["last_seen"] > FLOW_TIMEOUT):
            stale_keys.append(key)
    
    for key in stale_keys:
        f = flows.pop(key)
        # Convert set to list for JSON serialization if needed
        f["ports"] = list(f["ports"])
        finalized_flows.append(f)

def handle_packet(packet):
    if packet.haslayer(IP):
        src = packet[IP].src
        dst = packet[IP].dst
        packet_len = len(packet[IP])
    elif packet.haslayer(IPv6):
        src = packet[IPv6].src
        dst = packet[IPv6].dst
        packet_len = len(packet[IPv6])
    else:
        return

    # Extract native timestamp from scapy packet (defaults to current time if live capture)
    pkt_time = float(packet.time) if hasattr(packet, 'time') else time.time()
    
    # Flush old flows based on packet timestamp progression
    flush_stale_flows(pkt_time)

    key = (src, dst)
    flow = flows[key]
    
    flow["packet_count"] += 1
    flow["total_bytes"] += packet_len
    flow["sum_of_squares"] += packet_len ** 2

    if flow["first_seen"] is None:
        flow["first_seen"] = pkt_time
        flow["last_seen"] = pkt_time
    else:
        gap = pkt_time - flow["last_seen"]
        flow["sum_of_gaps"] += gap
        flow["sum_of_squared_gaps"] += gap ** 2
        flow["last_seen"] = pkt_time

    # Extract port information if TCP/UDP
    if packet.haslayer(TCP):
        sport = packet[TCP].sport
        dport = packet[TCP].dport
        flow["ports"].add(dport)
        
        # Track TCP flags
        flags = str(packet[TCP].flags)
        for flag in flags:
            flow["flag_counts"][flag] = flow["flag_counts"].get(flag, 0) + 1
            
    elif packet.haslayer(UDP):
        dport = packet[UDP].dport
        flow["ports"].add(dport)
