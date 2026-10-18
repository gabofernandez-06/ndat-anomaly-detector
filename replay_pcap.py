import time
import json
import os
from scapy.all import PcapReader
from capture import handle_packet, flush_stale_flows, flows, finalized_flows, FLOW_TIMEOUT

def serialize_flow(flow):
    """Convert non-JSON-serializable types (like sets) to lists before saving."""
    f_copy = flow.copy()
    # Convert set of ports to a sorted list
    if "ports" in f_copy and isinstance(f_copy["ports"], set):
        f_copy["ports"] = sorted(list(f_copy["ports"]))
    return f_copy

def replay_pcap(pcap_path, output_jsonl="replayed_flows.jsonl", max_packets=None):
    if not os.path.exists(pcap_path):
        print(f"Error: Pcap file not found at {pcap_path}")
        return

    print(f"Starting replay of: {pcap_path}")
    start_time = time.time()
    packet_count = 0

    # Open output file for streaming writes
    with open(output_jsonl, "w") as f_out:
        # Use PcapReader to stream packets lazily (memory efficient)
        with PcapReader(pcap_path) as pcap_reader:
            for packet in pcap_reader:
                packet_count += 1
                
                # Pass packet through our state machine (uses packet.time natively)
                handle_packet(packet)
                
                # Periodically dump finalized flows to disk to free memory
                if len(finalized_flows) >= 1000:
                    for flow in finalized_flows:
                        serialized = serialize_flow(flow)
                        f_out.write(json.dumps(serialized) + "\n")
                    finalized_flows.clear()

                if max_packets and packet_count >= max_packets:
                    print(f"Reached max packet limit ({max_packets}). Stopping early.")
                    break

                if packet_count % 50000 == 0:
                    print(f"Processed {packet_count} packets...")

    # Flush any remaining active flows in memory using their true packet-native last_seen time
    for key in list(flows.keys()):
        flow = flows.pop(key)
        flow["src"], flow["dst"] = key
        # Guard against None if a flow only got 1 packet
        first_t = flow["first_seen"] if flow["first_seen"] is not None else 0.0
        last_t = flow["last_seen"] if flow["last_seen"] is not None else first_t
        flow["duration"] = max(0.001, last_t - first_t)
        finalized_flows.append(flow)

    # Write remaining finalized flows
    with open(output_jsonl, "a") as f_out:
        for flow in finalized_flows:
            serialized = serialize_flow(flow)
            f_out.write(json.dumps(serialized) + "\n")
    
    elapsed = time.time() - start_time
    print(f"\nReplay complete!")
    print(f"Total packets processed: {packet_count}")
    print(f"Time elapsed: {elapsed:.2f} seconds ({packet_count / max(1, elapsed):.1f} pkts/sec)")
    print(f"Saved flow records to {output_jsonl}")

if __name__ == "__main__":
    # Point directly to the verified pcap path
    PCAP_FILE = "~/anomaly-detector/cic_data/Friday-WorkingHours.pcap"
    PCAP_FILE = os.path.expanduser(PCAP_FILE)
    
    replay_pcap(PCAP_FILE, output_jsonl="test_replayed_flows.jsonl", max_packets=None)
