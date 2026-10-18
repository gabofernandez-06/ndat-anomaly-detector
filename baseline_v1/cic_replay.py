"""
cic_replay.py - replay a CIC-IDS2017 pcap through capture_cic using the PACKET
timestamps, and label each flow by (attacker->victim IP pair) AND time window.

Usage:
  python3 cic_replay.py --pcap cic_data/Friday-WorkingHours.pcap \
      --csv-glob "cic_data/TrafficLabelling/Friday*.csv" --out friday_flows.jsonl
  add --calibrate-only            to just print the clock offset and windows
  add --slice-label PortScan --duration 300   to replay only a slice
"""
import argparse, csv, glob, json, sys, time
from datetime import datetime
from scapy.all import RawPcapReader, Ether
import capture_cic as capture


def parse_csv_ts(s):
    s = s.strip()
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            break
        except ValueError:
            dt = None
    if dt is None:
        return None
    if 1 <= dt.hour <= 7:          # Friday CSVs have no AM/PM; 1-7 means afternoon
        dt = dt.replace(hour=dt.hour + 12)
    return dt.timestamp()           # naive local; only DIFFERENCES matter below


def load_labels(pattern):
    pairs, tmin, tmax, earliest = {}, {}, {}, None
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no CSVs match {pattern}")
    for path in files:
        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for row in reader:
                ts = parse_csv_ts(row.get("Timestamp", ""))
                if ts is None:
                    continue
                earliest = ts if earliest is None else min(earliest, ts)
                lab = row["Label"].strip()
                if lab == "BENIGN":
                    continue
                pairs.setdefault(lab, set()).add((row["Source IP"].strip(), row["Destination IP"].strip()))
                tmin[lab] = min(tmin.get(lab, ts), ts)
                tmax[lab] = max(tmax.get(lab, ts), ts)
    return pairs, tmin, tmax, earliest


def pkt_ts(meta):
    """Packet time in seconds. Handles classic pcap (sec/usec) AND pcapng (tshigh/tslow/tsresol)."""
    if hasattr(meta, "sec"):
        return meta.sec + meta.usec / 1e6
    return ((meta.tshigh << 32) | meta.tslow) / meta.tsresol


def first_pcap_ts(path):
    for _, meta in RawPcapReader(path):
        return pkt_ts(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--csv-glob", required=True)
    ap.add_argument("--out", default="replayed_flows.jsonl")
    ap.add_argument("--source", default="cic")
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--slice-label")
    ap.add_argument("--start-offset", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument('--max-duration', type=float, default=60.0)
    a = ap.parse_args()
    capture.MAX_FLOW_DURATION = a.max_duration
    print('active timeout    :', a.max_duration, 's')

    pairs, tmin, tmax, earliest = load_labels(a.csv_glob)
    t0 = first_pcap_ts(a.pcap)
    offset = round((t0 - earliest) / 900) * 900   # clock offset, snapped to 15 min
    windows = {l: (tmin[l] + offset, tmax[l] + offset + 60) for l in pairs}

    print(f"first pcap packet : {datetime.fromtimestamp(t0)}")
    print(f"earliest CSV time : {datetime.fromtimestamp(earliest)}")
    print(f"clock offset      : {offset:+d} s ({offset/3600:+.2f} h)")
    for l, (lo, hi) in windows.items():
        print(f"{l:10s} window pcap-clock {datetime.fromtimestamp(lo)} -> {datetime.fromtimestamp(hi)}"
              f"  pairs={len(pairs[l])}")
    if a.calibrate_only:
        return

    skip_until, stop_at = t0, None
    if a.slice_label:
        skip_until = windows[a.slice_label][0] + a.start_offset
        stop_at = skip_until + a.duration if a.duration else windows[a.slice_label][1]
        print(f"slice {a.slice_label}: {datetime.fromtimestamp(skip_until)} -> {datetime.fromtimestamp(stop_at)}")

    def label_flow(fl):
        key = (fl["src"], fl["dst"])
        for lab, ps in pairs.items():
            if key in ps:
                lo, hi = windows[lab]
                if fl["first_seen"] <= hi and fl["last_seen"] >= lo:
                    return lab
                stats["pair_outside_window"] += 1
        return "NORMAL"

    stats = {"pair_outside_window": 0}
    counts = {}
    n = 0
    start = time.time()
    with open(a.out, "w") as out:
        def drain():
            for fl in capture.finalized_flows:
                fl["label"] = label_flow(fl)
                fl["source"] = a.source
                fl["ports"] = sorted(fl["ports"])
                counts[fl["label"]] = counts.get(fl["label"], 0) + 1
                out.write(json.dumps(fl) + "\n")
            capture.finalized_flows.clear()

        for data, meta in RawPcapReader(a.pcap):
            ts = pkt_ts(meta)
            if ts < skip_until:
                continue
            if stop_at and ts > stop_at:
                break
            capture.process_packet(Ether(data), ts)   # PACKET clock, not wall clock
            capture.flush_stale_flows(ts)
            n += 1
            if n % 200000 == 0:
                drain()
                print(f"  {n:,} packets, {n/(time.time()-start):,.0f} pkt/s")
        capture.flush_stale_flows(ts, force=True)
        drain()

    print(f"done: {n:,} packets -> {sum(counts.values()):,} flows in {a.out}")
    for l, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {l}: {c}")
    print(f"pair-matched but outside time window (labeled NORMAL): {stats['pair_outside_window']}")


if __name__ == "__main__":
    main()
