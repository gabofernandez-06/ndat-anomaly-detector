#!/bin/bash
# run_attacks.sh — runs a long (~5-6 hour) sequence of repeated attack cycles
# against the VM, logging every run's start/end timestamps to attack_log.csv.
#
# Designed for an unattended overnight run:
#   - All output is written to a timestamped log file (in addition to the
#     terminal), so you can check progress anytime with:
#       tail -f attack_run_<timestamp>.log
#   - Runs the full attack sequence in repeated CYCLES, with an idle period
#     between cycles to also collect clean NORMAL background traffic.
#
# IMPORTANT — before running:
#   1. Make sure your Mac will NOT go to sleep (System Settings > Lock Screen,
#      or just run this script prefixed with `caffeinate -i`, see below).
#   2. Keep the lid open and the Mac plugged into power.
#   3. Make sure the VM and main.py are both already running.
#
# Recommended way to launch this (prevents sleep for the whole run):
#   caffeinate -i sudo ./run_attacks.sh

TARGET="192.168.64.3"
ATTACK_LOG="attack_log.csv"
RUN_LOG="attack_run_$(date +%Y%m%d_%H%M%S).log"

# Tee all output to both the terminal and a log file, so you can check
# progress later even if you're not watching this terminal live.
exec > >(tee -a "$RUN_LOG") 2>&1

FAST_REPS=15
SERVICE_REPS=15
FLOOD_REPS=8
SLOW_REPS=1
OUTER_CYCLES=8
CYCLE_IDLE_SECONDS=600   # 10 min of clean idle/normal traffic between cycles

if [ ! -f "$ATTACK_LOG" ]; then
    echo "attack_type,start_time,end_time" > "$ATTACK_LOG"
fi

run_attack () {
    local label="$1"
    shift
    local start
    start=$(date +%s)
    "$@" > /dev/null 2>&1
    local end
    end=$(date +%s)
    echo "$label,$start,$end" >> "$ATTACK_LOG"
    echo "  [$(date '+%H:%M:%S')] $label logged ($start -> $end)"
    sleep 12   # let the flow go stale (FLOW_TIMEOUT) and get evaluated
}

run_repeated () {
    local label="$1"
    local reps="$2"
    shift 2
    echo ""
    echo "=== [$(date '+%H:%M:%S')] $label: running $reps rep(s) ==="
    for i in $(seq 1 "$reps"); do
        echo "-- rep $i/$reps --"
        run_attack "$label" "$@"
    done
    echo "=== [$(date '+%H:%M:%S')] $label: done ($reps reps) ==="
}

echo "Starting long attack simulation run against $TARGET"
echo "Logging attacks to $ATTACK_LOG"
echo "Full run log: $RUN_LOG"
echo "Planned: $OUTER_CYCLES cycles, ~$((CYCLE_IDLE_SECONDS/60)) min idle between each"
echo "Started at $(date)"

for cycle in $(seq 1 "$OUTER_CYCLES"); do
    echo ""
    echo "########## CYCLE $cycle/$OUTER_CYCLES — $(date) ##########"

    run_repeated "PORT_SCAN_FAST" "$FAST_REPS" nmap -sS -p 1-1000 "$TARGET"
    run_repeated "PORT_SCAN_SLOW" "$SLOW_REPS" nmap -sS -T1 -p 1-100 "$TARGET"
    run_repeated "SERVICE_SCAN" "$SERVICE_REPS" nmap -sV -p 1-100 "$TARGET"
    run_repeated "FLOOD_BURST" "$FLOOD_REPS" nmap -sT -T5 --min-rate 5000 -p 1-1000 "$TARGET"

    if [ "$cycle" -lt "$OUTER_CYCLES" ]; then
        echo ""
        echo "--- Idle period: $CYCLE_IDLE_SECONDS sec of normal traffic before next cycle ---"
        sleep "$CYCLE_IDLE_SECONDS"
    fi
done

echo ""
echo "All attack cycles complete at $(date)."
echo "See $ATTACK_LOG for timestamps, $RUN_LOG for the full run log."
