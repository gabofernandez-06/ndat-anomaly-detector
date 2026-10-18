"""
dashboard.py — Day 5: live-updating visual on top of the detection engine.
 
Reads dashboard_state.json (written continuously by main.py) and renders:
  - a stats panel (total flows, total flagged, flags by rule)
  - a time-series chart of traffic volume + flags per check interval
  - a live table of recent flows, with flagged rows highlighted
 
Run with:  streamlit run dashboard.py
(No sudo needed here — only main.py, which does the actual sniffing, needs root.)
"""
 
import json
import time
 
import pandas as pd
import streamlit as st
 
STATE_FILE = "dashboard_state.json"
REFRESH_SECONDS = 3
 
st.set_page_config(page_title="Anomaly Detector Dashboard", layout="wide")
st.title("🛰️ Network Anomaly Detector")
 
 
def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
 
 
state = load_state()
 
if state is None:
    st.warning(
        "No data yet. Make sure `sudo python3 main.py` is running in another terminal "
        "in this same folder — the dashboard reads its live output from "
        f"`{STATE_FILE}`."
    )
    time.sleep(REFRESH_SECONDS)
    st.rerun()
 
# ---- Stats panel ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total flows seen", state["total_flows"])
col2.metric("Total flagged", state["total_flagged"])
fp_rate = (
    f"{state['total_flagged'] / state['total_flows'] * 100:.1f}%"
    if state["total_flows"] else "—"
)
col3.metric("Flag rate", fp_rate)
col4.metric("Last updated", state["last_updated"])
 
st.subheader("Flags by rule")
if state["rule_counts"]:
    rule_df = pd.DataFrame(
        list(state["rule_counts"].items()), columns=["Rule", "Times Fired"]
    ).sort_values("Times Fired", ascending=False)
    st.bar_chart(rule_df.set_index("Rule"))
else:
    st.caption("No rules have fired yet.")
 
# ---- Time-series chart ----
st.subheader("Traffic volume over time")
if state["history"]:
    hist_df = pd.DataFrame(state["history"])
    st.line_chart(hist_df.set_index("time")[["packets", "flows", "flagged"]])
else:
    st.caption("Waiting for enough data to plot...")
 
# ---- Recent flows table, flagged rows highlighted ----
st.subheader("Recent flows")
if state["recent_flows"]:
    flows_df = pd.DataFrame(state["recent_flows"])
    flows_df["flagged"] = flows_df["fired_rules"].apply(lambda r: len(r) > 0)
    flows_df["fired_rules"] = flows_df["fired_rules"].apply(lambda r: ", ".join(r) if r else "")
 
    def highlight_row(row):
        is_flagged = flows_df.loc[row.name, "flagged"]
        color = "background-color: #ffcccc" if is_flagged else ""
        return [color] * len(row)
 
    display_df = flows_df[
        ["src", "dst", "packet_count", "total_bytes", "duration", "port_count", "fired_rules"]
    ]
    styled = display_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled, width="stretch", height=400)
else:
    st.caption("No flows finalized yet.")
 
# Auto-refresh
time.sleep(REFRESH_SECONDS)
st.rerun()
 
