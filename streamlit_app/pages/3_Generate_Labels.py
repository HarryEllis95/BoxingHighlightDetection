import io
import os
import pandas as pd
import streamlit as st
from typing import List, Tuple

st.set_page_config(page_title="Create Highlight Labels", layout="wide")
st.title("Create Label CSV for Highlight Detection")

def parse_time_userinput(s: str) -> float:
    s = (s or "").strip()
    if ":" not in s:
        raise ValueError("Wrong format")
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError("Wrong format")
    m_str, s_str = parts
    try:
        minutes = int(m_str)
    except Exception:
        raise ValueError("Minutes must be an integer")
    try:
        seconds = float(s_str)
    except Exception:
        raise ValueError("Seconds must be numeric")
    if minutes < 0:
        raise ValueError("Minutes must be non-negative")
    if not (0 <= seconds < 60):
         raise ValueError("Seconds must be in 0-59 range")
    seconds = round(seconds, 2)
    if seconds >= 60:
        raise ValueError("Seconds round up to 60; use the next minute (e.g. 3:00)")
    return minutes * 60.0 + seconds

def parse_time_from_csv(s: str) -> float:
    s = (s or "").strip()
    if ":" not in s:
        raise ValueError("Expected MM:SS")
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError("Expected MM:SS")
    minutes, seconds = parts
    minutes = int(minutes)
    seconds = float(seconds)
    if not (0 <= seconds < 60):
        raise ValueError("Seconds must be in [0,60)")
    if minutes < 0:
        raise ValueError("Minutes must be non-negative")
    return minutes * 60.0 + seconds

def normalize_interval(start: float, end: float) -> Tuple[float, float]:
    if start < 0 or end < 0:
        raise ValueError("Times must be non-negative")
    if end < start:
        start, end = end, start
    if end - start <= 0:
        raise ValueError("Interval must have positive duration")
    return float(start), float(end)

def intervals_to_df(intervals: List[Tuple[float, float]]) -> pd.DataFrame:
    df = pd.DataFrame(intervals, columns=["start_time", "end_time"])
    df["label"] = 1
    df["start_time"] = df["start_time"].astype(float)
    df["end_time"] = df["end_time"].astype(float)
    return df[["start_time", "end_time", "label"]]

if "intervals" not in st.session_state:
    st.session_state["intervals"] = []

# load existing csvs
st.subheader("Load existing labels")
uploaded = st.file_uploader("Load label CSV", type=["csv"])
if uploaded is not None:
    try:
        df_loaded = pd.read_csv(uploaded)
        if {"start_time", "end_time"}.issubset(df_loaded.columns):
            loaded_intervals = []
            for r in df_loaded.itertuples(index=False):
                def to_sec(x):
                    try:
                        return float(x)
                    except Exception:
                        return parse_time_from_csv(str(x))
                try:
                    stime = to_sec(r.start_time)
                    etime = to_sec(r.end_time)
                    stime, etime = normalize_interval(stime, etime)
                    loaded_intervals.append((stime, etime))
                except Exception:
                    continue
            if loaded_intervals:
                st.session_state["intervals"].extend(loaded_intervals)
                st.success(f"Loaded {len(loaded_intervals)} intervals from CSV")
            else:
                st.error("No valid intervals found in uploaded CSV")
        else:
            st.error("CSV must contain start_time and end_time columns")
    except Exception as exc:
        st.error(f"Error reading CSV: {exc}")

# Add new intervals
st.header("Add a highlight interval")
with st.form("add_interval_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([1, 1, 0.6])
    with c1:
        start_str = st.text_input(
            "Start time (MM:SS.xx)",
        )
    with c2:
        end_str = st.text_input(
            "End time (MM:SS.xx)",
        )
    with c3:
        add_btn = st.form_submit_button("Add interval")

if add_btn:
    try:
        s = parse_time_userinput(start_str)
        e = parse_time_userinput(end_str)
        s, e = normalize_interval(s, e)
        st.session_state["intervals"].append((s, e))
        s_txt = f"{int(s//60)}:{(s%60):05.2f}"
        e_txt = f"{int(e//60)}:{(e%60):05.2f}"
        st.success(f"Added interval {s_txt} → {e_txt}")
    except Exception as exc:
        st.error(f"Could not add interval: {exc}")

if st.button("Clear all intervals"):
    st.session_state["intervals"] = []
    st.info("Cleared all intervals.")


st.header("Current intervals")

intervals = st.session_state["intervals"]

if not intervals:
    st.info("Add intervals using the form above.")
else:
    df_display = pd.DataFrame(
        [(f"{int(s//60)}:{(s%60):05.2f}", f"{int(e//60)}:{(e%60):05.2f}") for s, e in intervals],
        columns=["start_mmss", "end_mmss"]
    )
    st.table(df_display)

    col = st.container()
    with col:
        rem_idx = st.number_input("Remove interval index", min_value=0, max_value=max(0, len(intervals) - 1), value=0,
                                  step=1)
        cbtn1, cbtn2 = st.columns([1, 1])
        if cbtn1.button("Remove by index"):
            try:
                st.session_state["intervals"].pop(int(rem_idx))
            except Exception as exc:
                st.error(f"Could not remove: {exc}")
        if cbtn2.button("Remove last interval"):
            st.session_state["intervals"].pop()

# Export label CSV
st.header("Export labels CSV")

out_filename = st.text_input("Output filename", value="labels.csv")


if st.button("Generate CSV"):
    if not intervals:
        st.error("No intervals to export")
    else:
        df_out =  intervals_to_df(intervals)
        print(df_out.head(10))
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        # Download button
        st.download_button("Download labels CSV", data=csv_bytes, file_name=out_filename, mime="text/csv")
