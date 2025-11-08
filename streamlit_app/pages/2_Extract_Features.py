import os
import seaborn as sns
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from src.extraction.extract_features import compute_motion_features, compute_audio_features
from src.extraction.feauture_engineering import window_timeseries

st.set_page_config(page_title="Feature Tester", layout="wide")
st.title("🎬 Boxing Highlight Feature Tester")

with st.container(border=True):
    st.markdown("""
    ## Feature QA

    You can either:
    - Run feature extraction from raw frames/audio
    - Or upload precomputed feature CSVs to visualize and validate them

    Test highlight scoring, inspect motion spikes, and verify audio quality before passing to ML model.
    """)

    frame_col, audio_col = st.columns(2)
    frames_dir = frame_col.text_input("Frames folder location", value = "", key = "frames_out")
    audio_path = audio_col.text_input("Audio path location", value= "",key = "audio_out")

    frames_dir = frames_dir.strip() or None
    audio_path = audio_path.strip() or None

    with st.container(border=True):
        st.markdown("### Or upload precomputed feature files")

        upload_col1, upload_col2 = st.columns(2)
        uploaded_visual_csv = upload_col1.file_uploader("Upload visual features CSV", "csv")
        uploaded_audio_csv = upload_col2.file_uploader("Upload audio features CSV", "csv")

        if uploaded_visual_csv:
            try:
                st.session_state.motion_df = pd.read_csv(uploaded_visual_csv)
                st.success("Visual features loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load visual features: {e}")

        if uploaded_audio_csv:
            try:
                st.session_state.audio_df = pd.read_csv(uploaded_audio_csv)
                st.success("Audio features loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load audio features: {e}")

    buttons_col1, buttons_col2, buttons_col3 = st.columns(3)
    extract_visual_features_btn = buttons_col1.button("Extract Visual Features", disabled=frames_dir is None)
    extract_audio_features_btn = buttons_col2.button("Extract Audio Features", disabled=audio_path is None)
    extract_all_features_btn = buttons_col3.button("Extract All Features", disabled=audio_path is None or frames_dir is None)

    if "motion_df" not in st.session_state:
        st.session_state.motion_df = None
    if "audio_df" not in st.session_state:
        st.session_state.audio_df = None

    if extract_visual_features_btn or extract_all_features_btn:
        with st.spinner("Extracting Visual Features..."):
            try:
                st.session_state.motion_df = compute_motion_features(frames_dir).reset_index(drop=True)
            except Exception as e:
                st.error(f"Visual extraction failed: {e}")

    if extract_audio_features_btn or extract_all_features_btn:
        with st.spinner("Extracting audio features..."):
            try:
                st.session_state.audio_df = compute_audio_features(audio_path).reset_index(drop=True)
            except Exception as e:
                st.error(f"Audio extraction failed: {e}")

    motion_df = st.session_state.motion_df
    audio_df = st.session_state.audio_df

    # sanity check plots
    if motion_df is not None and not motion_df.empty:

        with st.expander("Preview visual data"):
            st.dataframe(motion_df.head(200), height=240)
            st.download_button("Download visual features",
                               data=motion_df.to_csv(index=False).encode("utf-8"),
                               file_name="visual_features_raw.csv",
                               mime="text/csv")
        with st.expander("Preview Highlight Scoring"):
            st.subheader("Highlight Scoring")
            highlight_thresh = st.slider("Highlight threshold (motion_mean)", min_value=0.0, max_value=100.0, value=30.0)
            highlight_df = motion_df[motion_df["motion_mean"] > highlight_thresh]
            st.write(f"Detected {len(highlight_df)} highlight candidates above threshold.")
            st.dataframe(highlight_df[["timestamp", "motion_mean"]].head(50),)

            if frames_dir and not highlight_df.empty:
                st.subheader("Preview Highlight Frames")
                preview_count = st.slider("Number of frames to preview", 1, min(10, len(highlight_df)), 5)
                for i in highlight_df.index[:preview_count]:
                    frame_path = os.path.join(frames_dir, f"frame_{i:05d}.jpg")
                    if os.path.exists(frame_path):
                        st.image(frame_path, caption=f"Frame {i} (timestamp: {highlight_df.loc[i, 'timestamp']:.2f}s)")
                    else:
                        st.warning(f"Frame not found: {frame_path}")
            else:
                st.markdown("""Choose frames folder relating to these features to see 'highlight frames'""")

        with st.expander("Preview Correlation Heatmap"):
            st.subheader("Feature Correlation Heatmap")
            corr_df = motion_df.drop(columns="timestamp").corr()

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig, width = 500)


    if audio_df is not None and not audio_df.empty:
        # st.subheader("Raw Audio Features")
        # aud_cols = [c for c in ["rms", "zcr"] if c in audio_df.columns]
        # plot_df = audio_df[["timestamp"] + aud_cols].set_index("timestamp")
        # st.line_chart(plot_df)
        with st.expander("Preview audio data"):
            st.dataframe(audio_df.head(200), height=240)
        st.download_button("Download audio features",
                           data=audio_df.to_csv(index=False).encode("utf-8"),
                           file_name="audio_features_raw.csv",
                           mime="text/csv")

    with st.expander("Windowed Features (ready model training)"):
        c1, c2, c3 = st.columns([1, 1, 2])
        win_s = c1.number_input("Window size (s)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
        hop_s = c2.number_input("Hop size (s)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        do_window = c3.toggle("Create windowed features now", value=False)

        motion_win_df = None
        audio_win_df = None

        if do_window:
            if motion_df is not None and not motion_df.empty:
                vis_cols = [c for c in motion_df.columns if c != "timestamp"]
                motion_win_df = window_timeseries(motion_df, vis_cols, win_s=win_s, hop_s=hop_s)
                st.subheader("Visual windows")
                st.caption(f"{len(motion_win_df)} windows @ {win_s}s / {hop_s}s")
                st.dataframe(motion_win_df.head(50), height=260)
                st.download_button("Download visual windows (CSV)",
                                   data=motion_win_df.to_csv(index=False).encode("utf-8"),
                                   file_name="visual_features_windows.csv",
                                   mime="text/csv")

            if audio_df is not None and not audio_df.empty:
                aud_cols = [c for c in audio_df.columns if c != "timestamp"]
                audio_win_df = window_timeseries(audio_df, aud_cols, win_s=win_s, hop_s=hop_s)
                st.subheader("Audio windows")
                st.caption(f"{len(audio_win_df)} windows @ {win_s}s / {hop_s}s")
                st.dataframe(audio_win_df.head(50), height=260)
                st.download_button("Download audio windows (CSV)",
                                   data=audio_win_df.to_csv(index=False).encode("utf-8"),
                                   file_name="audio_features_windows.csv",
                                   mime="text/csv")

            if motion_win_df is not None and audio_win_df is not None:
                st.subheader("Joined windows (left-join on nearest timestamp)")
                joined = pd.merge_asof(
                    motion_win_df.sort_values("timestamp"),
                    audio_win_df.sort_values("timestamp"),
                    on="timestamp", direction="nearest", tolerance=hop_s / 2,
                    suffixes=("_vis", "_aud")
                )
                st.dataframe(joined.head(50), height=260)
                st.download_button("Download joined windows (CSV)",
                                   data=joined.to_csv(index=False).encode("utf-8"),
                                   file_name="joined_windows.csv",
                                   mime="text/csv")