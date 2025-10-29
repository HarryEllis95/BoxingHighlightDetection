import streamlit as st

from src.extraction.extract_features import compute_motion_features, compute_audio_features

st.set_page_config(page_title="Feature Tester", layout="wide")
st.title("🎬 Boxing Highlight Feature Tester")

with st.container(border=True):
    st.markdown("### Choose extracted frames and audio - ensure they relate to the same video!")

    frame_col, audio_col = st.columns(2)
    frames_dir = frame_col.text_input("Frames folder location", value = "", key = "frames_out")
    audio_path = audio_col.text_input("Audio path location", value= "",key = "audio_out")

    frames_dir = frames_dir.strip() or None
    audio_path = audio_path.strip() or None

    buttons_col1, buttons_col2, buttons_col3 = st.columns(3)
    extract_visual_features_btn = buttons_col1.button("Extract Visual Features", use_container_width=True, disabled=frames_dir is None)
    extract_audio_features_btn = buttons_col2.button("Extract Audio Features", use_container_width=True, disabled=audio_path is None)
    extract_all_features_btn = buttons_col3.button("Extract All Features", use_container_width=True, disabled=audio_path is None or frames_dir is None)

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
        st.subheader("Raw Visual Features")
        vis_cols = [c for c in ["motion_mean","flow_mean","flow_pct_high","center_motion_ratio","flow_dir_consistency"] if c in motion_df.columns]
        plot_df = motion_df[["timestamp"] + vis_cols].set_index("timestamp")
        st.line_chart(plot_df)
        with st.expander("Preview visual data"):
            st.dataframe(motion_df.head(200), use_container_width=True, height=240)
            st.download_button("Download visual features",
                               data=motion_df.to_csv(index=False).encode("utf-8"),
                               file_name="visual_features_raw.csv",
                               mime="text/csv")

    if audio_df is not None and not audio_df.empty:
        st.subheader("Raw Audio Features")
        aud_cols = [c for c in ["rms", "zcr"] if c in audio_df.columns]
        plot_df = audio_df[["timestamp"] + aud_cols].set_index("timestamp")
        st.line_chart(plot_df)
        with st.expander("Preview audio data"):
            st.dataframe(audio_df.head(200), use_container_width=True, height=240)
        st.download_button("Download audio features",
                           data=audio_df.to_csv(index=False).encode("utf-8"),
                           file_name="audio_features_raw.csv",
                           mime="text/csv")
