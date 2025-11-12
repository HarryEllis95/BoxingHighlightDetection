from pathlib import Path

import streamlit as st

st.set_page_config(page_title = "Boxing Highlight Detection", layout="wide", initial_sidebar_state="expanded")

st.session_state.setdefault("active_video_path", None)  # chosen video path
st.session_state.setdefault("last_frames_result", None)  # (folder, count)
st.session_state.setdefault("last_audio_path", None)  # wav path
st.session_state.setdefault("features_path", None)       # where you save features
st.session_state.setdefault("trained_model_path", None)  # where you save model

st.title("🥊 Boxing Highlight Detector")

st.markdown("### What it does")
st.markdown(
    """
The purpose of this project is to utilise Machine Learning to find and extract highlight moments (knockdowns, big exchanges, KOs) in full-length boxing videos.
This involves extraction of features such as:
- **Visual motion** from video frames
- **Audio energy & reactions**

Currently a model is trained on those features to predict a per-second highlight probability.  
Current output is **timestamps** of predicted highlights.
"""
)

st.markdown("### How to Train and Use Model")
st.markdown(
    """
1. **Extract Frames & Audio** – point to a local video (no web uploads currently), then:
   - Extract **frames** at a chosen FPS  
   - Extract **mono 16 kHz WAV** audio (using ffmpeg if downloaded, otherwise moviepy - see readme)
3. **Extract Features** – compute features from frames and/or audio and save.  
4. **Generate Labeled Data** – See Create Highlight Labels page 
4. **Train Model** – feed features into model - save trained model.  
5. **Run Inference** – apply the trained model to any video; get **highlight timestamps** (and later, exported clips).
""")

st.markdown("### Notes")
st.markdown(
    """
- Optimized for **local big files**: everything runs on file paths; no browser uploads.  
- Outputs (frames, WAV, features, model) are saved **next to your video** by default.  
"""
)

st.divider()

st.markdown("#### Session status")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Video", f"{Path(st.session_state.active_video_path).name}" if st.session_state.active_video_path else "-")
with c2:
    if st.session_state.last_frames_result:
        folder, count = st.session_state.last_frames_result
        st.metric("Frames", f"{count:,}")
    else:
        st.metric("Frames", "-")
with c3:
    st.metric("Audio", "Ready" if st.session_state.last_audio_path else "-")
with c4:
    st.metric("Features", "Ready" if st.session_state.features_path else "-")
with c5:
    st.metric("Model", "Trained" if st.session_state.trained_model_path else "-")