from typing import Optional
import streamlit as st
import os
from src.preprocess_video import extract_frames, extract_audio
from src.streamlit_utils import normalize_path, find_videos

st.set_page_config(page_title = "Boxing Highlight Detection", layout="wide", initial_sidebar_state="collapsed")
st.title("🥊 Boxing Highlight Detector")

# Session state for paths/results
st.session_state.setdefault("active_video_path", None)     # the chosen video path
st.session_state.setdefault("scan_results", [])            # cached list from last scan
st.session_state.setdefault("last_frames_result", None)    # (folder, count)
st.session_state.setdefault("last_audio_path", None)       # wav path
st.session_state.setdefault("scan_base_dir", os.getcwd())

def reset_results():
    st.session_state.last_frames_result = None
    st.session_state.last_audio_path = None

def set_active_video(path: Optional[str]):
    """Change the active video and clear old previews only if it actually changed."""
    if path and path != st.session_state.active_video_path:
        reset_results()
        st.session_state.active_video_path = path

# for now just uploaded downloaded mp4 but would be nicer to have option to select youtube video
# and analyse it through temp storage user might not want it permanently downloaded
with st.container(border=True):
    st.markdown("### Choose Video input source")

    input_mode = st.radio("Input mode", ["Video file path", "Scan a directory"], horizontal=True,  key="input_mode")

    if input_mode == "Video file path":
        raw = st.text_input(
            "Enter full path to boxing video",
            key="video_file_path",
            placeholder=r"/path/to/video.mp4"
        )
        vid_path = normalize_path(raw)
        if vid_path and os.path.isfile(vid_path):
            set_active_video(vid_path)
        elif vid_path:
            st.warning("That path does not point to a found file")

    else:  # dir scan
        base_dir = st.text_input(
            "Directory to scan (recursive)",
            value=st.session_state.scan_base_dir,
            key="scan_base_dir",
            placeholder=r"/path/to/folder"
        )

        if st.button("Find videos"):
            base = normalize_path(st.session_state.scan_base_dir)
            if not os.path.isdir(base):
                st.error("That directory does not exist.")
                st.session_state.scan_results = []
                reset_results()
            else:
                reset_results()
                st.session_state.scan_results = find_videos(base)
                if not st.session_state.scan_results:
                    st.warning("No videos found.")

        if st.session_state.scan_results:
            base_abs = normalize_path(base_dir)

            def to_rel(base_dir: str, abs_path: str) -> str:
                try:
                    if os.path.commonpath([base_dir, abs_path]) == base_dir:
                        return os.path.relpath(abs_path, base_dir)
                except Exception:
                    pass
                return abs_path  # fallback to absolute

            rel_options = [to_rel(base_abs, p) for p in st.session_state.scan_results]
            choice = st.selectbox("Select a video", options=rel_options, index=0, key="scan_choice")
            new_active = st.session_state.scan_results[rel_options.index(choice)]
            set_active_video(new_active)

    frame_extraction_rate = st.number_input("Frame extraction rate (FPS)", min_value=1, max_value=60, value=1, step=1)

    buttons_disabled = not (st.session_state.active_video_path and os.path.isfile(st.session_state.active_video_path))

    buttons_col1, buttons_col2, buttons_col3 = st.columns(3)
    extract_frames_btn = buttons_col1.button("Extract Frames", use_container_width=True, disabled=buttons_disabled)
    extract_audio_btn = buttons_col2.button("Extract Audio", use_container_width=True, disabled=buttons_disabled)
    extract_both_btn = buttons_col3.button("Frames & Audio", use_container_width=True, disabled=buttons_disabled)

    try:
        if extract_frames_btn:
            with st.spinner("Extracting frames"):
                folder, count = extract_frames(st.session_state.active_video_path, frame_rate=frame_extraction_rate)
                st.session_state.last_frames_result = (folder, count)
            st.success(f"Extracted {count} frames to: {folder}")

        if extract_audio_btn:
            with st.spinner("Extracting audio"):
                audio_path = extract_audio(st.session_state.active_video_path)
                st.session_state.last_audio_path = audio_path
            st.success(f"Audio saved to: {audio_path}")

        if extract_both_btn:
            with st.spinner("Extracting frames and audio..."):
                folder, count = extract_frames(st.session_state.active_video_path, frame_rate=frame_extraction_rate)
                audio_path = extract_audio(st.session_state.active_video_path)
                st.session_state.last_frames_result = (folder, count)
                st.session_state.last_audio_path = audio_path
            st.success(f"Frames: {count} → {folder}\n\n Audio: {audio_path}")

    except Exception as e:
        st.error(f"Error: {e}")

    st.divider()
    st.subheader("Results")

    # Frames preview
    if st.session_state.last_frames_result:
        folder, count = st.session_state.last_frames_result
        st.write(f"**Frames Folder:** `{folder}`  —  **Count:** {count}")
        try:
            files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])[:6]
            if files:
                st.write("Preview of first frames:")
                cols = st.columns(min(3, len(files)))
                for i, fname in enumerate(files):
                    img_path = os.path.join(folder, fname)
                    with cols[i % len(cols)]:
                        st.image(img_path, caption=fname, use_container_width=True)
        except Exception as e:
            st.info(f"(Could not preview frames: {e})")

    # Audio preview
    if st.session_state.last_audio_path:
        audio_path = st.session_state.last_audio_path
        st.write(f"**Audio File:** `{audio_path}`")
        try:
            with open(audio_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")
        except Exception as e:
            st.info(f"(Could not preview audio: {e})")

    st.caption("ffmpeg must be installed and on PATH for audio extraction.")