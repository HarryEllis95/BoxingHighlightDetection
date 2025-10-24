import streamlit as st

from src.preprocess_video import download_youtube_video

st.set_page_config(page_title = "Boxing Highlight Detection", layout="wide", initial_sidebar_state="collapsed")

st.title("🥊 Boxing Highlight Detector")

# for now just uploaded downloaded mp4 but would be nicer to have option to select youtube video
# and analyse it through temp storage user might not want it permanently downloaded
uploaded_video = st.file_uploader("Upload a Boxing Video", type=["mp4"])
video_url = st.text_input("Or paste a YouTube URL")

video_path = None

if uploaded_video:
    st.video(uploaded_video)
    video_path = uploaded_video
elif video_url:
    st.video(video_url)
    with st.spinner("Downloading video..."):
        video_path = download_youtube_video(video_url)
else:
    st.write("Choose to start analysis")

if video_path:
    st.success("Ready for analysis")

