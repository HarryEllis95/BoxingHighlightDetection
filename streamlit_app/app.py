import streamlit as st

st.set_page_config(page_title = "Boxing Highlight Detection", layout="wide", initial_sidebar_state="collapsed")

st.title("🥊 Boxing Highlight Detector")

# for now just uploaded downloaded mp4 but would be nicer to have option to select youtube video
# and analyse it through temp storage user might not want it permanently downloaded
uploaded_video = st.file_uploader("Upload a Boxing Video", type=["mp4"])

if uploaded_video:
    st.video(uploaded_video)
else:
    st.write("Upload boxing match (.mp4) to start analysis")

