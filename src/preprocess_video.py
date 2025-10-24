import os
import tempfile
from typing import Optional
import cv2
from yt_dlp import YoutubeDL
import ffmpeg

from src.utils import get_max_upload_size


def download_youtube_video(youtube_url: str, output_path: Optional[str] = None) -> str:
    """download video from youtube - save to path if provided else uses temp file"""

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        video_path = os.path.join(output_path, "%(title)s.%(ext)s")
    else:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_path = temp_file.name

    max_size_mb = get_max_upload_size("../.streamlit/config.toml")
    ydl_opts = {
        "format": f"bestvideo[ext=mp4][filesize<={max_size_mb * 1024 * 1024}]+bestaudio[ext=m4a]/best[ext=mp4][filesize<={max_size_mb * 1024 * 1024}]",
        "outtmpl": video_path,
        "quiet": True,
        "merge_output_format": "mp4"
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        downloaded_path = ydl.prepare_filename(info)

    print(f"Downloaded: {downloaded_path}")
    return downloaded_path

def get_best_stream_within_constraints(yt, max_size_mb=1024):
    # note progressive=True → includes both video and audio in one stream (vs. DASH which separates them)
    streams = yt.streams.filter(file_extension="mp4", progressive=True).order_by("resolution").desc()
    for stream in streams:
        size_mb = stream.filesize_approx / (1024 * 1024)   # convert MB
        if size_mb <= max_size_mb:
            return stream
    return None

def extract_frames(video_path: str, frame_rate: int = 1, output_folder: Optional[str] = None):
    """Extract frames from the video at a given frame rate (fps)"""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)
    count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read
        if not ret:
            break
        if count % interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(frame_name, frame)
            frame_id += 1
        count += 1
    cap.release()
    print(f"Extracted {frame_id} frames to {output_folder}")

def extract_audio(video_path: str, output_path: Optional[str] = None):
    """Extract audio track from video using ffmpeg"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    (
        ffmpeg.input(video_path).output(output_path, ac=1, ar=16000).overwrite_output().run(quiet=True)
    )
    print(f"Audio save to {output_path}")

if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=2lAe1cqCOXo"  # use a short public video
    try:
        video_path = download_youtube_video(test_url)
        print(f"Test successful. Video saved at: {video_path}")
        print(f"File size: {os.path.getsize(video_path) / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"Test failed: {e}")