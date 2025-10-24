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
