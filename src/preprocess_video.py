import os
import tempfile
from typing import Optional
import cv2
from pytube import YouTube    # yt-dlp may be better option ??
import ffmpeg

from src.utils import get_max_upload_size


def download_youtube_video(youtube_url: str, output_path: Optional[str] = None) -> str:
    """download video from youtube - save to path if provided else uses temp file"""
    yt = YouTube(youtube_url)
    max_size_mb = get_max_upload_size()
    stream = get_best_stream_within_constraints(yt, max_size_mb=max_size_mb)

    if not stream:
        raise ValueError("No suitable stream found within size limit.")

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        video_path = stream.download(output_path=output_path)
    else:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_path = stream.download(output_path=os.path.dirname(temp_file.name), filename=os.path.basename(temp_file.name))
    print(f"Downloaded: {video_path}")
    return video_path

def get_best_stream_within_constraints(yt, max_size_mb=1024):
    # note progressive=True → includes both video and audio in one stream (vs. DASH which separates them)
    streams = yt.streams.filter(file_extension="mp4", progressive=True).order_by("resolution").desc()
    for stream in streams:
        size_mb = stream.filesize_approx / (1024 * 1024)   # convert MB
        if size_mb <= max_size_mb:
            return stream
    return None
