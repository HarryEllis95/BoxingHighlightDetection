import os
import cv2
from pytube import YouTube
import ffmpeg

from pytube import YouTube


def download_video(youtube_url: str, output_path: str = "data/raw") -> str:
    """download video from youtube"""
    os.makedirs(output_path, exist_ok=True)
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension="mp4").first()
    video_path = stream.download(output_path)
    print(f"Downloaded: {video_path}")
    return video_path