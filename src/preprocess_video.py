import os
import tempfile
from typing import Optional
import cv2
import argparse
from yt_dlp import YoutubeDL
import ffmpeg
import shutil

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

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_folder is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        parent = os.path.dirname(video_path) or "."
        output_folder = os.path.join(parent, f"{base}_frames")
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)    # open video with OpenCV
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)       # retrieve video's native framerate
    if not fps or fps <= 0:
        raise ValueError(f"Video reports invalid FPS: {fps}")

    interval = max(int(round(fps / frame_rate)), 1)

    count = 0
    frame_id = 0
    # frame extraction
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{frame_id:05d}.jpg")
            if not cv2.imwrite(frame_name, frame):
                raise RuntimeError(f"Failed to write frame: {frame_name}")
            frame_id += 1
        count += 1

    cap.release()
    if frame_id == 0:
        raise RuntimeError("No frames were extracted. Check video content or frame rate settings.")

    print(f"Extracted {frame_id} frames to {output_folder}")
    return output_folder, frame_id

def extract_audio(video_path: str, output_path: Optional[str] = None):
    """Extract audio track from video using ffmpeg (mono 16 kHz WAV)"""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        parent = os.path.dirname(video_path) or "."
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(parent, f"{base}.wav")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, ac=1, ar=16000)  # mono and 16khz - standard values
            .overwrite_output()
            .run(quiet=True)
        )
    except:
        raise RuntimeError(f"ffmpeg failed to extract audio")

    print(f"Audio save to {output_path}")
    return output_path

# basic test make sure we can extract
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames and audio from a video file.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--frame_rate", type=int, default=1, help="Frames per second to extract (default: 1)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save frames and audio (default: temp)")
    parser.add_argument("--cleanup", action="store_true", help="Delete output files after extraction")

    args = parser.parse_args()

    video_path = args.video_path
    frame_rate = args.frame_rate
    output_dir = args.output_dir
    cleanup = args.cleanup

    if output_dir is None:
        output_dir = os.path.dirname(video_path)

    frame_dir = os.path.join(output_dir, "frames")
    frame_dir = extract_frames(video_path, frame_rate=frame_rate, output_folder=frame_dir)
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    print(f"Frame extraction complete: {len(frame_files)} frames saved to {frame_dir}")
    assert len(frame_files) > 0

    # check first frame to sanity check extraction
    first_frame_path = os.path.join(frame_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    assert first_frame is not None

    audio_path = os.path.join(output_dir, "audio.wav")
    audio_path = extract_audio(video_path, output_path=audio_path)
    assert os.path.exists(audio_path)

    if cleanup:
        shutil.rmtree(frame_dir)
        os.remove(audio_path)
        print(f"🧹 Cleaned up extracted files from {output_dir}")

# if __name__ == "__main__":
#     test_url = "https://www.youtube.com/watch?v=2lAe1cqCOXo"  # use a short public video
#     try:
#         video_path = download_youtube_video(test_url)
#         print(f"Test successful. Video saved at: {video_path}")
#         print(f"File size: {os.path.getsize(video_path) / (1024 * 1024):.2f} MB")
#     except Exception as e:
#         print(f"Test failed: {e}")