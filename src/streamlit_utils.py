import os
from typing import List

ALLOWED_VIDEO_EXTENSIONS = {".mp4"} # just handle mp4 for now

def normalize_path(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    return os.path.abspath(os.path.expanduser(p)) if p else p

def find_videos(base_dir: str, max_files: int = 1000) -> List[str]:
    """Locate video files under a directory - only gets ones with allowed extension"""
    results = []
    if not os.path.isdir(base_dir):
        return results

    base_dir = os.path.abspath(os.path.expanduser(base_dir))
    for root, _, files in os.walk(base_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in ALLOWED_VIDEO_EXTENSIONS:
                results.append(os.path.join(root, name))
                if len(results) >= max_files:
                    return sorted(results)
    return sorted(results)