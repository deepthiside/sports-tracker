"""
Utility functions for the sports tracking pipeline.
"""
import cv2
import numpy as np
import os
from datetime import datetime

# Assign a unique color per tracker ID (for consistent coloring)
COLOR_PALETTE = np.random.default_rng(42).uniform(50, 230, (300, 3)).astype(int)

def get_color(track_id: int) -> tuple:
    """Return a consistent BGR color for a given track ID."""
    color = COLOR_PALETTE[int(track_id) % len(COLOR_PALETTE)]
    return (int(color[0]), int(color[1]), int(color[2]))

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def get_video_info(video_path: str) -> dict:
    """Extract video metadata."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    print(f"[INFO] Video Info: {info}")
    return info

def create_video_writer(output_path: str, fps: float, width: int, height: int):
    """Create an OpenCV VideoWriter."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")