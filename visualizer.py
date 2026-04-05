"""
Visualization utilities: bounding boxes, IDs, trajectories, heatmaps.
"""
import cv2
import numpy as np
from collections import defaultdict
from utils import get_color

class Visualizer:
    def __init__(self, draw_trails: bool = True, trail_length: int = 30):
        self.draw_trails = draw_trails
        self.trail_length = trail_length
        # Store center history per track ID for trajectory trails
        self.track_history = defaultdict(list)

    def draw(self, frame: np.ndarray, track_id: int,
             x1: int, y1: int, x2: int, y2: int,
             label: str = "Person") -> np.ndarray:
        """Draw bounding box, ID label, and trail on a frame."""
        color = get_color(track_id)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # --- Bounding box ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # --- ID Label with background ---
        text = f"ID {track_id}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Trail / Trajectory ---
        if self.draw_trails:
            self.track_history[track_id].append((cx, cy))
            if len(self.track_history[track_id]) > self.trail_length:
                self.track_history[track_id].pop(0)
            pts = self.track_history[track_id]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pts[i - 1], pts[i], trail_color, 2)

        return frame

    def draw_count(self, frame: np.ndarray, count: int) -> np.ndarray:
        """Draw total active tracked subjects count."""
        text = f"Active Subjects: {count}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def reset_trails(self):
        self.track_history.clear()