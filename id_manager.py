"""
Persistent ID Manager — 3-layer ID stability system.

Problem:  ByteTrack resets or swaps IDs after occlusion/disappearance.
Solution: We maintain our own ID registry that:
  1. Maps ByteTrack IDs → our stable "final" IDs
  2. Uses IoU overlap to re-associate lost tracks
  3. Uses color histogram similarity as appearance cue
  4. Holds IDs in a "grace period" buffer before retiring them
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class TrackRecord:
    """Everything we remember about one tracked subject."""
    final_id: int                      # Our stable ID (never changes)
    bytetrack_id: int                  # Current ByteTrack ID
    bbox: Tuple[int,int,int,int]       # Last known (x1,y1,x2,y2)
    color_hist: Optional[np.ndarray]   # BGR color histogram (appearance)
    last_seen_frame: int               # Frame index last detected
    active: bool = True                # Currently visible?
    total_detections: int = 1          # How many frames detected


# ── Core Manager ─────────────────────────────────────────────────────────────

class PersistentIDManager:
    """
    Maintains stable subject IDs across the full video.

    Key parameters
    --------------
    grace_period   : frames to keep a lost track alive before retiring
    iou_thresh     : min IoU to re-associate a returning track
    hist_thresh    : min histogram similarity for appearance match
    """

    def __init__(self,
                 grace_period: int   = 45,
                 iou_thresh: float   = 0.25,
                 hist_thresh: float  = 0.45):

        self.grace_period = grace_period
        self.iou_thresh   = iou_thresh
        self.hist_thresh  = hist_thresh

        # bytetrack_id → TrackRecord (active tracks)
        self._active: Dict[int, TrackRecord] = {}

        # retired tracks (lost > grace_period frames ago)
        # kept for potential long-term re-id
        self._retired: List[TrackRecord] = []

        self._next_id = 1          # monotonically increasing stable ID
        self._frame_idx = 0        # updated each frame

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self,
               detections: list,
               frame: np.ndarray) -> list:
        """
        Call once per frame with ByteTrack detections.

        Parameters
        ----------
        detections : list of dicts from tracker.py
                     [{track_id, x1, y1, x2, y2, conf, cls}, ...]
        frame      : current BGR frame (used for color histograms)

        Returns
        -------
        Same list with an added 'final_id' key in each dict.
        """
        self._frame_idx += 1
        current_bt_ids = {d["track_id"] for d in detections}

        # Step 1 — mark tracks not seen this frame as inactive
        self._mark_lost(current_bt_ids)

        # Step 2 — for each detection, resolve its stable final_id
        for det in detections:
            bt_id = det["track_id"]
            bbox  = (det["x1"], det["y1"], det["x2"], det["y2"])
            crop  = self._crop(frame, bbox)
            hist  = self._color_hist(crop) if crop is not None else None

            if bt_id in self._active:
                # Known active track — just update it
                rec = self._active[bt_id]
                rec.bbox               = bbox
                rec.color_hist         = hist
                rec.last_seen_frame    = self._frame_idx
                rec.active             = True
                rec.total_detections  += 1
                det["final_id"]        = rec.final_id

            else:
                # New ByteTrack ID — could be:
                #   (a) genuinely new subject  → assign next_id
                #   (b) returning subject whose track was lost → re-use old ID
                matched_id = self._try_reidentify(bbox, hist)

                if matched_id is not None:
                    final_id = matched_id
                else:
                    final_id = self._next_id
                    self._next_id += 1

                rec = TrackRecord(
                    final_id        = final_id,
                    bytetrack_id    = bt_id,
                    bbox            = bbox,
                    color_hist      = hist,
                    last_seen_frame = self._frame_idx,
                    active          = True,
                )
                self._active[bt_id] = rec
                det["final_id"] = final_id

        # Step 3 — retire tracks that exceeded grace period
        self._retire_old_tracks()

        return detections

    def summary(self) -> dict:
        """Return stats about tracking."""
        all_ids = (
            {r.final_id for r in self._active.values()} |
            {r.final_id for r in self._retired}
        )
        return {
            "total_unique_ids" : len(all_ids),
            "currently_active" : sum(1 for r in self._active.values() if r.active),
            "retired_tracks"   : len(self._retired),
        }

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _mark_lost(self, current_bt_ids: set):
        """Mark tracks not present in this frame as inactive."""
        for bt_id, rec in self._active.items():
            if bt_id not in current_bt_ids:
                rec.active = False

    def _try_reidentify(self,
                        bbox: Tuple,
                        hist: Optional[np.ndarray]) -> Optional[int]:
        """
        Try to match a new ByteTrack ID to a recently lost track.
        Uses IoU + color histogram similarity.
        Returns the matched final_id or None.
        """
        best_score  = -1
        best_id     = None

        candidates = [
            r for r in self._active.values()
            if not r.active   # currently lost tracks
        ] + self._retired[-30:]  # last 30 retired tracks

        for rec in candidates:
            iou = self._iou(bbox, rec.bbox)
            if iou < self.iou_thresh:
                continue

            # Appearance similarity
            hist_sim = 0.5  # neutral if no histogram
            if hist is not None and rec.color_hist is not None:
                hist_sim = cv2.compareHist(
                    hist, rec.color_hist,
                    cv2.HISTCMP_CORREL   # returns -1..1, higher = more similar
                )
                hist_sim = (hist_sim + 1) / 2  # normalize to 0..1

            if hist_sim < self.hist_thresh:
                continue

            # Combined score (weighted)
            score = 0.6 * iou + 0.4 * hist_sim

            if score > best_score:
                best_score = score
                best_id    = rec.final_id

        return best_id

    def _retire_old_tracks(self):
        """Move tracks unseen for > grace_period frames to retired list."""
        to_retire = []
        for bt_id, rec in self._active.items():
            frames_lost = self._frame_idx - rec.last_seen_frame
            if frames_lost > self.grace_period:
                to_retire.append(bt_id)

        for bt_id in to_retire:
            rec = self._active.pop(bt_id)
            rec.active = False
            self._retired.append(rec)

    @staticmethod
    def _iou(boxA: Tuple, boxB: Tuple) -> float:
        """Compute Intersection over Union between two boxes."""
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0

        areaA = max(1, (ax2-ax1) * (ay2-ay1))
        areaB = max(1, (bx2-bx1) * (by2-by1))
        return inter / (areaA + areaB - inter)

    @staticmethod
    def _crop(frame: np.ndarray,
              bbox: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
        """Safely crop a bounding box region from a frame."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _color_hist(crop: np.ndarray) -> np.ndarray:
        """
        Compute a normalized BGR color histogram for appearance matching.
        16 bins per channel → 16³ = 4096-dim vector (compact & fast).
        """
        hist = cv2.calcHist(
            [crop], [0, 1, 2], None,
            [16, 16, 16],
            [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten()   