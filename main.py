"""
Main pipeline: Multi-Object Detection & Persistent ID Tracking
Sports/Event Footage — YOLOv8 + ByteTrack + PersistentIDManager

Usage:
    python main.py --input input_video.mp4 --output output/tracked.mp4
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import argparse
import time
import pandas as pd
from id_manager import PersistentIDManager
from tracker import SportsTracker
from visualizer import Visualizer
from utils import get_video_info, create_video_writer, ensure_dir, timestamp


def parse_args():
    parser = argparse.ArgumentParser(description="Sports Multi-Object Tracker")
    parser.add_argument("--input",      type=str,   default="input_video.mp4")
    parser.add_argument("--output",     type=str,   default="output/tracked.mp4")
    parser.add_argument("--model",      type=str,   default="yolov8m.pt",
                        help="YOLOv8 model: yolov8n/s/m/l/x.pt")
    parser.add_argument("--conf",       type=float, default=0.35)
    parser.add_argument("--device",     type=str,   default="cuda")
    parser.add_argument("--skip",       type=int,   default=1,
                        help="Process every Nth frame (1 = all frames)")
    parser.add_argument("--max_frames", type=int,   default=-1,
                        help="Limit frames processed (-1 = all)")
    # PersistentIDManager tuning
    parser.add_argument("--grace",      type=int,   default=45,
                        help="Frames to hold a lost track alive")
    parser.add_argument("--iou_thresh", type=float, default=0.25,
                        help="Min IoU for re-association")
    parser.add_argument("--hist_thresh",type=float, default=0.45,
                        help="Min color-hist similarity for re-association")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir("output")
    ensure_dir("output/screenshots")

    # ── Video Info ───────────────────────────────────────────────────────────
    info  = get_video_info(args.input)
    W, H  = info["width"], info["height"]
    FPS   = info["fps"]
    TOTAL = info["total_frames"]

    cap    = cv2.VideoCapture(args.input)
    writer = create_video_writer(args.output, FPS, W, H)

    # ── Initialize Modules ───────────────────────────────────────────────────
    tracker = SportsTracker(
        model_path      = args.model,
        conf_threshold  = args.conf,
        device          = args.device
    )
    visualizer = Visualizer(draw_trails=True, trail_length=40)

    id_manager = PersistentIDManager(
        grace_period = args.grace,
        iou_thresh   = args.iou_thresh,
        hist_thresh  = args.hist_thresh
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    frame_log       = []   # per-detection rows
    count_over_time = []   # per-frame active count

    frame_idx  = 0
    total_proc = 0
    start_time = time.time()

    print(f"\n[START] Processing {TOTAL} frames ...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        # ── Frame-skip (optional speedup) ───────────────────────────────────
        if frame_idx % args.skip != 0:
            writer.write(frame)
            frame_idx += 1
            continue

        # ── Detect  (ByteTrack raw IDs) ──────────────────────────────────────
        detections = tracker.track(frame)

        # ── Persistent ID resolution  ────────────────────────────────────────
        # Adds 'final_id' to every detection dict.
        # final_id is stable even after occlusion / camera cuts.
        detections = id_manager.update(detections, frame)

        # ── Visualise + Log ──────────────────────────────────────────────────
        for det in detections:
            frame = visualizer.draw(
                frame,
                track_id = det["final_id"],          # ← stable ID
                x1=det["x1"], y1=det["y1"],
                x2=det["x2"], y2=det["y2"]
            )
            frame_log.append({
                "frame"       : frame_idx,
                "final_id"    : det["final_id"],     # ← stable ID in CSV too
                "bytetrack_id": det["track_id"],     # raw BT id kept for debug
                "x1"  : det["x1"],  "y1": det["y1"],
                "x2"  : det["x2"],  "y2": det["y2"],
                "conf": round(det["conf"], 3)
            })

        active = len(detections)
        frame  = visualizer.draw_count(frame, active)
        count_over_time.append({"frame": frame_idx, "count": active})

        # ── Screenshot every 200 frames ──────────────────────────────────────
        if frame_idx % 200 == 0:
            sc_path = f"output/screenshots/frame_{frame_idx:05d}.jpg"
            cv2.imwrite(sc_path, frame)
            elapsed = time.time() - start_time
            print(f"  Frame {frame_idx:>6}/{TOTAL} | "
                  f"Active: {active:>3} | "
                  f"Elapsed: {elapsed:.1f}s")

        writer.write(frame)
        frame_idx  += 1
        total_proc += 1

    cap.release()
    writer.release()

    # ── Final ID Manager Summary ─────────────────────────────────────────────
    id_summary = id_manager.summary()

    elapsed = time.time() - start_time
    print(f"\n[DONE] Processed {total_proc} frames in {elapsed:.1f}s")
    print(f"[OUT]  Annotated video → {args.output}")

    # ── Save Analytics CSVs ──────────────────────────────────────────────────
    if frame_log:
        df = pd.DataFrame(frame_log)
        df.to_csv("output/tracking_log.csv", index=False)
        print(f"[OUT]  Tracking log    → output/tracking_log.csv")

        cdf = pd.DataFrame(count_over_time)
        cdf.to_csv("output/count_over_time.csv", index=False)
        print(f"[OUT]  Count over time → output/count_over_time.csv")

        # ── Console Summary ──────────────────────────────────────────────────
        print(f"\n{'='*45}")
        print(f"  TRACKING SUMMARY")
        print(f"{'='*45}")
        print(f"  Unique stable IDs assigned : {id_summary['total_unique_ids']}")
        print(f"  Currently active tracks    : {id_summary['currently_active']}")
        print(f"  Retired tracks             : {id_summary['retired_tracks']}")
        print(f"  Total detections logged    : {len(df)}")
        print(f"  Avg detections / frame     : {len(df)/total_proc:.2f}")
        print(f"  Processing speed           : {total_proc/elapsed:.1f} FPS")
        print(f"{'='*45}\n")


if __name__ == "__main__":
    main()