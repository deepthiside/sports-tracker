"""
Team Clustering via Jersey Color
Uses K-Means clustering on dominant jersey colors to separate
tracked subjects into 2 teams (+ optionally referees).

Usage: python team_clustering.py
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────────────────────
VIDEO_PATH   = "input_video.mp4"
LOG_CSV      = "output/tracking_log.csv"
N_TEAMS      = 3          # 2 teams + 1 for referees/others
SAMPLE_EVERY = 5          # sample every 5 frames per player (faster)
OUTPUT_IMG   = "output/team_clustering.png"
OUTPUT_CSV   = "output/team_assignments.csv"

# ── Step 1 — Extract dominant jersey color per track ID ──────────────────────

def get_dominant_color(crop: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Extract dominant color from the CENTER of a bounding box crop.
    Center crop avoids background pixels at edges.
    Uses K-Means with k=1 to get single dominant color.
    """
    h, w = crop.shape[:2]
    # Take center 50% of the crop (torso area = jersey)
    cy1, cy2 = h // 4, 3 * h // 4
    cx1, cx2 = w // 4, 3 * w // 4
    center   = crop[cy1:cy2, cx1:cx2]

    if center.size == 0:
        return np.array([0, 0, 0])

    pixels = center.reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        return pixels.mean(axis=0)

    km = KMeans(n_clusters=k, n_init=3, random_state=42)
    km.fit(pixels)
    dominant = km.cluster_centers_[0]
    return dominant  # BGR


def extract_colors_per_id(video_path: str,
                           log_csv: str,
                           sample_every: int = 5) -> dict:
    """
    For each track ID, collect average dominant jersey color
    by sampling frames from the video.
    Returns dict: {final_id: mean_BGR_color}
    """
    df  = pd.read_csv(log_csv)
    cap = cv2.VideoCapture(video_path)

    # Build lookup: frame → list of detections
    frame_lookup = defaultdict(list)
    for _, row in df.iterrows():
        if int(row["frame"]) % sample_every == 0:
            frame_lookup[int(row["frame"])].append(row)

    id_colors = defaultdict(list)  # final_id → list of dominant colors
    current_frame = 0

    print("[INFO] Extracting jersey colors per player...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frame_lookup:
            H, W = frame.shape[:2]
            for row in frame_lookup[current_frame]:
                x1 = max(0, int(row["x1"]))
                y1 = max(0, int(row["y1"]))
                x2 = min(W, int(row["x2"]))
                y2 = min(H, int(row["y2"]))
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                color = get_dominant_color(crop)
                id_colors[int(row["final_id"])].append(color)

        current_frame += 1

    cap.release()

    # Average color per ID
    id_mean_color = {}
    for fid, colors in id_colors.items():
        id_mean_color[fid] = np.mean(colors, axis=0)

    print(f"[INFO] Extracted colors for {len(id_mean_color)} unique IDs")
    return id_mean_color


# ── Step 2 — K-Means Cluster IDs into Teams ──────────────────────────────────

def cluster_into_teams(id_mean_color: dict,
                        n_teams: int = 3) -> pd.DataFrame:
    """
    Cluster all player IDs into n_teams groups based on jersey color.
    Returns DataFrame with columns: final_id, team, R, G, B
    """
    ids    = list(id_mean_color.keys())
    colors = np.array([id_mean_color[i] for i in ids], dtype=np.float32)

    # Convert BGR → HSV for better color clustering
    colors_rgb = colors[:, ::-1]  # BGR → RGB
    colors_hsv = np.array([
        cv2.cvtColor(
            c.reshape(1,1,3).astype(np.uint8),
            cv2.COLOR_RGB2HSV
        ).reshape(3)
        for c in colors_rgb.astype(np.uint8)
    ])

    km = KMeans(n_clusters=n_teams, n_init=10, random_state=42)
    labels = km.fit_predict(colors_hsv)

    # Build result DataFrame
    rows = []
    team_names = [f"Team {chr(65+i)}" for i in range(n_teams)]
    # Rename last cluster as "Other" (referees/staff)
    team_names[-1] = "Other/Referee"

    for idx, (fid, label) in enumerate(zip(ids, labels)):
        bgr = id_mean_color[fid]
        rows.append({
            "final_id"  : fid,
            "team"      : team_names[label],
            "team_id"   : int(label),
            "B"         : int(bgr[0]),
            "G"         : int(bgr[1]),
            "R"         : int(bgr[2]),
        })

    return pd.DataFrame(rows).sort_values("final_id")


# ── Step 3 — Visualize Team Assignments ──────────────────────────────────────

def visualize_teams(df: pd.DataFrame,
                    id_mean_color: dict,
                    output_path: str):
    """
    Plot each player ID as a colored dot, grouped by team.
    Color of dot = actual jersey color detected.
    """
    teams      = df["team"].unique()
    n_teams    = len(teams)
    team_colors = plt.cm.Set1(np.linspace(0, 1, n_teams))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left plot: Player IDs colored by jersey color ───────────────────────
    ax = axes[0]
    for _, row in df.iterrows():
        bgr = id_mean_color[row["final_id"]]
        rgb = (bgr[2]/255, bgr[1]/255, bgr[0]/255)  # BGR→RGB normalize
        ax.scatter(row["final_id"], row["team_id"],
                   color=rgb, s=200, edgecolors="black", linewidth=0.5)
        ax.text(row["final_id"], row["team_id"] + 0.1,
                str(row["final_id"]), ha="center", fontsize=7)

    ax.set_yticks(range(n_teams))
    ax.set_yticklabels(df.groupby("team_id")["team"].first().values)
    ax.set_xlabel("Player Track ID", fontsize=11)
    ax.set_title("Player IDs Grouped by Jersey Color", fontsize=12)
    ax.grid(True, alpha=0.3)

    # ── Right plot: Team size bar chart ─────────────────────────────────────
    ax2     = axes[1]
    counts  = df["team"].value_counts()
    bars    = ax2.bar(counts.index, counts.values,
                      color=team_colors[:len(counts)], edgecolor="black")
    ax2.set_xlabel("Team", fontsize=11)
    ax2.set_ylabel("Number of Players", fontsize=11)
    ax2.set_title("Players per Team", fontsize=12)

    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 str(val), ha="center", fontsize=12, fontweight="bold")

    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Team Clustering via Jersey Color Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[OUT] Team clustering plot → {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Extract jersey colors
    id_mean_color = extract_colors_per_id(
        VIDEO_PATH, LOG_CSV, SAMPLE_EVERY
    )

    if len(id_mean_color) == 0:
        print("[ERROR] No colors extracted. Check video and CSV paths.")
        return

    # 2. Cluster into teams
    df = cluster_into_teams(id_mean_color, N_TEAMS)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OUT] Team assignments → {OUTPUT_CSV}")

    # 3. Print summary
    print(f"\n{'='*45}")
    print(f"  TEAM CLUSTERING RESULTS")
    print(f"{'='*45}")
    for team, group in df.groupby("team"):
        ids = group["final_id"].tolist()
        print(f"  {team:20s}: {len(ids):>3} players "
              f"→ IDs {ids[:8]}{'...' if len(ids)>8 else ''}")
    print(f"{'='*45}\n")

    # 4. Visualize
    visualize_teams(df, id_mean_color, OUTPUT_IMG)


if __name__ == "__main__":
    main()