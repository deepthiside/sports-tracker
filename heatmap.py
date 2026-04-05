"""
Generate movement heatmap from tracking_log.csv
Usage: python heatmap.py
"""
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_heatmap(log_csv: str = "output/tracking_log.csv",
                     frame_ref: str = "input_video.mp4",
                     output: str = "output/heatmap.png"):
    df = pd.read_csv(log_csv)
    df["cx"] = (df["x1"] + df["x2"]) // 2
    df["cy"] = (df["y1"] + df["y2"]) // 2

    # Get frame dimensions
    cap = cv2.VideoCapture(frame_ref)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    heatmap = np.zeros((H, W), dtype=np.float32)
    for _, row in df.iterrows():
        cx, cy = int(row["cx"]), int(row["cy"])
        if 0 <= cx < W and 0 <= cy < H:
            cv2.circle(heatmap, (cx, cy), 15, 1, -1)

    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255,
                                  cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    cv2.imwrite(output, colored)
    print(f"[OUT] Heatmap saved → {output}")

    plt.figure(figsize=(12, 7))
    plt.imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    plt.title("Movement Heatmap — All Tracked Subjects")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output.replace(".png", "_matplotlib.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    generate_heatmap()