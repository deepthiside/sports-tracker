# 🏀 Multi-Object Detection & Persistent ID Tracking
### Sports/Event Footage — YOLOv8 + ByteTrack Pipeline

## Overview
This pipeline detects and tracks all people in a sports/event video using:
- **Detector:** YOLOv8m (COCO-pretrained)
- **Tracker:** ByteTrack (via Ultralytics)
- **Visualization:** OpenCV — bounding boxes, persistent IDs, motion trails

## 📹 Video Used
- **Source:** [Insert your YouTube link here]
- **Sport:** Basketball / Football / Cricket
- **Duration:** ~2 minutes

## 🛠️ Installation
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/sports-tracker.git
cd sports-tracker

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch with CUDA (Windows, CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## ▶️ How to Run

### Step 1 — Download Video
```bash
python download_video.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

### Step 2 — Run Tracker
```bash
python main.py --input input_video.mp4 --output output/tracked.mp4 --model yolov8m.pt
```

### Step 3 — Generate Heatmap (Optional)
```bash
python heatmap.py
python count_plot.py
```

## ⚙️ Arguments
| Argument | Default | Description |
|---|---|---|
| `--input` | `input_video.mp4` | Input video path |
| `--output` | `output/tracked.mp4` | Output video path |
| `--model` | `yolov8m.pt` | YOLO model size (n/s/m/l/x) |
| `--conf` | `0.35` | Detection confidence threshold |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--skip` | `1` | Process every Nth frame |

## Assumptions
- Video contains primarily human subjects (players/athletes)
- YOLOv8 COCO weights used (no fine-tuning on sports-specific data)
- ByteTrack handles re-identification using motion prediction

## Limitations
- ID switches can occur during heavy occlusion
- Fast camera pans reduce tracking stability
- No appearance-based re-ID (ReID) model used

## Model Choices
- **YOLOv8m** — best balance of speed vs accuracy for real-time use
- **ByteTrack** — superior in crowded scenes vs SORT/DeepSORT