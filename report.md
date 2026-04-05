# Technical Report: Multi-Object Detection and Persistent ID Tracking

## 1. Model & Detector
**YOLOv8m** (Medium variant) was selected as the detector:
- Pretrained on COCO (includes 'person' class)
- Processes ~30 FPS on a mid-range NVIDIA GPU
- Strong performance on partially occluded subjects

## 2. Tracking Algorithm
**ByteTrack** was selected for multi-object tracking:
- Considers *both* high-confidence and low-confidence detections
- Uses Kalman filtering for motion prediction between frames
- Significantly outperforms SORT/DeepSORT in crowded sports scenes
- Native integration into Ultralytics YOLOv8 via `persist=True`

## 3. Why This Combination
| Factor | Reason |
|---|---|
| Speed | YOLOv8m runs in real-time on GPU |
| Occlusion handling | ByteTrack keeps low-conf detections as candidates |
| Ease of use | Single API call handles detection + tracking |
| ID stability | Kalman filter predicts position when detection is lost |

## 4. ID Consistency Strategy
- `persist=True` in YOLOv8 maintains ByteTrack state across frames
- Each track maintains a unique integer ID for its lifetime
- When a subject disappears and reappears within ~30 frames, ByteTrack attempts to re-associate based on IoU overlap of predicted vs detected bounding boxes

## 5. Challenges Faced
- **Heavy occlusion** (players overlapping): ID switches observed
- **Fast camera pans**: Sudden global motion caused brief tracking loss
- **Similar appearance**: Players in same uniform sometimes swapped IDs

## 6. Failure Cases
- Two players crossing paths at high speed → occasional ID swap
- Subjects partially leaving frame → track lost and re-initialized with new ID
- Very fast motion blur → detection confidence drops below threshold

## 7. Possible Improvements
- Add **ReID model** (e.g., OSNet) for appearance-based re-identification
- Use **BoT-SORT** instead of ByteTrack for better camera-motion compensation
- Fine-tune YOLOv8 on sports-specific dataset (e.g., SportsMOT)
- Add **team clustering** via jersey color segmentation
- Implement **speed estimation** using homography projection