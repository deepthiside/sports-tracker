"""
Detection + Tracking pipeline using YOLOv8 + ByteTrack.
"""
import numpy as np
from ultralytics import YOLO

class SportsTracker:
    def __init__(self,
                 model_path: str = "yolov8m.pt",
                 conf_threshold: float = 0.35,
                 iou_threshold: float = 0.45,
                 device: str = "cuda"):
        """
        Args:
            model_path:      YOLOv8 weights (auto-downloaded if not present)
            conf_threshold:  Minimum detection confidence
            iou_threshold:   IoU threshold for NMS
            device:          'cuda' for GPU, 'cpu' for CPU
        """
        print(f"[INFO] Loading YOLO model: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.iou  = iou_threshold
        self.device = device

        # COCO class index for 'person' is 0
        self.target_classes = [0]

    def track(self, frame: np.ndarray) -> list:
        """
        Run detection + ByteTrack on a single frame.

        Returns:
            List of dicts: [{track_id, x1, y1, x2, y2, conf, cls}, ...]
        """
        results = self.model.track(
            frame,
            persist=True,           # ByteTrack: maintain ID across frames
            conf=self.conf,
            iou=self.iou,
            classes=self.target_classes,
            device=self.device,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detections = []
        if results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        if boxes.id is None:
            return detections

        for box, track_id in zip(boxes, boxes.id.int().cpu().tolist()):
            x1, y1, x2, y2 = box.xyxy[0].int().cpu().tolist()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            detections.append({
                "track_id": track_id,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "conf": conf,
                "cls": cls
            })

        return detections