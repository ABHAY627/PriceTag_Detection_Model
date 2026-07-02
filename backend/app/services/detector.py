"""
YOLOv8 detection service — loads model once, exposes detect() function.
Falls back to a mock detector if no trained weights exist yet (demo mode).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root
WEIGHTS = ROOT / "models" / "checkpoints" / "best.pt"


class DetectionBox:
    def __init__(self, x1, y1, x2, y2, confidence, class_id=0):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.confidence = float(confidence)
        self.class_id = int(class_id)

    @property
    def cx(self): return (self.x1 + self.x2) / 2
    @property
    def cy(self): return (self.y1 + self.y2) / 2
    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1

    def crop(self, img: np.ndarray) -> np.ndarray:
        x1, y1 = max(0, int(self.x1)), max(0, int(self.y1))
        x2 = min(img.shape[1], int(self.x2))
        y2 = min(img.shape[0], int(self.y2))
        return img[y1:y2, x1:x2]


class YOLODetector:
    def __init__(self, weights_path: Path):
        from ultralytics import YOLO
        self.model = YOLO(str(weights_path))
        self._name = weights_path.name

    def detect(self, img: np.ndarray, conf: float = 0.35) -> list[DetectionBox]:
        results = self.model.predict(img, conf=conf, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu())
                class_id = int(box.cls[0].cpu())
                boxes.append(DetectionBox(x1, y1, x2, y2, confidence, class_id))
        return boxes

    @property
    def model_name(self): return self._name


class MockDetector:
    """
    Used when no trained weights exist yet.
    Returns synthetic bounding boxes for demo/testing purposes.
    """

    def detect(self, img: np.ndarray, conf: float = 0.35) -> list[DetectionBox]:
        h, w = img.shape[:2]
        # Simulate 2-4 evenly-spaced price tags across the bottom third
        boxes = []
        n = 3
        tag_w = int(w * 0.12)
        tag_h = int(h * 0.06)
        spacing = w // (n + 1)
        for i in range(n):
            x1 = spacing * (i + 1) - tag_w // 2
            y1 = int(h * 0.65)
            boxes.append(DetectionBox(x1, y1, x1 + tag_w, y1 + tag_h,
                                      confidence=0.70 + i * 0.05))
        return boxes

    @property
    def model_name(self): return "mock_detector (no weights found)"


_detector: Optional[object] = None


def get_detector():
    global _detector
    if _detector is None:
        if WEIGHTS.exists():
            print(f"✅  Loading YOLO detector from {WEIGHTS}")
            _detector = YOLODetector(WEIGHTS)
        else:
            print(f"⚠️   No weights at {WEIGHTS} — using MockDetector (demo mode)")
            print("     Run: python scripts/train_detector.py to train the real model.")
            _detector = MockDetector()
    return _detector
