"""视频跟踪适配器：逐帧 track，维持跨帧状态"""
from __future__ import annotations

from pathlib import Path

import cv2
import supervision as sv
from PySide6.QtCore import QRectF
from ultralytics import YOLO

from app.adapters.base import InferenceAdapter
from app.services.label_reader import get_color


class TrackingAdapter(InferenceAdapter):
    """BoT-SORT / ByteTrack 逐帧跟踪，作为 benchmark 的一种推理方式"""

    def __init__(
            self,
            model_path: str,
            tracker: str = "botsort.yaml",
            conf: float = 0.25,
            iou: float = 0.7,
            imgsz: int = 640,
    ) -> None:
        self._model_path = model_path
        self._tracker = tracker
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._model: YOLO | None = None
        self._class_names: dict[int, str] = {}

        # 跟踪器名称显示
        self._tracker_name = "BoT-SORT" if "botsort" in tracker else "ByteTrack"

    @property
    def name(self) -> str:
        return f"YOLO + {self._tracker_name} 跟踪"

    def load_model(self) -> None:
        self._model = YOLO(str(self._model_path))
        self._class_names = self._model.names or {}
        # reset tracking state for new benchmark run

    def infer(self, image_path: Path) -> list[dict]:
        if self._model is None:
            raise RuntimeError("模型未加载")

        # OpenCV 读帧；track 保持跨帧 persist
        frame = cv2.imread(str(image_path))
        if frame is None:
            return []

        results = self._model.track(
            frame,
            persist=True,
            tracker=self._tracker,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            verbose=False,
        )

        detections = sv.Detections.from_ultralytics(results[0])
        annotations: list[dict] = []

        if detections.xyxy is None or len(detections.xyxy) == 0:
            return annotations

        has_mask = detections.mask is not None

        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i].tolist()
            class_id = (
                int(detections.class_id[i])
                if detections.class_id is not None
                else 0
            )
            confidence = (
                float(detections.confidence[i])
                if detections.confidence is not None
                else 0.0
            )

            class_name = self._class_names.get(class_id, f"cls_{class_id}")
            color = get_color(class_id)
            label = f"{class_name} {confidence:.2f}"

            annotations.append({
                "type": "bbox",
                "rect": QRectF(x1, y1, x2 - x1, y2 - y1),
                "class_id": class_id,
                "color": color,
                "label": label,
            })

            if has_mask:
                mask = detections.mask[i].astype("uint8")
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                for contour in contours:
                    if len(contour) >= 3:
                        from PySide6.QtCore import QPointF
                        annotations.append({
                            "type": "polygon",
                            "points": [QPointF(float(p[0][0]), float(p[0][1])) for p in contour],
                            "class_id": class_id,
                            "color": color,
                            "label": "",
                        })

        return annotations

    @property
    def config_summary(self) -> str:
        return (
            f"{self._tracker_name} Conf={self._conf} IoU={self._iou} ImgSz={self._imgsz}"
        )