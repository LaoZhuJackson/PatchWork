"""YOLO 推理引擎：模型加载 + 推理 + supervision 转换"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PySide6.QtCore import QRectF, QPointF
from ultralytics import YOLO

from app.services.label_reader import get_color


class InferenceEngine:
    """YOLO 推理引擎，模型只加载一次，复用"""

    def __init__(self) -> None:
        self._model: YOLO | None = None
        self._model_path: str = ""
        self._class_names: dict[int, str] = {}
        self._task: str = ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    @property
    def task(self) -> str:
        """模型任务类型: 'detect' | 'segment' | 'classify' | ''"""
        return self._task

    def load_model(self, model_path: str | Path) -> None:
        """加载YOLO模型"""
        try:
            self._model = YOLO(str(model_path))
        except AttributeError as e:
            msg = str(e)
            if "qkv" in msg or "qk" in msg or "AAttn" in msg:
                raise RuntimeError(
                    "模型权重与当前 ultralytics 版本不兼容。\n\n"
                    "可能原因：\n"
                    "  • YOLOv12 老版权重 + 新版 ultralytics\n"
                    "  • YOLOv12 Turbo 权重 + 旧版 ultralytics\n\n"
                    "解决方法：\n"
                    f"  1. 升级 ultralytics: pip install ultralytics --upgrade\n"
                    f"  2. 重新下载匹配的权重文件\n"
                ) from e
            raise
        self._model_path = str(model_path)
        self._class_names = self._model.names or {}
        # 获取任务类型
        overrides = self._model.overrides
        self._task = overrides.get("task", "")

    def infer(self, image_path: str | Path, conf: float = 0.25, iou: float = 0.45) -> list[dict]:
        """对单张图片推理，返回标注列表。

        - detect 模型: 返回 bbox
        - segment 模型: 返回 bbox + 多边形
        """
        if self._model is None:
            raise RuntimeError("模型未加载")

        results = self._model(str(image_path), conf=conf, iou=iou, verbose=False)
        results = results[0]

        detections = sv.Detections.from_ultralytics(results)
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

            class_name = self._class_names.get(class_id, f"class_{class_id}")
            color = get_color(class_id)
            label = f"{class_name} {confidence:.2f}"

            # bbox
            annotations.append({
                "type": "bbox",
                "rect": QRectF(x1, y1, x2 - x1, y2 - y1),
                "class_id": class_id,
                "color": color,
                "label": label,
            })

            # 分割多边形（仅 seg 模型）
            if has_mask:
                mask = detections.mask[i].astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    if len(contour) >= 3:
                        points = [
                            QPointF(float(p[0][0]), float(p[0][1]))
                            for p in contour
                        ]
                        annotations.append({
                            "type": "polygon",
                            "points": points,
                            "class_id": class_id,
                            "color": color,
                            "label": "",
                        })
        return annotations
