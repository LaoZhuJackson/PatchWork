"""SAHI 切片推理服务"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QRectF, QPointF
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from app.services.label_reader import get_color


@dataclass
class SahiConfig:
    model_path: Path
    model_type: str = "ultralytics"
    confidence: float = 0.25
    iou: float = 0.7
    slice_width: int = 640
    slice_height: int = 640
    overlap_width_ratio: float = 0.25
    overlap_height_ratio: float = 0.25
    perform_standard_pred: bool = True
    device: str = "cuda:0"


class SahiInferenceService:
    """SAHI 切片推理"""

    def __init__(self, config: SahiConfig) -> None:
        self.config = config
        self._model: AutoDetectionModel | None = None

    def load_model(self) -> None:
        self._model = AutoDetectionModel.from_pretrained(
            model_type=self.config.model_type,
            model_path=str(self.config.model_path),
            confidence_threshold=self.config.confidence,
            device=self.config.device,
        )

    def infer_image(self, image_path: Path) -> list[dict]:
        """对单张图片执行切片推理，返回统一标注列表"""
        if self._model is None:
            raise RuntimeError("SAHI 模型未加载")

        result = get_sliced_prediction(
            image=str(image_path),
            detection_model=self._model,
            slice_height=self.config.slice_height,
            slice_width=self.config.slice_width,
            overlap_width_ratio=self.config.overlap_width_ratio,
            overlap_height_ratio=self.config.overlap_height_ratio,
            perform_standard_pred=self.config.perform_standard_pred,
            verbose=0,
        )

        return self._convert_result(result)

    def _convert_result(self, result) -> list[dict]:
        """将 SAHI 结果转为统一标注格式"""
        annotations: list[dict] = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            x1, y1 = bbox.minx, bbox.miny
            x2, y2 = bbox.maxx, bbox.maxy

            class_id = pred.category.id
            class_name = pred.category.name
            confidence = pred.score.value

            color = get_color(class_id)
            label = f"{class_name} {confidence:.2f}"

            annotations.append({
                "type": "bbox",
                "rect": QRectF(x1, y1, x2 - x1, y2 - y1),
                "class_id": class_id,
                "color": color,
                "label": label,
            })

            # 分割多边形（如果有 mask）
            if hasattr(pred, 'mask') and pred.mask is not None:
                mask = pred.mask.bool_mask
                if mask is not None:
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                    )
                    for contour in contours:
                        if len(contour) >= 3:
                            points = [QPointF(float(p[0][0]), float(p[0][1])) for p in contour]
                            annotations.append({
                                "type": "polygon",
                                "points": points,
                                "class_id": class_id,
                                "color": color,
                                "label": "",
                            })
        return annotations

    def save_visualization(self, image_path: Path, output_path: Path) -> None:
        """保存带标注的可视化图片"""
        result = get_sliced_prediction(
            image=str(image_path),
            detection_model=self._model,
            slice_height=self.config.slice_height,
            slice_width=self.config.slice_width,
            overlap_height_ratio=self.config.overlap_height_ratio,
            overlap_width_ratio=self.config.overlap_width_ratio,
            perform_standard_pred=self.config.perform_standard_pred,
            verbose=0,
        )
        result.export_visuals(export_dir=str(output_path.parent), file_name=output_path.name)
