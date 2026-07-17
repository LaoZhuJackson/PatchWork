from __future__ import annotations

from pathlib import Path

from app.adapters.base import InferenceAdapter
from app.services.inference import InferenceEngine

class NormalAdapter(InferenceAdapter):
    """包装现有 InferenceEngine"""
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45) -> None:
        self._engine = InferenceEngine()
        self._model_path = model_path
        self._conf = conf
        self._iou = iou

    @property
    def name(self) -> str:
        return "YOLO 普通推理"

    def load_model(self) -> None:
        self._engine.load_model(self._model_path)

    def infer(self, image_path: Path) -> list[dict]:
        return self._engine.infer(str(image_path), conf=self._conf, iou=self._iou)

    @property
    def config_summary(self) -> str:
        return f"Conf={self._conf} IoU={self._iou}"