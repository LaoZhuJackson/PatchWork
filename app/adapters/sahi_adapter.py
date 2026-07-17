"""SAHI 切片推理适配器"""
from __future__ import annotations

from pathlib import Path

from app.adapters.base import InferenceAdapter
from app.services.sahi_inference import SahiConfig, SahiInferenceService


class SahiAdapter(InferenceAdapter):
    """包装 SahiInferenceService"""

    def __init__(
            self,
            model_path: str,
            model_type: str = "ultralytics",
            conf: float = 0.25,
            iou: float = 0.7,
            slice_w: int = 640,
            slice_h: int = 640,
            overlap_w: float = 0.25,
            overlap_h: float = 0.25,
            standard_pred: bool = True,
    ) -> None:
        self._config = SahiConfig(
            model_path=Path(model_path),
            model_type=model_type,
            confidence=conf,
            iou=iou,
            slice_width=slice_w,
            slice_height=slice_h,
            overlap_width_ratio=overlap_w,
            overlap_height_ratio=overlap_h,
            perform_standard_pred=standard_pred,
        )
        self._service = SahiInferenceService(self._config)

    @property
    def name(self) -> str:
        return "SAHI 切片推理"

    def load_model(self) -> None:
        self._service.load_model()

    def infer(self, image_path: Path) -> list[dict]:
        return self._service.infer_image(image_path)

    @property
    def config_summary(self) -> str:
        c = self._config
        return (
            f"Conf={c.confidence} IoU={c.iou} "
            f"Slice={c.slice_width}×{c.slice_height} "
            f"Overlap={c.overlap_width_ratio}/{c.overlap_height_ratio}"
        )
