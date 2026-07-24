"""YOLOE 开放词汇推理引擎 —— 输入任意类别名，零样本检测"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PySide6.QtCore import QRectF, QPointF
from ultralytics import YOLO

from app.services.label_reader import get_color

class OpenVocabEngine:
    """YOLOE 开放词汇推理引擎

    与 InferenceEngine 的核心区别：
    - 不依赖训练时固定的 class_names，推理时动态输入类别名
    - 通过 model.get_text_pe() 把文字编码为嵌入 → 注入检测头
    - 一次加载模型，可随时切换提示词

    用法:
      engine = OpenVocabEngine()
      engine.load_model("yoloe-26n-seg.pt")
      engine.set_prompts(["person", "excavator"])
      results = engine.infer("thermal.jpg", conf=0.25, iou=0.45)
    """

    def __init__(self) -> None:
        self._model: YOLO | None = None
        self._model_path: str = ""
        self._class_names: list[str] = []
        self._prompt_set: bool = False

    # 属性
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def prompt_set(self) -> bool:
        """是否已设置文本提示并注入了类别嵌入"""
        return self._prompt_set

    # 公共方法
    def load_model(self, model_path: str | Path) -> None:
        """加载 YOLOE .pt 模型文件。

        注意：必须是支持文本提示的 YOLOE 模型（非 prompt-free 变体）,
        例如 yoloe-26n-seg.pt 而不是 yoloe-26n-seg-pf.pt。
        """
        try:
            self._model = YOLO(str(model_path))
        except AttributeError as e:
            msg = str(e)
            if "qkv" in msg or "qk" in msg or "AAttn" in msg:
                raise RuntimeError(
                    "模型权重与当前 ultralytics 版本不兼容。\n"
                    "请升级 ultralytics 或使用匹配的权重文件。"
                ) from e
            raise
        self._model_path = str(model_path)
        self._prompt_set = False
        self._class_names = []

        # 检查是否支持文本提示
        if not hasattr(self._model.model, "get_text_pe"):
            raise RuntimeError(
                f"该模型不支持文本提示（{Path(model_path).name} 可能不是 YOLOE 模型）。\n"
                "请使用 yoloe-*-seg.pt 而非 -pf 变体。"
            )

    def set_prompts(self, class_names: list[str]) -> None:
        """设置文本提示：输入类别名称列表。

        每次调用会重新编码文本嵌入并注入模型检测头，
        之后 infer() 将只检测这些类别的目标。

        例:
          engine.set_prompts(["person", "excavator", "helmet"])
          engine.set_prompts(["car", "bus", "motorcycle"])
        """
        if self._model is None:
            raise RuntimeError("模型未加载")
        if not class_names:
            raise ValueError("至少需要一个类别名称")
        # 核心：YOLOE 的文本编码 → 嵌入向量
        tpe = self._model.model.get_text_pe(class_names)
        # 注入检测头的分类器权重
        self._model.model.set_classes(class_names, tpe)

        self._class_names = class_names
        self._prompt_set = True

    def infer(self, image_path:str | Path, conf: float = 0.25, iou: float = 0.45) -> list[dict]:
        """推理单张图片，返回标注列表。

        返回格式与 InferenceEngine.infer() 完全一致，
        可直接用于 ImageViewer.add_bbox() / add_polygon()。
        """
        if self._model is None:
            raise RuntimeError("模型未加载")
        if not self._prompt_set:
            raise RuntimeError("请先调用 set_prompts() 设置文本提示")

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

            # class_id 是 set_prompts 时类别列表的索引
            class_name = (
                self._class_names[class_id]
                if 0 <= class_id < len(self._class_names)
                else f"class_{class_id}"
            )
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

            # 分割多边形（仅 seg 模型有 mask）
            if has_mask:
                mask = detections.mask[i].astype(np.uint8)
            confidence = (
                float(detections.confidence[i])
                if detections.confidence is not None
                else 0.0
            )

            # class_id 是 set_prompts 时类别列表的索引
            class_name = (
                self._class_names[class_id]
                if 0 <= class_id < len(self._class_names)
                else f"class_{class_id}"
            )
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

            # 分割多边形（仅 seg 模型有 mask）
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
