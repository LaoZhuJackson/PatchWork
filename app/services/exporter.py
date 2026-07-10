"""ONNX 导出服务：封装 ultralytics model.export"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from ultralytics import YOLO


class ONNXExporter:
    """YOLO → ONNX 导出器"""

    def __init__(self) -> None:
        self._model: YOLO | None = None

    def export(self, model_path: str | Path, output_dir: str | Path | None = None, imgsz: int | None = None,
               simplify: bool = True, dynamic: bool = True,
               progress_callback: Callable[[int], None] | None = None) -> Path:
        """导出 ONNX 模型（在后台线程中调用）"""
        self._model = YOLO(str(model_path))

        if imgsz is None:
            overrides = self._model.overrides
            imgsz = overrides.get("imgsz", 640)
            if isinstance(imgsz, (list, tuple)):
                imgsz = imgsz[0]
        # 如果指定了输出目录，先 cd 进去（ultralytics export 输出到 CWD）
        cwd = os.getcwd()
        try:
            if output_dir:
                os.chdir(str(output_dir))

            if progress_callback:
                progress_callback(10)
            export_path = self._model.export(
                format="onnx",
                imgsz = imgsz,
                simplify = simplify,
                dynamic = dynamic,
                verbose = False
            )

            if progress_callback:
                progress_callback(90)

            # export() 返回导出文件的路径（可能是 str 或 Path）
            result = Path(export_path) if isinstance(export_path, str) else export_path
            if not result.is_absolute():
                result = Path.cwd() / result
            if progress_callback:
                progress_callback(100)

            return result
        finally:
            os.chdir(cwd)


