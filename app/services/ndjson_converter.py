"""NDJSON → YOLO 格式转换服务"""
from __future__ import annotations

import asyncio
from pathlib import Path

from ultralytics.data.converter import convert_ndjson_to_yolo


def convert(ndjson_path: str | Path, output_dir: str | Path) -> Path:
    """将 NDJSON 数据集转换为 YOLO 格式。

    Args:
      ndjson_path: NDJSON 文件路径
      output_dir:  输出目录

    Returns:
      生成的 data.yaml 文件路径
    """
    return asyncio.run(
        convert_ndjson_to_yolo(
            ndjson_path=Path(ndjson_path),
            output_path=Path(output_dir),
        )
    )


