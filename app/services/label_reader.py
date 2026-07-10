"""YOLO 标签读取与坐标反归一化"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QLineF, QPointF, QRectF
from PySide6.QtGui import QColor

from app.utils.logger import get_logger

logger = get_logger(__name__)

# 20 种颜色循环使用，区分不同类别
CLASS_COLORS = [
  QColor("#FF6B6B"), QColor("#4ECDC4"), QColor("#45B7D1"),
  QColor("#96CEB4"), QColor("#FFEAA7"), QColor("#DDA0DD"),
  QColor("#98D8C8"), QColor("#F7DC6F"), QColor("#BB8FCE"),
  QColor("#85C1E9"), QColor("#F8C471"), QColor("#82E0AA"),
  QColor("#F1948A"), QColor("#AED6F1"), QColor("#D7BDE2"),
  QColor("#A3E4D7"), QColor("#FAD7A0"), QColor("#ABEBC6"),
  QColor("#C39BD3"), QColor("#7FB3D8"),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def get_color(class_id: int) -> QColor:
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]

def parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """解析 YOLO 标签文件，返回检测框/多边形列表（像素坐标）。

    每项 dict 结构:
      type: "bbox" | "polygon"
      rect: QRectF          # 检测框 (bbox 类型)
      points: list[QPointF]  # 顶点 (polygon 类型)
      class_id: int
      color: QColor
      label: str
    """
    results: list[dict] = []

    if not label_path.exists():
        logger.warning("标签路径不存在")
        return results
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        color = get_color(class_id)

        if len(coords) == 4:
            # 检测框：cx cy w h(归一化)
            cx, cy, w, h = coords
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            results.append({
                "type": "bbox",
                "rect": QRectF(x1, y1, x2 - x1, y2 - y1),
                "class_id": class_id,
                "color": color,
                "label": f"class_{class_id}",
            })
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            # 分割多边形: x1 y1 x2 y2 ... (归一化)
            points = []
            for i in range(0, len(coords), 2):
                px = coords[i] * img_w
                py = coords[i + 1] * img_h
                points.append(QPointF(px, py))
            results.append({
                "type": "polygon",
                "points": points,
                "class_id": class_id,
                "color": color,
                "label": f"class_{class_id}",
            })
    return results
