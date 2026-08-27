"""AI 目标合成：标注导出（YOLO label / X-AnyLabeling JSON）"""
from __future__ import annotations

import json
from pathlib import Path

# 默认 X-AnyLabeling 版本号（仅写入 JSON version 字段，格式固定为 4 角点）
DEFAULT_XAL_VERSION = "4.0.0-beta.13"


def _iter_records(records):
    """records: list of
    {
      "stem": "frame_000000",           # 文件名（无扩展）
      "result_name": "frame_000000.png",  # 导出目录 images/ 下的结果图文件名
      "width": int, "height": int,
      "boxes": [{"cls_id": int, "cls_name": str, "xyxy": (x1,y1,x2,y2)}, ...]
    }
    """
    return list(records)


def export_yolo(records, out_dir) -> dict:
    """写 YOLO 格式：labels/*.txt（归一化 cx cy w h）+ data.yaml

    结果图已由面板保存到 out_dir/images/，这里只补标签和类别名表。
    """
    out = Path(out_dir)
    lbl = out / "labels"
    img_dir = out / "images"
    lbl.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # 类别名表（按 cls_id 排序）
    cls_map: dict[int, str] = {}
    for rec in _iter_records(records):
        for b in rec["boxes"]:
            cls_map[int(b["cls_id"])] = b.get("cls_name") or f"class_{b['cls_id']}"

    written = 0
    for rec in _iter_records(records):
        W, H = rec["width"], rec["height"]
        lines = []
        for b in rec["boxes"]:
            x1, y1, x2, y2 = b["xyxy"]
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            lines.append(f"{int(b['cls_id'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl / f"{rec['stem']}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    names = "\n".join(f"  {k}: {v}" for k, v in sorted(cls_map.items()))
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images\nval: images\n\nnames:\n{names}\n",
        encoding="utf-8")

    return {"labels": written, "classes": len(cls_map)}


def export_xanylabeling(records, out_dir, version: str = DEFAULT_XAL_VERSION) -> dict:
    """写 X-AnyLabeling JSON（与结果图同目录），严格按 4 角点格式：
    shape.points = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]（顺时针）
    """
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for rec in _iter_records(records):
        shapes = []
        for b in rec["boxes"]:
            x1, y1, x2, y2 = b["xyxy"]
            shapes.append({
                "label": b.get("cls_name") or f"class_{b['cls_id']}",
                "score": None,
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            })
        data = {
            "version": version,
            "flags": {},
            "checked": False,
            "shapes": shapes,
            "imagePath": rec.get("result_name") or f"{rec['stem']}.png",
            "imageData": None,
            "imageHeight": rec["height"],
            "imageWidth": rec["width"],
        }
        (img_dir / f"{rec['stem']}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1

    return {"json": written}
