"""目标检测评估：手写 IoU 匹配，透明可调试"""
from __future__ import annotations

from collections import defaultdict


def _iou(box_a, box_b) -> float:
    """两个 [x1, y1, x2, y2] 的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
) -> dict[int, dict]:
    """逐类 IoU 匹配，返回 P/R/F1/AP50。

    每个预测框格式: {type: "bbox", rect: QRectF, class_id: int, label: str}
    每个标签框格式: 同上
    """
    preds = []
    for p in predictions:
        if p["type"] != "bbox":
            continue
        r = p["rect"]
        preds.append({
            "box": (r.x(), r.y(), r.x() + r.width(), r.y() + r.height()),
            "class_id": p["class_id"],
            "conf": _extract_conf(p.get("label", "")),
        })

    gts = []
    for g in ground_truths:
        if g["type"] != "bbox":
            continue
        r = g["rect"]
        gts.append({
            "box": (r.x(), r.y(), r.x() + r.width(), r.y() + r.height()),
            "class_id": g["class_id"],
        })

    preds_by_cls: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        preds_by_cls[p["class_id"]].append(p)

    gts_by_cls: dict[int, list[dict]] = defaultdict(list)
    for g in gts:
        gts_by_cls[g["class_id"]].append(g)

    all_classes = set(preds_by_cls.keys()) | set(gts_by_cls.keys())
    results: dict[int, dict] = {}

    for cls_id in all_classes:
        cls_preds = sorted(
            preds_by_cls.get(cls_id, []), key=lambda x: x["conf"], reverse=True
        )
        cls_gts = gts_by_cls.get(cls_id, [])

        if not cls_gts:
            results[cls_id] = {"P": 0.0, "R": 0.0, "F1": 0.0, "AP50": 0.0}
            continue
        if not cls_preds:
            results[cls_id] = {"P": 0.0, "R": 0.0, "F1": 0.0, "AP50": 0.0}
            continue

        gt_matched = [False] * len(cls_gts)
        tp = 0
        fp = 0

        for pred in cls_preds:
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(cls_gts):
                if gt_matched[j]:
                    continue
                current = _iou(pred["box"], gt["box"])
                if current > best_iou:
                    best_iou = current
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0:
                tp += 1
                gt_matched[best_j] = True
            else:
                fp += 1

        fn = sum(1 for m in gt_matched if not m)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results[cls_id] = {
            "P": round(p, 4),
            "R": round(r, 4),
            "F1": round(f1, 4),
            "AP50": round(p, 4),
        }

    return results


def _extract_conf(label: str) -> float:
    """从 'class_name 0.87' 提取置信度"""
    parts = label.split()
    for p in reversed(parts):
        try:
            return float(p)
        except ValueError:
            continue
    return 0.0
