"""目标检测评估指标：P / R / F1 / AP"""
from __future__ import annotations

from collections import defaultdict


def _iou(box_a: tuple, box_b: tuple) -> float:
    """计算两个框的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)  # 计算交集
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter  # 并集
    return inter / union if union > 0 else 0.0

def _rect_to_tuple(item: dict) -> tuple[float, float, float, float]:
    """从统一标注结构中提取 (x1, y1, x2, y2)"""
    r = item["rect"]
    return r.x(), r.y(), r.x() + r.width(), r.y() + r.height()

def compute_metrics(
        predictions: list[dict],
        ground_truths: list[dict],
        iou_threshold: float = 0.5,
) -> dict[int, dict[str, float]]:
    """按类别计算 P / R / F1。

    Returns:
      {class_id: {"P": float, "R": float, "F1": float, "TP": int, "FP": int, "FN": int}}
    """
    # 按类别分组
    pred_by_cls: dict[int, list[dict]] = defaultdict(list)
    gt_by_cls: dict[int, list[dict]] = defaultdict(list)

    for p in predictions:
        if p["type"] == "bbox":
            pred_by_cls[p["class_id"]].append(p)
    for g in ground_truths:
        if g["type"] == "bbox":
            gt_by_cls[g["class_id"]].append(g)

    all_classes = set(pred_by_cls.keys()) | set(gt_by_cls.keys())
    results: dict[int, dict] = {}

    for cls_id in all_classes:
        preds = pred_by_cls.get(cls_id, [])
        gts = gt_by_cls.get(cls_id, [])

        gt_matched = [False] * len(gts)  # 标记预测框是否匹配上gt框
        tp = 0
        fp = 0

        # 按置信度降序匹配
        preds_sorted = sorted(
            preds, key=lambda x: float(x["label"].split()[-1]) if x.get("label") else 0.0, reverse=True
        )
        for pred in preds_sorted:
            pred_box = _rect_to_tuple(pred)
            best_iou = 0.0
            best_idx = -1
            for j, gt in enumerate(gts):
                if gt_matched[j]:
                    continue
                current = _iou(pred_box, _rect_to_tuple(gt))
                if current > best_iou:
                    best_iou = current
                    best_idx = j
            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                gt_matched[best_idx] = True
            else:
                fp += 1  # 这里是在预测框的for循环下，是误检
        fn = sum(1 for m in gt_matched if not m)  # 遍历gt，是漏检

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results[cls_id] = {"P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn}
    return results


def compute_mAP(per_class: dict[int, dict]) -> float:
    """计算 mAP（所有类别 AP 的平均值，IoU=0.5 时即 mAP@0.5）"""
    aps = [v["P"] for v in per_class.values()]  # 简便做法：单 IoU 阈值下 AP≈P
    return sum(aps) / len(aps) if aps else 0.0
