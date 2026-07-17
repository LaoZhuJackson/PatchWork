"""目标检测评估：基于 ultralytics 内置指标"""
from __future__ import annotations

import numpy as np
from ultralytics.utils.metrics import ap_per_class


def evaluate(
        predictions: list[dict],
        ground_truths: list[dict],
        class_names: dict[int, str] | None = None,
) -> dict[int, dict]:
    """按类别计算 P / R / AP@0.5 / AP@0.5:0.95。

    预测框和标签框都是统一格式 [{type, rect, class_id, label, ...}]。
    """
    if not predictions:
        return {}

    # 转为 numpy 矩阵: [N, 6] → [x1, y1, x2, y2, confidence, class_id]
    preds = []
    for p in predictions:
        if p["type"] != "bbox":
            continue
        r = p["rect"]
        conf = _extract_conf(p.get("label", ""))
        preds.append([r.x(), r.y(), r.x() + r.width(), r.y() + r.height(), conf, p["class_id"]])

    gts = []
    for g in ground_truths:
        if g["type"] != "bbox":
            continue
        r = g["rect"]
        gts.append([g["class_id"], r.x(), r.y(), r.x() + r.width(), r.y() + r.height()])

    if not preds:
        return {}

    preds_np = np.array(preds, dtype=np.float32)
    gts_np = np.array(gts, dtype=np.float32)

    # 调 ultralytics 核心函数
    # 返回: tp, fp, p, r, f1, ap, ap_class, precision_per_class, recall_per_class, ..
    tp, fp, p_arr, r_arr, f1_arr, ap_arr, ap_cls, *_ = ap_per_class(
        preds_np[:, :4],  # (N, 4) boxes
        preds_np[:, 4],  # (N,)  conf
        preds_np[:, 5],  # (N,)  class_id
        gts_np[:, 0:5],  # (M, 5) [class_id, x1, y1, x2, y2]
        names=class_names or {},
    )

    results: dict[int, dict] = {}
    for i, cls_id in enumerate(ap_cls):
        cls_id = int(cls_id)
        # ap_arr[i] 可能是数组（多 IoU 阈值），取第一个 = AP@0.5
        ap_val = ap_arr[i]
        if hasattr(ap_val, '__len__'):
            ap_val = float(ap_val[0]) if len(ap_val) > 0 else 0.0
        else:
            ap_val = float(ap_val)

        results[cls_id] = {
            "P": round(float(p_arr[i]) if i < len(p_arr) else 0, 4),
            "R": round(float(r_arr[i]) if i < len(r_arr) else 0, 4),
            "F1": round(float(f1_arr[i]) if i < len(f1_arr) else 0, 4),
            "AP50": round(ap_val, 4),
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