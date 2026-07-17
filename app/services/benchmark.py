"""推理基准对比服务"""
from __future__ import annotations

from pathlib import Path

from app.adapters.base import InferenceAdapter
from app.services.label_reader import IMAGE_EXTS, parse_yolo_label
from app.services.metrics import compute_metrics, compute_mAP
from app.utils.worker import Worker

class BenchmarkRunner(Worker):
    """对比多种推理方式的 Worker"""

    def __init__(self, adapters: list[InferenceAdapter], image_dir: Path, label_dir: Path, iou_threshold: float = 0.5) -> None:
        super().__init__()
        self.adapters = adapters
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.iou_threshold = iou_threshold

    def do_work(self) -> list[dict]:
        images = sorted(
            f for f in self.image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )

        pairs: list[tuple[Path, Path]] = []
        for img in images:
            label = self.label_dir / f"{img.stem}.txt"
            if label.exists():
                pairs.append((img, label))

        if not pairs:
            raise RuntimeError("没有找到配对成功的图片和标签")

        total = len(pairs)
        adapter_count = len(self.adapters)
        results: list[dict] = []

        import time
        from PySide6.QtGui import QPixmap

        for a_idx, adapter in enumerate(self.adapters):
            adapter.load_model()

            start = time.perf_counter()

            # 按类别汇总 TP / FP / FN（逐图计算）
            cls_tp: dict[int, int] = {}
            cls_fp: dict[int, int] = {}
            cls_fn: dict[int, int] = {}

            for i, (img_path, lbl_path) in enumerate(pairs):
                pix = QPixmap(str(img_path))
                img_w, img_h = pix.width(), pix.height()

                preds = adapter.infer(img_path)
                gts = parse_yolo_label(lbl_path, img_w, img_h)

                # 逐图算 metrics
                per_img = compute_metrics(preds, gts, self.iou_threshold)
                for cls_id, m in per_img.items():
                    cls_tp[cls_id] = cls_tp.get(cls_id, 0) + m["TP"]
                    cls_fp[cls_id] = cls_fp.get(cls_id, 0) + m["FP"]
                    cls_fn[cls_id] = cls_fn.get(cls_id, 0) + m["FN"]

                self.progress.emit(
                    int((a_idx * total + i + 1) / (adapter_count * total) * 100)
                )

            elapsed = time.perf_counter() - start

            # 汇总后算 P / R / F1
            all_cls = set(cls_tp.keys()) | set(cls_fp.keys()) | set(cls_fn.keys())
            per_class: dict[int, dict] = {}
            p_sum = 0.0
            r_sum = 0.0
            cls_count = 0
            for cls_id in all_cls:
                tp = cls_tp.get(cls_id, 0)
                fp = cls_fp.get(cls_id, 0)
                fn = cls_fn.get(cls_id, 0)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                per_class[cls_id] = {"P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn}
                p_sum += p
                r_sum += r
                cls_count += 1

            mAP = p_sum / cls_count if cls_count > 0 else 0.0

            results.append({
                "name": adapter.name,
                "summary": adapter.config_summary,
                "mAP": round(mAP, 4),
                "per_class": per_class,
                "time": round(elapsed, 1),
            })

        return results

