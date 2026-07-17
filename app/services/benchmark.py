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

    def do_work(self) ->list[dict]:
        """返回 [{"name": str, "summary": str, "mAP": float, "per_class": dict, "time": float}, ...]"""
        # 收集图片和标签
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

            # 收集所有预测和标签
            all_preds: dict[int, list[dict]] = {}
            all_gts: dict[int, list[dict]] = {}

            # 取第一张图的尺寸作为参考
            sample_pix = QPixmap(str(pairs[0][0]))
            ref_w, ref_h = sample_pix.width(), sample_pix.height()

            for i, (img_path, lbl_path) in enumerate(pairs):
                pix = QPixmap(str(img_path))
                img_w, img_h = pix.width(), pix.height()

                preds = adapter.infer(img_path)
                gts = parse_yolo_label(lbl_path, img_w, img_h)

                for p in preds:
                    if p["type"] == "bbox":
                        cls = p["class_id"]
                        all_preds.setdefault(cls, []).append(p)
                for g in gts:
                    if g["type"] == "bbox":
                        cls = g["class_id"]
                        all_gts.setdefault(cls, []).append(g)

                self.progress.emit(
                    int((a_idx * total + i + 1) / (adapter_count * total) * 100)
                )
            elapsed = time.perf_counter() - start

            # 全量拼接后算一次
            per_class_combined: dict[int, dict] = {}
            all_preds_list = [p for v in all_preds.values() for p in v]
            all_gts_list = [g for v in all_gts.values() for g in v]
            per_class_combined = compute_metrics(all_preds_list, all_gts_list, self.iou_threshold)
            mAP = compute_mAP(per_class_combined)

            results.append({
                "name": adapter.name,
                "summary": adapter.config_summary,
                "mAP": round(mAP, 4),
                "per_class": per_class_combined,
                "time": round(elapsed, 1),
            })

