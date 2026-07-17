"""推理基准对比服务"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.adapters.base import InferenceAdapter
from app.services.label_reader import IMAGE_EXTS, parse_yolo_label
from app.services.metrics import evaluate
from app.utils.logger import get_logger
from app.utils.worker import Worker


logger = get_logger(__name__)

class BenchmarkRunner(Worker):
    """对比多种推理方式的 Worker"""

    def __init__(
      self,
      adapters: list[InferenceAdapter],
      image_dir: Path,
      label_dir: Path,
      iou_threshold: float = 0.5,
    ) -> None:
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

        for a_idx, adapter in enumerate(self.adapters):
            adapter.load_model()
            start = time.perf_counter()

            # 收集该 adapter 对所有图片的预测和标签
            all_preds: list[dict] = []
            all_gts: list[dict] = []

            for i, (img_path, lbl_path) in enumerate(pairs):
              from PySide6.QtGui import QPixmap
              pix = QPixmap(str(img_path))
              img_w, img_h = pix.width(), pix.height()

              preds = adapter.infer(img_path)
              gts = parse_yolo_label(lbl_path, img_w, img_h)

              all_preds.extend(preds)
              all_gts.extend(gts)

              self.progress.emit(
                  int((a_idx * total + i + 1) / (adapter_count * total) * 100)
              )

            elapsed = time.perf_counter() - start

            # 一次性评估
            per_class = evaluate(all_preds, all_gts)

            p_mean = np.mean([v["P"] for v in per_class.values()]) if per_class else 0.0
            r_mean = np.mean([v["R"] for v in per_class.values()]) if per_class else 0.0
            f1_mean = np.mean([v["F1"] for v in per_class.values()]) if per_class else 0.0
            ap_mean = np.mean([v["AP50"] for v in per_class.values()]) if per_class else 0.0

            results.append({
              "name": adapter.name,
              "summary": adapter.config_summary,
              "mAP": round(ap_mean, 4),
              "per_class": per_class,
              "time": round(elapsed, 1),
            })

        logger.info(results)
        return results