"""确认是否配对并划分数据集"""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Callable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_pairs(image_dir: Path, label_dir: Path) -> tuple[list[tuple[Path, Path]], list[Path], list[Path]]:
    """扫描目录，返回 (配对成功列表, 缺label的图片, 缺图片的label)"""
    images = {f.stem: f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS}
    labels = {f.stem: f for f in label_dir.iterdir() if f.suffix.lower() == ".txt"}

    pairs: list[tuple[Path, Path]] = []
    missing_label: list[Path] = []
    missing_image: list[Path] = []

    for stem, img in images.items():
        if stem in labels:
            pairs.append((img, labels[stem]))
        else:
            missing_label.append(img)
    for stem, lbl in labels.items():
        if stem not in images:
            missing_image.append(lbl)

    return pairs, missing_label, missing_image


def split_dataset(
        image_dir: Path,
        label_dir: Path,
        output_dir: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        mode: str = "copy",
        progress_callback: Callable[[int], None] | None = None
) -> dict:
    """按比例划分数据集，返回各集合的文件数量
    先校验配对完整性——存在任何缺失则拒绝划分并返回缺失信息
    """
    pairs, missing_label, missing_image = find_pairs(image_dir, label_dir)
    if missing_label or missing_image:
        return {
            "ok": False,
            "missing_label": missing_label,
            "missing_image": missing_image,
        }
    if not pairs:
        raise ValueError(f"在 {image_dir} 中没有找到任何配对成功的图片和标签")

    # 洗牌+按比例切分数据集
    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    n_train = max(1, round(total * train_ratio))
    n_val = max(1, round(total * val_ratio))
    n_test = total - n_train - n_val

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train: n_train + n_val],
        "test": pairs[n_train+n_val:],
    }

    # 复制到输出目录
    processed = 0
    for split_name, split_pairs in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img, lbl in split_pairs:
            if mode == "move":
                shutil.move(str(img), img_out / img.name)
                shutil.move(str(lbl), lbl_out / lbl.name)
            else:
                shutil.copy2(img, img_out / img.name)
                shutil.copy2(lbl, lbl_out / lbl.name)
            processed += 1
            if progress_callback:
                progress_callback(int(processed / total * 100))

    return {
        "ok": True,
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }
