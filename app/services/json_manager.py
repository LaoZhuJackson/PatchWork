"""JSON 标注文件管理：按帧间隔删除 / 生成空 X-AnyLabeling JSON"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_filename(filename: str) -> tuple[str, int, str] | None:
    """解析 {prefix}_{frame:06d}.{ext} → (prefix, frame_num, ext)"""
    match = re.match(r"(.+)_(\d{6})\.(\w+)$", filename)
    if not match:
        return None
    return match.group(1), int(match.group(2)), "." + match.group(3)


def scan_directory(target_dir: Path) -> list[dict]:
    """扫描目录，返回每个前缀的统计，按前缀名排序

    Returns:
      [{prefix, image_count, json_count, total_frames, min_frame, max_frame}, ...]
    """
    prefixes: dict[str, dict] = {}

    for path in sorted(target_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        prefix, frame_num, ext = parsed
        info = prefixes.setdefault(prefix, {
            "prefix": prefix,
            "image_count": 0,
            "json_count": 0,
            "total_frames": 0,
            "min_frame": frame_num,
            "max_frame": frame_num,
        })
        info["total_frames"] += 1
        info["min_frame"] = min(info["min_frame"], frame_num)
        info["max_frame"] = max(info["max_frame"], frame_num)
        if ext.lower() in SUPPORTED_IMAGE_SUFFIXES:
            info["image_count"] += 1
        elif ext.lower() == ".json":
            info["json_count"] += 1

    return sorted(prefixes.values(), key=lambda x: x["prefix"])


def delete_by_interval(target_dir: Path, prefix_intervals: dict[str, int], apply: bool = False, ) -> dict:
    """按帧间隔删除多余 JSON。

    Args:
      target_dir: 目标目录
      prefix_intervals: {前缀: 间隔帧数}，帧号不为间隔倍数的 JSON 将被删除
      apply: False=干运行，True=实际删除

    Returns:
      {prefix: {total: int, kept: int, deleted: int, files: [str, ...]}}
    """
    result: dict[str, dict] = {}
    for path in sorted(target_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        prefix, frame_num, ext = parsed

        if ext.lower() != ".json":
            continue

        interval = prefix_intervals.get(prefix)
        if interval is None:
            continue  # 不在配置中的前缀不处理

        info = result.setdefault(prefix, {
            "total": 0, "kept": 0, "deleted": 0, "files": [],
        })
        info["total"] += 1

        keep = frame_num > 0 and frame_num % interval == 0
        if keep:
            info["kept"] += 1
        else:
            info["deleted"] += 1
            info["files"].append(path.name)
            if apply:
                try:
                    path.unlink()
                except OSError:
                    pass

    return result


def generate_empty_json(
        target_dir: Path,
        prefix_intervals: dict[str, int],
        version: str = "4.0.0-beta.13",
) -> dict:
    """为缺少 JSON 的图片生成空 X-AnyLabeling JSON。

    仅对帧间隔命中的图片（帧号 % interval == 0）补 JSON。
    已有同名 JSON 的跳过。

    Returns:
        {prefix: {total_images: int, created: int, skipped: int}}
    """
    result: dict[str, dict] = {}

    for path in sorted(target_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        prefix, frame_num, ext = parsed

        if ext.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue

        interval = prefix_intervals.get(prefix)
        if interval is None or frame_num <= 0 or frame_num % interval != 0:
            continue

        info = result.setdefault(prefix, {
            "total_images": 0, "created": 0, "skipped": 0,
        })
        info["total_images"] += 1

        json_path = target_dir / f"{Path(path.stem).stem}.json"

        # 重新确认：stem 相同
        json_path = target_dir / f"{path.stem}.json"
        if json_path.exists():
            info["skipped"] += 1
            continue

        # 读取图片尺寸
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            info["skipped"] += 1
            continue
        h, w = img.shape[:2]

        empty = {
            "version": version,
            "flags": {},
            "shapes": [],
            "imagePath": path.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
            "description": "",
        }
        json_path.write_text(
            json.dumps(empty, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        info["created"] += 1

    return result
