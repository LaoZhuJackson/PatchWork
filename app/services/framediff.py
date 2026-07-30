"""Frame Dynamics 数据集生成：当前帧 + 帧差三通道合成"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---- 文件名解析 ----

def parse_frame_number(filename: str) -> int:
    """从 video_name_000123.jpg 解析帧号"""
    match = re.search(r"_(\d+)$", Path(filename).stem)
    if match is None:
        raise ValueError(f"无法从文件名解析帧号: {filename}")
    return int(match.group(1))


def parse_video_id(filename: str) -> str:
    """从 video_name_000123.jpg 解析 video_name"""
    stem = Path(filename).stem
    match = re.match(r"^(.+?)_(\d+)$", stem)
    if match is None:
        raise ValueError(f"无法从文件名解析视频 ID: {filename}")
    return match.group(1)


# ---- 帧索引 ----

def build_full_frame_index(images_dir: Path) -> dict[str, dict[int, Path]]:
    """建立 {video_id: {frame_number: path}} 索引"""
    index: dict[str, dict[int, Path]] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        try:
            vid = parse_video_id(path.name)
            fn = parse_frame_number(path.name)
        except ValueError:
            continue
        index.setdefault(vid, {})[fn] = path
    return index


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """确保图像为单通道灰度图"""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        # fallback: 取第一个通道
        return img[:, :, 0]
    return img


# ---- 差分增强 ----

# 全局 CLAHE 实例（复用，避免每次创建）
_clahe = None


def _get_clahe(clip_limit: float = 2.0, tile_size: int = 8) -> cv2.CLAHE:
    """延迟创建 CLAHE 实例"""
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return _clahe


def _enhance_diff(diff: np.ndarray, mode: str, gain: float) -> np.ndarray:
    """增强差分通道"""
    if gain != 1.0:
        diff = np.clip(diff.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    if mode == "clahe":
        diff = _get_clahe().apply(diff)
    elif mode == "normalize":
        vmin, vmax = diff.min(), diff.max()
        if vmax > vmin:
            diff = ((diff.astype(np.float32) - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    return diff


# ---- 帧差合成 ----

def build_frame_dynamics(
        current: np.ndarray,
        previous_a: np.ndarray,
        previous_b: np.ndarray,
        registration: str = "none",
        blur_kernel: int = 0,
        diff_enhance: str = "clahe",
        diff_gain: float = 1.0,
) -> np.ndarray:
    """三通道合成：BGR = [diff_b, diff_a, current]

    Args:
        diff_enhance: 差分增强模式 — "none" | "clahe" | "normalize"
        diff_gain: 差分增益乘数 (0.5~5.0)
    """
    current = _to_grayscale(current)
    previous_a = _to_grayscale(previous_a)
    previous_b = _to_grayscale(previous_b)

    h, w = current.shape

    if previous_a.shape != (h, w):
        previous_a = cv2.resize(previous_a, (w, h), interpolation=cv2.INTER_LINEAR)
    if previous_b.shape != (h, w):
        previous_b = cv2.resize(previous_b, (w, h), interpolation=cv2.INTER_LINEAR)

    if registration == "phase":
        previous_a = _phase_correlation_register(previous_a, current)
        previous_b = _phase_correlation_register(previous_b, current)

    diff_a = cv2.absdiff(current, previous_a)
    diff_b = cv2.absdiff(current, previous_b)

    if blur_kernel > 0:
        diff_a = cv2.GaussianBlur(diff_a, (blur_kernel, blur_kernel), 0)
        diff_b = cv2.GaussianBlur(diff_b, (blur_kernel, blur_kernel), 0)

    diff_a = _enhance_diff(diff_a, mode=diff_enhance, gain=diff_gain)
    diff_b = _enhance_diff(diff_b, mode=diff_enhance, gain=diff_gain)

    return cv2.merge([diff_b, diff_a, current])


def _phase_correlation_register(
        source: np.ndarray,
        target: np.ndarray,
        min_response: float = 0.05,
        max_shift_ratio: float = 0.20,
) -> np.ndarray:
    """相位相关全局平移补偿"""
    if source.shape != target.shape:
        source = cv2.resize(
            source, (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    h, w = target.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    try:
        (dx, dy), response = cv2.phaseCorrelate(
            source.astype(np.float32), target.astype(np.float32), window,
        )
    except cv2.error:
        return source

    if not (np.isfinite(dx) and np.isfinite(dy) and np.isfinite(response)):
        return source
    if response < min_response:
        return source
    if abs(dx) > w * max_shift_ratio or abs(dy) > h * max_shift_ratio:
        return source

    transform = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(
        source, transform, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ---- 图片写入（绕过 cv2.imwrite 的中文路径问题） ----

def _imwrite(
    path: str,
    img: np.ndarray,
    image_format: str = "jpg",
    jpg_quality: int = 95,
) -> bool:
    """用 cv2.imencode + Python open() 写图片，兼容 Windows 中文路径"""
    if image_format == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
        ext = ".jpg"
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ext = ".png"
    success, buf = cv2.imencode(ext, img, params)
    if not success:
        return False
    try:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except OSError:
        return False


# ---- 数据集生成 ----

def generate_framediff_dataset(
        image_dir: Path,
        output_dir: Path,
        label_dir: Path | None = None,
        gaps: tuple[int, int] = (1, 2),
        registration: str = "none",
        blur_kernel: int = 0,
        diff_enhance: str = "clahe",
        diff_gain: float = 1.0,
        image_format: str = "jpg",
        jpg_quality: int = 95,
        overwrite: bool = False,
        progress_callback: Callable[[int], None] | None = None,
) -> dict:
    """生成 Frame Dynamics 数据集

    Args:
        image_dir: 图片目录（连续帧）
        output_dir: 输出根目录
        label_dir: 标签目录（可选，None 则对全部图片生成不复制标签）
    Returns:
        {total, success, missing_current, missing_history, unreadable, write_error, other_error}
    """
    gap_a, gap_b = gaps
    output_dir.mkdir(parents=True, exist_ok=True)

    out_img_dir = output_dir / "images"
    out_lbl_dir = output_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    has_labels = label_dir is not None and label_dir.is_dir()
    if has_labels:
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        label_paths = {p.stem: p for p in label_dir.glob("*.txt")}
        stems = sorted(label_paths.keys())
    else:
        label_paths = {}
        stems = sorted(
            p.stem for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )

    total = len(stems)
    frame_index = build_full_frame_index(image_dir)

    stats = {
        "total": total, "success": 0, "missing_current": 0,
        "missing_history": 0, "unreadable": 0, "write_error": 0, "other_error": 0,
    }

    for idx, stem in enumerate(stems):
        try:
            video_id = parse_video_id(stem)
            frame_number = parse_frame_number(stem)
        except ValueError:
            stats["other_error"] += 1
            continue

        video_frames = frame_index.get(video_id)
        if video_frames is None:
            stats["missing_current"] += 1
            continue

        current_path = video_frames.get(frame_number)
        history_a_path = video_frames.get(frame_number - gap_a)
        history_b_path = video_frames.get(frame_number - gap_b)

        if current_path is None:
            stats["missing_current"] += 1
            continue
        if history_a_path is None or history_b_path is None:
            stats["missing_history"] += 1
            continue

        suffix = ".jpg" if image_format == "jpg" else ".png"
        dest_img = out_img_dir / f"{stem}{suffix}"

        if dest_img.exists() and not overwrite:
            if not has_labels or (out_lbl_dir / f"{stem}.txt").exists():
                stats["success"] += 1
                if progress_callback and total > 0:
                    progress_callback(int((idx + 1) / total * 100))
                continue

        cur = cv2.imread(str(current_path), cv2.IMREAD_GRAYSCALE)
        pa = cv2.imread(str(history_a_path), cv2.IMREAD_GRAYSCALE)
        pb = cv2.imread(str(history_b_path), cv2.IMREAD_GRAYSCALE)

        if cur is None or pa is None or pb is None:
            stats["unreadable"] += 1
            if progress_callback and total > 0:
                progress_callback(int((idx + 1) / total * 100))
            continue

        try:
            fd = build_frame_dynamics(
                cur, pa, pb, registration=registration, blur_kernel=blur_kernel,
                diff_enhance=diff_enhance, diff_gain=diff_gain,
            )
            if not _imwrite(str(dest_img), fd, image_format, jpg_quality):
                stats["write_error"] += 1
                if progress_callback and total > 0:
                    progress_callback(int((idx + 1) / total * 100))
                continue

            if has_labels and stem in label_paths:
                shutil.copy2(label_paths[stem], out_lbl_dir / f"{stem}.txt")

            stats["success"] += 1

        except Exception:
            stats["other_error"] += 1
            if stats["other_error"] <= 3:
                logger.exception("生成帧差失败 [%s]: %s", stem, dest_img)

        if progress_callback and total > 0:
            progress_callback(int((idx + 1) / total * 100))

    return stats


