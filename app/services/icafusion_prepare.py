"""ICAFusion 数据准备：IR warp 对齐 + VIS 裁剪到重叠区

输入 IR / VIS 两个目录（文件名为 PREFIX_T_000000.jpg ↔ PREFIX_V_000000.jpg），
用固定映射矩阵 H（VIS→IR）把 IR warp 到 VIS 坐标系，再裁剪 VIS 到 IR 有效重叠区，
统一 resize 到目标尺寸（默认 1280×1024），输出：
    <output>/
        visible/            VIS 裁剪图
        ir/                 IR warp 对齐图
        crop_params.json    每帧裁剪参数（可用于标签重映射）

H 矩阵来源优先级：npz_path 指定的 XoFTR NPZ > 传入的 H 参数 > 内置固定 DEFAULT_H。
DEFAULT_H 为同型号镜头固定安装下的中值映射（DJI Matrice 4TD，白天 0803 的 29 个
XoFTR NPZ 中值），跨视频共用。
"""
from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 固定映射矩阵 H（VIS → IR 坐标）：DJI Matrice 4TD 固定安装，白天 0803 的 29 个 NPZ 中值
DEFAULT_H = np.array(
    [
        [7.51254694e-01, 1.21547467e-03, -8.05672055e02],
        [1.08882761e-03, 7.55416214e-01, -3.14674810e02],
        [1.27764981e-06, 6.16127711e-06, 1.00000000e00],
    ]
)


# ---- 读写（兼容 Windows 中文路径） ----

def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imread 在 Windows 中文路径下失败，改用 imdecode"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray, jpg_quality: int = 95) -> None:
    """cv2.imwrite 中文路径兼容写法"""
    ext = Path(path).suffix.lower()
    if ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
    success, buf = cv2.imencode(ext, img, params)
    if not success:
        raise IOError(f"编码图像失败: {path}")
    buf.tofile(path)


def load_h_from_npz(npz_path: str) -> np.ndarray:
    """从 XoFTR 结果 NPZ 读取 H 矩阵（含 'H' 键，VIS→IR）"""
    with open(npz_path, "rb") as f:
        data = np.load(io.BytesIO(f.read()), allow_pickle=True)
    return np.asarray(data["H"], dtype=np.float64)


# ---- 几何 ----

def find_ir_bbox(ir_warped_gray: np.ndarray, margin_px: int = 5) -> tuple[int, int, int, int]:
    """在 warp 后的图像上找 IR 有效内容包围盒 (xmin, ymin, xmax, ymax)"""
    rows = np.any(ir_warped_gray > 5, axis=1)
    cols = np.any(ir_warped_gray > 5, axis=0)
    if not rows.any() or not cols.any():
        h, w = ir_warped_gray.shape
        return 0, 0, w, h
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    xmin = max(0, xmin - margin_px)
    xmax = min(ir_warped_gray.shape[1], xmax + margin_px)
    ymin = max(0, ymin - margin_px)
    ymax = min(ir_warped_gray.shape[0], ymax + margin_px)
    return int(xmin), int(ymin), int(xmax), int(ymax)


def discover_pairs(ir_dir: Path, vis_dir: Path) -> list[tuple[str, str, str]]:
    """自动发现 IR-VIS 帧对：PREFIX_T_000000 ↔ PREFIX_V_000000

    返回 [(frame_key, vis_path, ir_path)]，frame_key = "PREFIX_000000"
    """
    ir_map = {
        p.stem: p
        for p in Path(ir_dir).glob("*_T_*.*")
        if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }
    pairs: list[tuple[str, str, str]] = []
    for p in sorted(Path(vis_dir).glob("*_V_*.*")):
        if p.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        ir_stem = p.stem.replace("_V_", "_T_")
        if ir_stem in ir_map:
            fk = p.stem.replace("_V_", "_")
            pairs.append((fk, str(p), str(ir_map[ir_stem])))
    return pairs


def process_frame(
    vis_path: str,
    ir_path: str,
    H: np.ndarray,
    target_w: int,
    target_h: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
    """处理一对图像。

    Returns:
        (vis_img, ir_img, crop_params)  成功
        (None, None, None)             裁剪区为空（IR 未落在 VIS 内）
    Raises:
        FileNotFoundError: 图像读取失败
    """
    vis = imread_unicode(vis_path)
    ir = imread_unicode(ir_path, cv2.IMREAD_GRAYSCALE)

    # H 映射 VIS→IR，warp IR→VIS 需用逆变换
    H_inv = np.linalg.inv(H)
    ir_warped = cv2.warpPerspective(
        ir, H_inv, (vis.shape[1], vis.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    xmin, ymin, xmax, ymax = find_ir_bbox(ir_warped, margin_px=5)
    vis_crop = vis[ymin:ymax, xmin:xmax]
    ir_crop = ir_warped[ymin:ymax, xmin:xmax]
    if vis_crop.size == 0 or ir_crop.size == 0:
        return None, None, None

    vis_resized = cv2.resize(vis_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    ir_rgb = cv2.cvtColor(ir_crop, cv2.COLOR_GRAY2BGR)
    ir_resized = cv2.resize(ir_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    crop_params = {
        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        "vis_orig_w": vis.shape[1], "vis_orig_h": vis.shape[0],
        "target_w": target_w, "target_h": target_h,
    }
    return vis_resized, ir_resized, crop_params


# ---- 主入口 ----

def prepare_icafusion_pairs(
    ir_dir: str,
    vis_dir: str,
    output_dir: str,
    H: np.ndarray | None = None,
    npz_path: str | None = None,
    target_w: int = 1280,
    target_h: int = 1024,
    overwrite: bool = False,
    progress_callback: Callable[[int], None] | None = None,
) -> dict:
    """对齐 IR-VIS 数据并输出 visible/ + ir/。

    H 来源优先级：npz_path > H 参数 > 内置固定 DEFAULT_H。
    """
    if npz_path:
        H = load_h_from_npz(npz_path)
    elif H is None:
        H = DEFAULT_H

    output_dir = Path(output_dir)
    out_vis = output_dir / "visible"
    out_ir = output_dir / "ir"
    out_vis.mkdir(parents=True, exist_ok=True)
    out_ir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(ir_dir, vis_dir)
    total = len(pairs)
    stats = {
        "total": total, "success": 0, "crop_empty": 0,
        "read_error": 0, "write_error": 0, "other_error": 0,
    }
    crop_params: dict = {}

    def _report(idx: int) -> None:
        if progress_callback and total > 0:
            progress_callback(int((idx + 1) / total * 100))

    for idx, (fk, vis_path, ir_path) in enumerate(pairs):
        stem = Path(vis_path).stem
        try:
            out_v = out_vis / f"{stem}.jpg"
            out_i = out_ir / f"{stem}.jpg"

            if out_v.exists() and out_i.exists() and not overwrite:
                stats["success"] += 1
                _report(idx)
                continue

            vis_img, ir_img, cp = process_frame(vis_path, ir_path, H, target_w, target_h)
            if vis_img is None:
                stats["crop_empty"] += 1
                _report(idx)
                continue

            imwrite_unicode(str(out_v), vis_img)
            imwrite_unicode(str(out_i), ir_img)
            crop_params[stem] = cp
            stats["success"] += 1

        except FileNotFoundError:
            stats["read_error"] += 1
        except Exception:
            stats["other_error"] += 1
            if stats["other_error"] <= 3:
                logger.exception("处理失败 [%s]", fk)

        _report(idx)

    (output_dir / "crop_params.json").write_text(
        json.dumps(crop_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats
