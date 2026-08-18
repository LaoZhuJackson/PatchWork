"""SD 边界自然化：用 Stable Diffusion inpaint 对贴图目标的边缘做自然过渡处理

原理：不重画整张图，也不直接用矩形框。对每只目标：
1. 在框内用阈值提取目标亮区轮廓（白热目标=亮块），得到贴合姿态的 silhouette mask
2. 以目标为中心裁剪局部区域 → 缩放到 512×512（SD 原生分辨率）→ 生成"轮廓骑边环"
   （轮廓内收缩 keep_px 作保留区保护姿态 + 轮廓外膨胀 band_px 作重画区）
3. SD inpaint 只重画这圈环带，让贴入目标与背景自然过渡
4. 只把环带区域写回原图，目标内部像素完全不动，姿态细节不丢失

输入：合成图目录 + YOLO 标签目录（class_id 指定贴入目标类别，默认 0 = 猫）
输出：处理后的图片目录（与输入同名），无目标框的图原样复制
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_PROMPT = "infrared thermal image, monochrome, smooth natural edge transition"
DEFAULT_NEGATIVE_PROMPT = (
    "color, vivid colors, saturation, artifacts, blurry, deformed, "
    "close-up, big head, huge face"
)
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-inpainting"
DEFAULT_PROXY = "http://127.0.0.1:7897"  # 工作区 Clash 代理

_LOCAL_TARGET = 512  # 局部裁剪的 SD 处理分辨率


# ---- 依赖检查 ----

def sd_availability() -> tuple[bool, str]:
    """检查 diffusers 是否可用，返回 (可用?, 不可用原因)"""
    try:
        import diffusers  # noqa: F401
        return True, ""
    except ImportError:
        return False, (
            "缺少 diffusers 库，请先在 laozhu311 环境安装：\n"
            "pip install diffusers\n"
            "（首次运行会从 HuggingFace 下载 SD inpaint 权重 ~4GB，需联网）"
        )


# ---- 读写（兼容 Windows 中文/空格路径） ----

def imread_u(path: str, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def imwrite_u(path: str, img: np.ndarray) -> None:
    ext = Path(path).suffix.lower()
    if ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"编码图像失败: {path}")
    buf.tofile(path)


# ---- 标签解析 ----

def load_boxes(label_path: str, class_id: int) -> list[tuple[float, float, float, float]]:
    """读取 YOLO 标签，返回指定类别的归一化框列表 [(cx, cy, w, h)]"""
    boxes: list[tuple[float, float, float, float]] = []
    try:
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5 or int(parts[0]) != class_id:
                    continue
                cx, cy, w, h = (float(v) for v in parts[1:5])
                boxes.append((cx, cy, w, h))
    except (OSError, ValueError):
        return []
    return boxes


def discover_pairs(
    img_dir: Path, label_dir: Path,
) -> list[tuple[str, Path, Path]]:
    """配对合成图与标签：(stem, img_path, label_path)，标签缺失则跳过"""
    pairs: list[tuple[str, Path, Path]] = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        label_path = label_dir / f"{p.stem}.txt"
        if label_path.is_file():
            pairs.append((p.stem, p, label_path))
    return pairs


# ---- 核心 ----

class BoundaryRefiner:
    """SD inpaint 边界环自然化（轮廓 mask + 局部 512 处理）"""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        proxy: str = "",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.proxy = proxy
        self._pipe: object | None = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        ok, msg = sd_availability()
        if not ok:
            raise RuntimeError(msg)
        # 未显式设置代理时，用工作区代理走外网（Clash 默认 7897）
        if self.proxy and not os.environ.get("HTTPS_PROXY"):
            os.environ["HTTPS_PROXY"] = self.proxy
            os.environ["HTTP_PROXY"] = self.proxy
            os.environ["https_proxy"] = self.proxy
            os.environ["http_proxy"] = self.proxy
        import torch
        from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
        # safety_checker 对灰度/热像小目标极易误报 NSFW → 返回黑图，此处禁用
        self._pipe = AutoPipelineForInpainting.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config
        )
        self._pipe.enable_attention_slicing()

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    # ---- mask 构造 ----

    @staticmethod
    def _polarity_is_white_hot(crop: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
        """判断目标极性：框内均值 >= 周边背景均值 = 白热（亮目标），否则黑热（暗目标）"""
        h, w = crop.shape
        x, y, bw, bh = rect
        x0 = max(0, x - bw)          # 外扩一个框宽作为背景参考带
        y0 = max(0, y - bh)
        x1 = min(w, x + 2 * bw)
        y1 = min(h, y + 2 * bh)
        inner = crop[y:y + bh, x:x + bw]
        outer = crop[y0:y1, x0:x1].copy()
        om = np.ones_like(outer, dtype=bool)
        om[y - y0:y - y0 + bh, x - x0:x - x0 + bw] = False
        bg = outer[om]
        if bg.size == 0:
            return True
        return float(inner.mean()) >= float(bg.mean())

    @staticmethod
    def _silhouette_mask(crop: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray | None:
        """提取目标轮廓 mask，自动适应白热（亮目标/暗背景）与黑热（暗目标/亮背景）。

        失败返回 None（调用方跳过该目标，**绝不退回整个 bbox 矩形**——
        围绕矩形生成 ring 会改到靠近框边的猫身体、漏掉真实猫边缘）。

        rect = (x, y, w, h) 框在 crop 内的坐标
        """
        h, w = crop.shape
        x, y, bw, bh = rect
        if bw <= 0 or bh <= 0 or x < 0 or y < 0 or x + bw > w or y + bh > h:
            return None

        hi = int(crop.max())
        if hi < 50:  # 整块对比度太低，阈值不可靠
            return None

        # 自动判断极性：白热找亮块，黑热反转后同样找亮块
        is_white_hot = BoundaryRefiner._polarity_is_white_hot(crop, rect)
        work = crop if is_white_hot else (255 - crop).astype(np.uint8)
        work_hi = int(work.max())
        otsu_val, _ = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu_val <= 0:  # 常数图（无对比度），Otsu 返回 0
            return None
        # Otsu 可能返回背景值（低对比度时尤其），用 0.5*max 作下限兜底；
        # 用严格 > 避免含入等于背景值的像素。不再设上限——设上限会在低对比度
        # 图里把阈值拉到背景以下，导致整框全亮、误判为无目标。
        th = max(otsu_val, int(0.5 * work_hi))
        m = (work > th).astype(np.uint8) * 255

        # 只保留目标框附近（外扩 ~25%）的亮区，避免远处背景渗入 mask
        ext = max(5, bw // 4, bh // 4)
        keep_region = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(
            keep_region,
            (max(0, x - ext), max(0, y - ext)),
            (min(w, x + bw + ext), min(h, y + bh + ext)),
            255, -1,
        )
        m = cv2.bitwise_and(m, keep_region)

        # 自适应形态学：小目标用小核；只 CLOSE 连接轮廓断点，不做 OPEN
        #（OPEN 会删除尾巴/脚等 1~3px 细结构）
        min_dim = min(bw, bh)
        ksize = 1 if min_dim < 40 else 3
        if ksize >= 3:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

        # 保留所有"与目标框相交 + 面积达标"的连通域，而不是只留最大一个
        #（阈值后尾巴可能和身体断开 1px，只留最大连通域会把尾巴丢掉）
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return None
        out = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 2:
                continue
            comp = (labels == i).astype(np.uint8)
            inside = comp[y:y + bh, x:x + bw]
            if inside.any():
                out = cv2.bitwise_or(out, comp * 255)
        if out.max() == 0:
            return None
        # 合理性检查（在组件筛选后）：
        # - 轮廓过小（<3px）或几乎占满整框（>97%）都视为无有效目标
        if int((out > 0).sum()) < 3:
            return None
        fill = float((out[y:y + bh, x:x + bw] > 0).mean())
        if fill > 0.97:
            return None
        return out

    @staticmethod
    def _box_to_ring(sil_sq: np.ndarray, band_px: int, keep_px: int) -> np.ndarray:
        """轮廓 → 非对称骑边环（白 = SD 重生成区域），羽化后返回 float32 0~255

        外侧（背景侧）全强度 100% 重画让边缘融入背景；
        内侧（目标侧）只允许 ~20% 轻微处理，保护猫姿态/白热信息；核心 0% 完全不动。
        """
        # 目标核心：轮廓向内收缩 keep_px（保护姿态细节）
        keep = sil_sq
        if keep_px > 0:
            kk = 2 * keep_px + 1
            keep = cv2.erode(keep, np.ones((kk, kk), np.uint8), iterations=1)
            if keep.max() == 0:  # 轮廓太细被腐蚀空 → 退回原轮廓
                keep = sil_sq

        # 外侧环：轮廓外扩 band_px → 背景侧 100%
        kb = 2 * band_px + 1
        outer = cv2.dilate(sil_sq, np.ones((kb, kb), np.uint8), iterations=1)
        outer_ring = cv2.subtract(outer, sil_sq).astype(np.float32)
        # 内侧环：轮廓与核心之间 → 目标侧 20%
        inner_ring = cv2.subtract(sil_sq, keep).astype(np.float32)
        ring = outer_ring + 0.2 * inner_ring
        # 轻羽化：sigma 取 band/4（小目标上 sigma 太大会让模糊尾巴渗进核心区）
        return cv2.GaussianBlur(ring, (0, 0), max(1.0, band_px / 4.0))

    def refine(
        self,
        gray: np.ndarray,
        boxes: list[tuple[float, float, float, float]],
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        steps: int = 30,
        strength: float = 0.4,
        guidance: float = 7.5,
        band_px: int = 4,
        keep_px: int = 1,
        obj_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """逐目标局部 512 处理边缘环，只把环带写回，目标内部不变。

        obj_mask: 可选，合成时保存的整图目标 mask（H×W uint8），有则直接用
                  精确轮廓，不再从灰度图猜；缺失/失败的目标跳过。
        """
        if self._pipe is None:
            raise RuntimeError("模型未加载，请先调用 load()")
        from PIL import Image

        result = gray.copy()
        H, W = gray.shape
        tgt = _LOCAL_TARGET
        skipped = 0

        for cx, cy, bw, bh in boxes:
            x1 = int((cx - bw / 2) * W)
            x2 = int((cx + bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            y2 = int((cy + bh / 2) * H)
            if x2 <= x1 or y2 <= y1:
                continue
            bw_px, bh_px = x2 - x1, y2 - y1

            # 1. 带边距裁剪局部区域
            pad = max(8, int(max(bw_px, bh_px) * 0.9))
            x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
            x2c, y2c = min(W, x2 + pad), min(H, y2 + pad)
            crop = gray[y1c:y2c, x1c:x2c].copy()
            ch, cw = crop.shape

            # 2. 目标轮廓：优先用合成时保存的精确 mask，否则灰度阈值提取
            if obj_mask is not None and obj_mask.shape == (H, W):
                m_box = obj_mask[y1:y2, x1:x2]
                if m_box.max() == 0:
                    skipped += 1
                    continue
                sil = np.zeros((ch, cw), dtype=np.uint8)
                sil[y1 - y1c:y1 - y1c + bh_px, x1 - x1c:x1 - x1c + bw_px] = \
                    (m_box > 128).astype(np.uint8) * 255
            else:
                sil = self._silhouette_mask(crop, (x1 - x1c, y1 - y1c, bw_px, bh_px))
            if sil is None:
                skipped += 1  # 轮廓提取失败 → 跳过，不围矩形生成错误环
                continue

            # 3. letterbox 缩放 + 补边到 512
            scale = tgt / max(cw, ch)
            nw = max(8, int(round(cw * scale)))
            nh = max(8, int(round(ch * scale)))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
            sil_r = cv2.resize(sil, (nw, nh), interpolation=cv2.INTER_NEAREST)
            top, left = (tgt - nh) // 2, (tgt - nw) // 2
            img_sq = np.full((tgt, tgt), 127, dtype=np.uint8)
            img_sq[top:top + nh, left:left + nw] = resized
            sil_sq = np.zeros((tgt, tgt), dtype=np.uint8)
            sil_sq[top:top + nh, left:left + nw] = sil_r

            # 4. 骑边环 + SD inpaint（band_px/keep_px 以原图像素为单位，换算到 512 空间）
            keep_512 = max(1, int(round(keep_px * scale)))
            band_512 = max(1, int(round(band_px * scale)))
            ring = self._box_to_ring(sil_sq, band_512, keep_512)
            if ring.max() < 1:
                continue
            rgb_sq = cv2.cvtColor(img_sq, cv2.COLOR_GRAY2RGB)
            out = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=Image.fromarray(rgb_sq),
                mask_image=Image.fromarray(ring.astype(np.uint8)),
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=guidance,
            ).images[0]
            out_gray = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2GRAY)

            # 5. 去掉补边 → 缩放回裁剪坐标 → 仅环带写回（保护目标内部清晰度）
            out_crop_sq = out_gray[top:top + nh, left:left + nw]
            out_crop = cv2.resize(out_crop_sq, (cw, ch), interpolation=cv2.INTER_LINEAR)
            ring_crop = ring[top:top + nh, left:left + nw]
            ring_crop = cv2.resize(ring_crop, (cw, ch), interpolation=cv2.INTER_LINEAR)

            alpha = (ring_crop / 255.0).astype(np.float32)
            region = result[y1c:y2c, x1c:x2c].astype(np.float32)
            region = region * (1.0 - alpha) + out_crop.astype(np.float32) * alpha
            result[y1c:y2c, x1c:x2c] = np.clip(region, 0, 255).astype(np.uint8)

        if skipped:
            logger.warning("轮廓提取失败，跳过 %d 个目标（避免围矩形生成错误环）", skipped)
        return result

    def regenerate(
        self,
        gray: np.ndarray,
        boxes: list[tuple[float, float, float, float]],
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        steps: int = 30,
        strength: float = 0.7,
        guidance: float = 7.5,
        band_px: int = 8,
    ) -> np.ndarray:
        """整框重画：把目标框内内容整体交给 SD 重新生成（提示词告知是猫）

        边缘问题构造性解决——SD 画出的目标与背景一体，无贴图缝。
        band_px = 框外扩的融合边距：重画范围延伸到框外 band_px 像素，
                  让新目标边缘在框外背景中自然过渡。原目标被替换为 SD 生成的版本。
        """
        if self._pipe is None:
            raise RuntimeError("模型未加载，请先调用 load()")
        from PIL import Image

        result = gray.copy()
        H, W = gray.shape
        tgt = _LOCAL_TARGET

        for cx, cy, bw, bh in boxes:
            x1 = int((cx - bw / 2) * W)
            x2 = int((cx + bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            y2 = int((cy + bh / 2) * H)
            if x2 <= x1 or y2 <= y1:
                continue
            bw_px, bh_px = x2 - x1, y2 - y1

            # 1. 带边距裁剪局部区域（给 SD 足够的场景上下文）
            pad = max(10, band_px + max(bw_px, bh_px) // 3)
            x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
            x2c, y2c = min(W, x2 + pad), min(H, y2 + pad)
            crop = gray[y1c:y2c, x1c:x2c].copy()
            ch, cw = crop.shape
            rx, ry = x1 - x1c, y1 - y1c

            # 提示词直接使用用户在面板中填写的文本（不内置类别/明暗假设，通用）
            box_prompt = prompt

            # 3. mask = 框外扩 band_px（软边），整框内容由 SD 重画
            ext = max(1, band_px)
            mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.rectangle(
                mask,
                (max(0, rx - ext), max(0, ry - ext)),
                (min(cw, rx + bw_px + ext), min(ch, ry + bh_px + ext)),
                255, -1,
            )
            mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(1.0, ext / 2.0))

            # 4. letterbox 缩放 + 补边到 512
            scale = tgt / max(cw, ch)
            nw = max(8, int(round(cw * scale)))
            nh = max(8, int(round(ch * scale)))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
            mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_LINEAR)
            top, left = (tgt - nh) // 2, (tgt - nw) // 2
            img_sq = np.full((tgt, tgt), 127, dtype=np.uint8)
            img_sq[top:top + nh, left:left + nw] = resized
            mask_sq = np.zeros((tgt, tgt), dtype=np.float32)
            mask_sq[top:top + nh, left:left + nw] = mask_r

            # 5. SD inpaint（整框重画，strength 需要偏高）
            rgb_sq = cv2.cvtColor(img_sq, cv2.COLOR_GRAY2RGB)
            out = self._pipe(
                prompt=box_prompt,
                negative_prompt=negative_prompt,
                image=Image.fromarray(rgb_sq),
                mask_image=Image.fromarray(mask_sq.astype(np.uint8)),
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=guidance,
            ).images[0]
            out_gray = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2GRAY)

            # 6. 去补边 → 缩回裁剪坐标 → 整框写回（mask 内替换，外保持原图）
            out_crop_sq = out_gray[top:top + nh, left:left + nw]
            out_crop = cv2.resize(out_crop_sq, (cw, ch), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(mask_r, (cw, ch), interpolation=cv2.INTER_LINEAR)
            alpha = (alpha / 255.0).astype(np.float32)
            region = result[y1c:y2c, x1c:x2c].astype(np.float32)
            result[y1c:y2c, x1c:x2c] = np.clip(
                region * (1.0 - alpha) + out_crop.astype(np.float32) * alpha,
                0, 255,
            ).astype(np.uint8)

        return result


# ---- 批量入口 ----

def refine_dir(
    img_dir: str,
    label_dir: str,
    out_dir: str,
    refiner: BoundaryRefiner,
    *,
    mode: str = "edge",
    class_id: int = 0,
    band_px: int = 4,
    keep_px: int = 1,
    steps: int = 30,
    strength: float = 0.4,
    guidance: float = 7.5,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    mask_dir: str = "",
    overwrite: bool = False,
    progress_callback=None,
) -> dict:
    """批量处理：对每张合成图中 class_id 的目标框做 SD 处理。

    mode="edge"：轮廓边缘自然化（只重画轮廓骑边环带）
    mode="regen"：整框重画（框内内容整体由 SD 重新生成）
    mask_dir：可选，合成时保存的整图目标 mask 目录（`<stem>_mask.png`），
              edge 模式优先用精确 mask，缺失则灰度阈值提取。
    返回统计 dict；无目标框的图原样复制。
    """
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    out_dir = Path(out_dir)
    mask_dir = Path(mask_dir) if mask_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(img_dir, label_dir)
    total = len(pairs)
    stats = {
        "total": total, "refined": 0, "no_target": 0, "skipped": 0,
        "read_error": 0, "write_error": 0, "other_error": 0,
    }

    for i, (stem, img_path, label_path) in enumerate(pairs):
        out_path = out_dir / img_path.name
        try:
            if out_path.exists() and not overwrite:
                stats["skipped"] += 1  # 已有输出且未勾覆盖 → 跳过
                if progress_callback and total:
                    progress_callback(int((i + 1) / total * 100))
                continue

            boxes = load_boxes(str(label_path), class_id)
            gray = imread_u(str(img_path), cv2.IMREAD_GRAYSCALE)
            if not boxes:
                imwrite_u(str(out_path), gray)
                stats["no_target"] += 1
                if progress_callback and total:
                    progress_callback(int((i + 1) / total * 100))
                continue

            if mode == "regen":
                refined = refiner.regenerate(
                    gray, boxes, prompt, negative_prompt, steps, strength, guidance,
                    band_px,
                )
            else:
                # edge 模式：优先加载合成时保存的精确 mask
                obj_mask = None
                if mask_dir is not None:
                    mask_path = mask_dir / f"{stem}_mask.png"
                    if mask_path.is_file():
                        obj_mask = imread_u(str(mask_path), cv2.IMREAD_GRAYSCALE)
                refined = refiner.refine(
                    gray, boxes, prompt, negative_prompt, steps, strength, guidance,
                    band_px, keep_px, obj_mask,
                )
            imwrite_u(str(out_path), refined)
            stats["refined"] += 1

        except FileNotFoundError:
            stats["read_error"] += 1
        except Exception:
            stats["other_error"] += 1
            if stats["other_error"] <= 3:
                logger.exception("SD 自然化失败 [%s]", stem)

        if progress_callback and total:
            progress_callback(int((i + 1) / total * 100))

    return stats
