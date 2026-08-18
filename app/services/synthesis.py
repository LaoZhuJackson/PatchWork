"""热成像目标合成：SAM 抠图 + 随机贴图 + 标注合并"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# SD 边缘自然化默认参数（refiner 传参时仅覆盖显式给定的项）
from app.services.sd_refine import (
    DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT,
)  # noqa: E402


# ---- 抠图引擎 ----

SAM_VARIANTS: dict[str, str] = {
    "MobileSAM (轻量, ~38.8MB)": "mobile_sam.pt",
    "SAM 2.1 Small (高精度, ~88MB)": "sam2.1_s.pt",
}
SAM_VARIANT_NAMES = list(SAM_VARIANTS.keys())


class BackgroundRemover:
    """SAM 自动抠图 —— 输入原图，输出透明背景 PNG"""

    def __init__(self) -> None:
        self._model: object | None = None  # ultralytics SAM
        self._model_name: str = ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def load_model(self, model_path: str) -> None:
        from ultralytics import SAM
        self._model = SAM(model_path)
        self._model_name = model_path

    def remove_background(self, image_path: Path, output_path: Path) -> Path:
        """抠掉背景，保存 RGBA PNG 到 output_path"""
        if self._model is None:
            raise RuntimeError("模型未加载")

        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 统一为 3 通道 BGR（处理灰度图 / 2通道热力图）
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] >= 3:
            bgr = img[:, :, :3]
        else:
            # 2 通道等异常情况 → 取第一通道转 BGR
            bgr = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)

        results = self._model(str(image_path), verbose=False)
        result = results[0]

        if result.masks is None:
            # 没检测到前景 → 整张保留（全不透明）
            alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
        else:
            masks = result.masks.data.cpu().numpy()
            # 合并所有 mask → 前景 alpha 通道
            alpha = (np.any(masks, axis=0).astype(np.uint8)) * 255
            # SAM 返回的 mask 尺寸可能和原图不一致，缩放到一致
            if alpha.shape != bgr.shape[:2]:
                alpha = cv2.resize(
                    alpha, (bgr.shape[1], bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

        rgba = np.dstack([bgr, alpha])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # OpenCV imwrite 对 RGBA PNG 支持不稳定，用 PIL 写
        from PIL import Image
        Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)).save(str(output_path))
        return output_path


# ---- 贴图合成器 ----

class ImageCompositor:
    """随机贴图合成：把实例库中的目标贴到背景图上

    支持：
    - 尺寸模式：比例缩放 / 指定最长边像素
    - 色阶模式：白热/黑热/保持原样（自动反色纠正）
    - 融合模式：Alpha 混合 / 泊松融合（无缝边缘）
    - 直方图匹配：把实例像素分布对齐到背景区域
    - 随机旋转、模糊、噪声、翻转
    - 自定义类别 ID
    """

    def __init__(
        self,
        instance_dir: Path,
        *,
        size_mode: str = "scale",
        scale_range: tuple[float, float] = (0.5, 1.5),
        pixel_size: int = 0,
        color_mode: str = "keep",
        blend_mode: str = "alpha",
        match_histogram: bool = False,
        rotation_range: tuple[float, float] = (-30, 30),
        blur_range: tuple[float, float] = (0.0, 0.0),
        noise_range: tuple[float, float] = (0.0, 0.0),
        flip_h_prob: float = 0.0,
        flip_v_prob: float = 0.0,
        contrast_range: tuple[float, float] = (0.8, 1.2),
        class_id: int = 0,
        opacity: float = 1.0,
    ) -> None:
        self.instances = sorted(instance_dir.glob("*.png"))
        if not self.instances:
            raise ValueError(f"实例库中没有 .png 文件: {instance_dir}")
        self._size_mode = size_mode
        self.scale_range = scale_range
        self._pixel_size = pixel_size
        self._color_mode = color_mode
        self._blend_mode = blend_mode
        self._match_histogram = match_histogram
        self.rotation_range = rotation_range
        self._blur_range = blur_range
        self._noise_range = noise_range
        self._flip_h_prob = flip_h_prob
        self._flip_v_prob = flip_v_prob
        self.contrast_range = contrast_range
        self.class_id = class_id
        self._opacity = opacity
    @property
    def instance_count(self) -> int:
        return len(self.instances)

    def composite(
        self, target_path: Path, output_path: Path, num_instances: int = 1,
        save_mask_to: Path | None = None,
        refiner=None, sd_params: dict | None = None,
    ) -> list[dict]:
        """贴入随机实例，返回新增 YOLO 标注列表。

        save_mask_to: 可选，同时保存整图目标 alpha mask PNG。
        refiner: 可选 BoundaryRefiner，传入则在**合成时**用精确 alpha 对贴图
                 边缘做 SD 自然化（不再单独后处理已生成图像）。
        sd_params: 可选 dict（band_px/keep_px/steps/strength/guidance/prompt/
                   negative_prompt），仅覆盖 refiner.refine() 默认值。
        """
        target = cv2.imread(str(target_path))
        if target is None:
            raise ValueError(f"无法读取图片: {target_path}")
        h_t, w_t = target.shape[:2]
        annotations: list[dict] = []
        mask_acc = np.zeros((h_t, w_t), dtype=np.uint8)  # 目标精确 alpha 累计

        for _ in range(num_instances):
            inst_path = random.choice(self.instances)
            instance = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED)
            if instance is None or instance.shape[2] < 4:
                continue

            # ── 色阶模式 ──
            instance = _apply_color_mode(instance, self._color_mode)

            # ── 尺寸 ──
            if self._size_mode == "pixel" and self._pixel_size > 0:
                max_dim = max(instance.shape[0], instance.shape[1])
                scale = self._pixel_size / max_dim
            else:
                scale = random.uniform(*self.scale_range)

            angle = random.uniform(*self.rotation_range)
            contrast = random.uniform(*self.contrast_range)
            blur_sigma = random.uniform(*self._blur_range)
            noise_sigma = random.uniform(*self._noise_range)
            flip_h = random.random() < self._flip_h_prob
            flip_v = random.random() < self._flip_v_prob

            instance = _augment(instance, scale, angle, contrast,
                                blur_sigma, flip_h, flip_v)
            h_i, w_i = instance.shape[:2]
            min_dim = min(h_i, w_i)

            # 核大小 = 实例短边的 1/20，最小 1，最大 5
            ks = max(1, min(5, min_dim // 20))
            if ks % 2 == 0:
                ks += 1 # 保持奇数

            if h_i >= h_t or w_i >= w_t:
                continue

            x = random.randint(0, w_t - w_i)
            y = random.randint(0, h_t - h_i)

            # ── 融合 ──
            raw_alpha = instance[:, :, 3].copy()  # 变换后的原始 alpha（未腐蚀羽化）
            alpha = instance[:, :, 3]
            # 腐蚀
            if ks >= 3:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
                alpha = cv2.erode(alpha, kernel, iterations=1)

            # 羽化
            sigma = ks / 3.0
            alpha = cv2.GaussianBlur(alpha, (ks, ks), sigma)

            # 累计精确 mask（供 SD 边缘自然化）
            mask_acc[y:y + h_i, x:x + w_i] = np.maximum(
                mask_acc[y:y + h_i, x:x + w_i], raw_alpha
            )

            # 应用不透明度
            if self._opacity < 1.0:
                alpha = (alpha.astype(np.float32) * self._opacity).astype(np.uint8)

            fg = instance[:, :, :3]
            roi = target[y:y + h_i, x:x + w_i]

            if self._blend_mode == "poisson":
                # 泊松融合：需要 mask（alpha>0 的区域）
                mask = alpha
                center = (x + w_i // 2, y + h_i // 2)
                try:
                    # 转为 3 通道灰度供 seamlessClone 用
                    if roi.ndim == 2:
                        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    if fg.ndim == 2:
                        fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
                    target[y:y + h_i, x:x + w_i] = cv2.seamlessClone(
                        fg, roi, mask, center, cv2.NORMAL_CLONE,
                    )
                except cv2.error:
                    # 泊松融合失败 → 回退 Alpha
                    self._alpha_blend(target, fg, alpha, x, y, h_i, w_i)
            else:
                self._alpha_blend(target, fg, alpha, x, y, h_i, w_i)

            # ── 直方图匹配 ──
            if self._match_histogram:
                roi2 = target[y:y + h_i, x:x + w_i]
                roi2 = _match_histogram_region(roi2, roi)
                target[y:y + h_i, x:x + w_i] = roi2

            # ── 噪声 ──
            if noise_sigma > 0:
                roi3 = target[y:y + h_i, x:x + w_i].astype(np.float32)
                gauss = np.random.normal(0, noise_sigma, roi3.shape).astype(np.float32)
                target[y:y + h_i, x:x + w_i] = np.clip(roi3 + gauss, 0, 255).astype(np.uint8)

            annotations.append({
                "class_id": self.class_id,
                "cx": (x + w_i / 2) / w_t,
                "cy": (y + h_i / 2) / h_t,
                "w": w_i / w_t,
                "h": h_i / h_t,
            })

        # ── SD 边缘自然化（合成时，用精确 alpha 对贴图边缘自然过渡） ──
        if refiner is not None and annotations:
            p = sd_params or {}
            boxes = [(a["cx"], a["cy"], a["w"], a["h"]) for a in annotations]
            try:
                gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
                refined = refiner.refine(
                    gray, boxes,
                    prompt=p.get("prompt", DEFAULT_PROMPT),
                    negative_prompt=p.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
                    steps=p.get("steps", 30),
                    strength=p.get("strength", 0.4),
                    guidance=p.get("guidance", 7.5),
                    band_px=p.get("band_px", 4),
                    keep_px=p.get("keep_px", 1),
                    obj_mask=mask_acc,
                )
                target = cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)
            except Exception:
                logger.exception("SD 边缘自然化失败，保留原始合成结果 [%s]", target_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), target)
        if save_mask_to is not None:
            save_mask_to.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".png", mask_acc)
            if ok:
                buf.tofile(str(save_mask_to))
        return annotations

    @staticmethod
    def _alpha_blend(
        target: np.ndarray, fg: np.ndarray, alpha: np.ndarray,
        x: int, y: int, h: int, w: int,
    ) -> None:
        a = alpha.astype(np.float32)[:, :, np.newaxis] / 255.0
        roi = target[y:y + h, x:x + w].astype(np.float32)
        fg_f = fg[:, :, :3].astype(np.float32)
        # 确保 channel 维度对齐
        if roi.shape[2] == 1:
            roi = np.repeat(roi, 3, axis=2)
        if fg_f.shape[2] == 1:
            fg_f = np.repeat(fg_f, 3, axis=2)
        target[y:y + h, x:x + w] = (fg_f * a + roi * (1 - a)).astype(np.uint8)


def _augment(
    img: np.ndarray,
    scale: float,
    angle: float,
    contrast: float,
    blur_sigma: float,
    flip_h: bool,
    flip_v: bool,
) -> np.ndarray:
    """缩放 + 翻转 + 旋转 + 模糊 + 对比度调整，保持 RGBA"""
    result = img.copy()

    # 1. 翻转
    if flip_h:
        result = cv2.flip(result, 1)
    if flip_v:
        result = cv2.flip(result, 0)

    # 2. 缩放 + 旋转
    h, w = result.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    matrix[0, 2] += nw / 2 - center[0]
    matrix[1, 2] += nh / 2 - center[1]

    result = cv2.warpAffine(
        result, matrix, (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # 3. 高斯模糊
    if blur_sigma > 0:
        result[:, :, :3] = cv2.GaussianBlur(
            result[:, :, :3], (0, 0), sigmaX=blur_sigma,
        )

    # 4. 对比度调整
    if abs(contrast - 1.0) > 1e-6:
        rgb = result[:, :, :3].astype(np.float32)
        mean = rgb.mean()
        result[:, :, :3] = np.clip((rgb - mean) * contrast + mean, 0, 255).astype(np.uint8)

    return result


def _apply_color_mode(img: np.ndarray, mode: str) -> np.ndarray:
    """色阶模式：白热（亮目标+暗背景）/ 黑热（暗目标+亮背景）/ 保持"""
    if mode == "keep":
        return img

    rgb = img[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.shape[2] == 3 else rgb.squeeze()
    alpha = img[:, :, 3] if img.shape[2] == 4 else np.full(gray.shape, 255, dtype=np.uint8)

    mean_val = gray[alpha > 30].mean() if alpha.max() > 30 else gray.mean()

    should_invert = False
    if mode == "white_hot" and mean_val < 128:
        should_invert = True  # 暗目标 → 反色变亮
    elif mode == "black_hot" and mean_val > 128:
        should_invert = True  # 亮目标 → 反色变暗

    if should_invert:
        rgb = 255 - rgb
        result = np.dstack([rgb, alpha])
    else:
        result = img

    return result


def _match_histogram_region(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """把 src 的直方图匹配到 ref 的分布，用于消除贴图与背景的温差"""
    if src.shape != ref.shape:
        return src

    result = np.empty_like(src)
    for c in range(src.shape[2] if src.ndim == 3 else 1):
        if src.ndim == 3:
            s_ch = src[:, :, c].ravel()
            r_ch = ref[:, :, c].ravel()
        else:
            s_ch = src.ravel()
            r_ch = ref.ravel()

        # CDF 匹配
        s_sorted = np.sort(s_ch)
        r_sorted = np.sort(r_ch)
        # 线性映射：src 的每个值 → ref 中对应分位数的值
        s_cdf = np.searchsorted(s_sorted, s_ch, side='right').astype(float)
        s_cdf = s_cdf / len(s_sorted)
        mapped = np.interp(s_cdf, np.linspace(0, 1, len(r_sorted)), r_sorted)

        if src.ndim == 3:
            result[:, :, c] = mapped.reshape(src.shape[:2])
        else:
            result = mapped.reshape(src.shape)

    return np.clip(result, 0, 255).astype(np.uint8)


# ---- 标签合并工具 ----

def merge_labels(
    existing_label_dir: Path,
    image_name: str,
    new_annotations: list[dict],
    output_label_dir: Path,
) -> None:
    """把新增标注合并到已有 YOLO 标签文件

    existing_label_dir: 已有标签目录（与目标图片同名 .txt）
    image_name: 目标图片文件名（含扩展名）
    new_annotations: ImageCompositor.composite() 返回的新标注
    output_label_dir: 合并后的标签输出目录
    """
    stem = Path(image_name).stem
    output_label_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_label_dir / f"{stem}.txt"

    lines: list[str] = []

    # 读取已有标注
    existing = existing_label_dir / f"{stem}.txt"
    if existing.is_file():
        lines.extend(existing.read_text(encoding="utf-8").strip().splitlines())

    # 追加新标注
    for ann in new_annotations:
        lines.append(
            f"{ann['class_id']} "
            f"{ann['cx']:.6f} {ann['cy']:.6f} "
            f"{ann['w']:.6f} {ann['h']:.6f}"
        )

    out_path.write_text("\n".join(line for line in lines if line), encoding="utf-8")
