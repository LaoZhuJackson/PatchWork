"""热成像目标合成：SAM 抠图 + 随机贴图 + 标注合并"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np


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
    - 两种尺寸模式：比例缩放 / 指定最长边像素
    - 随机旋转、模糊、亮度微调
    - 水平/垂直翻转
    - 自定义类别 ID
    """

    def __init__(
        self,
        instance_dir: Path,
        *,
        size_mode: str = "scale",
        scale_range: tuple[float, float] = (0.5, 1.5),
        pixel_size: int = 0,
        rotation_range: tuple[float, float] = (-30, 30),
        blur_range: tuple[float, float] = (0.0, 0.0),
        flip_h_prob: float = 0.0,
        flip_v_prob: float = 0.0,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        class_id: int = 0,
    ) -> None:
        self.instances = sorted(instance_dir.glob("*.png"))
        if not self.instances:
            raise ValueError(f"实例库中没有 .png 文件: {instance_dir}")
        self._size_mode = size_mode
        self.scale_range = scale_range
        self._pixel_size = pixel_size
        self.rotation_range = rotation_range
        self._blur_range = blur_range
        self._flip_h_prob = flip_h_prob
        self._flip_v_prob = flip_v_prob
        self.brightness_range = brightness_range
        self.class_id = class_id

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    def composite(
        self, target_path: Path, output_path: Path, num_instances: int = 1,
    ) -> list[dict]:
        """贴入随机实例，返回新增 YOLO 标注列表"""
        target = cv2.imread(str(target_path))
        if target is None:
            raise ValueError(f"无法读取图片: {target_path}")
        h_t, w_t = target.shape[:2]
        annotations: list[dict] = []

        for _ in range(num_instances):
            inst_path = random.choice(self.instances)
            instance = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED)
            if instance is None or instance.shape[2] < 4:
                continue

            # 尺寸：像素模式按最长边算比例，比例模式随机取
            if self._size_mode == "pixel" and self._pixel_size > 0:
                max_dim = max(instance.shape[0], instance.shape[1])
                scale = self._pixel_size / max_dim
            else:
                scale = random.uniform(*self.scale_range)

            angle = random.uniform(*self.rotation_range)
            brightness = random.uniform(*self.brightness_range)
            blur_sigma = random.uniform(*self._blur_range)
            flip_h = random.random() < self._flip_h_prob
            flip_v = random.random() < self._flip_v_prob

            instance = _augment(instance, scale, angle, brightness,
                                blur_sigma, flip_h, flip_v)
            h_i, w_i = instance.shape[:2]

            if h_i >= h_t or w_i >= w_t:
                continue

            x = random.randint(0, w_t - w_i)
            y = random.randint(0, h_t - h_i)

            # Alpha 混合
            alpha = instance[:, :, 3:4].astype(np.float32) / 255.0
            roi = target[y:y + h_i, x:x + w_i].astype(np.float32)
            fg = instance[:, :, :3].astype(np.float32)
            target[y:y + h_i, x:x + w_i] = (fg * alpha + roi * (1 - alpha)).astype(np.uint8)

            annotations.append({
                "class_id": self.class_id,
                "cx": (x + w_i / 2) / w_t,
                "cy": (y + h_i / 2) / h_t,
                "w": w_i / w_t,
                "h": h_i / h_t,
            })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), target)
        return annotations


def _augment(
    img: np.ndarray,
    scale: float,
    angle: float,
    brightness: float,
    blur_sigma: float,
    flip_h: bool,
    flip_v: bool,
) -> np.ndarray:
    """缩放 + 翻转 + 旋转 + 模糊 + 亮度调整，保持 RGBA"""
    result = img.copy()

    # 1. 翻转（不改变 alpha）
    if flip_h:
        result = cv2.flip(result, 1)   # 水平（左右）
    if flip_v:
        result = cv2.flip(result, 0)   # 垂直（上下）

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

    # 3. 高斯模糊（只处理 RGB，不动 alpha）
    if blur_sigma > 0:
        result[:, :, :3] = cv2.GaussianBlur(
            result[:, :, :3], (0, 0), sigmaX=blur_sigma,
        )

    # 4. 亮度调整
    rgb = result[:, :, :3].astype(np.float32)
    result[:, :, :3] = np.clip(rgb * brightness, 0, 255).astype(np.uint8)

    return result


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
