"""伪热红外图像增强：基于余弦变换的 RGB → 伪热红外风格化

源自 XoFTR (Tuzcuoglu et al., CVPRW 2024) 的 RGBThermalAug 模块。
用于数据增强、跨模态匹配训练、可视化预览。

算法流程:
  RGB → HSV微调 → 随机模糊 → 灰度化 → cos(x·w+φ) → 归一化 → 伪热红外图

纯 numpy + cv2，无需模型权重，即时执行。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PseudoThermalConfig:
    """伪热红外增强参数

    Attributes:
        blur_enabled: 是否启用随机模糊
        blur_p: 模糊触发概率
        blur_kernel_range: 模糊核范围 (min, max)
        hsv_enabled: 是否启用 HSV 微调
        hsv_p: HSV 触发概率
        hue_shift: 色调偏移范围 (±)
        sat_shift: 饱和度偏移范围 (±)
        val_shift: 明度偏移范围 (±)
        w0: 余弦频率基值 (默认 2π/3，对应论文)
        wr: 余弦频率随机范围 (默认 π/2)
        theta_r: 余弦相位随机范围 (默认 π/2)
        seed: 随机种子 (None=不固定)
    """
    blur_enabled: bool = True
    blur_p: float = 0.7
    blur_kernel_range: tuple[int, int] = (2, 4)

    hsv_enabled: bool = True
    hsv_p: float = 0.9
    hue_shift: int = 90
    sat_shift: int = 30
    val_shift: int = 30

    w0: float = np.pi * 2 / 3     # 余弦基频
    wr: float = np.pi / 2          # 频率随机幅度
    theta_r: float = np.pi / 2     # 相位随机幅度

    seed: int | None = None


# ═══════════════════════════════════════════════════════════════════════
# 增强器
# ═══════════════════════════════════════════════════════════════════════

class PseudoThermalAug:
    """伪热红外增强器

    用法:
        aug = PseudoThermalAug()
        result = aug.augment(rgb_image)           # 随机参数
        result = aug.augment(rgb_image, seed=42)  # 固定种子 (可复现)

    输入: RGB 图像 (H, W, 3), uint8
    输出: 伪热红外图像 (H, W, 3), uint8 (3通道灰度)
    """

    def __init__(self, config: PseudoThermalConfig | None = None) -> None:
        self.cfg = config or PseudoThermalConfig()
        self._rng: np.random.RandomState | None = None

    # ── 公开 API ──

    def augment(self, image: np.ndarray, seed: int | None = None) -> np.ndarray:
        """对 RGB 图像施加伪热红外增强

        Args:
            image: RGB 图像 (H, W, 3), uint8
            seed: 随机种子, 传入则固定随机性; 不传则使用 config.seed

        Returns:
            伪热红外图像 (H, W, 3), uint8
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"需要 3 通道 RGB 图像, 实际 shape={image.shape}")

        seed_val = seed if seed is not None else self.cfg.seed
        self._rng = np.random.RandomState(seed_val)
        result = image.copy()

        # 1. HSV 微调
        if self.cfg.hsv_enabled and self._rng.random() < self.cfg.hsv_p:
            result = self._augment_hsv(result)

        # 2. 随机模糊
        if self.cfg.blur_enabled and self._rng.random() < self.cfg.blur_p:
            result = self._augment_blur(result)

        # 3. 灰度化
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        # 4. 余弦变换 (核心)
        pseudo = self._cosine_transform(gray)

        # 5. 转回 3 通道
        return cv2.cvtColor(pseudo, cv2.COLOR_GRAY2RGB)

    def augment_batch(
        self,
        images: list[np.ndarray],
        seed: int | None = None,
    ) -> list[np.ndarray]:
        """批量增强 (每张使用不同随机参数但固定起点可复现)"""
        seed_val = seed if seed is not None else self.cfg.seed
        results = []
        for i, img in enumerate(images):
            s = seed_val + i if seed_val is not None else None
            results.append(self.augment(img, seed=s))
        return results

    # ── 内部方法 ──

    def _augment_hsv(self, image: np.ndarray) -> np.ndarray:
        """HSV 微调: 模拟热成像的色调/饱和度差异"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)

        dh = self._rng.randint(-self.cfg.hue_shift, self.cfg.hue_shift + 1)
        ds = self._rng.randint(-self.cfg.sat_shift, self.cfg.sat_shift + 1)
        dv = self._rng.randint(-self.cfg.val_shift, self.cfg.val_shift + 1)

        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + dh, 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + ds, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + dv, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _augment_blur(self, image: np.ndarray) -> np.ndarray:
        """随机核大小的高斯模糊"""
        k_min, k_max = self.cfg.blur_kernel_range
        # 确保核大小为奇数
        k = self._rng.randint(k_min, k_max + 1)
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(image, (k, k), 0)

    def _cosine_transform(self, gray: np.ndarray) -> np.ndarray:
        """余弦变换: 核心伪热红外映射

        cos(x * w + φ) 创建非线性的强度反转和纹理抑制,
        模拟热辐射图像的特征分布。
        """
        # 归一化到 [-0.5, 0.5]
        x = gray.astype(np.float32) / 255.0 - 0.5

        # 随机频率和相位
        phase = np.pi / 2 + self._rng.randn() * self.cfg.theta_r
        w = self.cfg.w0 + abs(self._rng.randn()) * self.cfg.wr

        # 余弦变换
        transformed = np.cos(x * w + phase)

        # Min-Max 归一化回 [0, 255]
        t_min = transformed.min()
        t_max = transformed.max()
        if t_max - t_min < 1e-8:
            transformed = np.zeros_like(transformed)
        else:
            transformed = (transformed - t_min) / (t_max - t_min) * 255.0

        return transformed.clip(0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════

def pseudo_thermal(image: np.ndarray, seed: int | None = None) -> np.ndarray:
    """一行调用: RGB → 伪热红外

    Args:
        image: RGB 图像 (H, W, 3), uint8
        seed: 随机种子 (可选)

    Returns:
        伪热红外图像 (H, W, 3), uint8

    Example:
        import cv2
        img = cv2.imread("photo.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        from pseudo_thermal import pseudo_thermal
        thermal = pseudo_thermal(img, seed=42)
    """
    return PseudoThermalAug().augment(image, seed=seed)


# ═══════════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 生成一张测试图 (彩色渐变)
    h, w = 480, 640
    y, x = np.ogrid[:h, :w]
    test_img = np.zeros((h, w, 3), dtype=np.uint8)
    test_img[:, :, 0] = (x / w * 255).astype(np.uint8)       # R 水平渐变
    test_img[:, :, 1] = (y / h * 255).astype(np.uint8)       # G 垂直渐变
    test_img[:, :, 2] = (np.sin(x / 30) * 127 + 128).astype(np.uint8)  # B 条纹

    aug = PseudoThermalAug()

    # 固定种子生成 3 张对比
    for i in range(3):
        result = aug.augment(test_img, seed=42 + i)
        out_path = Path(__file__).parent / f"_test_thermal_{i}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"[{i}] seed={42+i} → {out_path}")

    # 演示 config 自定义
    cfg = PseudoThermalConfig(blur_enabled=False, hsv_enabled=False, w0=np.pi, wr=0, theta_r=0)
    deterministic = PseudoThermalAug(cfg).augment(test_img, seed=0)
    det_out = Path(__file__).parent / "_test_thermal_det.png"
    cv2.imwrite(str(det_out), cv2.cvtColor(deterministic, cv2.COLOR_RGB2BGR))
    print(f"[det] 无随机性 → {det_out}")

    print("\n✅ 自测通过")
