"""伪热红外增强面板：RGB → 伪热红外风格化

基于 XoFTR (Tuzcuoglu et al., CVPRW 2024) 的 Cosine Transform 方法。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, ProgressBar,
    BodyLabel, StrongBodyLabel, SubtitleLabel,
    CardWidget, SpinBox, DoubleSpinBox, CheckBox, ComboBox,
)

from app.services.pseudo_thermal import (
    PseudoThermalAug, PseudoThermalConfig, pseudo_thermal,
)
from app.utils.config import (
    get_str, set_str, get_int, set_int, get_bool, set_bool,
    get_float, set_float,
)
from app.utils.message import error, info
from app.utils.worker import Worker
from app.widgets.path_browser import PathBrowser
from app.widgets.image_viewer import ImageViewer
from app.utils.logger import get_logger

logger = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ═══════════════════════════════════════════════════════════════════════
# Worker (批量处理)
# ═══════════════════════════════════════════════════════════════════════

class PseudoThermalWorker(Worker):
    """后台批量伪热红外增强"""

    def __init__(
        self,
        image_paths: list[Path],
        output_dir: Path,
        config: PseudoThermalConfig,
        seed: int,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.config = config
        self.seed = seed

    def do_work(self) -> dict:
        aug = PseudoThermalAug(self.config)
        total = len(self.image_paths)
        success = 0
        failed: list[str] = []

        for i, path in enumerate(self.image_paths):
            img = cv2.imread(str(path))
            if img is None:
                failed.append(f"{path.name}: 无法读取")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                result = aug.augment(img_rgb, seed=self.seed + i if self.seed else None)
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")
                continue

            out_path = self.output_dir / f"{path.stem}_thermal.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            success += 1
            self.progress.emit(int((i + 1) / total * 100))

        return {"total": total, "success": success, "failed": failed}


# ═══════════════════════════════════════════════════════════════════════
# 面板
# ═══════════════════════════════════════════════════════════════════════

class PseudoThermalPanel(QWidget):
    """伪热红外增强面板"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("pseudo_thermal_panel")
        self._worker: PseudoThermalWorker | None = None
        self._aug = PseudoThermalAug()
        self._current_result: np.ndarray | None = None

        self._setup_ui()
        self._load_settings()

    # ── UI 搭建 ──

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("伪热红外增强 (Pseudo-Thermal)"))

        # ---- 路径 ----
        layout.addWidget(StrongBodyLabel("输入输出"))
        path_card = CardWidget()
        path_form = QFormLayout(path_card)

        self.input_browser = PathBrowser(
            label="", mode="file",
            placeholder="选择 RGB 图像...",
            config_key="pt_input",
        )
        self.input_browser.path_changed.connect(self._on_input_changed)
        path_form.addRow(BodyLabel("输入图像:"), self.input_browser)

        self.output_browser = PathBrowser(
            label="", mode="dir",
            placeholder="输出目录（默认输入同目录）...",
            config_key="pt_output",
        )
        path_form.addRow(BodyLabel("输出目录:"), self.output_browser)

        layout.addWidget(path_card)

        # ---- 参数 ----
        layout.addWidget(StrongBodyLabel("增强参数"))
        param_card = CardWidget()
        param_layout = QVBoxLayout(param_card)

        row1 = QHBoxLayout()
        row1.addWidget(BodyLabel("随机种子:"))
        self.seed_spin = SpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        self.seed_spin.setToolTip("0=每次随机, 其他值=固定可复现")
        self.seed_spin.valueChanged.connect(lambda v: set_int("pt_seed", v))
        row1.addWidget(self.seed_spin)

        self.seed_random_check = CheckBox("随机 (不固定)")
        self.seed_random_check.stateChanged.connect(
            lambda: set_bool("pt_seed_random", self.seed_random_check.isChecked())
        )
        row1.addWidget(self.seed_random_check)
        row1.addStretch()
        param_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(BodyLabel("模糊概率:"))
        self.blur_p_spin = DoubleSpinBox()
        self.blur_p_spin.setRange(0.0, 1.0)
        self.blur_p_spin.setSingleStep(0.1)
        self.blur_p_spin.setValue(0.7)
        self.blur_p_spin.setToolTip("对图像施加高斯模糊的概率")
        self.blur_p_spin.valueChanged.connect(lambda v: set_float("pt_blur_p", v))
        row2.addWidget(self.blur_p_spin)

        row2.addSpacing(16)
        row2.addWidget(BodyLabel("模糊核上限:"))
        self.blur_k_spin = SpinBox()
        self.blur_k_spin.setRange(1, 11)
        self.blur_k_spin.setSingleStep(2)
        self.blur_k_spin.setValue(4)
        self.blur_k_spin.setToolTip("模糊核最大尺寸 (奇数)")
        self.blur_k_spin.valueChanged.connect(lambda v: set_int("pt_blur_k", v))
        row2.addWidget(self.blur_k_spin)
        row2.addStretch()
        param_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.hsv_check = CheckBox("HSV 微调")
        self.hsv_check.setChecked(True)
        self.hsv_check.stateChanged.connect(
            lambda: set_bool("pt_hsv", self.hsv_check.isChecked())
        )
        row3.addWidget(self.hsv_check)

        row3.addWidget(BodyLabel("色调范围:"))
        self.hue_spin = SpinBox()
        self.hue_spin.setRange(0, 180)
        self.hue_spin.setValue(90)
        self.hue_spin.valueChanged.connect(lambda v: set_int("pt_hue", v))
        row3.addWidget(self.hue_spin)

        row3.addWidget(BodyLabel("明度范围:"))
        self.val_spin = SpinBox()
        self.val_spin.setRange(0, 100)
        self.val_spin.setValue(30)
        self.val_spin.valueChanged.connect(lambda v: set_int("pt_val", v))
        row3.addWidget(self.val_spin)
        row3.addStretch()
        param_layout.addLayout(row3)

        layout.addWidget(param_card)

        # ---- 预览 ----
        layout.addWidget(StrongBodyLabel("预览"))
        preview_card = CardWidget()
        preview_layout = QHBoxLayout(preview_card)

        # 原图
        left_col = QVBoxLayout()
        left_col.addWidget(BodyLabel("原图 (RGB)"))
        self.original_viewer = ImageViewer()
        self.original_viewer.setMinimumHeight(250)
        left_col.addWidget(self.original_viewer, 1)
        preview_layout.addLayout(left_col, 1)

        # → 箭头
        arrow_label = BodyLabel("→")
        arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(arrow_label)

        # 结果
        right_col = QVBoxLayout()
        right_col.addWidget(BodyLabel("伪热红外"))
        self.result_viewer = ImageViewer()
        self.result_viewer.setMinimumHeight(250)
        right_col.addWidget(self.result_viewer, 1)
        preview_layout.addLayout(right_col, 1)

        layout.addWidget(preview_card)

        # ---- 操作按钮 ----
        btn_row = QHBoxLayout()
        self.status_label = BodyLabel("")
        btn_row.addWidget(self.status_label, 1)

        self.preview_btn = PrimaryPushButton("生成预览")
        self.preview_btn.clicked.connect(self._on_preview)
        btn_row.addWidget(self.preview_btn)

        self.save_btn = PushButton("保存当前结果")
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setEnabled(False)
        btn_row.addWidget(self.save_btn)

        self.batch_btn = PushButton("批量处理")
        self.batch_btn.clicked.connect(self._on_batch)
        btn_row.addWidget(self.batch_btn)

        layout.addLayout(btn_row)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addStretch()

    # ── 插槽 ──

    def _on_input_changed(self, path: str) -> None:
        if not path:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        self.original_viewer.set_image(pixmap)
        # 自动更新预览
        self._update_preview()

    def _on_preview(self) -> None:
        self._update_preview()

    def _on_save(self) -> None:
        if self._current_result is None:
            error("无结果", "请先生成预览", self)
            return

        input_path = Path(self.input_browser.path)
        out_dir_s = self.output_browser.path.strip()
        if out_dir_s:
            out_dir = Path(out_dir_s)
        else:
            out_dir = input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{input_path.stem}_thermal.jpg"
        cv2.imwrite(str(out_path),
                    cv2.cvtColor(self._current_result, cv2.COLOR_RGB2BGR))
        self.status_label.setText(f"已保存: {out_path.name}")
        logger.info(f"Saved pseudo-thermal → {out_path}")

    def _on_batch(self) -> None:
        input_path = self.input_browser.path.strip()
        out_dir_s = self.output_browser.path.strip()

        if not input_path:
            error("路径错误", "请先选择一张图片（批量将处理同目录所有图片）", self)
            return

        src_dir = Path(input_path).parent
        out_dir = Path(out_dir_s) if out_dir_s else src_dir / "pseudo_thermal"
        out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in src_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )
        if not image_paths:
            error("找不到图片", f"目录 {src_dir} 中没有支持的图片文件", self)
            return

        cfg = self._build_config()
        seed = 0 if self.seed_random_check.isChecked() else self.seed_spin.value()

        self.batch_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText(f"批量处理 {len(image_paths)} 张...")

        self._worker = PseudoThermalWorker(image_paths, out_dir, cfg, seed)
        self._worker.finished.connect(self._on_batch_done)
        self._worker.error.connect(self._on_batch_error)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.start()

    def _on_batch_done(self, result: dict) -> None:
        self.batch_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.save_btn.setEnabled(self._current_result is not None)
        self.progress.setVisible(False)

        failed = result["failed"]
        msg = f"成功: {result['success']}/{result['total']}"
        if failed:
            msg += f"\n失败 ({len(failed)}):\n" + "\n".join(failed[:10])
        self.status_label.setText(msg)
        info("批量处理完成", msg, self)

    def _on_batch_error(self, err: str) -> None:
        self.batch_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("处理失败")
        error("批量处理失败", err, self)

    # ── 内部 ──

    def _update_preview(self) -> None:
        path = self.input_browser.path.strip()
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cfg = self._build_config()
        seed = 0 if self.seed_random_check.isChecked() else self.seed_spin.value()
        self._aug = PseudoThermalAug(cfg)
        self._current_result = self._aug.augment(img_rgb, seed=seed)

        h, w, _ = self._current_result.shape
        qimg = self._ndarray_to_qpixmap(self._current_result)
        self.result_viewer.set_image(qimg)
        self.save_btn.setEnabled(True)
        self.status_label.setText(f"预览已更新 ({w}×{h})")

    def _build_config(self) -> PseudoThermalConfig:
        return PseudoThermalConfig(
            blur_enabled=True,
            blur_p=self.blur_p_spin.value(),
            blur_kernel_range=(2, self.blur_k_spin.value()),
            hsv_enabled=self.hsv_check.isChecked(),
            hsv_p=0.9,
            hue_shift=self.hue_spin.value(),
            sat_shift=30,
            val_shift=self.val_spin.value(),
        )

    @staticmethod
    def _ndarray_to_qpixmap(img: np.ndarray) -> QPixmap:
        """RGB uint8 ndarray → QPixmap"""
        h, w, c = img.shape
        from PySide6.QtGui import QImage
        qimg = QImage(img.data, w, h, c * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    # ── 持久化 ──

    def _load_settings(self) -> None:
        self.input_browser.path = get_str("pt_input")
        self.output_browser.path = get_str("pt_output")
        self.seed_spin.setValue(get_int("pt_seed", 42))
        self.seed_random_check.setChecked(get_bool("pt_seed_random", False))
        self.blur_p_spin.setValue(get_float("pt_blur_p", 0.7))
        self.blur_k_spin.setValue(get_int("pt_blur_k", 4))
        self.hsv_check.setChecked(get_bool("pt_hsv", True))
        self.hue_spin.setValue(get_int("pt_hue", 90))
        self.val_spin.setValue(get_int("pt_val", 30))
