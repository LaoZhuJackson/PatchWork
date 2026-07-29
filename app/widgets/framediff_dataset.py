"""Frame Dynamics 数据集生成面板"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, ProgressBar,
    BodyLabel, StrongBodyLabel, SubtitleLabel,
    CardWidget, SpinBox, ComboBox, CheckBox,
)

from app.services.framediff import generate_framediff_dataset
from app.utils.config import (
    get_str, set_str, get_int, set_int, get_bool, set_bool,
)
from app.utils.message import info, error
from app.utils.worker import Worker
from app.widgets.path_browser import PathBrowser
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FrameDiffWorker(Worker):
    """后台生成 Frame Dynamics 数据集"""

    def __init__(
            self,
            dataset_dir: Path,
            output_dir: Path,
            gap_a: int,
            gap_b: int,
            registration: str,
            blur_kernel: int,
            image_format: str,
            jpg_quality: int,
            splits: list[str],
            overwrite: bool,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.gap_a = gap_a
        self.gap_b = gap_b
        self.registration = registration
        self.blur_kernel = blur_kernel
        self.image_format = image_format
        self.jpg_quality = jpg_quality
        self.splits = splits
        self.overwrite = overwrite

    def do_work(self) ->dict:
        return generate_framediff_dataset(
            self.dataset_dir,
            self.output_dir,
            gaps=(self.gap_a, self.gap_b),
            registration=self.registration,
            blur_kernel=self.blur_kernel,
            image_format=self.image_format,
            jpg_quality=self.jpg_quality,
            splits=tuple(self.splits),
            overwrite=self.overwrite,
            progress_callback=lambda p: self.progress.emit(p),
        )

class FrameDiffDatasetPanel(QWidget):
    """Frame Dynamics 数据集生成面板"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("framediff_dataset_panel")
        self._worker: FrameDiffWorker | None = None

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("Frame Dynamics 数据集生成"))

        # ---- 路径 ----
        layout.addWidget(StrongBodyLabel("路径设置"))
        path_card = CardWidget()
        path_form = QFormLayout(path_card)

        self.dataset_browser = PathBrowser(
            label="", mode="dir",
            placeholder="YOLO 数据集根目录（含 images/ 和 labels/）...",
            config_key="fd_dataset_dir",
        )
        path_form.addRow(BodyLabel("数据集目录:"), self.dataset_browser)

        self.output_browser = PathBrowser(
            label="", mode="dir",
            placeholder="输出目录（留空则自动生成在同级目录）...",
            config_key="fd_output_dir",
        )
        path_form.addRow(BodyLabel("输出目录:"), self.output_browser)

        layout.addWidget(path_card)

        # ---- 帧差参数 ----
        layout.addWidget(StrongBodyLabel("帧差参数"))
        param_card = CardWidget()
        param_layout = QVBoxLayout(param_card)

        gap_row = QHBoxLayout()
        gap_row.addWidget(BodyLabel("历史帧间隔:"))
        self.gap_a_spin = SpinBox()
        self.gap_a_spin.setRange(1, 100)
        self.gap_a_spin.setValue(1)
        self.gap_a_spin.setToolTip("第一张历史帧距当前帧的间隔")
        self.gap_a_spin.valueChanged.connect(lambda v: set_int("fd_gap_a", v))
        gap_row.addWidget(self.gap_a_spin)
        gap_row.addWidget(BodyLabel("、"))
        self.gap_b_spin = SpinBox()
        self.gap_b_spin.setRange(1, 100)
        self.gap_b_spin.setValue(2)
        self.gap_b_spin.setToolTip("第二张历史帧距当前帧的间隔")
        self.gap_b_spin.valueChanged.connect(lambda v: set_int("fd_gap_b", v))
        gap_row.addWidget(self.gap_b_spin)
        gap_row.addStretch()
        param_layout.addLayout(gap_row)

        mid_row = QHBoxLayout()
        mid_row.addWidget(BodyLabel("配准方式:"))
        self.reg_combo = ComboBox()
        self.reg_combo.addItems(["不配准", "phase"])
        self.reg_combo.setCurrentText("不配准")
        self.reg_combo.currentTextChanged.connect(
            lambda v: set_str("fd_registration", v)
        )
        mid_row.addWidget(self.reg_combo)

        mid_row.addSpacing(16)
        mid_row.addWidget(BodyLabel("模糊核:"))
        self.blur_spin = SpinBox()
        self.blur_spin.setRange(0, 31)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setValue(0)
        self.blur_spin.setToolTip("0=关闭，正奇数")
        self.blur_spin.valueChanged.connect(lambda v: set_int("fd_blur", v))
        mid_row.addWidget(self.blur_spin)
        mid_row.addStretch()
        param_layout.addLayout(mid_row)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(BodyLabel("输出格式:"))
        self.fmt_combo = ComboBox()
        self.fmt_combo.addItems(["jpg", "png"])
        self.fmt_combo.setCurrentText("jpg")
        self.fmt_combo.currentTextChanged.connect(
            lambda v: set_str("fd_format", v)
        )
        fmt_row.addWidget(self.fmt_combo)

        fmt_row.addSpacing(16)
        fmt_row.addWidget(BodyLabel("JPG 质量:"))
        self.quality_spin = SpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)
        self.quality_spin.valueChanged.connect(lambda v: set_int("fd_quality", v))
        fmt_row.addWidget(self.quality_spin)
        fmt_row.addStretch()
        param_layout.addLayout(fmt_row)

        layout.addWidget(param_card)

        # ---- 选项 ----
        layout.addWidget(StrongBodyLabel("选项"))
        opt_card = CardWidget()
        opt_layout = QHBoxLayout(opt_card)

        self.train_check = CheckBox("train")
        self.train_check.setChecked(True)
        self.train_check.stateChanged.connect(
            lambda: set_bool("fd_split_train", self.train_check.isChecked())
        )
        opt_layout.addWidget(self.train_check)

        self.val_check = CheckBox("val")
        self.val_check.setChecked(True)
        self.val_check.stateChanged.connect(
            lambda: set_bool("fd_split_val", self.val_check.isChecked())
        )
        opt_layout.addWidget(self.val_check)

        opt_layout.addSpacing(16)
        self.overwrite_check = CheckBox("覆盖已有文件")
        self.overwrite_check.stateChanged.connect(
            lambda: set_bool("fd_overwrite", self.overwrite_check.isChecked())
        )
        opt_layout.addWidget(self.overwrite_check)
        opt_layout.addStretch()

        layout.addWidget(opt_card)

        # ---- 操作 ----
        btn_row = QHBoxLayout()
        self.status_label = BodyLabel("")
        btn_row.addWidget(self.status_label, 1)
        self.run_btn = PrimaryPushButton("开始生成")
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)
        layout.addLayout(btn_row)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addStretch()

    # ---- 运行 ----

    def _on_run(self) -> None:
        dataset_dir = Path(self.dataset_browser.path)
        out_path = self.output_browser.path.strip()
        output_dir = Path(out_path) if out_path else None

        if not dataset_dir.is_dir():
            error("路径错误", "请选择有效的数据集目录(YOLO目录格式)", self)
            return

        gap_a = self.gap_a_spin.value()
        gap_b = self.gap_b_spin.value()
        if gap_a <= 0 or gap_b <= 0 or gap_a == gap_b:
            error("参数错误", "两个历史帧间隔必须为正整数且不能相同", self)
            return

        if output_dir is None:
            suffix = f"framediff_g{gap_a}_g{gap_b}"
            output_dir = dataset_dir.parent / f"{dataset_dir.name}_{suffix}"
        elif output_dir == dataset_dir:
            error("路径错误", "输出目录不能与数据集目录相同", self)
            return

        splits = []
        if self.train_check.isChecked():
            splits.append("train")
        if self.val_check.isChecked():
            splits.append("val")
        if not splits:
            error("选项错误", "请至少选择一个划分 (train/val)", self)
            return

        blur_kernel = self.blur_spin.value()
        if blur_kernel > 0 and blur_kernel % 2 == 0:
            blur_kernel += 1

        self.run_btn.setEnabled(False)
        self._set_inputs_enabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("正在生成...")

        self._worker = FrameDiffWorker(
            dataset_dir, output_dir, gap_a, gap_b,
            self.reg_combo.currentText(),
            blur_kernel,
            self.fmt_combo.currentText(),
            self.quality_spin.value(),
            splits,
            self.overwrite_check.isChecked(),
        )
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.start()

    def _on_done(self, all_stats: dict) -> None:
        self._set_inputs_enabled(True)
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("生成完成")

        parts = ["Frame Dynamics 数据集生成完成\n"]
        for split, s in all_stats.items():
            parts.append(
                f"  [{split}] 成功 {s['success']}/{s['total']}"
                f"（缺历史帧 {s['missing_history']}）"
            )
        info("完成", "\n".join(parts), self)

    def _on_error(self, err: str) -> None:
        self._set_inputs_enabled(True)
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("生成失败")
        error("生成失败", err, self)

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.dataset_browser.setEnabled(enabled)
        self.output_browser.setEnabled(enabled)
        self.gap_a_spin.setEnabled(enabled)
        self.gap_b_spin.setEnabled(enabled)
        self.reg_combo.setEnabled(enabled)
        self.blur_spin.setEnabled(enabled)
        self.fmt_combo.setEnabled(enabled)
        self.quality_spin.setEnabled(enabled)
        self.train_check.setEnabled(enabled)
        self.val_check.setEnabled(enabled)
        self.overwrite_check.setEnabled(enabled)

    # ---- 持久化 ----

    def _load_settings(self) -> None:
        self.dataset_browser.path = get_str("fd_dataset_dir")
        self.output_browser.path = get_str("fd_output_dir")
        self.gap_a_spin.setValue(get_int("fd_gap_a", 1))
        self.gap_b_spin.setValue(get_int("fd_gap_b", 2))
        self.reg_combo.setCurrentText(get_str("fd_registration", "不配准"))
        self.blur_spin.setValue(get_int("fd_blur", 0))
        self.fmt_combo.setCurrentText(get_str("fd_format", "jpg"))
        self.quality_spin.setValue(get_int("fd_quality", 95))
        self.train_check.setChecked(get_bool("fd_split_train", True))
        self.val_check.setChecked(get_bool("fd_split_val", True))
        self.overwrite_check.setChecked(get_bool("fd_overwrite", False))
