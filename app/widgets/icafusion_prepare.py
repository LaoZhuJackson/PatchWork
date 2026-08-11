"""ICAFusion 数据准备面板：IR warp 对齐 + VIS 裁剪到重叠区"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, ProgressBar,
    BodyLabel, StrongBodyLabel, SubtitleLabel,
    CardWidget, SpinBox, CheckBox,
)

from app.services.icafusion_prepare import prepare_icafusion_pairs
from app.utils.config import (
    get_str, set_str, get_int, set_int, get_bool, set_bool,
)
from app.utils.message import info, error
from app.utils.worker import Worker
from app.widgets.path_browser import PathBrowser
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ICAFusionPrepareWorker(Worker):
    """后台对齐 IR-VIS 数据"""

    def __init__(
        self,
        ir_dir: Path,
        vis_dir: Path,
        output_dir: Path,
        npz_path: str,
        target_w: int,
        target_h: int,
        overwrite: bool,
    ) -> None:
        super().__init__()
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.output_dir = output_dir
        self.npz_path = npz_path
        self.target_w = target_w
        self.target_h = target_h
        self.overwrite = overwrite

    def do_work(self) -> dict:
        return prepare_icafusion_pairs(
            ir_dir=str(self.ir_dir),
            vis_dir=str(self.vis_dir),
            output_dir=str(self.output_dir),
            npz_path=self.npz_path or None,  # None → 用内置固定 H
            target_w=self.target_w,
            target_h=self.target_h,
            overwrite=self.overwrite,
            progress_callback=lambda p: self.progress.emit(p),
        )


class ICAFusionPreparePanel(QWidget):
    """ICAFusion 数据准备面板"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("icafusion_prepare_panel")
        self._worker: ICAFusionPrepareWorker | None = None

        self._setup_ui()
        self._load_settings()
        self._on_fixed_h_toggled()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("ICAFusion 数据准备（IR 对齐 + VIS 裁剪）"))

        # ---- 路径 ----
        layout.addWidget(StrongBodyLabel("路径设置"))
        path_card = CardWidget()
        path_form = QFormLayout(path_card)

        self.ir_browser = PathBrowser(
            label="", mode="dir",
            placeholder="IR 目录（..._T_000000.jpg）...",
            config_key="icp_ir_dir",
        )
        path_form.addRow(BodyLabel("IR 目录:"), self.ir_browser)

        self.vis_browser = PathBrowser(
            label="", mode="dir",
            placeholder="VIS 目录（..._V_000000.jpg）...",
            config_key="icp_vis_dir",
        )
        path_form.addRow(BodyLabel("VIS 目录:"), self.vis_browser)

        self.output_browser = PathBrowser(
            label="", mode="dir",
            placeholder="输出目录（生成 visible/ + ir/）...",
            config_key="icp_output_dir",
        )
        path_form.addRow(BodyLabel("输出目录:"), self.output_browser)

        layout.addWidget(path_card)

        # ---- 映射矩阵 ----
        layout.addWidget(StrongBodyLabel("映射矩阵"))
        h_card = CardWidget()
        h_layout = QVBoxLayout(h_card)

        h_row = QHBoxLayout()
        self.fixed_h_check = CheckBox("使用固定映射矩阵（同镜头固定安装）")
        self.fixed_h_check.setChecked(True)
        self.fixed_h_check.setToolTip("默认使用内置固定 H（白天 0803 的 29 个 NPZ 中值，跨视频共用）")
        self.fixed_h_check.toggled.connect(self._on_fixed_h_toggled)
        self.fixed_h_check.toggled.connect(
            lambda v: set_bool("icp_fixed_h", v)
        )
        h_row.addWidget(self.fixed_h_check)
        h_row.addStretch()
        h_layout.addLayout(h_row)

        self.npz_browser = PathBrowser(
            label="", mode="file",
            file_filter="XoFTR NPZ (*.npz);;All Files (*)",
            placeholder="选择 XoFTR 结果 NPZ（未勾选固定 H 时生效）...",
            config_key="icp_npz_path",
        )
        h_layout.addWidget(self.npz_browser)

        layout.addWidget(h_card)

        # ---- 输出参数 ----
        layout.addWidget(StrongBodyLabel("输出参数"))
        param_card = CardWidget()
        param_layout = QHBoxLayout(param_card)

        param_layout.addWidget(BodyLabel("输出尺寸:"))
        self.w_spin = SpinBox()
        self.w_spin.setRange(64, 4096)
        self.w_spin.setValue(1280)
        self.w_spin.valueChanged.connect(lambda v: set_int("icp_w", v))
        param_layout.addWidget(self.w_spin)
        param_layout.addWidget(BodyLabel("×"))
        self.h_spin = SpinBox()
        self.h_spin.setRange(64, 4096)
        self.h_spin.setValue(1024)
        self.h_spin.valueChanged.connect(lambda v: set_int("icp_h", v))
        param_layout.addWidget(self.h_spin)
        param_layout.addSpacing(16)

        self.overwrite_check = CheckBox("覆盖已有文件")
        self.overwrite_check.stateChanged.connect(
            lambda: set_bool("icp_overwrite", self.overwrite_check.isChecked())
        )
        param_layout.addWidget(self.overwrite_check)
        param_layout.addStretch()

        layout.addWidget(param_card)

        # ---- 操作 ----
        btn_row = QHBoxLayout()
        self.status_label = BodyLabel("")
        btn_row.addWidget(self.status_label, 1)
        self.run_btn = PrimaryPushButton("开始处理")
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)
        layout.addLayout(btn_row)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addStretch()

    # ---- 交互 ----

    def _on_fixed_h_toggled(self) -> None:
        """勾选固定 H 时禁用 NPZ 选择器"""
        use_fixed = self.fixed_h_check.isChecked()
        self.npz_browser.setEnabled(not use_fixed)

    def _on_run(self) -> None:
        ir_dir = Path(self.ir_browser.path)
        vis_dir = Path(self.vis_browser.path)
        out_path = self.output_browser.path.strip()

        if not ir_dir.is_dir():
            error("路径错误", "请选择有效的 IR 目录", self)
            return
        if not vis_dir.is_dir():
            error("路径错误", "请选择有效的 VIS 目录", self)
            return

        output_dir = Path(out_path) if out_path else ir_dir.parent / "icafusion_data"

        npz_path = ""
        if not self.fixed_h_check.isChecked():
            npz_path = self.npz_browser.path.strip()
            if not npz_path or not Path(npz_path).is_file():
                error("参数错误", "未勾选固定 H 时，必须选择有效的 NPZ 文件", self)
                return

        target_w = self.w_spin.value()
        target_h = self.h_spin.value()

        self.run_btn.setEnabled(False)
        self._set_inputs_enabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("正在处理...")

        self._worker = ICAFusionPrepareWorker(
            ir_dir, vis_dir, output_dir, npz_path,
            target_w, target_h,
            self.overwrite_check.isChecked(),
        )
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.start()

    def _on_done(self, stats: dict) -> None:
        self._set_inputs_enabled(True)
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("处理完成")

        info(
            "完成",
            f"ICAFusion 数据准备完成\n\n"
            f"总数: {stats['total']}\n"
            f"成功: {stats['success']}\n"
            f"裁剪区为空: {stats['crop_empty']}\n"
            f"读取失败: {stats['read_error']}\n"
            f"写入失败: {stats['write_error']}\n"
            f"其他错误: {stats['other_error']}",
            self,
        )

    def _on_error(self, err: str) -> None:
        self._set_inputs_enabled(True)
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("处理失败")
        error("处理失败", err, self)

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.ir_browser.setEnabled(enabled)
        self.vis_browser.setEnabled(enabled)
        self.output_browser.setEnabled(enabled)
        self.fixed_h_check.setEnabled(enabled)
        if enabled:
            self._on_fixed_h_toggled()  # 恢复 NPZ 可用性
        else:
            self.npz_browser.setEnabled(False)
        self.w_spin.setEnabled(enabled)
        self.h_spin.setEnabled(enabled)
        self.overwrite_check.setEnabled(enabled)

    # ---- 持久化 ----

    def _load_settings(self) -> None:
        self.ir_browser.path = get_str("icp_ir_dir")
        self.vis_browser.path = get_str("icp_vis_dir")
        self.output_browser.path = get_str("icp_output_dir")
        self.npz_browser.path = get_str("icp_npz_path")
        self.fixed_h_check.setChecked(get_bool("icp_fixed_h", True))
        self.w_spin.setValue(get_int("icp_w", 1280))
        self.h_spin.setValue(get_int("icp_h", 1024))
        self.overwrite_check.setChecked(get_bool("icp_overwrite", False))
