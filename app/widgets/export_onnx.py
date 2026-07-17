"""F5: 导出 ONNX 面板"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, LineEdit, ProgressBar,
    BodyLabel, StrongBodyLabel, SubtitleLabel,
    CheckBox, CardWidget, SpinBox,
)

from app.services.exporter import ONNXExporter
from app.utils.config import get_str, set_str, get_bool, set_bool, set_int
from app.utils.message import info, error
from app.utils.worker import Worker


class ExportWorker(Worker):
    """后台导出 ONNX"""

    def __init__(self, model_path: str, output_dir: str, imgsz: int, simplify: bool, dynamic: bool) -> None:
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir
        self.imgsz = imgsz
        self.simplify = simplify
        self.dynamic = dynamic

    def do_work(self) -> Path:
        exporter = ONNXExporter()
        return exporter.export(
            self.model_path,
            self.output_dir or None,
            self.imgsz,
            self.simplify,
            self.dynamic,
            progress_callback=lambda p: self.progress.emit(p)
        )


class ExportONNXPanel(QWidget):
    """导出 ONNX 面板"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("export_onnx_panel")
        self._worker: ExportWorker | None = None

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # ---- 标题 ----
        layout.addWidget(SubtitleLabel("导出 ONNX"))

        # ---- 路径设置 ----
        layout.addWidget(StrongBodyLabel("路径设置"))
        path_card = CardWidget()
        path_form = QFormLayout(path_card)

        model_row = QHBoxLayout()
        self.model_edit = LineEdit()
        self.model_edit.setPlaceholderText("选择 YOLO .pt 模型文件...")
        model_btn = PushButton("📄")
        model_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self.model_edit, 1)
        model_row.addWidget(model_btn)
        path_form.addRow(BodyLabel("模型文件:"), model_row)

        out_row = QHBoxLayout()
        self.out_edit = LineEdit()
        self.out_edit.setPlaceholderText("选择输出目录（留空则保存在模型同级目录）...")
        out_btn = PushButton("📁")
        out_btn.clicked.connect(self._browse_out)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(out_btn)
        path_form.addRow(BodyLabel("输出目录:"), out_row)

        layout.addWidget(path_card)

        # ---- 导出选项 ----
        layout.addWidget(StrongBodyLabel("导出选项"))
        opt_card = CardWidget()
        opt_layout = QVBoxLayout(opt_card)

        imgsz_row = QHBoxLayout()
        imgsz_row.addWidget(BodyLabel("输入尺寸 (imgsz):"))
        self.imgsz_spin = SpinBox()
        self.imgsz_spin.setRange(0, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setToolTip("模型的输入尺寸，与训练时保持一致，设0为自动获取")
        self.imgsz_spin.valueChanged.connect(
            lambda v: set_int("onnx_imgsz", v)
        )
        imgsz_row.addWidget(self.imgsz_spin)
        imgsz_row.addStretch()
        opt_layout.addLayout(imgsz_row)

        self.simplify_check = CheckBox("简化模型（simplify，推荐开启）")
        self.simplify_check.setChecked(True)
        self.simplify_check.stateChanged.connect(
            lambda: set_bool("onnx_simplify", self.simplify_check.isChecked())
        )
        opt_layout.addWidget(self.simplify_check)

        self.dynamic_check = CheckBox("动态输入尺寸（dynamic axes）")
        self.dynamic_check.setChecked(True)
        self.dynamic_check.stateChanged.connect(
            lambda: set_bool("onnx_dynamic", self.dynamic_check.isChecked())
        )
        opt_layout.addWidget(self.dynamic_check)

        layout.addWidget(opt_card)

        # ---- 执行 ----
        self.export_btn = PrimaryPushButton("开始导出")
        self.export_btn.clicked.connect(self._on_export)
        layout.addWidget(self.export_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self.status_label = BodyLabel("")
        layout.addWidget(self.status_label)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addStretch()

    # ---- 事件 ----
    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型", self.model_edit.text(),
            "Model Files (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.model_edit.setText(path)
            set_str("onnx_model_path", path)

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", self.out_edit.text()
        )
        if path:
            self.out_edit.setText(path)
            set_str("onnx_output_dir", path)

    def _on_export(self) -> None:
        model = self.model_edit.text().strip()
        if not model or not Path(model).is_file():
            error("错误", "请选择有效的模型文件", self)
            return

        out = self.out_edit.text().strip()
        simplify = self.simplify_check.isChecked()
        dynamic = self.dynamic_check.isChecked()
        imgsz = self.imgsz_spin.value()

        self.export_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("正在导出...")

        self._worker = ExportWorker(model, out, imgsz, simplify, dynamic)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.start()

    def _on_finished(self, export_path: Path) -> None:
        self.export_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(f"✅ 导出成功: {export_path}")
        info("导出完成", f"ONNX 模型已保存到:\n{export_path}", self)

    def _on_error(self, err_msg: str) -> None:
        self.export_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("❌ 导出失败")
        error("导出失败", err_msg, self)

    # ---- 持久化 ----
    def _load_settings(self) -> None:
        self.model_edit.setText(get_str("onnx_model_path"))
        self.out_edit.setText(get_str("onnx_output_dir"))
        self.simplify_check.setChecked(get_bool("onnx_simplify", True))
        self.dynamic_check.setChecked(get_bool("onnx_dynamic", True))
