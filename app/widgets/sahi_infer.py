"""SAHI 切片推理面板"""
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
    CardWidget, CheckBox, ComboBox, DoubleSpinBox, SpinBox,
)

from app.services.sahi_inference import SahiConfig, SahiInferenceService
from app.services.label_reader import IMAGE_EXTS
from app.utils.config import get_str, set_str, get_float, set_float, get_int, set_int, get_bool, set_bool
from app.utils.message import error, info
from app.utils.worker import Worker
from app.widgets.image_browser import ImageBrowser


class SahiInferWorker(Worker):
    """后台 SAHI 推理（单张）"""

    def __init__(self, service: SahiInferenceService, image_path: Path) -> None:
        super().__init__()
        self.service = service
        self.image_path = image_path

    def do_work(self) -> list[dict]:
        return self.service.infer_image(self.image_path)


class SahiFolderWorker(Worker):
    """后台 SAHI 批量推理"""

    def __init__(self, service: SahiInferenceService, image_paths: list[Path], output_dir: Path, save_vis: bool,
                 save_txt: bool) -> None:
        super().__init__()
        self.service = service
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.save_vis = save_vis
        self.save_txt = save_txt

    def do_work(self) -> dict:
        """返回 {"total": int, "annotations": dict[str, list[dict]]}"""
        total = len(self.image_paths)
        all_anns: dict[str, list[dict]] = {}
        for i, img_path in enumerate(self.image_paths):
            annotations = self.service.infer_image(img_path)
            all_anns[str(img_path)] = annotations

            # 保存可视化
            if self.save_vis:
                vis_dir = self.output_dir / "visuals"
                vis_dir.mkdir(parents=True, exist_ok=True)
                self.service.save_visualization(img_path, vis_dir / img_path.name)

            # 保存 YOLO TXT
            if self.save_txt:
                txt_dir = self.output_dir / "labels"
                txt_dir.mkdir(parents=True, exist_ok=True)
                self._save_yolo_txt(txt_dir / f"{img_path.stem}.txt", annotations)

            self.progress.emit(int((i + 1) / total * 100))
        return {"total": total, "annotations": all_anns}

    def _save_yolo_txt(self, path: Path, annotations: list[dict]) -> None:
        """保存 YOLO 格式标签"""
        lines = []
        for ann in annotations:
            if ann["type"] != "bbox":
                continue
            rect = ann["rect"]
            # 需要原图尺寸——这里简化处理，存相对坐标
            lines.append(
                f"{ann['class_id']} "
                f"{(rect.x() + rect.width() / 2):.6f} "
                f"{(rect.y() + rect.height() / 2):.6f} "
                f"{rect.width():.6f} "
                f"{rect.height():.6f}"
            )
        path.write_text("\n".join(lines), encoding="utf-8")


class SahiInferPanel(QWidget):
    """SAHI 切片推理面板"""

    status_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("sahi_infer_panel")
        self._service: SahiInferenceService | None = None
        self._worker: Worker | None = None
        self._image_files: list[Path] = []

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("SAHI 切片推理"))

        # ---- 模型 ----
        layout.addWidget(StrongBodyLabel("模型设置"))
        model_card = CardWidget()
        model_form = QFormLayout(model_card)

        model_row = QHBoxLayout()
        self.model_edit = LineEdit()
        self.model_edit.setPlaceholderText("选择 YOLO .pt 模型...")
        self.model_edit.setReadOnly(True)

        self.model_type_combo = ComboBox()
        self.model_type_combo.addItems([
            "ultralytics", "yolov5", "mmdet", "detectron2", "torchvision",
        ])
        self.model_type_combo.setCurrentText("ultralytics")
        self.model_type_combo.currentTextChanged.connect(
            lambda v: set_str("sahi_model_type", v)
        )

        model_btn = PushButton("📁")
        model_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self.model_edit, 1)
        model_row.addWidget(self.model_type_combo)
        model_row.addWidget(model_btn)
        model_form.addRow(BodyLabel("模型文件:"), model_row)
        
        layout.addWidget(model_card)

        # ---- 切片参数 ----
        layout.addWidget(StrongBodyLabel("切片参数"))
        slice_card = CardWidget()
        slice_form = QFormLayout(slice_card)

        # 切片宽高
        wh_row = QHBoxLayout()
        wh_row.addWidget(BodyLabel("宽:"))
        self.slice_w_spin = SpinBox()
        self.slice_w_spin.setRange(320, 2048)
        self.slice_w_spin.setSingleStep(32)
        self.slice_w_spin.setValue(640)
        self.slice_w_spin.valueChanged.connect(lambda v: set_int("sahi_slice_w", v))
        wh_row.addWidget(self.slice_w_spin)
        wh_row.addWidget(BodyLabel("高:"))
        self.slice_h_spin = SpinBox()
        self.slice_h_spin.setRange(320, 2048)
        self.slice_h_spin.setSingleStep(32)
        self.slice_h_spin.setValue(640)
        self.slice_h_spin.valueChanged.connect(lambda v: set_int("sahi_slice_h", v))
        wh_row.addWidget(self.slice_h_spin)
        wh_row.addStretch()
        slice_form.addRow(BodyLabel("切片尺寸:"), wh_row)

        # 重叠率
        ol_row = QHBoxLayout()
        ol_row.addWidget(BodyLabel("水平:"))
        self.overlap_w_spin = DoubleSpinBox()
        self.overlap_w_spin.setRange(0.0, 0.9)
        self.overlap_w_spin.setSingleStep(0.05)
        self.overlap_w_spin.setDecimals(2)
        self.overlap_w_spin.setValue(0.25)
        self.overlap_w_spin.valueChanged.connect(lambda v: set_float("sahi_overlap_w", v))
        ol_row.addWidget(self.overlap_w_spin)
        ol_row.addWidget(BodyLabel("垂直:"))
        self.overlap_h_spin = DoubleSpinBox()
        self.overlap_h_spin.setRange(0.0, 0.9)
        self.overlap_h_spin.setSingleStep(0.05)
        self.overlap_h_spin.setDecimals(2)
        self.overlap_h_spin.setValue(0.25)
        self.overlap_h_spin.valueChanged.connect(lambda v: set_float("sahi_overlap_h", v))
        ol_row.addWidget(self.overlap_h_spin)
        ol_row.addStretch()
        slice_form.addRow(BodyLabel("重叠率:"), ol_row)

        layout.addWidget(slice_card)

        # ---- 推理参数 ----
        layout.addWidget(StrongBodyLabel("推理参数"))
        infer_card = CardWidget()
        infer_form = QFormLayout(infer_card)

        thr_row = QHBoxLayout()
        thr_row.addWidget(BodyLabel("Conf:"))
        self.conf_spin = DoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.25)
        self.conf_spin.valueChanged.connect(lambda v: set_float("sahi_conf", v))
        thr_row.addWidget(self.conf_spin)
        thr_row.addWidget(BodyLabel("IoU:"))
        self.iou_spin = DoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.7)
        self.iou_spin.valueChanged.connect(lambda v: set_float("sahi_iou", v))

        self.standard_check = CheckBox("同时执行整图预测")
        self.standard_check.setChecked(True)
        self.standard_check.stateChanged.connect(
            lambda: set_bool("sahi_standard_pred", self.standard_check.isChecked()))

        thr_row.addWidget(self.iou_spin)
        thr_row.addWidget(self.standard_check)
        thr_row.addStretch()
        infer_form.addRow(BodyLabel("阈值:"), thr_row)

        layout.addWidget(infer_card)

        # ---- 输入 / 输出 ----
        layout.addWidget(StrongBodyLabel("输入 / 输出"))
        io_card = CardWidget()
        io_form = QFormLayout(io_card)

        # 输入
        in_row = QHBoxLayout()
        self.input_edit = LineEdit()
        self.input_edit.setPlaceholderText("选择图片文件或文件夹...")
        self.input_edit.setReadOnly(True)
        self.input_edit.textChanged.connect(lambda v: set_str("sahi_input", v))
        in_row.addWidget(self.input_edit, 1)
        file_btn = PushButton("📄")
        file_btn.setToolTip("选择单张图片")
        file_btn.clicked.connect(self._browse_file)
        in_row.addWidget(file_btn)
        folder_btn = PushButton("📁")
        folder_btn.setToolTip("选择文件夹")
        folder_btn.clicked.connect(self._browse_folder)
        in_row.addWidget(folder_btn)
        io_form.addRow(BodyLabel("输入:"), in_row)

        # 输出
        out_row = QHBoxLayout()
        self.output_edit = LineEdit()
        self.output_edit.setPlaceholderText("选择输出目录...")
        out_btn = PushButton("📁")
        out_btn.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(out_btn)
        io_form.addRow(BodyLabel("输出:"), out_row)

        # 输出选项
        opt_row = QHBoxLayout()
        self.vis_check = CheckBox("保存可视化")
        self.vis_check.setChecked(True)
        self.vis_check.stateChanged.connect(lambda: set_bool("sahi_save_vis", self.vis_check.isChecked()))
        opt_row.addWidget(self.vis_check)
        self.txt_check = CheckBox("保存 YOLO TXT")
        self.txt_check.setChecked(True)
        self.txt_check.stateChanged.connect(lambda: set_bool("sahi_save_txt", self.txt_check.isChecked()))
        opt_row.addWidget(self.txt_check)
        opt_row.addStretch()
        io_form.addRow(BodyLabel(""), opt_row)

        layout.addWidget(io_card)

        # ---- 操作按钮 ----
        btn_row = QHBoxLayout()
        self.status_label = BodyLabel("")
        btn_row.addWidget(self.status_label)
        btn_row.addStretch()

        self.preview_btn = PushButton("预览单张")
        self.preview_btn.clicked.connect(self._on_preview)
        btn_row.addWidget(self.preview_btn)

        self.batch_btn = PrimaryPushButton("批量推理")
        self.batch_btn.clicked.connect(self._on_batch)
        btn_row.addWidget(self.batch_btn)
        layout.addLayout(btn_row)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # ---- 预览区 ----
        self.browser = ImageBrowser()
        self.browser.image_selected.connect(self._on_browser_selected)
        layout.addWidget(self.browser, 1)

    # ---- 构建 service ----
    def _build_service(self) -> SahiInferenceService | None:
        model = self.model_edit.text().strip()
        if not model or not Path(model).is_file():
            return None
        return SahiInferenceService(SahiConfig(
            model_path=Path(model),
            model_type=self.model_type_combo.currentText(),
            confidence=self.conf_spin.value(),
            iou=self.iou_spin.value(),
            slice_width=self.slice_w_spin.value(),
            slice_height=self.slice_h_spin.value(),
            overlap_width_ratio=self.overlap_w_spin.value(),
            overlap_height_ratio=self.overlap_h_spin.value(),
            perform_standard_pred=self.standard_check.isChecked(),
        ))

    # ---- 路径 ----
    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型", self.model_edit.text(), "Model Files (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.model_edit.setText(path)
            set_str("sahi_model_path", path)

    def _browse_file(self) -> None:
        # 先试选文件
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if path:
            self.input_edit.setText(path)

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.input_edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_edit.setText(path)
            set_str("sahi_output", path)

    # ---- 预览 ----
    def _on_preview(self) -> None:
        service = self._build_service()
        if service is None:
            error("错误", "请选择有效的模型文件", self)
            return

        input_path = Path(self.input_edit.text().strip())
        if not input_path.is_file():
            # 如果是文件夹，取第一张
            imgs = sorted(f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            if not imgs:
                error("错误", "未找到图片文件", self)
                return
            input_path = imgs[0]

        self.status_label.setText("正在加载模型并推理...")
        self.preview_btn.setEnabled(False)

        self._worker = SahiInferWorker(service, input_path)
        self._worker.finished.connect(self._on_preview_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_preview_done(self, annotations):
        self.preview_btn.setEnabled(True)

        input_path = Path(self.input_edit.text().strip())
        if input_path.is_file():
            path = input_path
        else:
            imgs = sorted(f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            path = imgs[0] if imgs else None

        if path:
            self.browser.set_images([path])
            self.browser.show_annotations(annotations)
            self._all_annotations = {str(path): annotations}

        self.status_label.setText(f"检测到 {len(annotations)} 个目标")

    # ---- 批量 ----
    def _on_batch(self) -> None:
        self._all_annotations = {}

        service = self._build_service()
        if service is None:
            error("错误", "请选择有效的模型文件", self)
            return

        input_path = Path(self.input_edit.text().strip())
        output_dir = Path(self.output_edit.text().strip())

        if input_path.is_file():
            images = [input_path]
        elif input_path.is_dir():
            images = sorted(f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        else:
            error("错误", "请选择有效的输入路径", self)
            return

        if not images:
            error("错误", "未找到图片文件", self)
            return
        self._last_images = images
        output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText(f"批量推理 {len(images)} 张...")

        self._last_output_dir = output_dir
        self._last_save_vis = self.vis_check.isChecked()

        self._worker = SahiFolderWorker(
            service, images, output_dir,
            self.vis_check.isChecked(), self.txt_check.isChecked(),
        )
        self._worker.progress.connect(self.progress.setValue)
        self._worker.finished.connect(self._on_batch_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_batch_done(self, result: dict) -> None:
        total = result["total"]
        self._all_annotations = result["annotations"]

        self.batch_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(f"✅ 完成 {total} 张")

        if self._last_images:
            self.browser.set_images(self._last_images)
        info("批量推理完成", f"共处理 {total} 张图片", self)

    def _on_browser_selected(self, idx: int, path: Path) -> None:
        """点击缩略图 → 取缓存标注叠加"""
        key = str(path)
        anns = self._all_annotations.get(key, []) if hasattr(self, '_all_annotations') else []
        self.browser.show_annotations(anns)

    def _on_error(self, err_msg: str) -> None:
        self.preview_btn.setEnabled(True)
        self.batch_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("❌ 推理失败")
        error("SAHI 推理失败", err_msg, self)

    # ---- 持久化 ----

    def _load_settings(self) -> None:
        self.model_edit.setText(get_str("sahi_model_path", ""))
        self.model_type_combo.setCurrentText(get_str("sahi_model_type", "ultralytics"))
        self.slice_w_spin.setValue(get_int("sahi_slice_w", 640))
        self.slice_h_spin.setValue(get_int("sahi_slice_h", 640))
        self.overlap_w_spin.setValue(get_float("sahi_overlap_w", 0.25))
        self.overlap_h_spin.setValue(get_float("sahi_overlap_h", 0.25))
        self.conf_spin.setValue(get_float("sahi_conf", 0.25))
        self.iou_spin.setValue(get_float("sahi_iou", 0.7))
        self.standard_check.setChecked(get_bool("sahi_standard_pred", True))
        self.input_edit.setText(get_str("sahi_input", ""))
        self.output_edit.setText(get_str("sahi_output", ""))
        self.vis_check.setChecked(get_bool("sahi_save_vis", True))
        self.txt_check.setChecked(get_bool("sahi_save_txt", True))
