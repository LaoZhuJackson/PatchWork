"""推理基准对比面板：多选推理方式 → 同一数据集评测 → 表格对比"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QSizePolicy,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    LineEdit, ProgressBar,
    BodyLabel, StrongBodyLabel, SubtitleLabel,
    CardWidget, CheckBox, DoubleSpinBox, SpinBox, TableWidget, FlowLayout, PushButton,
    PrimaryPushButton, ToolButton, ComboBox,
)

from app.adapters.base import InferenceAdapter
from app.adapters.normal_adapter import NormalAdapter
from app.adapters.sahi_adapter import SahiAdapter
from app.services.benchmark import BenchmarkRunner
from app.adapters.tracking_adapter import TrackingAdapter
from app.utils.config import (
    get_str, set_str, get_float, set_float,
    get_int, set_int, get_bool, )
from app.utils.message import error
from qfluentwidgets import FluentIcon as FIF

TRACKER_CHOICES = {
    "BoT-SORT": "botsort.yaml",
    "ByteTrack": "bytetrack.yaml",
}


class BenchmarkPanel(QWidget):
    """推理基准对比面板"""

    status_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("benchmark_panel")
        self._worker: BenchmarkRunner | None = None

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("推理基准对比"))

        # ============================================================
        # 路径设置
        # ============================================================
        layout.addWidget(StrongBodyLabel("路径设置"))
        ds_card = CardWidget()
        ds_form = QFormLayout(ds_card)

        mdl_row = QHBoxLayout()
        self.model_edit = LineEdit()
        self.model_edit.setPlaceholderText("选择 YOLO .pt 模型...")
        self.model_edit.setReadOnly(True)
        self.model_edit.textChanged.connect(lambda v: set_str("bm_model_path", v))
        mdl_btn = PushButton("📄")
        mdl_btn.clicked.connect(self._browse_model)
        mdl_row.addWidget(self.model_edit, 1)
        mdl_row.addWidget(mdl_btn)
        ds_form.addRow(BodyLabel("模型文件:"), mdl_row)

        img_row = QHBoxLayout()
        self.img_edit = LineEdit()
        self.img_edit.setPlaceholderText("选择图片目录...")
        self.img_edit.setReadOnly(True)
        self.img_edit.textChanged.connect(lambda v: set_str("bm_img_dir", v))
        img_btn = PushButton("📁")
        img_btn.clicked.connect(lambda: self._browse_dir(self.img_edit))
        img_row.addWidget(self.img_edit, 1)
        img_row.addWidget(img_btn)
        ds_form.addRow(BodyLabel("图片目录:"), img_row)

        lbl_row = QHBoxLayout()
        self.lbl_edit = LineEdit()
        self.lbl_edit.setPlaceholderText("选择 YOLO 标签目录...")
        self.lbl_edit.setReadOnly(True)
        self.lbl_edit.textChanged.connect(lambda v: set_str("bm_lbl_dir", v))
        lbl_btn = PushButton("📁")
        lbl_btn.clicked.connect(lambda: self._browse_dir(self.lbl_edit))
        lbl_row.addWidget(self.lbl_edit, 1)
        lbl_row.addWidget(lbl_btn)
        ds_form.addRow(BodyLabel("标签目录:"), lbl_row)

        iou_row = QHBoxLayout()
        iou_row.addWidget(BodyLabel("IoU 阈值:"))
        self.iou_spin = DoubleSpinBox()
        self.iou_spin.setRange(0.1, 0.95)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.5)
        self.iou_spin.valueChanged.connect(lambda v: set_float("bm_iou", v))
        iou_row.addWidget(self.iou_spin)
        iou_row.addStretch()
        ds_form.addRow(iou_row)

        layout.addWidget(ds_card)

        # ============================================================
        # 推理方式多选（可折叠）
        # ============================================================
        layout.addWidget(StrongBodyLabel("推理方式（可多选）"))
        method_card = CardWidget()
        method_layout = QVBoxLayout(method_card)

        # ---- 普通推理 ----
        self._build_method_row(
            method_layout,
            check_attr="normal_check",
            toggle_attr="_normal_toggle",
            config_attr="_normal_config",
            name="YOLO 普通推理",
            configs=[
                ("Conf:", "normal_conf", DoubleSpinBox, 0.01, 1.0, 0.25, 0.05, 2,
                 lambda v: set_float("bm_normal_conf", v)),
                ("IoU:", "normal_iou", DoubleSpinBox, 0.01, 1.0, 0.45, 0.05, 2,
                 lambda v: set_float("bm_normal_iou", v)),
            ],
        )

        # ---- SAHI 推理 ----
        self._build_method_row(
            method_layout,
            check_attr="sahi_check",
            toggle_attr="_sahi_toggle",
            config_attr="_sahi_config",
            name="SAHI 切片推理",
            configs=[
                ("Conf:", "sahi_conf", DoubleSpinBox, 0.01, 1.0, 0.25, 0.05, 2,
                 lambda v: set_float("bm_sahi_conf", v)),
                ("切片宽:", "sahi_sw", SpinBox, 320, 2048, 640, 32, 0,
                 lambda v: set_int("bm_sahi_sw", v)),
                ("切片高:", "sahi_sh", SpinBox, 320, 2048, 640, 32, 0,
                 lambda v: set_int("bm_sahi_sh", v)),
                ("重叠W:", "sahi_ow", DoubleSpinBox, 0.0, 0.9, 0.25, 0.05, 2,
                 lambda v: set_float("bm_sahi_ow", v)),
                ("重叠H:", "sahi_oh", DoubleSpinBox, 0.0, 0.9, 0.25, 0.05, 2,
                 lambda v: set_float("bm_sahi_oh", v)),
            ],
        )

        # ---- 视频跟踪 ----
        self._build_method_row(
            method_layout,
            check_attr="track_check",
            toggle_attr="_track_toggle",
            config_attr="_track_config",
            name="YOLO + 跟踪器",
            configs=[
                ("跟踪器:", "track_tracker", ComboBox, 0, 0, "BoT-SORT", 0, 0,
                 lambda v: set_str("bm_tracker", v)),
                ("Conf:", "track_conf", DoubleSpinBox, 0.01, 1.0, 0.25, 0.05, 2,
                 lambda v: set_float("bm_track_conf", v)),
                ("IoU:", "track_iou", DoubleSpinBox, 0.01, 1.0, 0.7, 0.05, 2,
                 lambda v: set_float("bm_track_iou", v)),
                ("ImgSz:", "track_imgsz", SpinBox, 320, 2048, 640, 32, 0,
                 lambda v: set_int("bm_track_imgsz", v)),
            ],
        )

        layout.addWidget(method_card)

        # ============================================================
        # 操作
        # ============================================================
        btn_row = QHBoxLayout()
        self.status_label = BodyLabel("")
        btn_row.addWidget(self.status_label)
        btn_row.addStretch()
        self.run_btn = PrimaryPushButton("开始对比")
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)
        layout.addLayout(btn_row)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # ============================================================
        # 结果表格
        # ============================================================
        self.table = TableWidget()
        self.table.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

    # ============================================================
    # 构建可折叠方法行
    # ============================================================

    def _make_toggle_btn(self) -> ToolButton:
        """创建折叠按钮"""
        btn = ToolButton(FIF.CARE_LEFT_SOLID, self)
        btn.setFixedSize(20, 20)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    def _build_method_row(
            self,
            parent_layout: QVBoxLayout,
            check_attr: str,
            toggle_attr: str,
            config_attr: str,
            name: str,
            configs: list[tuple],
    ) -> None:
        """构建一行：[CheckBox 名称] ---- [▼]

        configs: [(label, attr_name, widget_class, min, max, default,
                   step, decimals, callback), ...]
        """
        # 顶行
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)

        check = CheckBox(name)
        setattr(self, check_attr, check)
        top.addWidget(check)
        top.addStretch()

        toggle = self._make_toggle_btn()
        setattr(self, toggle_attr, toggle)
        top.addWidget(toggle)

        parent_layout.addLayout(top)

        # 配置区（默认隐藏）
        config_widget = QWidget()
        config_widget.setVisible(False)
        config_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )
        config_layout = FlowLayout(config_widget)
        config_layout.setContentsMargins(8, 4, 0, 4)
        config_layout.setSpacing(10)
        setattr(self, config_attr, config_widget)

        # 根据配置列表创建控件。
        # 标签和 SpinBox 必须放进同一个容器，否则 FlowLayout 会把二者
        # 当成两个独立元素换行，SAHI 参数较多时就会出现错位和挤压。
        for item in configs:
            label, attr_name, widget_class, rmin, rmax, default, step, decimals, callback = item

            field_widget = QWidget(config_widget)
            field_widget.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            field_layout = QHBoxLayout(field_widget)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(6)

            lbl = BodyLabel(label)
            self._normalize_point_font(lbl)
            field_layout.addWidget(lbl)

            widget = widget_class()
            if widget_class is ComboBox:
                widget.addItems(["BoT-SORT", "ByteTrack"])
                widget.setCurrentText(str(default))
                widget.currentTextChanged.connect(callback)
            elif widget_class is DoubleSpinBox:
                widget.setRange(rmin, rmax)
                widget.setSingleStep(step)
                widget.setDecimals(decimals)
                widget.setValue(default)
                widget.valueChanged.connect(callback)
            elif widget_class is SpinBox:
                widget.setRange(int(rmin), int(rmax))
                widget.setSingleStep(int(step))
                widget.setValue(int(default))
                widget.valueChanged.connect(callback)

            # qfluentwidgets 默认使用像素字号；部分 PySide6/Qt 版本在控件
            # 从隐藏变为显示时会读取 pointSize()，此时返回 -1 并触发警告。
            self._normalize_point_font(widget)
            if hasattr(widget, "lineEdit") and widget.lineEdit() is not None:
                widget.lineEdit().setFont(widget.font())

            setattr(self, attr_name, widget)
            field_layout.addWidget(widget)
            config_layout.addWidget(field_widget)

        parent_layout.addWidget(config_widget)

        # 折叠切换
        toggle.clicked.connect(
            lambda: self._toggle_config(config_widget, toggle)
        )

    @staticmethod
    def _normalize_point_font(widget: QWidget) -> None:
        """把像素字号转换为等效 point size，避免 Qt 读取到 pointSize=-1。"""
        font = widget.font()
        if font.pointSizeF() > 0:
            return

        pixel_size = font.pixelSize()
        if pixel_size <= 0:
            return

        dpi = widget.logicalDpiY()
        point_size = pixel_size * 72.0 / dpi if dpi > 0 else 10.5
        font.setPointSizeF(max(point_size, 1.0))
        widget.setFont(font)

    @staticmethod
    def _toggle_config(widget: QWidget, btn: ToolButton) -> None:
        visible = not widget.isVisible()
        widget.setVisible(visible)
        btn.setIcon(FIF.CARE_DOWN_SOLID if visible else FIF.CARE_LEFT_SOLID)

    # ============================================================
    # 路径
    # ============================================================

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型", self.model_edit.text(),
            "Model Files (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.model_edit.setText(path)

    def _browse_dir(self, edit: LineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录", edit.text())
        if path:
            edit.setText(path)

    # ============================================================
    # 运行
    # ============================================================

    def _on_run(self) -> None:
        model_path = self.model_edit.text().strip()
        img_dir = Path(self.img_edit.text().strip())
        lbl_dir = Path(self.lbl_edit.text().strip())
        iou = self.iou_spin.value()

        if not model_path or not Path(model_path).is_file():
            error("错误", "请选择有效的模型文件", self)
            return
        if not img_dir.is_dir():
            error("错误", "请选择有效的图片目录", self)
            return
        if not lbl_dir.is_dir():
            error("错误", "请选择有效的标签目录", self)
            return

        adapters: list[InferenceAdapter] = []

        if self.normal_check.isChecked():
            adapters.append(NormalAdapter(
                model_path,
                conf=self.normal_conf.value(),
                iou=self.normal_iou.value(),
            ))

        if self.sahi_check.isChecked():
            adapters.append(SahiAdapter(
                model_path,
                conf=self.sahi_conf.value(),
                slice_w=self.sahi_sw.value(),
                slice_h=self.sahi_sh.value(),
                overlap_w=self.sahi_ow.value(),
                overlap_h=self.sahi_oh.value(),
            ))

        if self.track_check.isChecked():
            tracker_name = self.track_tracker.currentText()
            adapters.append(TrackingAdapter(
                model_path,
                tracker=TRACKER_CHOICES.get(tracker_name, "botsort.yaml"),
                conf=self.track_conf.value(),
                iou=self.track_iou.value(),
                imgsz=self.track_imgsz.value(),
            ))

        if not adapters:
            error("错误", "请至少选择一种推理方式", self)
            return

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("正在对比...")

        self._worker = BenchmarkRunner(adapters, img_dir, lbl_dir, iou)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, results: list[dict]) -> None:
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("✅ 对比完成")
        self._build_table(results)

    def _on_error(self, err_msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("❌ 对比失败")
        error("对比出错", err_msg, self)

    # ============================================================
    # 表格
    # ============================================================

    def _build_table(self, results: list[dict]) -> None:
        if not results:
            return

        all_cls: set[int] = set()
        for r in results:
            all_cls.update(r["per_class"].keys())

        columns = ["方法", "mAP", "P_mean", "R_mean", "F1_mean", "耗时(s)"]
        cls_sorted = sorted(all_cls)
        for cls_id in cls_sorted:
            columns.append(f"Cls_{cls_id}_P")
            columns.append(f"Cls_{cls_id}_R")

        self.table.setRowCount(len(results))
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)

        for row_idx, r in enumerate(results):
            pc = r["per_class"]
            self._set_cell(row_idx, 0, r["name"])
            self._set_cell(row_idx, 1, f"{r['mAP']:.4f}")

            p_mean = np.mean([v["P"] for v in pc.values()]) if pc else 0.0
            r_mean = np.mean([v["R"] for v in pc.values()]) if pc else 0.0
            f1_mean = np.mean([v["F1"] for v in pc.values()]) if pc else 0.0
            self._set_cell(row_idx, 2, f"{p_mean:.4f}")
            self._set_cell(row_idx, 3, f"{r_mean:.4f}")
            self._set_cell(row_idx, 4, f"{f1_mean:.4f}")
            self._set_cell(row_idx, 5, f"{r['time']:.1f}")

            for j, cls_id in enumerate(cls_sorted):
                v = pc.get(cls_id, {})
                self._set_cell(row_idx, 6 + j * 2, f"{v.get('P', 0):.4f}")
                self._set_cell(row_idx, 6 + j * 2 + 1, f"{v.get('R', 0):.4f}")

        self._format_table()

    def _set_cell(self, row: int, col: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, col, item)

    def _format_table(self) -> None:
        for col in range(1, self.table.columnCount()):
            header = self.table.horizontalHeaderItem(col)
            if header is None:
                continue
            values = []
            for row in range(self.table.rowCount()):
                item = self.table.item(row, col)
                try:
                    values.append(float(item.text()) if item else 0.0)
                except ValueError:
                    values.append(0.0)
            if not values:
                continue
            ascending = "耗时" in header.text()
            ranks = np.argsort(values)
            if not ascending:
                ranks = ranks[::-1]
            if len(ranks) >= 1:
                self._apply_font(ranks[0], col, bold=True)
            if len(ranks) >= 2:
                self._apply_font(ranks[1], col, italic=True)

    def _apply_font(self, row: int, col: int, bold=False, italic=False) -> None:
        item = self.table.item(row, col)
        if item is None:
            return
        font = item.font()
        font.setBold(bold)
        font.setItalic(italic)
        item.setFont(font)

    # ============================================================
    # 持久化
    # ============================================================

    def _load_settings(self) -> None:
        self.normal_check.setChecked(get_bool("bm_normal_enabled", True))
        self.normal_conf.setValue(get_float("bm_normal_conf", 0.25))
        self.normal_iou.setValue(get_float("bm_normal_iou", 0.45))

        self.sahi_check.setChecked(get_bool("bm_sahi_enabled", True))
        self.sahi_conf.setValue(get_float("bm_sahi_conf", 0.25))
        self.sahi_sw.setValue(get_int("bm_sahi_sw", 640))
        self.sahi_sh.setValue(get_int("bm_sahi_sh", 640))
        self.sahi_ow.setValue(get_float("bm_sahi_ow", 0.25))
        self.sahi_oh.setValue(get_float("bm_sahi_oh", 0.25))

        self.track_check.setChecked(get_bool("bm_track_enabled", False))
        self.track_conf.setValue(get_float("bm_track_conf", 0.25))
        self.track_iou.setValue(get_float("bm_track_iou", 0.7))
        self.track_imgsz.setValue(get_int("bm_track_imgsz", 640))
        saved_tracker = get_str("bm_tracker", "BoT-SORT")
        idx = self.track_tracker.findText(saved_tracker)
        if idx >= 0:
            self.track_tracker.setCurrentIndex(idx)

        self.model_edit.setText(get_str("bm_model_path", ""))
        self.img_edit.setText(get_str("bm_img_dir", ""))
        self.lbl_edit.setText(get_str("bm_lbl_dir", ""))
        self.iou_spin.setValue(get_float("bm_iou", 0.5))
