"""热成像目标合成面板：SAM 抠图 → 实例库 → 随机贴图合成"""
from __future__ import annotations

from pathlib import Path

import cv2

from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import (
    QButtonGroup, QHBoxLayout, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, ComboBox, ProgressBar,
    BodyLabel, StrongBodyLabel, SubtitleLabel,
    CardWidget, SpinBox, DoubleSpinBox, RadioButton,
)

from app.services.label_reader import IMAGE_EXTS, get_color
from app.services.synthesis import (
    BackgroundRemover, ImageCompositor, SAM_VARIANTS, SAM_VARIANT_NAMES,
    merge_labels,
)
from app.utils.config import (
    get_str, set_str, get_int, set_int, get_float, set_float, get_bool, set_bool,
)
from app.utils.message import error, info
from app.utils.worker import Worker
from app.widgets.image_browser import ImageBrowser
from app.widgets.path_browser import PathBrowser
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---- Workers ----

class RemoveWorker(Worker):
    """后台批量抠图"""

    def __init__(
        self, remover: BackgroundRemover,
        image_paths: list[Path], output_dir: Path,
    ) -> None:
        super().__init__()
        self.remover = remover
        self.image_paths = image_paths
        self.output_dir = output_dir

    def do_work(self) -> int:
        total = len(self.image_paths)
        for i, path in enumerate(self.image_paths):
            out = self.output_dir / f"{path.stem}.png"
            self.remover.remove_background(path, out)
            self.progress.emit(int((i + 1) / total * 100))
        return total


class CompositeWorker(Worker):
    """后台批量贴图合成"""

    def __init__(
        self,
        compositor: ImageCompositor,
        target_paths: list[Path],
        output_dir: Path,
        label_dir: Path | None,
        merge_label_dir: Path | None,
        num_per_image: int,
    ) -> None:
        super().__init__()
        self.compositor = compositor
        self.target_paths = target_paths
        self.output_dir = output_dir
        self.label_dir = label_dir
        self.merge_label_dir = merge_label_dir
        self.num_per_image = num_per_image

    def do_work(self) -> dict:
        total = len(self.target_paths)
        all_annotations: dict[str, list[dict]] = {}
        for i, path in enumerate(self.target_paths):
            anns = self.compositor.composite(
                path, self.output_dir / path.name, self.num_per_image,
            )
            all_annotations[str(path)] = anns
            if self.label_dir and self.merge_label_dir:
                merge_labels(self.label_dir, path.name, anns, self.merge_label_dir)
            self.progress.emit(int((i + 1) / total * 100))
        return {"total": total, "annotations": all_annotations}


# ---- 转换工具 ----

def _yolo_to_annotations(yolo_anns: list[dict], image_path: Path) -> list[dict]:
    """把 YOLO 归一化标注 (cx,cy,w,h) 转为 ImageViewer 格式 (QRectF)"""
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    annotations: list[dict] = []
    for ann in yolo_anns:
        cx, cy, bw, bh = ann["cx"], ann["cy"], ann["w"], ann["h"]
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        cid = ann["class_id"]
        annotations.append({
            "type": "bbox",
            "rect": QRectF(x1, y1, bw * w, bh * h),
            "class_id": cid,
            "color": get_color(cid),
            "label": f"cls_{cid}",
        })
    return annotations


# ---- 面板 ----

class ImageSynthesisPanel(QWidget):
    """热成像目标合成"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("image_synthesis_panel")

        self._remover = BackgroundRemover()
        self._compositor: ImageCompositor | None = None
        self._remove_worker: RemoveWorker | None = None
        self._composite_worker: CompositeWorker | None = None

        self._setup_ui()
        self._load_settings()

    # ============================================================
    # UI
    # ============================================================

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("热成像目标合成器"))

        # ── 第一步：抠图 ──
        layout.addWidget(StrongBodyLabel("自动抠图"))
        layout.addWidget(self._build_remove_card())

        # ── 第二步：贴图合成 ──
        layout.addWidget(StrongBodyLabel("贴图合成"))
        layout.addWidget(self._build_composite_card())

        # ── 进度条 ──
        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # ── 预览 ──
        layout.addWidget(StrongBodyLabel("预览"))
        self.browser = ImageBrowser()
        self.browser.image_selected.connect(self._on_preview_image_selected)
        layout.addWidget(self.browser, 1)

    def _build_remove_card(self) -> CardWidget:
        card = CardWidget()
        ly = QVBoxLayout(card)
        ly.setContentsMargins(12, 8, 12, 8)
        ly.setSpacing(8)

        # 模型选择
        row = QHBoxLayout()
        row.addWidget(BodyLabel("模型:"))
        self.model_combo = ComboBox()
        self.model_combo.addItems(SAM_VARIANT_NAMES)
        self.model_combo.setCurrentIndex(0)
        row.addWidget(self.model_combo, 1)
        self.load_model_btn = PushButton("加载模型")
        self.load_model_btn.clicked.connect(self._on_load_model)
        row.addWidget(self.load_model_btn)
        ly.addLayout(row)

        # 源图片 → 抠图输出
        self.source_browser = PathBrowser(
            label="源图片目录:", mode="dir",
            placeholder="选择要抠图的图片目录...",
            config_key="syn_source_dir",
        )
        self.source_browser.path_changed.connect(self._on_source_selected)
        ly.addWidget(self.source_browser)

        self.instance_out_browser = PathBrowser(
            label="抠图输出目录:", mode="dir",
            placeholder="透明 PNG 保存到...",
            config_key="syn_instance_out_dir",
        )
        ly.addWidget(self.instance_out_browser)

        row2 = QHBoxLayout()
        self.remove_status = BodyLabel("")
        row2.addWidget(self.remove_status, 1)
        self.remove_btn = PrimaryPushButton("开始批量抠图")
        self.remove_btn.clicked.connect(self._on_remove)
        row2.addWidget(self.remove_btn)
        ly.addLayout(row2)

        return card

    def _build_composite_card(self) -> CardWidget:
        card = CardWidget()
        ly = QVBoxLayout(card)
        ly.setContentsMargins(12, 8, 12, 8)
        ly.setSpacing(8)

        # ── 路径 ──
        # 实例库（独立于抠图步骤）
        self.comp_instance_browser = PathBrowser(
            label="实例库目录:", mode="dir",
            placeholder="透明 PNG 实例库目录...",
            config_key="syn_comp_instance_dir",
        )
        ly.addWidget(self.comp_instance_browser)

        self.target_browser = PathBrowser(
            label="目标图片目录:", mode="dir",
            placeholder="背景图片目录...",
            config_key="syn_target_dir",
        )
        self.target_browser.path_changed.connect(self._on_target_selected)
        ly.addWidget(self.target_browser)

        self.target_label_browser = PathBrowser(
            label="目标标签目录:", mode="dir",
            placeholder="已有 YOLO 标签（可选）...",
            config_key="syn_target_label_dir",
        )
        ly.addWidget(self.target_label_browser)

        self.composite_out_browser = PathBrowser(
            label="图片输出目录:", mode="dir",
            placeholder="合成图片保存到...",
            config_key="syn_composite_out",
        )
        ly.addWidget(self.composite_out_browser)

        self.composite_label_browser = PathBrowser(
            label="标签输出目录:", mode="dir",
            placeholder="合并后的 YOLO 标签保存到...",
            config_key="syn_merge_label_dir",
        )
        ly.addWidget(self.composite_label_browser)

        # ── 数量 + 类别 ──
        row1 = QHBoxLayout()
        row1.addWidget(BodyLabel("每图贴入:"))
        self.num_spin = SpinBox()
        self.num_spin.setRange(1, 20)
        self.num_spin.setValue(2)
        self.num_spin.valueChanged.connect(lambda v: set_int("syn_num", v))
        row1.addWidget(self.num_spin)
        row1.addSpacing(16)
        row1.addWidget(BodyLabel("类别 ID:"))
        self.class_id_spin = SpinBox()
        self.class_id_spin.setRange(0, 99)
        self.class_id_spin.setValue(0)
        self.class_id_spin.setToolTip("生成标签的类别序号，避免和已有标签重叠")
        self.class_id_spin.valueChanged.connect(lambda v: set_int("syn_class_id", v))
        row1.addWidget(self.class_id_spin)
        row1.addStretch()
        ly.addLayout(row1)

        # ── 尺寸模式 ──
        row2 = QHBoxLayout()
        row2.addWidget(BodyLabel("尺寸模式:"))
        self.scale_radio = RadioButton("比例")
        self.pixel_radio = RadioButton("最长边(px)")
        self.scale_radio.setChecked(True)
        self._size_group = QButtonGroup(self)
        self._size_group.addButton(self.scale_radio, 0)
        self._size_group.addButton(self.pixel_radio, 1)
        self._size_group.buttonClicked.connect(self._on_size_mode_changed)
        row2.addWidget(self.scale_radio)
        row2.addWidget(self.pixel_radio)

        row2.addSpacing(8)
        self.scale_min = DoubleSpinBox()
        self.scale_min.setRange(0.1, 5.0)
        self.scale_min.setSingleStep(0.1)
        self.scale_min.setDecimals(2)
        self.scale_min.setValue(0.5)
        self.scale_min.valueChanged.connect(lambda v: set_float("syn_scale_min", v))
        row2.addWidget(self.scale_min)
        row2.addWidget(BodyLabel("~"))
        self.scale_max = DoubleSpinBox()
        self.scale_max.setRange(0.1, 5.0)
        self.scale_max.setSingleStep(0.1)
        self.scale_max.setDecimals(2)
        self.scale_max.setValue(1.5)
        self.scale_max.valueChanged.connect(lambda v: set_float("syn_scale_max", v))
        row2.addWidget(self.scale_max)

        self.pixel_size_spin = SpinBox()
        self.pixel_size_spin.setRange(10, 2048)
        self.pixel_size_spin.setValue(200)
        self.pixel_size_spin.setVisible(False)
        self.pixel_size_spin.valueChanged.connect(
            lambda v: set_int("syn_pixel_size", v)
        )
        row2.addWidget(self.pixel_size_spin)

        row2.addStretch()
        ly.addLayout(row2)

        # ── 旋转 + 模糊 ──
        row3 = QHBoxLayout()
        row3.addWidget(BodyLabel("旋转:"))
        self.rot_min = DoubleSpinBox()
        self.rot_min.setRange(-180, 180)
        self.rot_min.setSingleStep(5)
        self.rot_min.setDecimals(0)
        self.rot_min.setValue(-30)
        self.rot_min.valueChanged.connect(lambda v: set_float("syn_rot_min", v))
        row3.addWidget(self.rot_min)
        row3.addWidget(BodyLabel("~"))
        self.rot_max = DoubleSpinBox()
        self.rot_max.setRange(-180, 180)
        self.rot_max.setSingleStep(5)
        self.rot_max.setDecimals(0)
        self.rot_max.setValue(30)
        self.rot_max.valueChanged.connect(lambda v: set_float("syn_rot_max", v))
        row3.addWidget(self.rot_max)
        row3.addWidget(BodyLabel("度"))

        row3.addSpacing(16)
        row3.addWidget(BodyLabel("模糊:"))
        self.blur_min = DoubleSpinBox()
        self.blur_min.setRange(0, 10.0)
        self.blur_min.setSingleStep(0.1)
        self.blur_min.setDecimals(1)
        self.blur_min.setValue(0.0)
        self.blur_min.valueChanged.connect(lambda v: set_float("syn_blur_min", v))
        row3.addWidget(self.blur_min)
        row3.addWidget(BodyLabel("~"))
        self.blur_max = DoubleSpinBox()
        self.blur_max.setRange(0, 10.0)
        self.blur_max.setSingleStep(0.1)
        self.blur_max.setDecimals(1)
        self.blur_max.setValue(0.0)
        self.blur_max.valueChanged.connect(lambda v: set_float("syn_blur_max", v))
        row3.addWidget(self.blur_max)
        row3.addStretch()
        ly.addLayout(row3)

        # ── 翻转 ──
        row4 = QHBoxLayout()
        row4.addWidget(BodyLabel("水平翻转概率:"))
        self.flip_h_spin = DoubleSpinBox()
        self.flip_h_spin.setRange(0.0, 1.0)
        self.flip_h_spin.setSingleStep(0.1)
        self.flip_h_spin.setDecimals(2)
        self.flip_h_spin.setValue(0.0)
        self.flip_h_spin.valueChanged.connect(lambda v: set_float("syn_flip_h", v))
        row4.addWidget(self.flip_h_spin)

        row4.addSpacing(16)
        row4.addWidget(BodyLabel("垂直翻转概率:"))
        self.flip_v_spin = DoubleSpinBox()
        self.flip_v_spin.setRange(0.0, 1.0)
        self.flip_v_spin.setSingleStep(0.1)
        self.flip_v_spin.setDecimals(2)
        self.flip_v_spin.setValue(0.0)
        self.flip_v_spin.valueChanged.connect(lambda v: set_float("syn_flip_v", v))
        row4.addWidget(self.flip_v_spin)
        row4.addStretch()
        ly.addLayout(row4)

        # ── 操作 ──
        row5 = QHBoxLayout()
        self.composite_status = BodyLabel("")
        row5.addWidget(self.composite_status, 1)
        self.composite_btn = PrimaryPushButton("开始合成")
        self.composite_btn.clicked.connect(self._on_composite)
        row5.addWidget(self.composite_btn)
        ly.addLayout(row5)

        return card

    # ============================================================
    # 模型加载
    # ============================================================

    def _on_load_model(self) -> None:
        variant = self.model_combo.currentText()
        filename = SAM_VARIANTS.get(variant)
        if not filename:
            return
        set_str("syn_sam_variant", variant)
        try:
            self._remover.load_model(filename)
        except Exception as e:
            error("模型加载失败", str(e), self)
            return
        self.remove_status.setText(f"模型就绪: {filename}")

    # ============================================================
    # 路径回调 + 尺寸模式切换
    # ============================================================

    def _on_source_selected(self, path: str) -> None:
        if path:
            self._preview_dir(Path(path))

    def _on_target_selected(self, path: str) -> None:
        if path:
            self._preview_dir(Path(path))

    def _on_size_mode_changed(self) -> None:
        is_scale = self._size_group.checkedId() == 0
        set_bool("syn_size_scale", is_scale)
        self.scale_min.setVisible(is_scale)
        self.scale_max.setVisible(is_scale)
        self.pixel_size_spin.setVisible(not is_scale)

    def _preview_dir(self, directory: Path) -> None:
        images = sorted(
            f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )
        if images:
            self.browser.set_images(images, select_index=0)

    def _on_preview_image_selected(self, idx: int, path: Path) -> None:
        """预览图点击 → 叠加合成标注框"""
        anns = getattr(self, "_composite_annotations", {}).get(str(path))
        if anns:
            self.browser.show_annotations(
                _yolo_to_annotations(anns, path)
            )
        else:
            self.browser.show_annotations([])

    def _clear_composite_annotations(self) -> None:
        self._composite_annotations = {}
        self.browser.show_annotations([])

    # ============================================================
    # 批量抠图
    # ============================================================

    def _on_remove(self) -> None:
        if not self._remover.is_loaded:
            error("模型未加载", "请先选择并加载 SAM 模型", self)
            return

        source_dir = Path(self.source_browser.path)
        instance_dir = Path(self.instance_out_browser.path)

        if not source_dir.is_dir():
            error("路径错误", "请选择有效的源图片目录", self)
            return
        if not instance_dir.is_dir():
            error("路径错误", "请选择有效的抠图输出目录", self)
            return

        images = sorted(
            f for f in source_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )
        if not images:
            error("无图片", "源目录中未找到图片文件", self)
            return

        self.remove_btn.setEnabled(False)
        self.load_model_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.remove_status.setText(f"正在抠图 ({len(images)} 张)...")

        self._remove_worker = RemoveWorker(self._remover, images, instance_dir)
        self._remove_worker.finished.connect(self._on_remove_done)
        self._remove_worker.error.connect(self._on_remove_error)
        self._remove_worker.progress.connect(self.progress.setValue)
        self._remove_worker.start()

    def _on_remove_done(self, total: int) -> None:
        self._set_remove_enabled(True)
        self.progress.setVisible(False)
        self.remove_status.setText(f"抠图完成，共 {total} 张")

        instance_dir = Path(self.instance_out_browser.path)
        self._preview_dir(instance_dir)

        info("抠图完成", f"已处理 {total} 张图片\n输出目录: {instance_dir}", self)

    def _on_remove_error(self, err: str) -> None:
        self._set_remove_enabled(True)
        self.progress.setVisible(False)
        self.remove_status.setText("抠图失败")
        error("抠图失败", err, self)

    def _set_remove_enabled(self, enabled: bool) -> None:
        self.remove_btn.setEnabled(enabled)
        self.load_model_btn.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.source_browser.setEnabled(enabled)
        self.instance_out_browser.setEnabled(enabled)

    # ============================================================
    # 贴图合成
    # ============================================================

    def _on_composite(self) -> None:
        instance_dir = Path(self.comp_instance_browser.path)
        target_dir = Path(self.target_browser.path)
        out_dir = Path(self.composite_out_browser.path)

        if not instance_dir.is_dir():
            error("路径错误", "请选择有效的实例库目录", self)
            return
        if not target_dir.is_dir():
            error("路径错误", "请选择有效的目标图片目录", self)
            return
        if not out_dir.is_dir():
            error("路径错误", "请选择有效的输出目录", self)
            return

        target_images = sorted(
            f for f in target_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )
        if not target_images:
            error("无图片", "目标目录中未找到图片文件", self)
            return

        # 尺寸模式
        use_pixel = self._size_group.checkedId() == 1

        try:
            self._compositor = ImageCompositor(
                instance_dir,
                size_mode="pixel" if use_pixel else "scale",
                scale_range=(self.scale_min.value(), self.scale_max.value()),
                pixel_size=self.pixel_size_spin.value() if use_pixel else 0,
                rotation_range=(self.rot_min.value(), self.rot_max.value()),
                blur_range=(self.blur_min.value(), self.blur_max.value()),
                flip_h_prob=self.flip_h_spin.value(),
                flip_v_prob=self.flip_v_spin.value(),
                class_id=self.class_id_spin.value(),
            )
        except ValueError as e:
            error("实例库错误", str(e), self)
            return

        label_dir = Path(self.target_label_browser.path)
        merge_label_dir = Path(self.composite_label_browser.path)
        has_labels = label_dir.is_dir() and merge_label_dir.is_dir()

        self.composite_btn.setEnabled(False)
        self._set_remove_enabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.composite_status.setText(
            f"正在合成 ({len(target_images)} 张, 每张 {self.num_spin.value()} 个)..."
        )

        self._composite_worker = CompositeWorker(
            self._compositor, target_images, out_dir,
            label_dir if has_labels else None,
            merge_label_dir if has_labels else None,
            self.num_spin.value(),
        )
        self._composite_worker.finished.connect(self._on_composite_done)
        self._composite_worker.error.connect(self._on_composite_error)
        self._composite_worker.progress.connect(self.progress.setValue)
        self._composite_worker.start()

    def _on_composite_done(self, result: dict) -> None:
        self._set_composite_enabled(True)
        self.progress.setVisible(False)
        total = result["total"]
        self._composite_annotations = result["annotations"]
        self.composite_status.setText(f"合成完成，共 {total} 张")

        out_dir = Path(self.composite_out_browser.path)
        self._preview_dir(out_dir)

        msg = f"已处理 {total} 张图片\n输出: {out_dir}"
        merge_dir = Path(self.composite_label_browser.path)
        if merge_dir.is_dir():
            msg += f"\n合并标签: {merge_dir}"
        info("合成完成", msg, self)

    def _on_composite_error(self, err: str) -> None:
        self._set_composite_enabled(True)
        self.progress.setVisible(False)
        self.composite_status.setText("合成失败")
        error("合成失败", err, self)

    def _set_composite_enabled(self, enabled: bool) -> None:
        self.composite_btn.setEnabled(enabled)
        self.comp_instance_browser.setEnabled(enabled)
        self.target_browser.setEnabled(enabled)
        self.target_label_browser.setEnabled(enabled)
        self.composite_out_browser.setEnabled(enabled)
        self.composite_label_browser.setEnabled(enabled)
        self.num_spin.setEnabled(enabled)
        self.class_id_spin.setEnabled(enabled)
        self.scale_radio.setEnabled(enabled)
        self.pixel_radio.setEnabled(enabled)
        self.scale_min.setEnabled(enabled)
        self.scale_max.setEnabled(enabled)
        self.pixel_size_spin.setEnabled(enabled)
        self.rot_min.setEnabled(enabled)
        self.rot_max.setEnabled(enabled)
        self.blur_min.setEnabled(enabled)
        self.blur_max.setEnabled(enabled)
        self.flip_h_spin.setEnabled(enabled)
        self.flip_v_spin.setEnabled(enabled)

    # ============================================================
    # 持久化
    # ============================================================

    def _load_settings(self) -> None:
        # 模型变体
        saved_variant = get_str("syn_sam_variant", SAM_VARIANT_NAMES[0])
        idx = 0
        for i, name in enumerate(SAM_VARIANT_NAMES):
            if name == saved_variant:
                idx = i
                break
        self.model_combo.setCurrentIndex(idx)

        # 路径
        self.source_browser.path = get_str("syn_source_dir")
        self.instance_out_browser.path = get_str("syn_instance_out_dir")
        self.comp_instance_browser.path = get_str("syn_comp_instance_dir")
        self.target_browser.path = get_str("syn_target_dir")
        self.target_label_browser.path = get_str("syn_target_label_dir")
        self.composite_out_browser.path = get_str("syn_composite_out")
        self.composite_label_browser.path = get_str("syn_merge_label_dir")

        # 基础参数
        self.num_spin.setValue(get_int("syn_num", 2))
        self.class_id_spin.setValue(get_int("syn_class_id", 0))

        # 尺寸模式
        if get_bool("syn_size_scale", True):
            self.scale_radio.setChecked(True)
        else:
            self.pixel_radio.setChecked(True)
        self._on_size_mode_changed()
        self.scale_min.setValue(get_float("syn_scale_min", 0.5))
        self.scale_max.setValue(get_float("syn_scale_max", 1.5))
        self.pixel_size_spin.setValue(get_int("syn_pixel_size", 200))

        # 增强参数
        self.rot_min.setValue(get_float("syn_rot_min", -30))
        self.rot_max.setValue(get_float("syn_rot_max", 30))
        self.blur_min.setValue(get_float("syn_blur_min", 0.0))
        self.blur_max.setValue(get_float("syn_blur_max", 0.0))
        self.flip_h_spin.setValue(get_float("syn_flip_h", 0.0))
        self.flip_v_spin.setValue(get_float("syn_flip_v", 0.0))
