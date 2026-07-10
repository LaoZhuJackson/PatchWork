"""F4: Label 标注预览面板"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
    QWidget, QPushButton,
)
from qfluentwidgets import (
    PushButton,
    LineEdit,
    SubtitleLabel,
    BodyLabel,
    FluentIcon as FIF,
)

from app.services.label_reader import IMAGE_EXTS, parse_yolo_label
from app.utils.config import get_str, set_str
from app.utils.message import info, warning
from app.widgets.image_viewer import ImageViewer

THUMB_SIZE = QSize(100, 80)


class LabelPreviewPanel(QWidget):
    """Label 标注预览面板"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("label_preview_panel")

        self._image_files: list[Path] = []  # 所有图片
        self._label_map: dict[str, Path] = {}  # stem → label 路径
        self._current_index: int = -1

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # ---- 标题 ----
        layout.addWidget(SubtitleLabel("Label 标注预览"))

        # ---- 路径选择 ----
        path_group = QGroupBox("路径设置")
        path_form = QFormLayout(path_group)

        # 图片目录
        img_row = QHBoxLayout()
        self.img_edit = LineEdit()
        self.img_edit.setPlaceholderText("选择图片所在的文件夹...")
        img_btn = PushButton("浏览...")
        img_btn.clicked.connect(lambda: self._browse_dir(self.img_edit, "label_preview_img_dir", self._on_path_changed))
        img_row.addWidget(self.img_edit, 1)
        img_row.addWidget(img_btn)
        path_form.addRow("图片目录:", img_row)

        # 标签目录
        lbl_row = QHBoxLayout()
        self.lbl_edit = LineEdit()
        self.lbl_edit.setPlaceholderText("选择标签所在的文件夹...")
        lbl_btn = PushButton("浏览...")
        lbl_btn.clicked.connect(lambda: self._browse_dir(self.lbl_edit, "label_preview_lbl_dir", self._on_path_changed))
        lbl_row.addWidget(self.lbl_edit, 1)
        lbl_row.addWidget(lbl_btn)
        path_form.addRow("标签目录:", lbl_row)

        self.info_label = BodyLabel("")
        path_form.addRow("", self.info_label)

        layout.addWidget(path_group)

        # ---- 上方: 水平缩略图列表 ----
        layout.addWidget(BodyLabel("图片列表:"))

        self.image_list = QListWidget()
        self.image_list.setFlow(QListWidget.Flow.LeftToRight)
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setIconSize(THUMB_SIZE)
        self.image_list.setGridSize(QSize(THUMB_SIZE.width() + 12, THUMB_SIZE.height() + 30))
        self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list.setWrapping(False)
        self.image_list.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.image_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_list.setFixedHeight(THUMB_SIZE.height() + 40)
        self.image_list.itemClicked.connect(self._on_item_clicked)
        self.image_list.wheelEvent = self._list_wheel_event # 替换滚轮逻辑
        layout.addWidget(self.image_list)

        # ---- 下方: 导航 + 预览 ----
        preview_container = QWidget()
        preview_container.setMinimumHeight(300)
        container_layout = QHBoxLayout(preview_container)
        container_layout.setContentsMargins(0,0,0,0)

        self.viewer = ImageViewer()
        container_layout.addWidget(self.viewer)

        # 叠加在 viewer 上的圆形按钮样式
        btn_style = """
            QPushButton {
              background: rgba(0, 0, 0, 0.35);
              color: white;
              border: none;
              border-radius: 24px;
              font-size: 20px;
              min-width: 48px;
              max-width: 48px;
              min-height: 48px;
              max-height: 48px;
            }
            QPushButton:hover {
              background: rgba(0, 0, 0, 0.65);
            }
        """
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setToolTip("上一张")
        self.prev_btn.setStyleSheet(btn_style)
        self.prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_btn.setParent(self.viewer)
        self.prev_btn.clicked.connect(self._prev_image)
        self.prev_btn.move(12, 0) # y 在 resizeEvent 里居中

        self.next_btn = QPushButton("▶")
        self.next_btn.setToolTip("下一张")
        self.next_btn.setStyleSheet(btn_style)
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.setParent(self.viewer)
        self.next_btn.clicked.connect(self._next_image)
        self.next_btn.move(0, 0)

        layout.addWidget(preview_container, 1)

    # ---- 路径 ----

    def _browse_dir(self, edit: LineEdit, key: str, callback=None) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录", edit.text())
        if not path:
            return
        edit.setText(path)
        set_str(key, path)
        if callback:
            callback()

    def _on_path_changed(self) -> None:
        """任一目录变更时重新加载"""
        img_dir = self.img_edit.text().strip()
        lbl_dir = self.lbl_edit.text().strip()
        if img_dir and lbl_dir:
            self._load_dataset(Path(img_dir), Path(lbl_dir))
        elif img_dir:
            # 仅图片目录，无标签
            self._load_dataset(Path(img_dir), None)

    def _load_dataset(self, img_dir: Path, lbl_dir: Path | None) -> None:
        """加载数据集，配对 images/ 和 labels/"""
        self._image_files.clear()
        self._label_map.clear()
        self.image_list.clear()
        self.viewer.set_image(None)
        self._current_index = -1
        self._update_nav_buttons()

        if not img_dir.is_dir():
            return

        images = sorted(
            f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )

        if not images:
            self.info_label.setText("❌ 图片目录下未找到图片")
            return

        self._image_files = images

        # 配对
        missing_label: list[str] = []
        if lbl_dir and lbl_dir.is_dir():
            for img in images:
                label = lbl_dir / f"{img.stem}.txt"
                if label.exists():
                    self._label_map[img.stem] = label
                else:
                    missing_label.append(img.name)
        # 缩略图列表
        for img in images:
            icon = QPixmap(str(img)).scaled(
                THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            item = QListWidgetItem(icon, img.name)
            item.setData(Qt.ItemDataRole.UserRole, str(img))
            item.setToolTip(img.name)
            self.image_list.addItem(item)

        # 汇总信息
        paired = len(self._label_map)
        msg = f"共 {len(images)} 张图片，{paired} 张已配对"
        if missing_label:
            msg += f"，{len(missing_label)} 张缺少标签"
            detail = (
                f"以下 {len(missing_label)} 张图片缺少对应标签，\n"
                f"将仅显示原图（无标注叠加）:\n\n"
            )
            for name in missing_label[:10]:
                detail += f"  • {name}\n"
            if len(missing_label) > 10:
                detail += f"  ... 等共 {len(missing_label)} 张\n"
            info("配对提示", detail, self)
        self.info_label.setText(msg)

        # 选中第一张
        if self._image_files:
            self.image_list.setCurrentRow(0)
            self._show_image_at(0)

    # ---- 导航 ----
    def _prev_image(self) -> None:
        if not self._image_files:
            return
        new_idx = max(0, self._current_index - 1)
        self.image_list.setCurrentRow(new_idx)
        self._show_image_at(new_idx)

    def _next_image(self) -> None:
        if not self._image_files:
            return
        new_idx = min(len(self._image_files) - 1, self._current_index + 1)
        self.image_list.setCurrentRow(new_idx)
        self._show_image_at(new_idx)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        idx = self.image_list.row(item)
        self._show_image_at(idx)

    def _show_image_at(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._image_files):
            return
        self._current_index = idx
        self._update_nav_buttons()

        img_path = self._image_files[idx]
        if not img_path.exists():
            return
        pixmap = QPixmap(str(img_path))
        if pixmap.isNull():
            warning("错误", f"无法加载图片: {img_path.name}", self)
            return

        # 查找对应label文件
        label_path = self._label_map.get(img_path.stem)
        if label_path:
            annotations = parse_yolo_label(label_path, pixmap.width(), pixmap.height())
        else:
            annotations = []
        self.viewer.set_image(pixmap)
        for ann in annotations:
            if ann["type"] == "bbox":
                self.viewer.add_bbox(ann["rect"], ann["color"], ann["label"])
            elif ann["type"] == "polygon":
                self.viewer.add_polygon(ann["points"], ann["color"], ann["label"])

    def _update_nav_buttons(self) -> None:
        total = len(self._image_files)
        self.prev_btn.setEnabled(self._current_index > 0)
        self.next_btn.setEnabled(self._current_index < total - 1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_nav_buttons()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, self._position_nav_buttons)

    def _position_nav_buttons(self):
        """将导航按钮放在 viewer 左右两侧垂直居中"""
        vw = self.viewer.width()
        vh = self.viewer.height()
        btn_h = 48
        center_y = (vh - btn_h) // 2
        self.prev_btn.move(12, center_y)
        self.next_btn.move(vw - 60, center_y)

    def _list_wheel_event(self, event) -> None:
        """将垂直滚轮转为水平滚动"""
        delta = event.angleDelta().y()
        bar = self.image_list.horizontalScrollBar()
        bar.setValue(bar.value() - delta)

    # ---- 持久化 ----
    def _load_settings(self) -> None:
        self.img_edit.setText(get_str("label_preview_img_dir"))
        self.lbl_edit.setText(get_str("label_preview_lbl_dir"))
        # 恢复上次的两个路径
        img_dir = get_str("label_preview_img_dir")
        lbl_dir = get_str("label_preview_lbl_dir")
        if img_dir:
            p_img = Path(img_dir)
            if p_img.is_dir():
                p_lbl = Path(lbl_dir) if lbl_dir else None
                self._load_dataset(p_img, p_lbl if p_lbl and p_lbl.is_dir() else None)
