"""通用缩略图列表组件：懒加载 + 水平滚轮"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

THUMB_SIZE = QSize(100, 80)
BATCH_SIZE = 80


class ThumbnailList(QWidget):
    """缩略图列表。

    用法:
        thumbs = ThumbnailList()
        thumbs.image_clicked.connect(self._on_clicked)
        thumbs.set_images(image_paths)
    """

    image_clicked = Signal(int, Path)   # index, path

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image_files: list[Path] = []
        self._thumbs_loaded: int = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget()
        self._list.setFlow(QListWidget.Flow.LeftToRight)
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(THUMB_SIZE)
        self._list.setGridSize(QSize(THUMB_SIZE.width() + 12, THUMB_SIZE.height() + 30))
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setWrapping(False)
        self._list.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFixedHeight(THUMB_SIZE.height() + 40)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.horizontalScrollBar().valueChanged.connect(self._on_scrolled)
        self._list.wheelEvent = self._wheel_event

        layout.addWidget(self._list)

    # ---- 公共 API ----

    def set_images(self, paths: list[Path]) -> None:
        """设置图片列表（全量文件名秒显，缩略图分批加载）"""
        self._image_files = paths
        self._thumbs_loaded = 0
        self._list.clear()

        for img in paths:
            item = QListWidgetItem(img.name)
            item.setData(Qt.ItemDataRole.UserRole, str(img))
            item.setToolTip(img.name)
            self._list.addItem(item)

        self._load_batch()

    def clear(self) -> None:
        self._image_files.clear()
        self._thumbs_loaded = 0
        self._list.clear()

    @property
    def count(self) -> int:
        return len(self._image_files)

    @property
    def current_index(self) -> int:
        return self._list.currentRow()

    def set_current_row(self, index: int) -> None:
        self._list.setCurrentRow(index)
        self._ensure_thumbnail(index)

    def path_at(self, index: int) -> Path | None:
        if 0 <= index < len(self._image_files):
            return self._image_files[index]
        return None

    # ---- 内部 ----

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        idx = self._list.row(item)
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        self._ensure_thumbnail(idx)
        self.image_clicked.emit(idx, path)

    def _ensure_thumbnail(self, idx: int) -> None:
        """确保指定行有缩略图"""
        if idx < 0 or idx >= len(self._image_files):
            return
        item = self._list.item(idx)
        if item is None or not item.icon().isNull():
            return
        img_path = self._image_files[idx]
        icon = QPixmap(str(img_path)).scaled(
            THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        item.setIcon(icon)

    def _load_batch(self) -> None:
        """加载下一批缩略图"""
        if self._thumbs_loaded >= len(self._image_files):
            return
        end = min(self._thumbs_loaded + BATCH_SIZE, len(self._image_files))
        for i in range(self._thumbs_loaded, end):
            img_path = self._image_files[i]
            item = self._list.item(i)
            if item is None:
                continue
            icon = QPixmap(str(img_path)).scaled(
                THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item.setIcon(icon)
        self._thumbs_loaded = end

    def _on_scrolled(self, value: int) -> None:
        bar = self._list.horizontalScrollBar()
        if value >= bar.maximum() - 300:
            self._load_batch()

    def _wheel_event(self, event) -> None:
        delta = event.angleDelta().y()
        bar = self._list.horizontalScrollBar()
        bar.setValue(bar.value() - delta)
