"""通用图片浏览器：缩略图列表 + ImageViewer + 圆形导航按钮"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.widgets.image_viewer import ImageViewer
from app.widgets.thumbnail_list import ThumbnailList

NAV_BTN_STYLE = """
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


class ImageBrowser(QWidget):
    """缩略图列表 + 图片预览 + 左右导航。

    用法:
        browser = ImageBrowser()
        browser.image_selected.connect(self._on_selected)
        layout.addWidget(browser)

        browser.set_images(paths)
        browser.current_path
        browser.show_annotations(anns)
    """

    # 用户选中了某张图片（缩略图点击 / 导航按钮）
    image_selected = Signal(int, Path)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._current_index: int = -1
        self._current_pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ---- 缩略图列表 ----
        self.thumb_list = ThumbnailList()
        self.thumb_list.image_clicked.connect(self._on_thumb_clicked)
        layout.addWidget(self.thumb_list)

        # ---- 预览区 + 导航按钮 ----
        preview_container = QWidget()
        preview_container.setMinimumHeight(300)
        container_layout = QHBoxLayout(preview_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self.viewer = ImageViewer()
        container_layout.addWidget(self.viewer)

        self.prev_btn = QPushButton("◀")
        self.prev_btn.setToolTip("上一张")
        self.prev_btn.setStyleSheet(NAV_BTN_STYLE)
        self.prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_btn.setParent(self.viewer)
        self.prev_btn.clicked.connect(self.go_prev)

        self.next_btn = QPushButton("▶")
        self.next_btn.setToolTip("下一张")
        self.next_btn.setStyleSheet(NAV_BTN_STYLE)
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.setParent(self.viewer)
        self.next_btn.clicked.connect(self.go_next)

        layout.addWidget(preview_container, 1)

    # ---- 公共 API ----

    def set_images(self, paths: list[Path], select_index: int = 0) -> None:
        """加载图片列表，选中指定索引。路径为空则清空。"""
        self._current_index = -1
        self._current_pixmap = None

        if not paths:
            self.thumb_list.clear()
            self.viewer.set_image(None)
            self._update_nav_buttons()
            return

        self.thumb_list.set_images(paths)
        idx = max(0, min(select_index, len(paths) - 1))
        self._select(idx, emit=True)

    def clear(self) -> None:
        self.set_images([])

    def go_prev(self) -> None:
        if self._current_index > 0:
            self._select(self._current_index - 1, emit=True)

    def go_next(self) -> None:
        if self._current_index < self.thumb_list.count - 1:
            self._select(self._current_index + 1, emit=True)

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def current_path(self) -> Path | None:
        return self.thumb_list.path_at(self._current_index)

    @property
    def current_pixmap(self) -> QPixmap | None:
        return self._current_pixmap

    @property
    def count(self) -> int:
        return self.thumb_list.count

    def show_annotations(self, annotations: list[dict]) -> None:
        """替换叠加层"""
        self.viewer.clear_overlays()
        for ann in annotations:
            if ann["type"] == "bbox":
                self.viewer.add_bbox(ann["rect"], ann["color"], ann["label"])
            elif ann["type"] == "polygon":
                self.viewer.add_polygon(ann["points"], ann["color"], ann["label"])

    # ---- 内部 ----

    def _on_thumb_clicked(self, idx: int, path: Path) -> None:
        self._select(idx, emit=True)

    def _select(self, idx: int, *, emit: bool = False) -> None:
        """选中某张图，显示大图"""
        if idx < 0 or idx >= self.thumb_list.count:
            return

        path = self.thumb_list.path_at(idx)
        if path is None or not path.exists():
            return

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return

        self._current_index = idx
        self._current_pixmap = pixmap
        self.thumb_list.set_current_row(idx)
        self.viewer.set_image(pixmap)
        self._update_nav_buttons()

        if emit:
            self.image_selected.emit(idx, path)

    def _update_nav_buttons(self) -> None:
        total = self.thumb_list.count
        self.prev_btn.setEnabled(self._current_index > 0)
        self.next_btn.setEnabled(self._current_index < total - 1)

    def set_nav_enabled(self, enabled: bool) -> None:
        """推理中禁用列表和导航按钮"""
        self.prev_btn.setEnabled(enabled and self._current_index > 0)
        self.next_btn.setEnabled(enabled and self._current_index < self.thumb_list.count - 1)

    # ---- 按钮定位 ----

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_nav_buttons()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, self._position_nav_buttons)

    def _position_nav_buttons(self) -> None:
        vw = self.viewer.width()
        vh = self.viewer.height()
        center_y = max(0, (vh - 48) // 2)
        self.prev_btn.move(12, center_y)
        self.next_btn.move(max(0, vw - 60), center_y)
