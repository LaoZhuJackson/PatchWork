"""通用图片查看器：QGraphicsView 封装，支持缩放、拖拽、叠加绘制"""
from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

ZOOM_FACTOR = 1.15


class ImageViewer(QGraphicsView):
    """可缩放、拖拽的图片查看器，支持在图片上叠加框、多边形、文字"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # 背景图 item
        self._pixmap_item = self._scene.addPixmap(QPixmap())

        # 交互设置
        self.setRenderHint(QPainter.RenderHint.Antialiasing)  # 抗锯齿：让图形边缘平滑，消除锯齿
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)  # 平滑像素图变换：缩放/旋转图片时保持平滑，避免像素化

        self.setDragMode(QGraphicsView.DragMode.NoDrag)  # 拖拽模式：禁用视图的默认拖拽（后续自定义拖拽逻辑）

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)  # 变换锚点：缩放/旋转时以鼠标位置为中心
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)  # 调整锚点：视图大小变化时锚点在鼠标位置

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # 水平滚动条：始终隐藏
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # 垂直滚动条：始终隐藏

        self.setFrameShape(QGraphicsView.Shape.NoFrame)  # 边框样式：无边框，让视图融入界面

        self.setBackgroundBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setStyleSheet("QGraphicsView { background: transparent; }")  # 透明背景

        self._overlay_items: list = []  # 叠加层列表：存储所有叠加在场景上的图形项（如标注、选框等）
        self._is_dragging = False  # 拖拽状态标志：记录当前是否正在拖拽
        self._last_mouse_pos = QPointF()  # 上次鼠标位置：用于计算拖拽时的移动偏移量

    # ---- 公共 API ----
    def set_image(self, pixmap: QPixmap | None) -> None:
        """设置图片并自适应窗口"""
        self._clear_overlays()
        if pixmap is None or pixmap.isNull():
            self._pixmap_item.setPixmap(QPixmap())
            self._scene.setSceneRect(QRectF())
            return
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def fit_to_window(self) -> None:
        """缩放至适应窗口"""
        rect = self._scene.sceneRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def show_original(self) -> None:
        """1:1 显示"""
        self.resetTransform()

    def clear_overlays(self) -> None:
        self._clear_overlays()

    def add_bbox(self, rect: QRectF, color: QColor = QColor("#FF6B6B"), label: str = "",
                 line_width: float = 2.0) -> None:
        """叠加一个矩形检测框"""
        pen = QPen(color, line_width)
        pen.setCosmetic(True)  # 线宽不随缩放变化
        item = self._scene.addRect(rect, pen)
        self._overlay_items.append(item)

        if label:
            font = QFont("Microsoft YaHei", 10)
            text = self._scene.addSimpleText(label, font)
            text.setBrush(QBrush(color))
            text.setPen(QPen(color))
            text.setPos(rect.x(), rect.y() - 18)
            text.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, False)
            self._overlay_items.append(text)

    def add_polygon(self, points: list[QPointF], color: QColor = QColor("#4ECDC4"), label: str = "",
                    line_width: float = 2.0) -> None:
        """叠加一个多边形"""
        pen = QPen(color, line_width)
        pen.setCosmetic(True)
        poly = self._scene.addPolygon(points, pen)
        self._overlay_items.append(poly)

        if label and points:
            font = QFont("Microsoft YaHei", 10)
            text = self._scene.addSimpleText(label, font)
            text.setBrush(QBrush(color))
            text.setPen(QPen(color))
            text.setPos(points[0])
            text.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, False)
            self._overlay_items.append(text)

    # ---- 内部方法 ----
    def _clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    # ---- 事件处理 ----
    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = ZOOM_FACTOR if event.angleDelta().y() > 0 else 1 / ZOOM_FACTOR
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_dragging:
            delta = event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)
