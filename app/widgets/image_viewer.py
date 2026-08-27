"""通用图片查看器：QGraphicsView 封装，支持缩放、拖拽、叠加绘制"""
from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
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
    """可缩放、拖拽的图片查看器，支持在图片上叠加框、多边形、文字

    交互模式:
      - "pan" (默认): 左键拖拽平移, 滚轮缩放
      - "pick": 左键发射 clicked 信号(场景坐标), 中键拖拽平移, 滚轮缩放
    """

    clicked = Signal(QPointF)  # 点击信号（仅在 pick 模式下发射场景坐标）
    rect_drawn = Signal(QRectF)  # 拖拽画框完成信号（仅在 draw 模式下发射场景坐标矩形）

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # 背景图 item
        self._pixmap_item = self._scene.addPixmap(QPixmap())

        # 交互设置
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        self.setBackgroundBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setStyleSheet("QGraphicsView { background: transparent; }")

        self._overlay_items: list = []
        self._is_dragging = False
        self._last_mouse_pos = QPointF()
        self._interaction_mode = "pan"  # "pan" | "pick" | "draw"
        self._draw_start: QPointF | None = None
        self._draw_item = None

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

    def set_interaction_mode(self, mode: str) -> None:
        """设置交互模式: "pan"(拖拽平移) | "pick"(点击选点) | "draw"(拖拽画框)"""
        # 离开 draw 时清理未完成的预览矩形
        if mode != "draw" and self._draw_item is not None:
            self._scene.removeItem(self._draw_item)
            self._draw_item = None
            self._draw_start = None
        self._interaction_mode = mode
        if mode in ("pick", "draw"):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

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

    def add_text(self, pos: QPointF, text: str, color: QColor = QColor("#FFFFFF"),
                 size: int = 12) -> None:
        """叠加纯文字（用于标注编号等）"""
        font = QFont()
        font.setPointSize(size)
        item = self._scene.addSimpleText(text, font)
        item.setBrush(QBrush(color))
        item.setPen(QPen(color))
        item.setPos(pos)
        item.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, False)
        self._overlay_items.append(item)

    def add_bbox(self, rect: QRectF, color: QColor = QColor("#FF6B6B"), label: str = "",
                 line_width: float = 2.0) -> None:
        """叠加一个矩形检测框"""
        pen = QPen(color, line_width)
        pen.setCosmetic(True)  # 线宽不随缩放变化
        item = self._scene.addRect(rect, pen)
        self._overlay_items.append(item)

        if label:
            font = QFont()
            font.setPointSize(10)
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
            font = QFont()
            font.setPointSize(10)
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
        if self._interaction_mode == "draw":
            if event.button() == Qt.MouseButton.LeftButton:
                self._draw_start = self.mapToScene(event.position().toPoint())
                pen = QPen(QColor(255, 0, 0), 2)
                pen.setCosmetic(True)
                self._draw_item = self._scene.addRect(
                    QRectF(self._draw_start, self._draw_start), pen)
                event.accept()
                return
            elif event.button() == Qt.MouseButton.MiddleButton:
                self._is_dragging = True
                self._last_mouse_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        elif self._interaction_mode == "pick":
            if event.button() == Qt.MouseButton.LeftButton:
                scene_pos = self.mapToScene(event.position().toPoint())
                self.clicked.emit(scene_pos)
                event.accept()
                return
            elif event.button() == Qt.MouseButton.MiddleButton:
                self._is_dragging = True
                self._last_mouse_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        else:
            if event.button() == Qt.MouseButton.LeftButton:
                self._is_dragging = True
                self._last_mouse_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._interaction_mode == "draw" and self._draw_start is not None:
            cur = self.mapToScene(event.position().toPoint())
            self._draw_item.setRect(QRectF(self._draw_start, cur).normalized())
            event.accept()
            return
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
        if self._interaction_mode == "draw":
            if event.button() == Qt.MouseButton.LeftButton and self._draw_start is not None:
                cur = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._draw_start, cur).normalized()
                if self._draw_item is not None:
                    self._scene.removeItem(self._draw_item)
                    self._draw_item = None
                self._draw_start = None
                if rect.width() >= 3 and rect.height() >= 3:
                    self.rect_drawn.emit(rect)
                event.accept()
                return
            elif event.button() == Qt.MouseButton.MiddleButton:
                self._is_dragging = False
                self.setCursor(Qt.CursorShape.CrossCursor)
                event.accept()
                return
        elif self._interaction_mode == "pick":
            if event.button() == Qt.MouseButton.MiddleButton:
                self._is_dragging = False
                self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            if event.button() == Qt.MouseButton.LeftButton:
                self._is_dragging = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)
