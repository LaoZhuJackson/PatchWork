"""MessageBox 封装，统一 Fluent 风格，长内容自动滚动"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QApplication
from qfluentwidgets import MessageBox, ScrollArea


def _setup_scrollable(w: MessageBox, parent=None) -> None:
    """限制对话框高度不超过父窗口 85%，内容过长时包装为可滚动区域"""
    # 计算最大高度
    if parent is not None:
        max_h = int(parent.window().height() * 0.85)
    else:
        screen = QApplication.primaryScreen()
        max_h = int(screen.availableGeometry().height() * 0.85) if screen else 600

    # 去掉 MessageBox 的固定尺寸，改为动态高度（固定尺寸会让长内容撑破屏幕）
    w.widget.setMinimumSize(300, 200)
    w.widget.setMaximumHeight(max_h)

    # 把 contentLabel 包装进 QScrollArea（无论长短都包，短内容时滚动条不出现）
    scroll = ScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(ScrollArea.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    # scroll.setStyleSheet(
    #     "QScrollArea { background: transparent; border: none; }"
    # )

    # 从 textLayout 中取出 contentLabel，换成 scroll area
    idx = w.textLayout.indexOf(w.contentLabel)
    if idx >= 0:
        w.textLayout.removeWidget(w.contentLabel)
        w.contentLabel.setParent(None)
        scroll.setWidget(w.contentLabel)
        w.textLayout.insertWidget(idx, scroll, 1)


def info(title: str, content: str, parent=None) -> None:
    w = MessageBox(title, content, parent)
    w.cancelButton.hide()
    _setup_scrollable(w, parent)
    w.exec()


def warning(title: str, content: str, parent=None) -> None:
    w = MessageBox(title, content, parent)
    w.yesButton.setText("确定")
    w.cancelButton.hide()
    _setup_scrollable(w, parent)
    w.exec()


def error(title: str, content: str, parent=None) -> None:
    w = MessageBox(title, content, parent)
    w.yesButton.setText("确定")
    w.cancelButton.hide()
    _setup_scrollable(w, parent)
    w.exec()


def confirm(title: str, content: str, parent=None) -> bool:
    """确认对话框，返回 True(是) / False(否)"""
    w = MessageBox(title, content, parent)
    w.yesButton.setText("是")
    w.cancelButton.setText("否")
    _setup_scrollable(w, parent)
    return bool(w.exec())
