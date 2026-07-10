"""MessageBox 封装，统一 Fluent 风格"""
from __future__ import annotations

from qfluentwidgets import MessageBox


def info(title: str, content: str, parent=None) -> None:
  w = MessageBox(title, content, parent)
  w.cancelButton.hide()
  w.exec()


def warning(title: str, content: str, parent=None) -> None:
  w = MessageBox(title, content, parent)
  w.yesButton.setText("确定")
  w.cancelButton.hide()
  w.exec()


def error(title: str, content: str, parent=None) -> None:
  w = MessageBox(title, content, parent)
  w.yesButton.setText("确定")
  w.cancelButton.hide()
  w.exec()


def confirm(title: str, content: str, parent=None) -> bool:
  """确认对话框，返回 True(是) / False(否)"""
  w = MessageBox(title, content, parent)
  w.yesButton.setText("是")
  w.cancelButton.setText("否")
  return bool(w.exec())