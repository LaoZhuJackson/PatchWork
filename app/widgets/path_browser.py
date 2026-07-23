"""可复用的路径浏览控件：模式选择 + 标签 + 路径输入框 + 浏览按钮

支持两种模式：
  - "file": 选择单个文件（可指定后缀过滤）
  - "dir":  选择文件夹

可选 config_key 自动将路径写入 QSettings。

用法:
    from app.widgets.path_browser import PathBrowser

    # 文件模式
    model_browser = PathBrowser(
        label="模型文件:",
        mode="file",
        file_filter="Model Files (*.pt *.pth);;All Files (*)",
        placeholder="选择 YOLO .pt 模型...",
        config_key="infer_model_path",
    )
    model_browser.path_changed.connect(self._on_model_selected)

    # 目录模式
    folder_browser = PathBrowser(
        label="图片目录:", mode="dir",
        placeholder="选择图片文件夹...",
        config_key="infer_folder_path",
    )
    folder_browser.path_changed.connect(self._on_folder_selected)

    # 读写当前路径
    print(model_browser.path)
    model_browser.path = "D:/models/yolo.pt"
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    PushButton, LineEdit, BodyLabel,
)

from app.utils.config import set_str


class PathBrowser(QWidget):
    """可复用路径浏览器。

    path_changed 信号在用户通过浏览按钮选择新路径后发射。
    通过程序设置 path 属性不会触发信号。
    """

    path_changed = Signal(str)

    def __init__(
        self,
        *,
        label: str = "",
        mode: str = "file",
        file_filter: str = "All Files (*)",
        placeholder: str = "",
        config_key: str = "",
        dialog_title: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._file_filter = file_filter
        self._config_key = config_key
        self._dialog_title = dialog_title or label

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if label:
            layout.addWidget(BodyLabel(label))

        self._edit = LineEdit()
        self._edit.setPlaceholderText(placeholder)
        self._edit.setReadOnly(True)
        layout.addWidget(self._edit, 1)

        icon = "📄" if mode == "file" else "📁"
        btn = PushButton(icon)
        btn.clicked.connect(self._browse)
        layout.addWidget(btn)

    # ---- 属性 ----

    @property
    def path(self) -> str:
        """当前路径"""
        return self._edit.text().strip()

    @path.setter
    def path(self, value: str) -> None:
        """设置路径（不触发 path_changed 信号）"""
        self._edit.setText(value)
        if self._config_key and value:
            set_str(self._config_key, value)

    # ---- 内部 ----

    def _browse(self) -> None:
        if self._mode == "dir":
            path = QFileDialog.getExistingDirectory(
                self, self._dialog_title, self.path
            )
            if not path:
                return
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, self._dialog_title, self.path, self._file_filter
            )
            if not path:
                return

        self._edit.setText(path)
        if self._config_key:
            set_str(self._config_key, path)
        self.path_changed.emit(path)
