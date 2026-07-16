"""F7: X-AnyLabeling 启动面板"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
  QFileDialog,
  QFormLayout,
  QHBoxLayout,
  QVBoxLayout,
  QWidget,
)
from qfluentwidgets import (
  PushButton, PrimaryPushButton, LineEdit,
  BodyLabel, StrongBodyLabel, SubtitleLabel, CardWidget,
)

from app.services.xanylabeling import launch
from app.utils.config import get_str, set_str
from app.utils.message import info, warning, error

class XAnyLabelingPanel(QWidget):
    """X-AnyLabeling 启动面板"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("xanylabeling_panel")
        self._process = None

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("X-AnyLabeling"))

        # ---- exe 路径 ----
        layout.addWidget(StrongBodyLabel("配置"))
        exe_card = CardWidget()
        exe_form = QFormLayout(exe_card)

        exe_row = QHBoxLayout()
        self.exe_edit = LineEdit()
        self.exe_edit.setPlaceholderText("选择 X-AnyLabeling.exe ...")
        exe_btn = PushButton("📁")
        exe_btn.clicked.connect(self._browse_exe)
        exe_row.addWidget(self.exe_edit, 1)
        exe_row.addWidget(exe_btn)
        exe_form.addRow(BodyLabel("exe 路径:"), exe_row)

        layout.addWidget(exe_card)

        # ---- 文件夹 ----
        layout.addWidget(StrongBodyLabel("启动"))
        folder_card = CardWidget()
        folder_form = QFormLayout(folder_card)

        folder_row = QHBoxLayout()
        self.folder_edit = LineEdit()
        self.folder_edit.setPlaceholderText("选择要加载的数据集文件夹...")
        folder_btn = PushButton("📁")
        folder_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(folder_btn)
        folder_form.addRow(BodyLabel("数据集目录:"), folder_row)

        layout.addWidget(folder_card)

        # ---- 启动按钮 ----
        self.launch_btn = PrimaryPushButton("启动 X-AnyLabeling")
        self.launch_btn.clicked.connect(self._on_launch)
        layout.addWidget(self.launch_btn, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addStretch()

    # ---- 事件 ----
    def _browse_exe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 X-AnyLabeling 可执行文件", "", "Executable Files (*.exe);;All Files (*)"
        )
        if path:
            self.exe_edit.setText(path)
            set_str("xanylabeling_exe", path)

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择数据集文件夹", self.folder_edit.text()
        )
        if path:
            self.folder_edit.setText(path)
            set_str("xanylabeling_folder", path)

    def _on_launch(self) -> None:
        exe = self.exe_edit.text().strip()
        folder = self.folder_edit.text().strip()

        if not exe or not Path(exe).is_file():
            warning("错误", "请先选择 X-AnyLabeling 可执行文件路径", self)
            return
        if not folder or not Path(folder).is_dir():
            warning("错误", "请先选择要加载的数据集文件夹", self)
            return

        try:
            self._process = launch(exe, folder)
            info("已启动", f"X-AnyLabeling 正在加载:\n{folder}", self)
        except FileNotFoundError as e:
            error("启动失败", str(e), self)
        except OSError as e:
            error("启动失败", f"无法启动 X-AnyLabeling:\n{e}", self)

    # ---- 持久化 ----
    def _load_settings(self) -> None:
        self.exe_edit.setText(get_str("xanylabeling_exe", ""))
        self.folder_edit.setText(get_str("xanylabeling_folder", ""))