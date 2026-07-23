"""F7: X-AnyLabeling 启动面板"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    PrimaryPushButton,
    BodyLabel, StrongBodyLabel, SubtitleLabel, CardWidget,
)

from app.services.xanylabeling import launch
from app.utils.config import get_str
from app.utils.message import info, warning, error
from app.widgets.path_browser import PathBrowser

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

        self.exe_browser = PathBrowser(
            label="", mode="file",
            file_filter="Executable Files (*.exe);;All Files (*)",
            placeholder="选择 X-AnyLabeling.exe ...",
            config_key="xanylabeling_exe",
        )
        exe_form.addRow(BodyLabel("exe 路径:"), self.exe_browser)

        layout.addWidget(exe_card)

        # ---- 文件夹 ----
        layout.addWidget(StrongBodyLabel("启动"))
        folder_card = CardWidget()
        folder_form = QFormLayout(folder_card)

        self.folder_browser = PathBrowser(
            label="", mode="dir",
            placeholder="选择要加载的数据集文件夹...",
            config_key="xanylabeling_folder",
        )
        folder_form.addRow(BodyLabel("数据集目录:"), self.folder_browser)

        layout.addWidget(folder_card)

        # ---- 启动按钮 ----
        self.launch_btn = PrimaryPushButton("启动 X-AnyLabeling")
        self.launch_btn.clicked.connect(self._on_launch)
        layout.addWidget(self.launch_btn, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addStretch()

    # ---- 事件 ----
    def _on_launch(self) -> None:
        exe = self.exe_browser.path
        folder = self.folder_browser.path

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
        self.exe_browser.path = get_str("xanylabeling_exe", "")
        self.folder_browser.path = get_str("xanylabeling_folder", "")