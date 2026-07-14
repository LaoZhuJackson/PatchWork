"""主窗口：FluentWindow 侧边栏导航 + 页面切换"""
from __future__ import annotations

from qfluentwidgets import FluentWindow, NavigationItemPosition, setTheme, Theme
from qfluentwidgets import FluentIcon as FIF
from PySide6.QtWidgets import QLabel

from app.utils.config import get_str, set_str
from app.widgets.dataset_split import DatasetSplitPanel
from app.widgets.export_onnx import ExportONNXPanel
from app.widgets.label_preview import LabelPreviewPanel
from app.widgets.model_infer import ModelInferPanel
from app.widgets.video_extract import VideoExtractPanel


class MainWindow(FluentWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PatchWork")
        self.resize(900, 700)

        self.navigationInterface.setReturnButtonVisible(False)
        self.navigationInterface.setExpandWidth(160)

        # 先建占位页面，后续逐步替换为真实面板
        self._placeholder = {
            "dataset_split": DatasetSplitPanel(),
            "model_infer": ModelInferPanel(),
            "label_preview": LabelPreviewPanel(),
            "export_onnx": ExportONNXPanel(),
            "video_extract": VideoExtractPanel(),
            "gpu_monitor": QLabel("🖥️ GPU监控"),
            "xanylabeling": QLabel("🔧 X-AnyLabeling"),
        }

        for name, widget in self._placeholder.items():
            widget.setObjectName(name)

        self._register_navigation()

    def _register_navigation(self) -> None:
        """注册导航项和子页面"""

        # ----- 导航栏上半区（功能入口） -----
        self.addSubInterface(
            self._placeholder["dataset_split"],
            FIF.APPLICATION, "数据集划分",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["model_infer"],
            FIF.PHOTO, "模型推理",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["label_preview"],
            FIF.TILES, "Label预览",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["export_onnx"],
            FIF.SAVE_AS, "导出ONNX",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["video_extract"],
            FIF.MEDIA, "视频抽帧",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["gpu_monitor"],
            FIF.IOT, "GPU监控",
            position=NavigationItemPosition.TOP,
        )

        # ----- 导航栏下半区（工具入口） -----
        self.addSubInterface(
            self._placeholder["xanylabeling"],
            FIF.LINK, "X-AnyLabeling",
            position=NavigationItemPosition.BOTTOM,
        )
        # 主题切换按钮（不切换页面，仅触发回调）
        icon = FIF.CONSTRACT
        self.navigationInterface.addItem(
            routeKey="theme_toggle",
            icon=icon,
            text="切换主题",
            onClick=self._toggle_theme,
            selectable=False,
            position=NavigationItemPosition.BOTTOM,
        )

    def _toggle_theme(self) -> None:
        """切换浅色/深色主题"""
        current = get_str("app_theme", "light")
        new = "dark" if current == "light" else "light"
        set_str("app_theme", new)
        setTheme(Theme.DARK if new == "dark" else Theme.LIGHT)
