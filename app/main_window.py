"""主窗口：FluentWindow 侧边栏导航 + 页面切换（所有页面包在 ScrollArea 中）"""
from __future__ import annotations

from PySide6.QtCore import Qt
from qfluentwidgets import (
    FluentWindow, NavigationItemPosition, setTheme, Theme,
    ScrollArea,
)
from qfluentwidgets import FluentIcon as FIF

from app.utils.config import get_str, set_str
from app.widgets.dataset_split import DatasetSplitPanel
from app.widgets.export_onnx import ExportONNXPanel
from app.widgets.gpu_monitor import GPUMonitorPanel
from app.widgets.label_preview import LabelPreviewPanel
from app.widgets.model_infer import ModelInferPanel
from app.widgets.sahi_infer import SahiInferPanel
from app.widgets.video_extract import VideoExtractPanel
from app.widgets.xanylabeling import XAnyLabelingPanel


class MainWindow(FluentWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PatchWork")
        self.resize(900, 700)

        self.navigationInterface.setReturnButtonVisible(False)
        self.navigationInterface.setExpandWidth(160)

        self._placeholder = {
            "dataset_split": DatasetSplitPanel(),
            "model_infer": ModelInferPanel(),
            "label_preview": LabelPreviewPanel(),
            "export_onnx": ExportONNXPanel(),
            "video_extract": VideoExtractPanel(),
            "gpu_monitor": GPUMonitorPanel(),
            "xanylabeling": XAnyLabelingPanel(),
            "sahi_infer": SahiInferPanel(),
        }

        for name, widget in self._placeholder.items():
            widget.setObjectName(name)

        # 将所有面板包进 ScrollArea
        for name, panel in list(self._placeholder.items()):
            scroll = ScrollArea()
            scroll.setObjectName(name)
            scroll.setWidgetResizable(True)
            scroll.setWidget(panel)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setStyleSheet(
                "QScrollArea { background: transparent; border: none; }"
                "QScrollArea > QWidget > QWidget { background: transparent; }"
            )
            scroll.viewport().setStyleSheet("background: transparent;")
            self._placeholder[name] = scroll

        self._register_navigation()

    def _register_navigation(self) -> None:
        """注册导航项和子页面"""

        # ----- 导航栏上半区（功能入口） -----
        self.addSubInterface(
            self._placeholder["video_extract"],
            FIF.MEDIA, "视频抽帧",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["dataset_split"],
            FIF.APPLICATION, "数据集划分",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["gpu_monitor"],
            FIF.IOT, "GPU监控",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["model_infer"],
            FIF.PHOTO, "模型推理",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["sahi_infer"],
            FIF.ZOOM, "SAHI 推理",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["export_onnx"],
            FIF.SAVE_AS, "导出ONNX",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["label_preview"],
            FIF.TILES, "Label预览",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["xanylabeling"],
            FIF.LINK, "X-AnyLabeling",
            position=NavigationItemPosition.TOP,
        )

        # ----- 导航栏下半区（工具入口） -----
        # 主题切换按钮（不切换页面，仅触发回调）
        self.navigationInterface.addItem(
            routeKey="theme_toggle",
            icon=FIF.CONSTRACT,
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
