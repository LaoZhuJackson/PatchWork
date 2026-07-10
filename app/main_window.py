"""主窗口：FluentWindow 侧边栏导航 + 页面切换"""
from __future__ import annotations

from qfluentwidgets import FluentWindow, NavigationItemPosition
from qfluentwidgets import FluentIcon as FIF
from PySide6.QtWidgets import QLabel

from app.widgets.dataset_split import DatasetSplitPanel
from app.widgets.export_onnx import ExportONNXPanel
from app.widgets.label_preview import LabelPreviewPanel
from app.widgets.model_infer import ModelInferPanel


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
