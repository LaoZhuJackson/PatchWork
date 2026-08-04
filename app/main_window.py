"""主窗口：FluentWindow 侧边栏导航 + 页面切换（所有页面包在 ScrollArea 中）"""
from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication
from qfluentwidgets import (
    FluentWindow, NavigationItemPosition, setTheme, Theme,
    ScrollArea, SplashScreen,
)
from qfluentwidgets import FluentIcon as FIF

from app.utils.config import get_str, set_str
from app.widgets.benchmark import BenchmarkPanel
from app.widgets.dataset_split import DatasetSplitPanel
from app.widgets.export_onnx import ExportONNXPanel
from app.widgets.framediff_dataset import FrameDiffDatasetPanel
from app.widgets.gpu_monitor import GPUMonitorPanel
from app.widgets.home_panel import HomePanel
from app.widgets.image_synthesis import ImageSynthesisPanel
from app.widgets.json_manager import JsonManagerPanel
from app.widgets.label_preview import LabelPreviewPanel
from app.widgets.model_infer import ModelInferPanel
from app.widgets.ndjson_convert import NDJSONConvertPanel
from app.widgets.open_vocab_detect import OpenVocabDetectPanel
from app.widgets.presentation import PresentationPanel
from app.widgets.sahi_infer import SahiInferPanel
from app.widgets.video_extract import VideoExtractPanel
from app.widgets.xanylabeling import XAnyLabelingPanel
from app.widgets.pseudo_thermal import PseudoThermalPanel
from app.widgets.irvis_annotator import IRVISAnnotatorPanel

from pathlib import Path

_ICON_PATH = (
    Path(__file__).resolve().parent.parent / "resources" / "icon.svg"
)

class MainWindow(FluentWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PatchWork")

        # 显式覆盖最小尺寸，防止导航栏/面板的 minimumSizeHint 撑大窗口
        self.setMinimumSize(900, 700)

        self.setWindowIcon(QIcon(QPixmap(str(_ICON_PATH))))

        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(96, 96))

        screen = QApplication.primaryScreen()
        if screen:
            rect = screen.availableGeometry()
            self.move(
                rect.x() + (rect.width() - self.width()) // 2,
                rect.y() + (rect.height() - self.height()) // 2,
            )
        self.show()
        QApplication.processEvents()

        self.navigationInterface.setReturnButtonVisible(False)
        self.navigationInterface.setExpandWidth(160)

        self._placeholder = {
            "home": HomePanel(),
            "image_synthesis": ImageSynthesisPanel(),
            "dataset_split": DatasetSplitPanel(),
            "model_infer": ModelInferPanel(),
            "label_preview": LabelPreviewPanel(),
            "export_onnx": ExportONNXPanel(),
            "video_extract": VideoExtractPanel(),
            "gpu_monitor": GPUMonitorPanel(),
            "xanylabeling": XAnyLabelingPanel(),
            "sahi_infer": SahiInferPanel(),
            "benchmark": BenchmarkPanel(),
            "open_vocab_detect": OpenVocabDetectPanel(),
            "ndjson_convert": NDJSONConvertPanel(),
            "framediff_dataset": FrameDiffDatasetPanel(),
            "json_manager": JsonManagerPanel(),
            "presentation": PresentationPanel(),
            "pseudo_thermal": PseudoThermalPanel(),
            "irvis_annotator": IRVISAnnotatorPanel(),
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

        # 加载完毕，关闭splash
        self.splashScreen.finish()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, 'splashScreen') and self.splashScreen.isVisible():
            self.splashScreen.resize(self.size())

    def _register_navigation(self) -> None:
        """注册导航项和子页面"""

        self.addSubInterface(
            self._placeholder["home"],
            FIF.HOME, "首页",
            position=NavigationItemPosition.TOP,
        )
        # ----- 导航栏上半区（功能入口） -----
        self.addSubInterface(
            self._placeholder["presentation"],
            FIF.PROJECTOR,
            "PPT 汇报",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["pseudo_thermal"],
            FIF.CALORIES, "伪热红外增强",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["irvis_annotator"],
            FIF.PIN, "IR-VIS 标注",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["image_synthesis"],
            FIF.TRANSPARENT, "目标合成器",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["ndjson_convert"],
            FIF.CODE, "NDJSON转换",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["video_extract"],
            FIF.MEDIA, "视频抽帧",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["framediff_dataset"],
            FIF.VIDEO, "帧差数据集",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._placeholder["json_manager"],
            FIF.FOLDER, "JSON 管理",
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
            self._placeholder["open_vocab_detect"],
            FIF.TAG, "开放词汇检测",
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
            self._placeholder["benchmark"],
            FIF.ALBUM, "推理对比",
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
