"""首页展示面板 — Banner + 功能模块卡片"""
from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import (
    QPixmap, QPainter, QColor, QBrush,
    QPainterPath, QLinearGradient, QFont,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CardWidget, BodyLabel, StrongBodyLabel,
    FluentIcon as FIF, isDarkTheme, FlowLayout,
)


# ---- 模块卡片数据 ----

_MODULES = [
    # (图标, 标题, 描述, 分组)
    (FIF.MEDIA,       "视频抽帧",     "按时间或帧间隔从视频提取帧"),
    (FIF.APPLICATION, "数据集划分",   "图片-标签配对检查 + 训练/验证/测试集划分"),
    (FIF.DOCUMENT,    "NDJSON转换",   "NDJSON 格式数据集 → YOLO 格式"),
    (FIF.IOT,         "GPU监控",      "远程 GPU 状态监控 (nvidia-smi / gpustat / HTTP)"),
    (FIF.TAG,         "开放词汇检测", "YOLOE 文本提示零样本检测，支持任意类别名"),
    (FIF.PHOTO,       "模型推理",     "YOLO 模型加载 + 单图推理 + 结果叠加"),
    (FIF.ZOOM,        "SAHI 推理",    "切片辅助推理，提升小目标检测精度"),
    (FIF.ALBUM,       "推理对比",     "多推理方式对比评测，输出每类 P/R/F1 表格"),
    (FIF.SAVE_AS,     "导出ONNX",     ".pt → ONNX 转换，支持 simplify / dynamic"),
    (FIF.TILES,       "Label预览",    "YOLO 标注文件可视化预览"),
    (FIF.LINK,        "X-AnyLabeling","一键启动外部标注工具"),
]


class _Banner(QWidget):
    """顶部 Banner：渐变背景 + 标题 + 副标题"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(160)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 28, 36, 20)
        layout.setSpacing(8)

        self._title = QLabel("PatchWork")
        self._title.setObjectName("homeTitle")
        self._title.setStyleSheet(
            "QLabel#homeTitle { font-size: 32px; font-weight: bold; }"
        )
        layout.addWidget(self._title)

        self._subtitle = QLabel(
            "视觉检测工具集 — 一站式管理数据集、模型推理、标注与评测"
        )
        self._subtitle.setStyleSheet("QLabel { font-size: 14px; }")
        layout.addWidget(self._subtitle)
        layout.addStretch()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        w, h = self.width(), self.height()
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), 8, 8)

        gradient = QLinearGradient(0, 0, 0, h)
        if isDarkTheme():
            gradient.setColorAt(0, QColor(45, 55, 70, 200))
            gradient.setColorAt(1, QColor(30, 35, 45, 80))
        else:
            gradient.setColorAt(0, QColor(200, 215, 230, 180))
            gradient.setColorAt(1, QColor(220, 228, 240, 60))
        painter.fillPath(path, QBrush(gradient))


class _ModuleCard(CardWidget):
    """单个功能模块卡片：图标 + 标题 + 描述"""

    def __init__(
        self, icon: FIF, title: str, desc: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        # 图标 + 标题
        header = QHBoxLayout()
        icon_widget = QLabel()
        icon_widget.setPixmap(icon.icon().pixmap(24, 24))
        header.addWidget(icon_widget)

        title_label = StrongBodyLabel(title)
        header.addWidget(title_label, 1)
        layout.addLayout(header)

        # 描述
        desc_label = BodyLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("QLabel { font-size: 12px; }")
        layout.addWidget(desc_label)

        layout.addStretch()


class HomePanel(QWidget):
    """首页 — PatchWork 功能总览"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("home_panel")

        layout = QVBoxLayout(self)
        layout.setSpacing(24)
        layout.setContentsMargins(24, 24, 24, 24)

        # ---- Banner ----
        layout.addWidget(_Banner(self))

        # ---- 模块卡片 ----
        title = StrongBodyLabel("功能模块")
        title.setStyleSheet("QLabel { font-size: 16px; }")
        layout.addWidget(title)

        card_container = QWidget()
        card_flow = FlowLayout(card_container, needAni=True)
        card_flow.setContentsMargins(0, 0, 0, 0)
        card_flow.setSpacing(12)

        for icon, name, desc in _MODULES:
            card_flow.addWidget(_ModuleCard(icon, name, desc, self))

        layout.addWidget(card_container, 1)
