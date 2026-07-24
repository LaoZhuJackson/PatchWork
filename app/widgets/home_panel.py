"""首页展示面板 — Banner + 链接卡片 + 功能模块卡片"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QUrl
from PySide6.QtGui import (
    QPixmap, QPainter, QColor, QBrush,
    QPainterPath, QLinearGradient, QDesktopServices,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CardWidget, BodyLabel, StrongBodyLabel, TitleLabel, CaptionLabel,
    FluentIcon as FIF, isDarkTheme, FlowLayout, IconWidget,
)


# ---- 模块卡片数据 ----

_MODULES = [
    # (图标, 标题, 描述)
    (FIF.MEDIA,       "视频抽帧",     "按时间或帧间隔从视频提取帧"),
    (FIF.APPLICATION, "数据集划分",   "图片-标签配对检查 + 训练/验证/测试集划分"),
    (FIF.DOCUMENT,    "NDJSON转换",   "NDJSON 格式数据集 → YOLO 格式"),
    (FIF.IOT,         "GPU监控",      "远程 GPU 状态监控"),
    (FIF.TAG,         "开放词汇检测", "YOLOE 文本提示零样本检测，支持任意类别名"),
    (FIF.PHOTO,       "模型推理",     "YOLO 模型加载 + 单图推理 + 结果叠加"),
    (FIF.ZOOM,        "SAHI 推理",    "切片辅助推理，提升小目标检测精度"),
    (FIF.ALBUM,       "推理对比",     "多推理方式对比评测，输出每类 P/R/F1 表格"),
    (FIF.SAVE_AS,     "导出ONNX",     ".pt → ONNX 转换"),
    (FIF.TILES,       "Label预览",    "YOLO 标注文件可视化预览"),
    (FIF.LINK,        "X-AnyLabeling","一键启动外部标注工具"),
]

_BANNER_PATH = (
    Path(__file__).resolve().parent.parent.parent / "resources" / "banner.svg"
)

_GITHUB_URL = "https://github.com/LaoZhuJackson/PatchWork"


# ---- 链接卡片 ----

class _LinkCard(CardWidget):
    """可点击的外部链接卡片 — 仿 Fluent Gallery LinkCard 风格

    大图标 + 标题 + 描述 + 右下角外链图标。
    """

    def __init__(
        self, icon: FIF, title: str, content: str, url: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._url = QUrl(url)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(198, 220)

        vbox = QVBoxLayout(self)
        vbox.setSpacing(0)
        vbox.setContentsMargins(24, 24, 0, 13)
        vbox.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # 大图标（带圆形背景色）
        self._icon_widget = IconWidget(icon, self)
        self._icon_widget.setFixedSize(54, 54)
        vbox.addWidget(self._icon_widget)
        vbox.addSpacing(16)

        # 标题 — StrongBodyLabel 继承 FluentLabelBase，自动响应主题色
        title_label = StrongBodyLabel(title, self)
        vbox.addWidget(title_label)
        vbox.addSpacing(8)

        # 描述 — CaptionLabel 同样支持主题
        desc_label = CaptionLabel(content, self)
        desc_label.setWordWrap(True)
        desc_label.setFixedWidth(150)
        vbox.addWidget(desc_label)

        # 右下角外链图标
        self._link_icon = IconWidget(FIF.LINK, self)
        self._link_icon.setFixedSize(16, 16)
        self._link_icon.move(170, 192)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        QDesktopServices.openUrl(self._url)


# ---- Banner ----

class _Banner(QWidget):
    """顶部 Banner：SVG 背景 + 渐变遮罩 + 底部淡出 + 标题 + 链接卡片"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(336)

        self._bg = QPixmap(str(_BANNER_PATH))

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 20, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # 标题
        title = TitleLabel("PatchWork", self)
        title.setContentsMargins(24, 0, 0, 0)
        layout.addWidget(title)

        layout.addSpacing(4)

        subtitle = StrongBodyLabel(
            "视觉检测工具集 —— 一站式管理数据集、模型推理、标注与评测", self
        )
        subtitle.setContentsMargins(24, 0, 0, 0)
        layout.addWidget(subtitle)

        # 链接卡片 — 浮动在 banner 底部
        links_row = QHBoxLayout()
        links_row.setContentsMargins(24, 12, 0, 24)
        links_row.setSpacing(12)
        links_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        links_row.addWidget(_LinkCard(
            FIF.GITHUB, "GitHub 仓库",
            "查看源码、提交 Issue\n或参与贡献",
            _GITHUB_URL, self,
        ))
        layout.addLayout(links_row, 1)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        w, h = self.width(), self.height()

        # 圆角裁剪路径
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), 8, 8)

        # 1. SVG 背景图
        if not self._bg.isNull():
            scaled = self._bg.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (scaled.width() - w) // 2
            y = (scaled.height() - h) // 2
            painter.setClipPath(path)
            painter.drawPixmap(-x, -y, scaled)
            painter.setClipping(False)

        # 2. 整体遮罩（保证文字可读）
        gradient = QLinearGradient(0, 0, 0, h)
        if isDarkTheme():
            gradient.setColorAt(0, QColor(30, 35, 45, 80))
            gradient.setColorAt(0.6, QColor(20, 22, 28, 160))
            gradient.setColorAt(1.0, QColor(20, 22, 28, 220))
        else:
            gradient.setColorAt(0, QColor(255, 255, 255, 40))
            gradient.setColorAt(0.6, QColor(220, 228, 240, 140))
            gradient.setColorAt(1.0, QColor(220, 228, 240, 200))
        painter.fillPath(path, QBrush(gradient))

        # 3. 底部淡出 — 遮罩从半透明渐变为完全透明，露出下方内容
        fade = QLinearGradient(0, h - 40, 0, h)
        fade.setColorAt(0, QColor(0, 0, 0, 0))
        if isDarkTheme():
            fade.setColorAt(1, QColor(20, 22, 28, 220))
        else:
            fade.setColorAt(1, QColor(220, 228, 240, 200))
        # 取交集: 只画 banner 底部 40px 的淡出带
        fade_path = QPainterPath()
        fade_path.addRect(QRectF(0, h - 40, w, 40))
        painter.fillPath(path.intersected(fade_path), QBrush(fade))


# ---- 模块卡片 ----

class _ModuleCard(CardWidget):
    """单个功能模块卡片：图标 + 标题 + 描述"""

    def __init__(
        self, icon: FIF, title: str, desc: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMinimumHeight(100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        header = QHBoxLayout()
        icon_widget = QLabel()
        icon_widget.setPixmap(icon.icon().pixmap(24, 24))
        header.addWidget(icon_widget)

        title_label = StrongBodyLabel(title)
        header.addWidget(title_label, 1)
        layout.addLayout(header)

        desc_label = BodyLabel(desc)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        layout.addStretch()


# ---- 首页面板 ----

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
        layout.addWidget(title)

        card_container = QWidget()
        card_flow = FlowLayout(card_container, needAni=True)
        card_flow.setContentsMargins(0, 0, 0, 0)
        card_flow.setSpacing(12)

        for icon, name, desc in _MODULES:
            card_flow.addWidget(_ModuleCard(icon, name, desc, self))

        layout.addWidget(card_container, 1)
