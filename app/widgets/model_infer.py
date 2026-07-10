"""F2: 模型推理 + 图片预览面板"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    PushButton,
    LineEdit,
    SubtitleLabel,
    BodyLabel,
    FluentIcon as FIF,
)

from app.services.inference import InferenceEngine
from app.services.label_reader import IMAGE_EXTS
from app.utils.config import get_str, set_str
from app.utils.message import error
from app.utils.worker import Worker
from app.widgets.image_viewer import ImageViewer

from app.utils.logger import get_logger

logger = get_logger(__name__)

THUMB_SIZE = QSize(100, 80)


# ---- Workers ----
class LoadModelWorker(Worker):
    """后台加载模型"""

    def __init__(self, engine: InferenceEngine, model_path: str) -> None:
        super().__init__()
        self.engine = engine
        self.model_path = model_path

    def do_work(self) -> str:
        self.engine.load_model(self.model_path)
        return self.model_path


class InferWorker(Worker):
    """后台推理单张图片"""

    def __init__(self, engine: InferenceEngine, image_path: Path) -> None:
        super().__init__()
        self.engine = engine
        self.image_path = image_path

    def do_work(self) -> list[dict]:
        return self.engine.infer(self.image_path)


# ---- Panel ----
class ModelInferPanel(QWidget):
    """模型推理 + 图片预览面板"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("model_infer_panel")

        self._engine = InferenceEngine()
        self._image_files: list[Path] = []
        self._current_index: int = -1
        self._load_worker: LoadModelWorker | None = None
        self._infer_worker: InferWorker | None = None

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # ---- 标题 ----
        layout.addWidget(SubtitleLabel("模型推理"))

        # ---- 工具栏 ----
        toolbar_group = QGroupBox("")
        toolbar = QHBoxLayout(toolbar_group)
        toolbar.setContentsMargins(12, 8, 12, 8)
        toolbar.setSpacing(12)

        toolbar.addWidget(BodyLabel("模型:"))
        self.model_edit = LineEdit()
        self.model_edit.setPlaceholderText("选择 YOLO .pt 模型文件...")
        self.model_edit.setReadOnly(True)
        toolbar.addWidget(self.model_edit, 1)

        model_btn = PushButton("浏览...")
        model_btn.clicked.connect(self._browse_model)
        toolbar.addWidget(model_btn)

        toolbar.addWidget(BodyLabel("图片目录:"))
        self.folder_edit = LineEdit()
        self.folder_edit.setPlaceholderText("选择图片文件夹...")
        self.folder_edit.setReadOnly(True)
        toolbar.addWidget(self.folder_edit, 1)

        folder_btn = PushButton("浏览...")
        folder_btn.clicked.connect(self._browse_folder)
        toolbar.addWidget(folder_btn)

        layout.addWidget(toolbar_group)

        # ---- 状态标签 ----
        self.status_label = BodyLabel("请先导入模型并选择图片目录")
        layout.addWidget(self.status_label)

        # ---- 缩略图列表 ----
        layout.addWidget(BodyLabel("图片列表:"))

        self.image_list = QListWidget()
        self.image_list.setFlow(QListWidget.Flow.LeftToRight)
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setIconSize(THUMB_SIZE)
        self.image_list.setGridSize(QSize(THUMB_SIZE.width() + 12, THUMB_SIZE.height() + 30))
        self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list.setWrapping(False)
        self.image_list.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.image_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_list.setFixedHeight(THUMB_SIZE.height() + 40)
        self.image_list.itemClicked.connect(self._on_item_clicked)
        self.image_list.wheelEvent = self._list_wheel_event
        layout.addWidget(self.image_list)

        # ---- 预览区 + 导航按钮 ----
        preview_container = QWidget()
        preview_container.setMinimumHeight(300)
        container_layout = QHBoxLayout(preview_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self.viewer = ImageViewer()
        container_layout.addWidget(self.viewer)

        btn_style = """
                      QPushButton {
                          background: rgba(0, 0, 0, 0.35);
                          color: white;
                          border: none;
                          border-radius: 24px;
                          font-size: 20px;
                          min-width: 48px;
                          max-width: 48px;
                          min-height: 48px;
                          max-height: 48px;
                      }
                      QPushButton:hover {
                          background: rgba(0, 0, 0, 0.65);
                      }
                  """

        self.prev_btn = QPushButton("◀")
        self.prev_btn.setToolTip("上一张")
        self.prev_btn.setStyleSheet(btn_style)
        self.prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_btn.setParent(self.viewer)
        self.prev_btn.clicked.connect(self._prev_image)

        self.next_btn = QPushButton("▶")
        self.next_btn.setToolTip("下一张")
        self.next_btn.setStyleSheet(btn_style)
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.setParent(self.viewer)
        self.next_btn.clicked.connect(self._next_image)

        layout.addWidget(preview_container, 1)

        # ---- 滚轮 ----

    def _list_wheel_event(self, event) -> None:
        delta = event.angleDelta().y()
        bar = self.image_list.horizontalScrollBar()
        bar.setValue(bar.value() - delta)

        # ---- 按钮定位 ----

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_nav_buttons()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, self._position_nav_buttons)

    def _position_nav_buttons(self) -> None:
        vw = self.viewer.width()
        vh = self.viewer.height()
        btn_h = 48
        center_y = (vh - btn_h) // 2
        self.prev_btn.move(12, center_y)
        self.next_btn.move(vw - 60, center_y)

    # ---- 模型加载 ----
    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型", self.model_edit.text(),
            "Model Files (*.pt *.pth);;All Files (*)"
        )
        if not path:
            return
        self.model_edit.setText(path)
        set_str("infer_model_path", path)
        self._load_model(path)

    def _load_model(self, path: str) -> None:
        self.status_label.setText("正在加载模型...")
        self._load_worker = LoadModelWorker(self._engine, path)
        self._load_worker.finished.connect(self._on_model_loaded)
        self._load_worker.error.connect(self._on_model_error)
        self._load_worker.start()

    def _on_model_loaded(self, path: str) -> None:
        names = self._engine.class_names
        task = self._engine.task
        count = len(names)
        task_label = f" ({task})" if task else ""
        logger.info(f"✅ 模型已加载: {Path(path).name}{task_label} ({count} 个类别)")
        # self.status_label.setText(f"✅ 模型已加载: {Path(path).name}{task_label} ({count} 个类别)")

        # 如果存在图片列表自动推理当前选中的图
        if 0 <= self._current_index < len(self._image_files):
            self._run_inference(self._current_index)

    def _on_model_error(self, err:str) -> None:
        self.status_label.setText("❌ 模型加载失败")
        error("模型加载失败", err, self)

    # ---- 文件夹加载 ----
    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择图片文件夹", self.folder_edit.text()
        )
        if not path:
            return
        self.folder_edit.setText(path)
        set_str("infer_folder_path", path)
        self._load_images(Path(path))

    def _load_images(self, directory: Path) -> None:
        self._image_files.clear()
        self.image_list.clear()
        self.viewer.set_image(None)
        self._current_index = -1
        self._update_nav_buttons()

        images = sorted(
            f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )

        if not images:
            self.status_label.setText("❌ 未找到图片文件")
            return

        self._image_files = images

        for img in images:
            icon = QPixmap(str(img)).scaled(
                THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item = QListWidgetItem(icon, img.name)
            item.setData(Qt.ItemDataRole.UserRole, str(img))
            item.setToolTip(img.name)
            self.image_list.addItem(item)

        model_status = "已加载" if self._engine.is_loaded else "未加载"
        self.status_label.setText(f"共 {len(images)} 张图片 | 模型: {model_status}")

        # 自动选中第一张
        if self._image_files:
            self.image_list.setCurrentRow(0)
            self._current_index = 0
            self._update_nav_buttons()
            if self._engine.is_loaded:
                self._run_inference(0)

    # ---- 推理 ----
    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        idx = self.image_list.row(item)
        if self._engine.is_loaded:
            self._run_inference(idx)

    def _run_inference(self, idx: int) -> None:
        if not self._engine.is_loaded:
            return
        if idx <0 or idx >= len(self._image_files):
            return

        self._current_index = idx
        self._update_nav_buttons()
        self.image_list.setCurrentRow(idx)

        img_path = self._image_files[idx]
        self.status_label.setText(f"正在推理: {img_path.name} ...")

        self._infer_worker = InferWorker(self._engine, img_path)
        self._infer_worker.finished.connect(self._on_infer_done)
        self._infer_worker.error.connect(self._on_infer_error)
        self._infer_worker.start()

    def _on_infer_done(self, annotations: list[dict]) -> None:
        idx = self._current_index
        if idx < 0 or idx >= len(self._image_files):
            return
        img_path = self._image_files[idx]
        pixmap = QPixmap(str(img_path))
        if pixmap.isNull():
            return

        self.viewer.set_image(pixmap)
        for ann in annotations:
            if ann["type"] == "bbox":
                self.viewer.add_bbox(ann["rect"], ann["color"], ann["label"])
            elif ann["type"] == "polygon":
                self.viewer.add_polygon(ann["points"], ann["color"], ann["label"])

        self.status_label.setText(
            f"{img_path.name} | 检测到 {len(annotations)} 个目标"
        )

    def _on_infer_error(self, err: str) -> None:
        self.status_label.setText("❌ 推理失败")
        error("推理失败", err, self)

    # ---- 导航 ----

    def _prev_image(self) -> None:
        if not self._image_files or not self._engine.is_loaded:
            return
        new_idx = max(0, self._current_index - 1)
        self._run_inference(new_idx)

    def _next_image(self) -> None:
        if not self._image_files or not self._engine.is_loaded:
            return
        new_idx = min(len(self._image_files) - 1, self._current_index + 1)
        self._run_inference(new_idx)

    def _update_nav_buttons(self) -> None:
        total = len(self._image_files)
        self.prev_btn.setEnabled(self._current_index > 0)
        self.next_btn.setEnabled(self._current_index < total - 1)

    # ---- 持久化 ----

    def _load_settings(self) -> None:
        self.model_edit.setText(get_str("infer_model_path"))
        self.folder_edit.setText(get_str("infer_folder_path"))

        # 先恢复模型（异步），再恢复文件夹
        model_path = get_str("infer_model_path")
        if model_path and Path(model_path).is_file():
            self._load_model(model_path)

        folder = get_str("infer_folder_path")
        if folder:
            p = Path(folder)
            if p.is_dir():
                self._load_images(p)
