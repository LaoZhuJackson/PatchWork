"""F2: 模型推理 + 图片预览面板"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, LineEdit,
    BodyLabel, SubtitleLabel, CardWidget, DoubleSpinBox,
)

from app.services.inference import InferenceEngine
from app.services.label_reader import IMAGE_EXTS
from app.utils.config import get_str, set_str, set_float, get_float
from app.utils.message import error
from app.utils.worker import Worker
from app.widgets.image_browser import ImageBrowser
from app.utils.logger import get_logger

logger = get_logger(__name__)


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

    def __init__(self, engine: InferenceEngine, image_path: Path, conf: float, iou: float) -> None:
        super().__init__()
        self.setTerminationEnabled(True)
        self.engine = engine
        self.image_path = image_path
        self.conf = conf
        self.iou = iou

    def do_work(self) -> list[dict]:
        return self.engine.infer(self.image_path, self.conf, self.iou)


# ---- Panel ----

class ModelInferPanel(QWidget):
    """模型推理 + 图片预览面板"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("model_infer_panel")

        self._engine = InferenceEngine()
        self._image_files: list[Path] = []
        self._load_worker: LoadModelWorker | None = None
        self._infer_worker: InferWorker | None = None
        self._inferring: bool = False

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # ---- 标题 ----
        layout.addWidget(SubtitleLabel("模型推理"))

        # ---- 工具栏 ----
        toolbar_card = CardWidget()
        toolbar = QVBoxLayout(toolbar_card)
        toolbar.setContentsMargins(12, 8, 12, 8)
        toolbar.setSpacing(8)

        model_row = QHBoxLayout()

        model_row.addWidget(BodyLabel("模型目录:"))
        self.model_edit = LineEdit()
        self.model_edit.setPlaceholderText("选择 YOLO .pt 模型文件...")
        self.model_edit.setReadOnly(True)
        model_row.addWidget(self.model_edit, 1)

        model_btn = PushButton("浏览...")
        model_btn.clicked.connect(self._browse_model)
        model_row.addWidget(model_btn)
        toolbar.addLayout(model_row)

        img_row = QHBoxLayout()
        img_row.addWidget(BodyLabel("图片目录:"))
        self.folder_edit = LineEdit()
        self.folder_edit.setPlaceholderText("选择图片文件夹...")
        self.folder_edit.setReadOnly(True)
        img_row.addWidget(self.folder_edit, 1)

        folder_btn = PushButton("浏览...")
        folder_btn.clicked.connect(self._browse_folder)
        img_row.addWidget(folder_btn)
        toolbar.addLayout(img_row)

        # ---- 阈值设置 ----
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(BodyLabel("Conf:"))
        self.conf_spin = DoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setToolTip("置信度阈值")
        self.conf_spin.valueChanged.connect(
            lambda v: set_float("infer_conf", v)
        )
        threshold_row.addWidget(self.conf_spin)

        threshold_row.addWidget(BodyLabel("IoU:"))
        self.iou_spin = DoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setToolTip("NMS IoU 阈值")
        self.iou_spin.valueChanged.connect(
            lambda v: set_float("infer_iou", v)
        )
        threshold_row.addWidget(self.iou_spin)

        self.reinfer_btn = PrimaryPushButton("重新推理")
        self.reinfer_btn.setToolTip("用当前阈值对当前图片重新推理")
        self.reinfer_btn.clicked.connect(self._on_reinfer)
        threshold_row.addWidget(self.reinfer_btn)

        threshold_row.addStretch()

        toolbar.addLayout(threshold_row)

        layout.addWidget(toolbar_card)

        # ---- 状态标签 ----
        self.status_label = BodyLabel("请先导入模型并选择图片目录")
        layout.addWidget(self.status_label)

        # ---- 图片浏览器（缩略图 + Viewer + 导航按钮） ----
        self.browser = ImageBrowser()
        self.browser.image_selected.connect(self._on_image_selected)
        layout.addWidget(self.browser, 1)

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
        logger.info(f"模型已加载: {Path(path).name}{task_label} ({count} 个类别)")
        self.status_label.setText(f"✅ 模型已加载: {Path(path).name}{task_label} ({count} 个类别)")

        # 如果已有图片列表，自动推理当前选中的图
        if self.browser.current_path is not None and self._engine.is_loaded:
            self._run_inference()

    def _on_model_error(self, err: str) -> None:
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
        images = sorted(
            f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )

        if not images:
            self.status_label.setText("❌ 未找到图片文件")
            self.browser.clear()
            return

        self._image_files = images

        model_status = "已加载" if self._engine.is_loaded else "未加载"
        self.status_label.setText(f"共 {len(images)} 张图片 | 模型: {model_status}")

        self.browser.set_images(images, select_index=0)

        # 自动推理第一张
        if self._engine.is_loaded:
            self._run_inference()

    # ---- 推理 ----

    def _on_image_selected(self, idx: int, path: Path) -> None:
        """缩略图点击或导航按钮"""
        if self._engine.is_loaded:
            self._run_inference()

    def _run_inference(self) -> None:
        if not self._engine.is_loaded:
            return

        path = self.browser.current_path
        if path is None:
            return

        # 正在推理则终止旧线程
        if self._inferring and self._infer_worker and self._infer_worker.isRunning():
            self._infer_worker.terminate()
            self._infer_worker.wait(3000)

        self._inferring = True
        self._update_ui_state()
        self.status_label.setText(f"正在推理: {path.name} ...")

        conf = self.conf_spin.value()
        iou = self.iou_spin.value()

        self._infer_worker = InferWorker(self._engine, path, conf, iou)
        self._infer_worker.finished.connect(self._on_infer_done)
        self._infer_worker.error.connect(self._on_infer_error)
        self._infer_worker.start()

    def _on_infer_done(self, annotations: list[dict]) -> None:
        path = self.browser.current_path
        self.browser.show_annotations(annotations)
        self.status_label.setText(
            f"{path.name if path else '?'} | 检测到 {len(annotations)} 个目标"
        )
        self._inferring = False
        self._update_ui_state()

    def _on_infer_error(self, err: str) -> None:
        self.status_label.setText("❌ 推理失败")
        error("推理失败", err, self)
        self._inferring = False
        self._update_ui_state()

    def _update_ui_state(self) -> None:
        """推理中禁用图片列表"""
        self.browser.thumb_list.setDisabled(self._inferring)
        self.browser.set_nav_enabled(not self._inferring)

    def _on_reinfer(self) -> None:
        """手动触发重新推理"""
        if self._engine.is_loaded and self.browser.current_path is not None:
            self._run_inference()

    # ---- 持久化 ----

    def _load_settings(self) -> None:
        self.model_edit.setText(get_str("infer_model_path"))
        self.folder_edit.setText(get_str("infer_folder_path"))
        self.conf_spin.setValue(get_float("infer_conf", 0.25))
        self.iou_spin.setValue(get_float("infer_iou", 0.45))

        model_path = get_str("infer_model_path")
        if model_path and Path(model_path).is_file():
            self._load_model(model_path)

        folder = get_str("infer_folder_path")
        if folder:
            p = Path(folder)
            if p.is_dir():
                self._load_images(p)
