"""开放词汇检测面板：YOLOE + 文本提示 → 零样本检测"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, LineEdit, ComboBox,
    BodyLabel, SubtitleLabel, CardWidget, DoubleSpinBox,
)

from app.services.open_vocab import OpenVocabEngine
from app.services.label_reader import IMAGE_EXTS
from app.utils.config import get_str, set_str, set_float, get_float
from app.utils.message import error
from app.utils.worker import Worker
from app.widgets.image_browser import ImageBrowser
from app.widgets.path_browser import PathBrowser
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---- 模型变体（显示名 → 文件名） ----
YOLOE_VARIANTS: dict[str, str] = {
    "YOLOE26 Nano": "yoloe-26n-seg.pt",
    "YOLOE26 Small": "yoloe-26s-seg.pt",
    "YOLOE26 Medium": "yoloe-26m-seg.pt",
    "YOLOE26 Large": "yoloe-26l-seg.pt",
    "YOLOE26 XLarge": "yoloe-26x-seg.pt",
}
YOLOE_VARIANT_NAMES = list(YOLOE_VARIANTS.keys())


# workers
class LoadYOLOEWorker(Worker):
    def __init__(self, engine: OpenVocabEngine, model_path: str) -> None:
        super().__init__()
        self.engine = engine
        self.model_path = model_path

    def do_work(self) -> str:
        self.engine.load_model(self.model_path)
        return self.model_path


class YOLOEInferWorker(Worker):
    """后台推理单张图片"""

    def __init__(self, engine: OpenVocabEngine, image_path: Path, conf: float, iou: float) -> None:
        super().__init__()
        self.setTerminationEnabled(True)
        self.engine = engine
        self.image_path = image_path
        self.conf = conf
        self.iou = iou

    def do_work(self) -> list[dict]:
        return self.engine.infer(self.image_path, self.conf, self.iou)


# 面板
class OpenVocabDetectPanel(QWidget):
    """开放词汇检测面板

    不绑定任何特定任务领域——用户自由输入文本提示，
    可收藏常用提示词组以便复用。
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_selected_saved = ""
        self.setObjectName("open_vocab_detect_panel")

        self._engine = OpenVocabEngine()
        self._image_files: list[Path] = []
        self._load_worker: LoadYOLOEWorker | None = None
        self._infer_worker: YOLOEInferWorker | None = None
        self._inferring: bool = False

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # ---- 标题 ----
        layout.addWidget(SubtitleLabel("开放词汇检测"))

        # ---- 工具栏卡片 ----
        toolbar_card = CardWidget()
        toolbar = QVBoxLayout(toolbar_card)
        toolbar.setContentsMargins(12, 8, 12, 8)
        toolbar.setSpacing(8)

        # 第一行：模型选择
        model_row = QHBoxLayout()
        model_row.addWidget(BodyLabel("模型:"))

        self.model_combo = ComboBox()
        self.model_combo.addItems(YOLOE_VARIANT_NAMES)
        self.model_combo.addItem("自定义模型...")
        self.model_combo.setCurrentIndex(1)  # 默认 Small
        self.model_combo.setToolTip(
            "选择 YOLOE 变体（首次使用会自动下载模型权重）\n"
            "Small 是速度与效果的最佳平衡，推荐作为默认"
        )
        self.model_combo.currentTextChanged.connect(self._on_model_variant_changed)
        model_row.addWidget(self.model_combo, 1)

        self.custom_model_browser = PathBrowser(
            label="", mode="file",
            file_filter="Model Files (*.pt *.pth);;All Files (*)",
            placeholder="选择自定义 .pt 模型文件...",
            config_key="ovd_custom_model",
        )
        self.custom_model_browser.setVisible(False)
        self.custom_model_browser.path_changed.connect(self._load_model)
        model_row.addWidget(self.custom_model_browser, 1)

        toolbar.addLayout(model_row)

        # 第二行：文本提示输入 + 收藏管理
        prompt_row = QHBoxLayout()
        prompt_row.addWidget(BodyLabel("提示词:"))

        self.prompt_edit = LineEdit()
        self.prompt_edit.setPlaceholderText(
            "输入目标描述，逗号分隔。例: person, crack, defect, construction vehicle"
        )
        self.prompt_edit.returnPressed.connect(self._apply_prompts)
        prompt_row.addWidget(self.prompt_edit, 1)

        self.apply_prompt_btn = PrimaryPushButton("应用")
        self.apply_prompt_btn.setToolTip(
            "将文本提示编码为类别嵌入并注入模型"
        )
        self.apply_prompt_btn.clicked.connect(self._apply_prompts)
        prompt_row.addWidget(self.apply_prompt_btn)

        toolbar.addLayout(prompt_row)

        # 第三行：收藏的提示词组
        saved_row = QHBoxLayout()
        saved_row.addWidget(BodyLabel("收藏:"))

        self.saved_prompt_combo = ComboBox()
        self.saved_prompt_combo.setMinimumWidth(200)
        self.saved_prompt_combo.setToolTip("已保存的提示词组，选中后自动填入上方输入框")
        self.saved_prompt_combo.currentTextChanged.connect(self._on_saved_prompt_selected)
        saved_row.addWidget(self.saved_prompt_combo, 1)

        self.save_prompt_btn = PushButton("💾 保存当前")
        self.save_prompt_btn.setToolTip("将当前提示词保存到收藏列表")
        self.save_prompt_btn.clicked.connect(self._save_current_prompt)
        saved_row.addWidget(self.save_prompt_btn)

        self.delete_prompt_btn = PushButton("🗑 删除")
        self.delete_prompt_btn.setToolTip("从收藏列表中删除当前选中的提示词组")
        self.delete_prompt_btn.clicked.connect(self._delete_saved_prompt)
        saved_row.addWidget(self.delete_prompt_btn)

        toolbar.addLayout(saved_row)

        # 第四行：图片目录 + 阈值
        bottom_row = QHBoxLayout()

        bottom_row.addWidget(BodyLabel("图片目录:"))
        self.folder_browser = PathBrowser(
            label="", mode="dir",
            placeholder="选择图片文件夹...",
            config_key="ovd_folder_path",
        )
        self.folder_browser.path_changed.connect(self._on_folder_selected)
        bottom_row.addWidget(self.folder_browser, 1)

        bottom_row.addSpacing(16)

        bottom_row.addWidget(BodyLabel("Conf:"))
        self.conf_spin = DoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setToolTip("置信度阈值")
        self.conf_spin.valueChanged.connect(
            lambda v: set_float("ovd_conf", v)
        )
        bottom_row.addWidget(self.conf_spin)

        bottom_row.addWidget(BodyLabel("IoU:"))
        self.iou_spin = DoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setToolTip("NMS IoU 阈值")
        self.iou_spin.valueChanged.connect(
            lambda v: set_float("ovd_iou", v)
        )
        bottom_row.addWidget(self.iou_spin)

        self.reinfer_btn = PrimaryPushButton("重新推理")
        self.reinfer_btn.setToolTip("用当前提示词和阈值重新推理")
        self.reinfer_btn.clicked.connect(self._on_reinfer)
        bottom_row.addWidget(self.reinfer_btn)

        toolbar.addLayout(bottom_row)

        layout.addWidget(toolbar_card)

        # ---- 状态标签 ----
        self.status_label = BodyLabel("请先选择模型并设置提示词")
        layout.addWidget(self.status_label)

        # ---- 图片浏览器 ----
        self.browser = ImageBrowser()
        self.browser.image_selected.connect(self._on_image_selected)
        layout.addWidget(self.browser, 1)

    # ================================================================
    # 模型加载
    # ================================================================

    def _on_model_variant_changed(self, text: str) -> None:
        """用户在下拉框中选择模型变体"""
        if text == "自定义模型...":
            self.custom_model_browser.setVisible(True)
            return

        self.custom_model_browser.setVisible(False)

        filename = YOLOE_VARIANTS.get(text)
        if filename:
            set_str("ovd_model_variant", text)
            self._load_model(filename)

    def _load_model(self, model_name_or_path: str) -> None:
        """后台加载模型（支持自动下载）"""
        self.status_label.setText(f"正在加载模型: {model_name_or_path} ...")
        self._set_inputs_enabled(False)
        self._load_worker = LoadYOLOEWorker(self._engine, model_name_or_path)
        self._load_worker.finished.connect(self._on_model_loaded)
        self._load_worker.error.connect(self._on_model_error)
        self._load_worker.start()

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.model_combo.setEnabled(enabled)
        self.custom_model_browser.setEnabled(enabled)
        self.prompt_edit.setEnabled(enabled)
        self.apply_prompt_btn.setEnabled(enabled)
        self.saved_prompt_combo.setEnabled(enabled)
        self.save_prompt_btn.setEnabled(enabled)
        self.delete_prompt_btn.setEnabled(enabled)
        self.folder_browser.setEnabled(enabled)
        self.conf_spin.setEnabled(enabled)
        self.iou_spin.setEnabled(enabled)
        self.reinfer_btn.setEnabled(enabled)

    def _on_model_loaded(self, path: str) -> None:
        self._set_inputs_enabled(True)
        logger.info(f"YOLOE 模型已加载: {path}")

        # 恢复历史提示词
        saved_prompt = get_str("ovd_prompts")
        if saved_prompt:
            self.prompt_edit.setText(saved_prompt)
            self._do_set_prompts(saved_prompt)

        status = f"✅ 模型已加载: {Path(path).name}"
        if self._engine.prompt_set:
            status += f" | 提示词: {', '.join(self._engine.class_names)}"
        self.status_label.setText(status)

        # 如果已有图片则自动推理
        if self.browser.current_path is not None:
            self._run_inference()

    def _on_model_error(self, err: str) -> None:
        self._set_inputs_enabled(True)
        self.status_label.setText("❌ 模型加载失败")
        error("模型加载失败", err, self)

    # ================================================================
    # 文本提示
    # ================================================================

    def _apply_prompts(self) -> None:
        """点击「应用」按钮"""
        text = self.prompt_edit.text().strip()
        if not text:
            error("提示词为空", "请输入至少一个目标描述", self)
            return
        if not self._engine.is_loaded:
            error("模型未加载", "请先选择并加载模型", self)
            return
        self._do_set_prompts(text)

    def _do_set_prompts(self, text: str) -> None:
        """解析文本提示并注入模型"""
        class_names = [name.strip() for name in text.split(",") if name.strip()]
        if not class_names:
            error("提示词为空", "请输入至少一个目标描述", self)
            return

        try:
            self._engine.set_prompts(class_names)
        except Exception as e:
            error("提示词设置失败", str(e), self)
            return

        set_str("ovd_prompts", text)
        logger.info(f"提示词已应用: {', '.join(class_names)}")
        self.status_label.setText(
            f"✅ 模型就绪 | 提示词: {', '.join(class_names)} ({len(class_names)} 类)"
        )
        # 提示词变化后自动推理当前图片
        if self.browser.current_path is not None:
            self._run_inference()

    # ================================================================
    # 收藏管理
    # ================================================================

    def _on_saved_prompt_selected(self, text: str) -> None:
        """从收藏下拉框选中时自动填入输入框"""
        if text and text != self._last_selected_saved:
            self._last_selected_saved = text
            self.prompt_edit.setText(text)

    def _load_saved_prompts(self) -> list[str]:
        """从 QSettings 加载收藏的提示词组"""
        raw = get_str("ovd_saved_prompts", "[]")
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    def _save_prompts_to_settings(self, prompts: list[str]) -> None:
        """保存收藏的提示词组到 QSettings"""
        set_str("ovd_saved_prompts", json.dumps(prompts, ensure_ascii=False))

    def _refresh_saved_combo(self) -> None:
        """刷新收藏下拉框"""
        self.saved_prompt_combo.blockSignals(True)
        self.saved_prompt_combo.clear()
        saved = self._load_saved_prompts()
        if saved:
            self.saved_prompt_combo.addItems(saved)
            self.saved_prompt_combo.setCurrentIndex(-1)
        self.saved_prompt_combo.blockSignals(False)

    def _save_current_prompt(self) -> None:
        """将当前提示词保存到收藏列表"""
        text = self.prompt_edit.text().strip()
        if not text:
            error("提示词为空", "没有可保存的内容", self)
            return

        saved = self._load_saved_prompts()
        if text in saved:
            return  # 已存在，不重复添加

        saved.append(text)
        self._save_prompts_to_settings(saved)
        self._refresh_saved_combo()
        logger.info(f"提示词已收藏: {text}")

    def _delete_saved_prompt(self) -> None:
        """从收藏列表中删除当前选中的提示词组"""
        current = self.saved_prompt_combo.currentText()
        if not current:
            return

        saved = self._load_saved_prompts()
        if current in saved:
            saved.remove(current)
            self._save_prompts_to_settings(saved)
            self._refresh_saved_combo()
            logger.info(f"提示词已删除: {current}")

    # ================================================================
    # 图片管理
    # ================================================================

    def _on_folder_selected(self, path: str) -> None:
        """图片目录选择回调"""
        if path:
            self._load_images(Path(path))

    def _load_images(self, directory: Path) -> None:
        self._cancel_inference()

        images = sorted(
            f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )

        if not images:
            self.status_label.setText("❌ 未找到图片文件")
            self.browser.clear()
            return

        self._image_files = images
        prompt_status = "已设置" if self._engine.prompt_set else "未设置"
        model_status = "已加载" if self._engine.is_loaded else "未加载"
        self.status_label.setText(
            f"共 {len(images)} 张图片 | 模型: {model_status} | 提示词: {prompt_status}"
        )
        self.browser.set_images(images, select_index=0)

    # ================================================================
    # 推理
    # ================================================================
    def _on_image_selected(self, idx: int, path: Path) -> None:
        if self._engine.is_loaded and self._engine.prompt_set:
            self._run_inference()

    def _run_inference(self) -> None:
        if not self._engine.is_loaded or not self._engine.prompt_set:
            return

        path = self.browser.current_path
        if path is None:
            return

        if self._inferring and self._infer_worker and self._infer_worker.isRunning():
            self._infer_worker.terminate()
            self._infer_worker.wait(3000)

        self._inferring = True
        self._set_inputs_enabled(False)
        self._update_ui_state()
        self.status_label.setText(f"正在推理: {path.name} ...")

        conf = self.conf_spin.value()
        iou = self.iou_spin.value()

        self._infer_worker = YOLOEInferWorker(self._engine, path, conf, iou)
        self._infer_worker.finished.connect(self._on_infer_done)
        self._infer_worker.error.connect(self._on_infer_error)
        self._infer_worker.start()

    def _on_infer_done(self, annotations: list[dict]) -> None:
        path = self.browser.current_path
        self.browser.show_annotations(annotations)
        bbox_count = sum(1 for a in annotations if a["type"] == "bbox")
        self.status_label.setText(
            f"{path.name if path else '?'} | 检测到 {bbox_count} 个目标 | "
            f"提示词: {', '.join(self._engine.class_names)}"
        )
        self._inferring = False
        self._set_inputs_enabled(True)
        self._update_ui_state()

    def _on_infer_error(self, err: str) -> None:
        self._set_inputs_enabled(True)
        self.status_label.setText("❌ 推理失败")
        error("推理失败", err, self)
        self._inferring = False
        self._update_ui_state()

    def _update_ui_state(self) -> None:
        self.browser.thumb_list.setDisabled(self._inferring)
        self.browser.set_nav_enabled(not self._inferring)

    def _on_reinfer(self) -> None:
        if (
                self._engine.is_loaded
                and self._engine.prompt_set
                and self.browser.current_path is not None
        ):
            self._run_inference()

    def _cancel_inference(self) -> None:
        if self._infer_worker and self._infer_worker.isRunning():
            self._infer_worker.terminate()
            self._infer_worker.wait(3000)
        self._inferring = False
        self._update_ui_state()

    # ================================================================
    # 持久化
    # ================================================================

    _last_selected_saved: str = ""

    def _load_settings(self) -> None:
        # 恢复模型选择
        saved_variant = get_str("ovd_model_variant", YOLOE_VARIANT_NAMES[1])  # 默认 Small
        idx = 1  # fallback: Small
        for i, name in enumerate(YOLOE_VARIANT_NAMES):
            if name == saved_variant:
                idx = i
                break
        self.model_combo.setCurrentIndex(idx)

        # 自定义模型路径
        custom_path = get_str("ovd_custom_model")
        if custom_path:
            self.custom_model_browser.path = custom_path

        # 启动时预加载模型
        idx = self.model_combo.currentIndex()
        if 0 <= idx < len(YOLOE_VARIANT_NAMES):
            # 预设 YOLOE 变体
            self._load_model(YOLOE_VARIANTS[YOLOE_VARIANT_NAMES[idx]])
        elif idx == len(YOLOE_VARIANT_NAMES) and custom_path:
            # 上次选了"自定义模型..."
            self._load_model(custom_path)

        # 图片目录
        saved_folder = get_str("ovd_folder_path")
        if saved_folder:
            self.folder_browser.path = saved_folder
            p = Path(saved_folder)
            if p.is_dir():
                self._load_images(p)

        # 提示词
        saved_prompt = get_str("ovd_prompts")
        if saved_prompt:
            self.prompt_edit.setText(saved_prompt)

        # 阈值
        self.conf_spin.setValue(get_float("ovd_conf", 0.25))
        self.iou_spin.setValue(get_float("ovd_iou", 0.45))

        # 收藏列表
        self._refresh_saved_combo()
