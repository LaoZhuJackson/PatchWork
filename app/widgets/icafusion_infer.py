"""ICAFusion 双流推理面板：VIS/IR 图片配对 + 模型推理 + 结果预览"""
from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    ComboBox,
    DoubleSpinBox,
    PrimaryPushButton,
    PushButton,
    SubtitleLabel,
    ToggleButton,
)

from app.services.icafusion_inference import ICAFusionEngine
from app.utils.config import get_str, set_str, get_float, set_float
from app.utils.message import error, info
from app.utils.worker import Worker
from app.widgets.image_browser import ImageBrowser
from app.widgets.path_browser import PathBrowser
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── 文件名匹配 ──────────────────────────────────────
# IR: ..._T_000042.jpg   VIS: ..._V_000042.jpg
_IR_RE = re.compile(r"^(.+)_T_(\d{6})\.\w+$", re.IGNORECASE)
_VIS_RE = re.compile(r"^(.+)_V_(\d{6})\.\w+$", re.IGNORECASE)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ── Workers ──────────────────────────────────────────

class LoadModelWorker(Worker):
    """后台加载 ICAFusion 模型"""

    def __init__(self, engine: ICAFusionEngine, model_path: str, device: str) -> None:
        super().__init__()
        self.engine = engine
        self.model_path = model_path
        self.device = device

    def do_work(self) -> str:
        self.engine.load_model(self.model_path, self.device)
        return self.model_path


class ICAFusionInferWorker(Worker):
    """后台推理一对 VIS+IR 图像"""

    def __init__(
        self,
        engine: ICAFusionEngine,
        vis_path: Path,
        ir_path: Path,
        conf: float,
        iou: float,
        img_size: int,
    ) -> None:
        super().__init__()
        self.setTerminationEnabled(True)
        self.engine = engine
        self.vis_path = vis_path
        self.ir_path = ir_path
        self.conf = conf
        self.iou = iou
        self.img_size = img_size

    def do_work(self) -> tuple[list[dict], list[dict]]:
        return self.engine.infer(
            self.vis_path, self.ir_path, self.conf, self.iou, self.img_size,
        )


# ── 配对工具 ─────────────────────────────────────────

def _scan_pairs(
    vis_dir: str, ir_dir: str,
) -> tuple[list[Path], dict[str, Path], str]:
    """扫描 VIS/IR 目录，优先按文件名配对，失败时按索引顺序配对。

    Returns:
        vis_files: 排序后的 VIS 文件列表（用于缩略图展示）
        ir_map: {vis_stem: ir_path} 映射
        mode: "name" 按文件名配对 / "index" 按索引顺序配对
    """
    vis_dir = Path(vis_dir)
    ir_dir = Path(ir_dir)

    # ── 1. 按文件名配对（IR: *_T_帧号, VIS: *_V_帧号）──
    ir_map: dict[str, Path] = {}
    for f in ir_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in _IMAGE_EXTS:
            continue
        m = _IR_RE.match(f.name)
        if m:
            ir_map[f"{m.group(1)}_{m.group(2)}"] = f

    vis_files: list[Path] = []
    pair_map: dict[str, Path] = {}

    for f in sorted(vis_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _IMAGE_EXTS:
            continue
        m = _VIS_RE.match(f.name)
        if m:
            key = f"{m.group(1)}_{m.group(2)}"
            if key in ir_map:
                vis_files.append(f)
                pair_map[f.stem] = ir_map[key]

    if vis_files:
        return vis_files, pair_map, "name"

    # ── 2. 回退：按索引顺序配对（同 detect_twostream.py 的 zip 逻辑）──
    vis_all = sorted(
        f for f in vis_dir.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
    )
    ir_all = sorted(
        f for f in ir_dir.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
    )
    n = min(len(vis_all), len(ir_all))
    for i in range(n):
        pair_map[vis_all[i].stem] = ir_all[i]

    return vis_all[:n], pair_map, "index"


# ── Panel ────────────────────────────────────────────

class ICAFusionInferPanel(QWidget):
    """ICAFusion 推理面板"""

    MODE_VIS = "可见光 (VIS)"
    MODE_IR = "红外 (IR)"

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("icafusion_infer_panel")

        # ── 状态 ──
        self._engine: ICAFusionEngine | None = None
        self._vis_files: list[Path] = []
        self._ir_map: dict[str, Path] = {}
        # 缓存：{vis_stem: vis_anns}
        self._vis_anns_cache: dict[str, list[dict]] = {}
        # 缓存：{vis_stem: ir_anns}
        self._ir_anns_cache: dict[str, list[dict]] = {}
        self._current_mode: str = self.MODE_VIS
        self._load_worker: LoadModelWorker | None = None
        self._infer_worker: ICAFusionInferWorker | None = None
        self._inferring: bool = False

        self._setup_ui()
        self._load_settings()

    # ── UI ────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("ICAFusion 双流推理"))

        # ── 工具栏 Card ──
        toolbar_card = CardWidget()
        toolbar = QVBoxLayout(toolbar_card)
        toolbar.setContentsMargins(12, 8, 12, 8)
        toolbar.setSpacing(8)

        # ICAFusion 仓库路径
        self.repo_browser = PathBrowser(
            label="ICAFusion 仓库:",
            mode="dir",
            placeholder="选择 ICAFusion 代码目录...",
            config_key="icafusion_repo_path",
        )
        self.repo_browser.path_changed.connect(self._on_repo_changed)
        toolbar.addWidget(self.repo_browser)

        # 模型文件
        self.model_browser = PathBrowser(
            label="模型文件:",
            mode="file",
            file_filter="Model Files (*.pt *.pth);;All Files (*)",
            placeholder="选择 ICAFusion .pt 权重...",
            config_key="icafusion_model_path",
        )
        self.model_browser.path_changed.connect(self._load_model)
        toolbar.addWidget(self.model_browser)

        # GPU 设备
        device_row = QHBoxLayout()
        device_row.addWidget(BodyLabel("GPU:"))
        self.device_combo = ComboBox()
        self.device_combo.addItems(["0", "1", "2", "3", "cpu"])
        self.device_combo.setCurrentIndex(0)
        self.device_combo.setToolTip("选择推理设备")
        self.device_combo.setFixedWidth(80)
        device_row.addWidget(self.device_combo)
        device_row.addStretch()
        toolbar.addLayout(device_row)

        # VIS + IR 目录
        dir_row = QHBoxLayout()
        dir_row.setSpacing(12)

        vis_container = QVBoxLayout()
        self.vis_browser = PathBrowser(
            label="VIS 目录:",
            mode="dir",
            placeholder="可见光图片文件夹...",
            config_key="icafusion_vis_dir",
        )
        self.vis_browser.path_changed.connect(self._on_vis_dir_changed)
        vis_container.addWidget(self.vis_browser)
        dir_row.addLayout(vis_container)

        ir_container = QVBoxLayout()
        self.ir_browser = PathBrowser(
            label="IR 目录:",
            mode="dir",
            placeholder="红外图片文件夹...",
            config_key="icafusion_ir_dir",
        )
        self.ir_browser.path_changed.connect(self._on_ir_dir_changed)
        ir_container.addWidget(self.ir_browser)
        dir_row.addLayout(ir_container)

        toolbar.addLayout(dir_row)
        self._pair_status = BodyLabel("")
        toolbar.addWidget(self._pair_status)

        # 阈值 + 操作按钮
        threshold_row = QHBoxLayout()

        threshold_row.addWidget(BodyLabel("Conf:"))
        self.conf_spin = DoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setToolTip("置信度阈值")
        self.conf_spin.valueChanged.connect(lambda v: set_float("icafusion_conf", v))
        threshold_row.addWidget(self.conf_spin)

        threshold_row.addWidget(BodyLabel("IoU:"))
        self.iou_spin = DoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setToolTip("NMS IoU 阈值")
        self.iou_spin.valueChanged.connect(lambda v: set_float("icafusion_iou", v))
        threshold_row.addWidget(self.iou_spin)

        self.reinfer_btn = PrimaryPushButton("重新推理")
        self.reinfer_btn.setToolTip("用当前阈值对当前图片重新推理")
        self.reinfer_btn.clicked.connect(self._on_reinfer)
        threshold_row.addWidget(self.reinfer_btn)

        # VIS/IR 视图切换
        self.mode_toggle = ToggleButton()
        self.mode_toggle.setText("切换 IR 视图")
        self.mode_toggle.setToolTip("在 VIS 和 IR 之间切换显示")
        self.mode_toggle.toggled.connect(self._on_mode_toggled)
        threshold_row.addWidget(self.mode_toggle)

        threshold_row.addStretch()
        toolbar.addLayout(threshold_row)

        layout.addWidget(toolbar_card)

        # ── 状态 ──
        self.status_label = BodyLabel("请设置 ICAFusion 仓库路径、模型文件、VIS/IR 目录")
        layout.addWidget(self.status_label)

        # ── 图片浏览器 ──
        self.browser = ImageBrowser()
        self.browser.image_selected.connect(self._on_image_selected)
        layout.addWidget(self.browser, 1)

    # ── 仓库路径 ───────────────────────────────

    def _on_repo_changed(self, path: str) -> None:
        """ICAFusion 仓库路径变更：重建 engine"""
        if not path or not Path(path).is_dir():
            return

        self._engine = ICAFusionEngine(path)
        logger.info(f"ICAFusion 仓库: {self._engine.root}")

        # 如果模型路径已有，尝试加载
        model_path = self.model_browser.path
        if model_path and Path(model_path).is_file():
            self._load_model(model_path)

    # ── 模型加载 ───────────────────────────────

    def _load_model(self, path: str) -> None:
        if not path or not Path(path).is_file():
            return
        if self._engine is None:
            self.status_label.setText("❌ 请先设置 ICAFusion 仓库路径")
            return

        self.status_label.setText("正在加载模型...")
        self._set_inputs_enabled(False)

        device = self.device_combo.currentText()
        self._load_worker = LoadModelWorker(self._engine, path, device)
        self._load_worker.finished.connect(self._on_model_loaded)
        self._load_worker.error.connect(self._on_model_error)
        self._load_worker.start()

    def _on_model_loaded(self, _path: str) -> None:
        self._set_inputs_enabled(True)
        engine = self._engine
        if engine is None:
            return
        names = engine.class_names
        count = len(names)
        dev = engine.device_str
        logger.info(f"ICAFusion 模型已加载 ({count} 类, device={dev})")
        self.status_label.setText(
            f"✅ 模型已加载: {Path(engine.model_path).name} | {count} 类 | device={dev}"
        )

        # 如果已有图片列表，自动推理当前图
        if self.browser.current_path is not None:
            self._run_inference()

    def _on_model_error(self, err: str) -> None:
        self._set_inputs_enabled(True)
        self.status_label.setText("❌ 模型加载失败")
        error("模型加载失败", err, self)

    # ── 目录选择 & 配对 ────────────────────────

    def _on_vis_dir_changed(self, path: str) -> None:
        self._try_pair()

    def _on_ir_dir_changed(self, path: str) -> None:
        self._try_pair()

    def _try_pair(self) -> None:
        """尝试配对 VIS + IR 目录中的图片"""
        vis_dir = self.vis_browser.path
        ir_dir = self.ir_browser.path

        if not vis_dir or not ir_dir or not Path(vis_dir).is_dir() or not Path(ir_dir).is_dir():
            return

        self._cancel_inference()
        vis_files, ir_map, pair_mode = _scan_pairs(vis_dir, ir_dir)

        self._vis_files = vis_files
        self._ir_map = ir_map
        self._vis_anns_cache.clear()
        self._ir_anns_cache.clear()

        if not vis_files:
            self._pair_status.setText("❌ 未找到可配对的 VIS-IR 图片对")
            self.browser.clear()
            return

        mode_text = "按文件名配对" if pair_mode == "name" else "按索引顺序配对"
        self._pair_status.setText(
            f"共 {len(vis_files)} 对 VIS-IR 图片 | {mode_text}"
        )

        model_status = "已加载" if (self._engine and self._engine.is_loaded) else "未加载"
        self.status_label.setText(
            f"共 {len(vis_files)} 对图片 | 模型: {model_status}"
        )

        self.browser.set_images(vis_files, select_index=0)

    # ── 推理 ───────────────────────────────────

    def _on_image_selected(self, idx: int, path: Path) -> None:
        """缩略图点击或导航"""
        # 检查缓存
        if path.stem in self._vis_anns_cache:
            anns = self._vis_anns_cache[path.stem] if self._current_mode == self.MODE_VIS else self._ir_anns_cache.get(path.stem, [])
            self.browser.show_annotations(anns)
            self.status_label.setText(
                f"{path.name} | 检测到 {len(anns)} 个目标"
            )
            return

        if self._engine and self._engine.is_loaded:
            self._run_inference()

    def _run_inference(self) -> None:
        if self._engine is None or not self._engine.is_loaded:
            return

        path = self.browser.current_path
        if path is None:
            return

        ir_path = self._ir_map.get(path.stem)
        if ir_path is None:
            self.status_label.setText(f"❌ 未找到 IR 配对: {path.name}")
            return

        # 终止旧推理
        if self._inferring and self._infer_worker and self._infer_worker.isRunning():
            self._infer_worker.terminate()
            self._infer_worker.wait(3000)

        self._inferring = True
        self._update_ui_state()
        self.status_label.setText(f"正在推理: {path.name} ...")

        conf = self.conf_spin.value()
        iou = self.iou_spin.value()

        self._infer_worker = ICAFusionInferWorker(
            self._engine, path, ir_path, conf, iou, img_size=1280,
        )
        self._infer_worker.finished.connect(self._on_infer_done)
        self._infer_worker.error.connect(self._on_infer_error)
        self._infer_worker.start()

    def _on_infer_done(self, result: tuple[list[dict], list[dict]]) -> None:
        vis_anns, ir_anns = result
        path = self.browser.current_path
        if path is not None:
            self._vis_anns_cache[path.stem] = vis_anns
            self._ir_anns_cache[path.stem] = ir_anns

        # 当前模式决定显示哪个
        anns = vis_anns if self._current_mode == self.MODE_VIS else ir_anns
        self.browser.show_annotations(anns)

        self.status_label.setText(
            f"{path.name if path else '?'} | {len(vis_anns)} 个目标"
        )
        self._inferring = False
        self._update_ui_state()

    def _on_infer_error(self, err: str) -> None:
        self.status_label.setText("❌ 推理失败")
        error("推理失败", err, self)
        self._inferring = False
        self._update_ui_state()

    # ── VIS/IR 切换 ───────────────────────────

    def _on_mode_toggled(self, checked: bool) -> None:
        """切换 VIS ↔ IR 视图"""
        self._current_mode = self.MODE_IR if checked else self.MODE_VIS
        self.mode_toggle.setText("切换 VIS 视图" if checked else "切换 IR 视图")

        path = self.browser.current_path
        if path is None:
            return

        if checked:
            # 切换到 IR：需要加载 IR 图片 + IR 标注
            ir_path = self._ir_map.get(path.stem)
            if ir_path is not None:
                from PySide6.QtGui import QPixmap
                pixmap = QPixmap(str(ir_path))
                if not pixmap.isNull():
                    self.browser.viewer.set_image(pixmap)
                    ir_anns = self._ir_anns_cache.get(path.stem, [])
                    self.browser.show_annotations(ir_anns)
        else:
            # 切换回 VIS
            from PySide6.QtGui import QPixmap
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                self.browser.viewer.set_image(pixmap)
                vis_anns = self._vis_anns_cache.get(path.stem, [])
                self.browser.show_annotations(vis_anns)

    # ── 其他 ───────────────────────────────────

    def _on_reinfer(self) -> None:
        if self._engine and self._engine.is_loaded and self.browser.current_path is not None:
            self._vis_anns_cache.clear()
            self._ir_anns_cache.clear()
            self._run_inference()

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.repo_browser.setEnabled(enabled)
        self.model_browser.setEnabled(enabled)
        self.device_combo.setEnabled(enabled)
        self.vis_browser.setEnabled(enabled)
        self.ir_browser.setEnabled(enabled)
        self.conf_spin.setEnabled(enabled)
        self.iou_spin.setEnabled(enabled)
        self.reinfer_btn.setEnabled(enabled)
        self.mode_toggle.setEnabled(enabled)

    def _update_ui_state(self) -> None:
        self.browser.thumb_list.setDisabled(self._inferring)
        self.browser.set_nav_enabled(not self._inferring)

    def _cancel_inference(self) -> None:
        if self._infer_worker and self._infer_worker.isRunning():
            self._infer_worker.terminate()
            self._infer_worker.wait(3000)
        self._inferring = False
        self._update_ui_state()

    # ── 持久化 ────────────────────────────────

    def _load_settings(self) -> None:
        self.repo_browser.path = get_str("icafusion_repo_path")
        self.model_browser.path = get_str("icafusion_model_path")
        self.vis_browser.path = get_str("icafusion_vis_dir")
        self.ir_browser.path = get_str("icafusion_ir_dir")
        self.conf_spin.setValue(get_float("icafusion_conf", 0.25))
        self.iou_spin.setValue(get_float("icafusion_iou", 0.45))

        # 恢复 engine
        repo = get_str("icafusion_repo_path")
        if repo and Path(repo).is_dir():
            self._on_repo_changed(repo)

        # 恢复图片列表
        vis_dir = get_str("icafusion_vis_dir")
        ir_dir = get_str("icafusion_ir_dir")
        if vis_dir and ir_dir:
            self._try_pair()
