"""AI 目标合成面板：SenseNova U1.5 云端红框控制编辑

交互式流程（类似 X-AnyLabeling，逐张标注）：
    选图片目录 → 选图 → 选类别 → 画框(可多个) → 自动拼提示词
    → AI 生成(云端) → 原图/结果切换 → 保存结果 → 导出 YOLO / X-AnyLabeling JSON

类别选择保持逻辑：ComboBox 选中某类别后持续生效，跨框/跨图不重置，
只有用户手动切换才换类别 —— 画完框无需再确认类别。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QKeyEvent
from PySide6.QtWidgets import (
    QFormLayout, QHBoxLayout, QVBoxLayout, QListWidget, QWidget,
    QPlainTextEdit,
)
from qfluentwidgets import (
    BodyLabel, CardWidget, CheckBox, ComboBox, InfoBar, InfoBarPosition,
    LineEdit, PasswordLineEdit, PrimaryPushButton, ProgressBar,
    PushButton, SpinBox, StrongBodyLabel, SubtitleLabel, ToggleButton,
)

from app.services.label_reader import IMAGE_EXTS, get_color
from app.services.sensenova import (
    DEFAULT_CLASSES, build_prompt, edit_image, resolve_key,
)
from app.services.sensenova_export import (
    DEFAULT_XAL_VERSION, export_xanylabeling, export_yolo,
)
from app.utils.config import (
    get_bool, get_str, get_int, set_bool, set_str, set_int,
)
from app.utils.logger import get_logger
from app.utils.message import error, info, warning
from app.utils.worker import Worker
from app.widgets.image_browser import ImageBrowser
from app.widgets.path_browser import PathBrowser

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 后台 Worker：云端生成
# ═══════════════════════════════════════════════════════════════════════

class SynthWorker(Worker):
    """后台调用 SenseNova 云端编辑 API"""

    def __init__(self, img_path: str, boxes: list, prompt: str, cfg: dict) -> None:
        super().__init__()
        self.img_path = img_path
        self.boxes = boxes
        self.prompt = prompt
        self.cfg = cfg

    def do_work(self):
        out, meta = edit_image(
            self.img_path, self.prompt, boxes=self.boxes,
            api_key=self.cfg.get("api_key", ""),
            model=self.cfg.get("model", ""),
            keep_size=self.cfg.get("keep_size", True),
            watermark=self.cfg.get("watermark", False),
            prompt_extend=self.cfg.get("prompt_extend", False),
            timeout=self.cfg.get("timeout", 180),
        )
        return out, meta


# ═══════════════════════════════════════════════════════════════════════
# 面板
# ═══════════════════════════════════════════════════════════════════════

class SenseNovaSynthPanel(QWidget):
    """AI 目标合成（SenseNova U1.5 云端 API）"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("sensenova_synth_panel")

        self._classes: list[dict] = []
        self._boxes: list[dict] = []     # {rect: QRectF, cls_id, cls_name}
        self._orig_path: Path | None = None
        self._orig_size = (0, 0)         # (w, h) 原图像素
        self._result_np: np.ndarray | None = None
        self._showing_result = False
        self._worker: SynthWorker | None = None
        self._records: dict[str, dict] = {}   # stem -> record（累计，供导出）
        self._result_pixmap: QPixmap | None = None

        self._setup_ui()
        self._load_settings()

    # ── UI 搭建 ──

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("AI 目标合成 (SenseNova U1.5)"))

        # ---- 图片目录 ----
        layout.addWidget(StrongBodyLabel("① 图片目录"))
        dir_card = CardWidget()
        dir_form = QFormLayout(dir_card)
        self.img_browser = PathBrowser(
            label="", mode="dir",
            placeholder="选择要标注的图片目录...",
            config_key="sensenova_img_dir",
        )
        self.img_browser.path_changed.connect(self._on_dir_changed)
        dir_form.addRow(BodyLabel("图片目录:"), self.img_browser)
        layout.addWidget(dir_card)

        # 图片浏览（缩略图 + 查看器）
        self.browser = ImageBrowser()
        self.browser.image_selected.connect(self._on_image_selected)
        self.browser.viewer.rect_drawn.connect(self._on_rect_drawn)
        layout.addWidget(self.browser, 1)

        # ---- 类别定义 ----
        layout.addWidget(StrongBodyLabel("② 类别定义 (id/name/prompt/detector_class)"))
        cls_card = CardWidget()
        cls_layout = QVBoxLayout(cls_card)
        self.classes_text = QPlainTextEdit()
        self.classes_text.setPlaceholderText(
            '[{"id": 0, "name": "光伏面板裂缝", '
            '"prompt": "在光伏面板上生成细长的裂缝，符合自然光伏板会产生的裂缝", '
            '"detector_class": "crack"}]')
        # self.classes_text.setMaximumHeight(120)
        cls_layout.addWidget(self.classes_text)

        cls_row = QHBoxLayout()
        self.load_demo_btn = PushButton("加载示例")
        self.load_demo_btn.clicked.connect(self._on_load_demo)
        cls_row.addWidget(self.load_demo_btn)
        self.apply_cls_btn = PushButton("应用类别")
        self.apply_cls_btn.clicked.connect(self._on_apply_classes)
        cls_row.addWidget(self.apply_cls_btn)
        cls_row.addStretch()
        cls_row.addWidget(BodyLabel("当前类别:"))
        self.class_combo = ComboBox()
        self.class_combo.setMinimumWidth(160)
        cls_row.addWidget(self.class_combo)
        cls_layout.addLayout(cls_row)
        layout.addWidget(cls_card)

        # ---- 画框 ----
        layout.addWidget(StrongBodyLabel("③ 画框 (选中类别后持续生效，直到手动切换)"))
        draw_card = CardWidget()
        draw_layout = QVBoxLayout(draw_card)
        draw_row = QHBoxLayout()
        self.draw_toggle = ToggleButton("画框模式")
        self.draw_toggle.setChecked(False)
        self.draw_toggle.toggled.connect(self._on_draw_toggled)
        draw_row.addWidget(self.draw_toggle)
        draw_row.addWidget(BodyLabel("开启后左键拖拽画框，中键平移，滚轮缩放"))
        draw_row.addStretch()
        self.box_count_label = BodyLabel("本图框数: 0")
        draw_row.addWidget(self.box_count_label)
        draw_layout.addLayout(draw_row)

        self.box_list = QListWidget()
        # self.box_list.setMaximumHeight(110)
        draw_layout.addWidget(self.box_list)

        box_btns = QHBoxLayout()
        self.undo_btn = PushButton("撤销最后一框")
        self.undo_btn.clicked.connect(self._on_undo_box)
        box_btns.addWidget(self.undo_btn)
        self.del_btn = PushButton("删除选中")
        self.del_btn.clicked.connect(self._on_del_box)
        box_btns.addWidget(self.del_btn)
        self.clear_btn = PushButton("清空本图框")
        self.clear_btn.clicked.connect(self._on_clear_boxes)
        box_btns.addWidget(self.clear_btn)
        box_btns.addStretch()
        draw_layout.addLayout(box_btns)
        layout.addWidget(draw_card)

        # ---- 生成 ----
        layout.addWidget(StrongBodyLabel("④ 生成"))
        gen_card = CardWidget()
        gen_layout = QVBoxLayout(gen_card)
        self.auto_append_chk = CheckBox("自动拼接红框约束提示词（物体在框内/移除红框/保持框外不变）")
        self.auto_append_chk.setChecked(True)
        gen_layout.addWidget(self.auto_append_chk)
        self.prompt_text = QPlainTextEdit()
        self.prompt_text.setReadOnly(True)
        self.prompt_text.setPlaceholderText("提示词预览（随框和类别自动生成）")
        # self.prompt_text.setMaximumHeight(90)
        gen_layout.addWidget(self.prompt_text)

        gen_row = QHBoxLayout()
        self.gen_btn = PrimaryPushButton("AI 生成")
        self.gen_btn.clicked.connect(self._on_generate)
        gen_row.addWidget(self.gen_btn)
        self.gen_status = BodyLabel("")
        gen_row.addWidget(self.gen_status, 1)
        gen_layout.addLayout(gen_row)
        self.progress = ProgressBar()
        self.progress.setVisible(False)
        gen_layout.addWidget(self.progress)
        layout.addWidget(gen_card)

        # ---- 结果查看与保存 ----
        layout.addWidget(StrongBodyLabel("⑤ 结果"))
        res_card = CardWidget()
        res_layout = QHBoxLayout(res_card)
        self.result_toggle = ToggleButton("显示生成结果")
        self.result_toggle.setChecked(False)
        self.result_toggle.toggled.connect(self._on_result_toggled)
        res_layout.addWidget(self.result_toggle)
        self.save_btn = PushButton("保存当前结果")
        self.save_btn.clicked.connect(self._on_save_result)
        res_layout.addWidget(self.save_btn)
        res_layout.addStretch()
        self.res_meta = BodyLabel("")
        res_layout.addWidget(self.res_meta)
        layout.addWidget(res_card)

        # ---- 导出 ----
        layout.addWidget(StrongBodyLabel("⑥ 导出"))
        exp_card = CardWidget()
        exp_form = QFormLayout(exp_card)
        self.export_browser = PathBrowser(
            label="", mode="dir",
            placeholder="选择导出目录（结果图存 images/，JSON 与图并列）...",
            config_key="sensenova_export_dir",
        )
        exp_form.addRow(BodyLabel("导出目录:"), self.export_browser)

        exp_row = QHBoxLayout()
        self.yolo_chk = CheckBox("导出 YOLO label")
        self.yolo_chk.setChecked(True)
        exp_row.addWidget(self.yolo_chk)
        self.xal_chk = CheckBox("导出 X-AnyLabeling JSON")
        self.xal_chk.setChecked(True)
        exp_row.addWidget(self.xal_chk)
        exp_row.addWidget(BodyLabel("XAL 版本号:"))
        self.version_edit = LineEdit()
        self.version_edit.setText(DEFAULT_XAL_VERSION)
        self.version_edit.setMaximumWidth(150)
        exp_row.addWidget(self.version_edit)
        exp_row.addStretch()
        self.export_btn = PrimaryPushButton("导出标注")
        self.export_btn.clicked.connect(self._on_export)
        exp_row.addWidget(self.export_btn)
        exp_form.addRow(BodyLabel("格式:"), exp_row)
        layout.addWidget(exp_card)

        # ---- 设置 ----
        layout.addWidget(StrongBodyLabel("⑦ 云端设置"))
        set_card = CardWidget()
        set_form = QFormLayout(set_card)
        self.key_edit = PasswordLineEdit()
        self.key_edit.setPlaceholderText("sk-... （留空则读环境变量 / sensenova_key.txt）")
        set_form.addRow(BodyLabel("API Key:"), self.key_edit)
        self.model_edit = LineEdit()
        self.model_edit.setText("sensenova-u1.5-lite")
        set_form.addRow(BodyLabel("模型:"), self.model_edit)
        self.keep_size_chk = CheckBox("保持输入输出分辨率一致（默认开）")
        self.keep_size_chk.setChecked(True)
        set_form.addRow(BodyLabel("分辨率:"), self.keep_size_chk)
        self.watermark_chk = CheckBox("开启水印")
        set_form.addRow(BodyLabel("水印:"), self.watermark_chk)
        self.extend_chk = CheckBox("开启 prompt_extend 自动扩写")
        set_form.addRow(BodyLabel("扩写:"), self.extend_chk)
        self.timeout_spin = SpinBox()
        self.timeout_spin.setRange(30, 600)
        self.timeout_spin.setValue(180)
        set_form.addRow(BodyLabel("超时(秒):"), self.timeout_spin)
        layout.addWidget(set_card)

        layout.addStretch()

    # ── 图片目录 → 扫描 ──

    def _on_dir_changed(self, _: str) -> None:
        path = self.img_browser.path.strip()
        if not path:
            return
        images = sorted(
            f for f in Path(path).iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        if images:
            self.browser.set_images(images, select_index=0)
        else:
            self.browser.clear()

    # ── 类别 ──

    def _on_load_demo(self) -> None:
        self.classes_text.setPlainText(
            json.dumps(DEFAULT_CLASSES, ensure_ascii=False, indent=2))

    def _on_apply_classes(self) -> None:
        try:
            classes = json.loads(self.classes_text.toPlainText())
        except json.JSONDecodeError as e:
            warning("类别 JSON 解析失败", str(e), self)
            return
        if not isinstance(classes, list) or not classes:
            warning("类别不能为空", "至少需要一个类别。", self)
            return
        for c in classes:
            c.setdefault("id", 0)
            c.setdefault("name", "class")
            c.setdefault("prompt", c["name"])
            c.setdefault("detector_class", c["name"])
        self._classes = classes
        self._refresh_class_combo()
        set_str("sensenova_classes_json", self.classes_text.toPlainText())
        info("类别已应用", f"共 {len(classes)} 个类别。", self)

    def _refresh_class_combo(self) -> None:
        """重建类别下拉；尽量保持当前选中（类别保持逻辑的载体）"""
        current = self.class_combo.currentText()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        for c in self._classes:
            self.class_combo.addItem(c["name"])
        if current:
            idx = self.class_combo.findText(current)
            if idx >= 0:
                self.class_combo.setCurrentIndex(idx)
        self.class_combo.blockSignals(False)

    def _current_class(self) -> dict | None:
        name = self.class_combo.currentText()
        for c in self._classes:
            if c["name"] == name:
                return c
        return self._classes[0] if self._classes else None

    # ── 画框 ──

    def _on_draw_toggled(self, checked: bool) -> None:
        # 画框必须在原图上：若当前显示的是生成结果，先切回原图
        if checked and self._showing_result:
            self.result_toggle.setChecked(False)
        self.browser.viewer.set_interaction_mode("draw" if checked else "pan")

    def _on_rect_drawn(self, rect: QRectF) -> None:
        cls = self._current_class()
        if cls is None:
            warning("未定义类别", "请先在②中定义并应用类别。", self)
            return
        self._boxes.append({
            "rect": rect,
            "cls_id": cls["id"],
            "cls_name": cls["name"],
        })
        self._refresh_boxes()

    def _on_undo_box(self) -> None:
        if self._boxes:
            self._boxes.pop()
            self._refresh_boxes()

    def _on_del_box(self) -> None:
        row = self.box_list.currentRow()
        if 0 <= row < len(self._boxes):
            del self._boxes[row]
            self._refresh_boxes()

    def _on_clear_boxes(self) -> None:
        if self._boxes:
            self._boxes.clear()
            self._refresh_boxes()

    def _refresh_boxes(self) -> None:
        """重绘叠加层 + 框列表 + 提示词"""
        if not self._showing_result:
            self._draw_overlays()
        self.box_list.clear()
        for i, b in enumerate(self._boxes):
            r = b["rect"]
            self.box_list.addItem(
                f"{i + 1}. {b['cls_name']}  ({r.x():.0f},{r.y():.0f},"
                f"{r.width():.0f}x{r.height():.0f})")
        self.box_count_label.setText(f"本图框数: {len(self._boxes)}")
        self._update_prompt()

    def _draw_overlays(self) -> None:
        viewer = self.browser.viewer
        viewer.clear_overlays()
        for b in self._boxes:
            cls_id = int(b["cls_id"])
            viewer.add_bbox(b["rect"], get_color(cls_id), b["cls_name"])

    # ── 提示词 ──

    def _class_prompt(self, cls_name: str) -> str:
        for c in self._classes:
            if c["name"] == cls_name:
                return c.get("prompt") or cls_name
        return cls_name

    def _update_prompt(self) -> None:
        if not self._boxes:
            self.prompt_text.setPlainText("")
            return
        items = [(b["cls_name"], self._class_prompt(b["cls_name"]))
                 for b in self._boxes]
        if self.auto_append_chk.isChecked():
            prompt = build_prompt(items)
        else:
            prompt = "；".join(f"在红框内生成{desc}" for _, desc in items)
        self.prompt_text.setPlainText(prompt)

    def _on_auto_append_changed(self) -> None:
        self._update_prompt()

    # ── 图切换 ──

    def _on_image_selected(self, idx: int, path: Path) -> None:
        self._orig_path = path
        pixmap = self.browser.current_pixmap
        self._orig_size = (pixmap.width(), pixmap.height()) if pixmap else (0, 0)
        self._boxes = []
        self._result_np = None
        self._result_pixmap = None
        self._showing_result = False
        self.result_toggle.setChecked(False)
        self.result_toggle.setText("显示生成结果")
        self.res_meta.setText("")
        self._refresh_boxes()

    # ── 生成 ──

    def _on_generate(self) -> None:
        if self._orig_path is None or self._orig_size == (0, 0):
            warning("未选择图片", "请先在①中选择图片目录并选一张图。", self)
            return
        if not self._boxes:
            warning("未画框", "请先在③中画至少一个红框。", self)
            return
        api_key = self.key_edit.text().strip() or resolve_key()
        if not api_key:
            warning("缺少 API Key",
                    "请在⑦设置中填入 sk- 开头的 Key，或设置环境变量 SENSENOVA_API_KEY。",
                    self)
            return
        if self._worker is not None and self._worker.isRunning():
            return

        prompt = self.prompt_text.toPlainText().strip()
        boxes_xyxy = [
            (int(b["rect"].x()), int(b["rect"].y()),
             int(b["rect"].x() + b["rect"].width()),
             int(b["rect"].y() + b["rect"].height()))
            for b in self._boxes
        ]
        cfg = {
            "api_key": api_key,
            "model": self.model_edit.text().strip() or "sensenova-u1.5-lite",
            "keep_size": self.keep_size_chk.isChecked(),
            "watermark": self.watermark_chk.isChecked(),
            "prompt_extend": self.extend_chk.isChecked(),
            "timeout": self.timeout_spin.value(),
        }

        self._set_gen_busy(True)
        self.gen_status.setText("正在生成...")
        self._worker = SynthWorker(str(self._orig_path), boxes_xyxy, prompt, cfg)
        self._worker.finished.connect(self._on_gen_done)
        self._worker.error.connect(self._on_gen_error)
        self._worker.start()

    def _set_gen_busy(self, busy: bool) -> None:
        self.gen_btn.setEnabled(not busy)
        self.progress.setVisible(busy)
        if busy:
            self.progress.setValue(0)

    def _on_gen_done(self, result) -> None:
        out, meta = result
        self._set_gen_busy(False)
        self._result_np = out
        self._result_pixmap = _np_to_qpixmap(out)
        # 尺寸校验：保持分辨率时输出必须 == 输入
        w, h = self._orig_size
        if meta.get("size_used") != "auto" and out.shape[1] != w:
            logger.warning("输出尺寸 %s != 输入 %s，已按原图缩放",
                           out.shape[1], w)
        self._showing_result = True
        self.result_toggle.setChecked(True)
        self.result_toggle.setText("显示生成结果")
        self.res_meta.setText(
            f"HTTP {meta.get('http_status')} | {meta.get('size_used')} | "
            f"{meta.get('elapsed_s')}s")
        info("生成完成",
             f"接口 {meta.get('size_used')}，用时 {meta.get('elapsed_s')}s。"
             f"用⑤切换原图/结果查看。", self)

    def _on_gen_error(self, err: str) -> None:
        self._set_gen_busy(False)
        self.gen_status.setText("生成失败")
        error("生成失败", err, self)

    # ── 结果切换 / 保存 ──

    def _on_result_toggled(self, checked: bool) -> None:
        if checked and self._result_pixmap is not None:
            self.browser.viewer.set_image(self._result_pixmap)
            self._showing_result = True
            self.result_toggle.setText("显示生成结果")
        else:
            self.browser.viewer.set_image(self.browser.current_pixmap)
            self._showing_result = False
            self.result_toggle.setText("显示生成结果")
            self._draw_overlays()

    def _on_save_result(self) -> None:
        if self._result_np is None:
            warning("无结果", "请先点击「AI 生成」。", self)
            return
        export_dir = self.export_browser.path.strip()
        if not export_dir:
            warning("未设置导出目录", "请在⑥中先选择导出目录。", self)
            return
        if self._orig_path is None:
            return

        img_dir = Path(export_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        stem = self._orig_path.stem
        result_name = f"{stem}.png"
        Image.fromarray(self._result_np).save(img_dir / result_name)

        w, h = self._orig_size
        record = {
            "stem": stem,
            "result_name": result_name,
            "width": w, "height": h,
            "boxes": [
                {
                    "cls_id": b["cls_id"],
                    "cls_name": b["cls_name"],
                    "xyxy": (
                        int(b["rect"].x()), int(b["rect"].y()),
                        int(b["rect"].x() + b["rect"].width()),
                        int(b["rect"].y() + b["rect"].height()),
                    ),
                }
                for b in self._boxes
            ],
        }
        self._records[stem] = record
        info("已保存", f"结果图 → {img_dir / result_name}", self)

    # ── 导出 ──

    def _on_export(self) -> None:
        export_dir = self.export_browser.path.strip()
        if not export_dir:
            warning("未设置导出目录", "请在⑥中先选择导出目录。", self)
            return
        if not self._records:
            warning("无标注数据", "请先逐张「AI 生成」并「保存当前结果」。", self)
            return

        records = list(self._records.values())
        msgs = []
        if self.yolo_chk.isChecked():
            r = export_yolo(records, export_dir)
            msgs.append(f"YOLO: {r['labels']} 个 label + data.yaml")
        if self.xal_chk.isChecked():
            version = self.version_edit.text().strip() or DEFAULT_XAL_VERSION
            r = export_xanylabeling(records, export_dir, version)
            msgs.append(f"X-AnyLabeling JSON: {r['json']} 个 (v{version})")
        if not msgs:
            warning("未选择格式", "请至少勾选一种导出格式。", self)
            return
        info("导出完成", "\n".join(msgs), self)

    # ── 键盘快捷键 ──

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_R, Qt.Key.Key_D):
            self.browser.go_next()
        elif key in (Qt.Key.Key_Q, Qt.Key.Key_A):
            self.browser.go_prev()
        elif key == Qt.Key.Key_U:
            self._on_undo_box()
        else:
            super().keyPressEvent(event)

    # ── 持久化 ──

    def _save_settings(self) -> None:
        """设置项改动即持久化"""
        set_str("sensenova_api_key", self.key_edit.text().strip())
        set_str("sensenova_model", self.model_edit.text().strip())
        set_bool("sensenova_keep_size", self.keep_size_chk.isChecked())
        set_bool("sensenova_watermark", self.watermark_chk.isChecked())
        set_bool("sensenova_extend", self.extend_chk.isChecked())
        set_int("sensenova_timeout", self.timeout_spin.value())
        set_str("sensenova_xal_version", self.version_edit.text().strip())
        set_bool("sensenova_export_yolo", self.yolo_chk.isChecked())
        set_bool("sensenova_export_xal", self.xal_chk.isChecked())
        set_bool("sensenova_auto_append", self.auto_append_chk.isChecked())

    def _load_settings(self) -> None:
        self.img_browser.path = get_str("sensenova_img_dir")
        self.export_browser.path = get_str("sensenova_export_dir")
        self.key_edit.setText(get_str("sensenova_api_key"))
        self.model_edit.setText(get_str("sensenova_model", "sensenova-u1.5-lite"))
        self.keep_size_chk.setChecked(get_bool("sensenova_keep_size", True))
        self.watermark_chk.setChecked(get_bool("sensenova_watermark", False))
        self.extend_chk.setChecked(get_bool("sensenova_extend", False))
        self.timeout_spin.setValue(get_int("sensenova_timeout", 180))
        self.version_edit.setText(get_str("sensenova_xal_version", DEFAULT_XAL_VERSION))
        self.yolo_chk.setChecked(get_bool("sensenova_export_yolo", True))
        self.xal_chk.setChecked(get_bool("sensenova_export_xal", True))
        self.auto_append_chk.setChecked(get_bool("sensenova_auto_append", True))
        self.auto_append_chk.toggled.connect(self._on_auto_append_changed)

        # 设置改动即持久化
        self.key_edit.textChanged.connect(lambda: self._save_settings())
        self.model_edit.textChanged.connect(lambda: self._save_settings())
        self.keep_size_chk.toggled.connect(lambda: self._save_settings())
        self.watermark_chk.toggled.connect(lambda: self._save_settings())
        self.extend_chk.toggled.connect(lambda: self._save_settings())
        self.timeout_spin.valueChanged.connect(lambda _: self._save_settings())
        self.version_edit.textChanged.connect(lambda: self._save_settings())
        self.yolo_chk.toggled.connect(lambda: self._save_settings())
        self.xal_chk.toggled.connect(lambda: self._save_settings())

        classes_json = get_str("sensenova_classes_json")
        if classes_json:
            try:
                self._classes = json.loads(classes_json)
                self.classes_text.setPlainText(
                    json.dumps(self._classes, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                self._classes = list(DEFAULT_CLASSES)
                self.classes_text.setPlainText(
                    json.dumps(DEFAULT_CLASSES, ensure_ascii=False, indent=2))
        else:
            self._classes = list(DEFAULT_CLASSES)
            self.classes_text.setPlainText(
                json.dumps(DEFAULT_CLASSES, ensure_ascii=False, indent=2))
        self._refresh_class_combo()

        # 目录已持久化则自动扫描
        if self.img_browser.path.strip():
            self._on_dir_changed("")


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

def _np_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """RGB ndarray -> QPixmap"""
    from PySide6.QtGui import QImage
    im = Image.fromarray(arr).convert("RGB")
    data = im.tobytes("raw", "RGB")
    qimg = QImage(data, im.width, im.height, im.width * 3,
                  QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)
