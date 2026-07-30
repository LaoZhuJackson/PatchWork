"""PPT 汇报生成面板。"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    CheckBox,
    ComboBox,
    LineEdit,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    StrongBodyLabel,
    SubtitleLabel,
)

from app.services.presentation import (
    PresentationRequest,
    PresentationResult,
    PresentationService,
    PresentationServiceError,
)
from app.utils.config import get_bool, get_str, set_bool, set_str
from app.utils.message import error, info, warning
from app.utils.worker import Worker
from app.widgets.path_browser import PathBrowser


class PresentationWorker(Worker):
    """后台调用工作区 PPT CLI。"""

    def __init__(
        self,
        operation: str,
        request: PresentationRequest,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.operation = operation
        self.request = request

    def do_work(self) -> PresentationResult:
        service = PresentationService()
        if self.operation == "validate":
            return service.validate(self.request)
        if self.operation == "inspect":
            return service.inspect(self.request)
        if self.operation == "build":
            return service.build(self.request)
        raise ValueError(f"未知 PPT 操作：{self.operation}")


class PresentationPanel(QWidget):
    """工作区 PPT 生成面板。"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("presentation_panel")

        self._service = PresentationService()
        self._worker: PresentationWorker | None = None
        self._last_outputs: dict[str, str] = {}

        self._setup_ui()
        self._load_settings()
        self._refresh_workspace()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("PPT 汇报生成"))
        intro = BodyLabel(
            "调用工作区中的 make_slides.py。PatchWork 只负责选择参数、"
            "执行和打开结果，不复制 Claude/Marp 的生成逻辑。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ---- 工作区 ----
        layout.addWidget(StrongBodyLabel("工作区"))
        workspace_card = CardWidget()
        workspace_form = QFormLayout(workspace_card)

        self.workspace_browser = PathBrowser(
            label="",
            mode="dir",
            placeholder="选择包含 .slides/ 和 utils/make_slides.py 的工作区...",
            config_key="presentation_workspace",
            dialog_title="选择 Markdown 项目工作区",
        )
        self.workspace_browser.path_changed.connect(self._on_workspace_changed)
        workspace_form.addRow(BodyLabel("工作区目录:"), self.workspace_browser)

        profile_row = QHBoxLayout()
        self.profile_combo = ComboBox()
        self.profile_combo.setMinimumWidth(220)
        self.refresh_btn = PushButton("刷新")
        self.refresh_btn.clicked.connect(self._refresh_workspace)
        profile_row.addWidget(self.profile_combo, 1)
        profile_row.addWidget(self.refresh_btn)
        workspace_form.addRow(BodyLabel("汇报类型:"), profile_row)

        self.python_edit = LineEdit()
        self.python_edit.setPlaceholderText(
            "留空使用 PatchWork 当前 Python；打包版可填写 python.exe"
        )
        self.python_edit.editingFinished.connect(
            lambda: set_str("presentation_python", self.python_edit.text().strip())
        )
        workspace_form.addRow(BodyLabel("Python解释器:"), self.python_edit)

        layout.addWidget(workspace_card)

        # ---- Deck ----
        layout.addWidget(StrongBodyLabel("汇报内容"))
        deck_card = CardWidget()
        deck_form = QFormLayout(deck_card)

        self.deck_browser = PathBrowser(
            label="",
            mode="file",
            file_filter="Deck YAML (*.yaml *.yml);;All Files (*)",
            placeholder="选择 Claude 生成的 deck.yaml...",
            config_key="presentation_deck",
            dialog_title="选择 deck.yaml",
        )
        deck_form.addRow(BodyLabel("Deck 文件:"), self.deck_browser)

        self.latest_deck_btn = PushButton("使用最近的 Deck")
        self.latest_deck_btn.clicked.connect(self._select_latest_deck)
        deck_form.addRow(BodyLabel("自动查找:"), self.latest_deck_btn)

        self.output_name_edit = LineEdit()
        self.output_name_edit.setPlaceholderText("留空时自动使用日期和汇报类型")
        self.output_name_edit.editingFinished.connect(
            lambda: set_str(
                "presentation_output_name",
                self.output_name_edit.text().strip(),
            )
        )
        deck_form.addRow(BodyLabel("输出文件名:"), self.output_name_edit)

        self.output_dir_browser = PathBrowser(
            label="",
            mode="dir",
            placeholder="留空时使用工作区 config.yaml 中的输出目录...",
            config_key="presentation_output_dir",
            dialog_title="选择 PPT 输出目录",
        )
        deck_form.addRow(BodyLabel("输出目录:"), self.output_dir_browser)

        layout.addWidget(deck_card)

        # ---- 输出格式 ----
        layout.addWidget(StrongBodyLabel("输出选项"))
        option_card = CardWidget()
        option_layout = QVBoxLayout(option_card)

        format_row = QHBoxLayout()
        self.pptx_check = CheckBox("PPTX")
        self.pdf_check = CheckBox("PDF")
        self.html_check = CheckBox("HTML")
        format_row.addWidget(self.pptx_check)
        format_row.addWidget(self.pdf_check)
        format_row.addWidget(self.html_check)
        format_row.addStretch()
        option_layout.addLayout(format_row)

        self.pptx_check.stateChanged.connect(self._save_format_settings)
        self.pdf_check.stateChanged.connect(self._save_format_settings)
        self.html_check.stateChanged.connect(self._save_format_settings)

        layout.addWidget(option_card)

        # ---- 操作 ----
        button_row = QHBoxLayout()
        self.validate_btn = PushButton("校验配置")
        self.inspect_btn = PushButton("提取项目上下文")
        self.build_btn = PrimaryPushButton("生成汇报")
        self.open_btn = PushButton("打开生成结果")
        self.open_btn.setEnabled(False)

        self.validate_btn.clicked.connect(lambda: self._start_operation("validate"))
        self.inspect_btn.clicked.connect(lambda: self._start_operation("inspect"))
        self.build_btn.clicked.connect(lambda: self._start_operation("build"))
        self.open_btn.clicked.connect(self._open_result)

        button_row.addWidget(self.validate_btn)
        button_row.addWidget(self.inspect_btn)
        button_row.addStretch()
        button_row.addWidget(self.open_btn)
        button_row.addWidget(self.build_btn)
        layout.addLayout(button_row)

        self.status_label = BodyLabel("请选择工作区")
        layout.addWidget(self.status_label)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addWidget(StrongBodyLabel("执行日志"))
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(220)
        self.log_edit.setPlaceholderText("校验、提取上下文和生成结果会显示在这里。")
        layout.addWidget(self.log_edit)

        layout.addStretch()

    def _load_settings(self) -> None:
        self.workspace_browser.path = get_str("presentation_workspace")
        self.deck_browser.path = get_str("presentation_deck")
        self.output_dir_browser.path = get_str("presentation_output_dir")
        self.output_name_edit.setText(get_str("presentation_output_name"))
        self.python_edit.setText(get_str("presentation_python"))

        self.pptx_check.setChecked(get_bool("presentation_pptx", True))
        self.pdf_check.setChecked(get_bool("presentation_pdf", True))
        self.html_check.setChecked(get_bool("presentation_html", False))

    def _save_format_settings(self) -> None:
        set_bool("presentation_pptx", self.pptx_check.isChecked())
        set_bool("presentation_pdf", self.pdf_check.isChecked())
        set_bool("presentation_html", self.html_check.isChecked())

    def _on_workspace_changed(self, _: str) -> None:
        self._refresh_workspace()

    def _refresh_workspace(self) -> None:
        workspace_text = self.workspace_browser.path.strip()
        previous_profile = (
            self.profile_combo.currentText()
            if self.profile_combo.count()
            else get_str("presentation_profile", "技术分享")
        )

        self.profile_combo.clear()

        if not workspace_text:
            self.status_label.setText("请选择工作区")
            return

        workspace = Path(workspace_text)
        ready, reason = self._service.is_workspace_ready(workspace)
        if not ready:
            self.status_label.setText(f"⚠️ {reason}")
            return

        profiles = self._service.discover_profiles(workspace)
        self.profile_combo.addItems(profiles)

        preferred = previous_profile if previous_profile in profiles else ""
        if not preferred:
            saved = get_str("presentation_profile", "技术分享")
            preferred = saved if saved in profiles else ""
        if not preferred and profiles:
            preferred = profiles[0]
        if preferred:
            self.profile_combo.setCurrentText(preferred)

        self.status_label.setText(
            f"✅ 工作区就绪，共发现 {len(profiles)} 个汇报类型"
        )

        if not self.deck_browser.path:
            decks = self._service.discover_decks(workspace)
            if decks:
                self.deck_browser.path = str(decks[0])

    def _select_latest_deck(self) -> None:
        workspace_text = self.workspace_browser.path.strip()
        if not workspace_text:
            warning("缺少工作区", "请先选择工作区目录。", self)
            return

        decks = self._service.discover_decks(workspace_text)
        if not decks:
            warning(
                "未找到 Deck",
                "没有在 .slides/runtime、.slides/example(s) 或汇报记录中找到 deck YAML。",
                self,
            )
            return

        self.deck_browser.path = str(decks[0])
        set_str("presentation_deck", str(decks[0]))
        self.status_label.setText(f"已选择最近 Deck：{decks[0].name}")

    def _start_operation(self, operation: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            warning("任务进行中", "请等待当前任务结束。", self)
            return

        try:
            request = self._build_request(operation)
        except ValueError as exc:
            error("参数错误", str(exc), self)
            return

        if self.profile_combo.currentText():
            set_str("presentation_profile", self.profile_combo.currentText())

        self._set_busy(True)
        action_name = {
            "validate": "校验配置",
            "inspect": "提取项目上下文",
            "build": "生成汇报",
        }[operation]
        self.status_label.setText(f"正在{action_name}...")
        self.log_edit.setPlainText(
            f"操作：{action_name}\n工作区：{request.workspace}\n"
        )

        self._worker = PresentationWorker(operation, request, self)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _build_request(self, operation: str) -> PresentationRequest:
        workspace_text = self.workspace_browser.path.strip()
        if not workspace_text:
            raise ValueError("请选择工作区目录。")

        workspace = Path(workspace_text)
        ready, reason = self._service.is_workspace_ready(workspace)
        if not ready:
            raise ValueError(f"工作区尚未准备好：{reason}")

        profile = self.profile_combo.currentText().strip()
        formats = tuple(
            name
            for name, checked in (
                ("pptx", self.pptx_check.isChecked()),
                ("pdf", self.pdf_check.isChecked()),
                ("html", self.html_check.isChecked()),
            )
            if checked
        )

        deck: Path | None = None
        if operation == "build":
            deck_text = self.deck_browser.path.strip()
            if not deck_text:
                raise ValueError("请选择 deck.yaml。")
            deck = Path(deck_text)
            if not deck.is_file():
                raise ValueError(f"deck 文件不存在：{deck}")
            if not formats:
                raise ValueError("请至少选择一种输出格式。")

        output_dir_text = self.output_dir_browser.path.strip()
        output_dir = Path(output_dir_text) if output_dir_text else None

        return PresentationRequest(
            workspace=workspace,
            profile=profile,
            deck=deck,
            output_dir=output_dir,
            output_name=self.output_name_edit.text().strip(),
            formats=formats or ("pptx",),
            python_executable=self.python_edit.text().strip(),
            timeout=600,
        )

    def _on_finished(self, result: PresentationResult) -> None:
        self._set_busy(False)
        self._last_outputs = result.outputs

        payload_text = json.dumps(
            result.payload,
            ensure_ascii=False,
            indent=2,
        )
        command_text = subprocess.list2cmdline(result.command)
        stderr_text = result.stderr.strip()

        log_parts = [
            f"命令：{command_text}",
            "",
            payload_text,
        ]
        if stderr_text:
            log_parts += ["", "标准错误：", stderr_text]
        self.log_edit.setPlainText("\n".join(log_parts))

        if result.operation == "validate":
            self.status_label.setText("✅ 配置校验通过")
            info("校验完成", "工作区 PPT 配置可正常读取。", self)
            return

        if result.operation == "inspect":
            context_file = result.payload.get("context_file")
            self.status_label.setText("✅ 项目上下文已生成")
            message = "项目概览上下文提取完成。"
            if context_file:
                message += f"\n\n{context_file}"
                self._last_outputs = {"context": str(context_file)}
                self.open_btn.setEnabled(True)
            info("提取完成", message, self)
            return

        self.status_label.setText("✅ 汇报生成完成")
        self.open_btn.setEnabled(bool(self._last_outputs))
        outputs = "\n".join(
            f"{key.upper()}: {value}"
            for key, value in self._last_outputs.items()
        )
        info("生成完成", outputs or "汇报文件已生成。", self)

    def _on_worker_error(self, traceback_text: str) -> None:
        self._set_busy(False)
        self.status_label.setText("❌ 操作失败")
        self.log_edit.setPlainText(traceback_text)

        # Worker 会把异常 traceback 传回来；最后一行通常最适合弹窗展示。
        lines = [line.strip() for line in traceback_text.splitlines() if line.strip()]
        message = lines[-1] if lines else traceback_text
        error("PPT 操作失败", message, self)

    def _set_busy(self, busy: bool) -> None:
        for widget in (
            self.workspace_browser,
            self.profile_combo,
            self.refresh_btn,
            self.python_edit,
            self.deck_browser,
            self.latest_deck_btn,
            self.output_name_edit,
            self.output_dir_browser,
            self.pptx_check,
            self.pdf_check,
            self.html_check,
            self.validate_btn,
            self.inspect_btn,
            self.build_btn,
        ):
            widget.setEnabled(not busy)

        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

    def _open_result(self) -> None:
        if not self._last_outputs:
            return

        preferred_keys = ("pptx", "pdf", "html", "context", "markdown", "manifest")
        for key in preferred_keys:
            raw_path = self._last_outputs.get(key)
            if not raw_path:
                continue
            path = Path(raw_path)
            target = path if path.exists() else path.parent
            if target.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))
                return

        warning("文件不存在", "没有找到可打开的生成结果。", self)
