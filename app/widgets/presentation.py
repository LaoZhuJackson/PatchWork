"""PPT 汇报生成面板"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

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
    BodyLabel, CardWidget, CheckBox, ComboBox, LineEdit,
    PrimaryPushButton, ProgressBar, PushButton,
    StrongBodyLabel, SubtitleLabel,
)

from app.services.presentation import (
    PresentationRequest,
    PresentationResult,
    PresentationService,
)
from app.utils.config import get_bool, get_str, set_bool, set_str
from app.utils.message import error, info, warning
from app.utils.worker import Worker
from app.widgets.path_browser import PathBrowser


class PresentationWorker(Worker):
    """后台调用工作区 PPT CLI"""

    def __init__(
        self,
        operation: str,
        request: PresentationRequest,
    ) -> None:
        super().__init__()
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
    """工作区 PPT 生成面板"""

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
        self.profile_combo.currentTextChanged.connect(
            lambda v: set_str("presentation_profile", v)
        )
        self.refresh_btn = PushButton("刷新")
        self.refresh_btn.clicked.connect(self._refresh_workspace)
        profile_row.addWidget(self.profile_combo, 1)
        profile_row.addWidget(self.refresh_btn)
        workspace_form.addRow(BodyLabel("汇报类型:"), profile_row)

        theme_row = QHBoxLayout()
        self.theme_combo = ComboBox()
        self.theme_combo.setMinimumWidth(220)
        self.theme_combo.currentTextChanged.connect(
            lambda v: set_str("presentation_theme", v)
        )
        theme_row.addWidget(self.theme_combo, 1)
        theme_row.addStretch()
        workspace_form.addRow(BodyLabel("PPT 主题:"), theme_row)

        python_row = QHBoxLayout()
        self.python_edit = LineEdit()
        self.python_edit.setPlaceholderText(
            "留空使用 PatchWork 当前 Python；打包版可填写 python.exe"
        )
        self.python_edit.textChanged.connect(
            lambda v: set_str("presentation_python", v)
        )
        python_row.addWidget(self.python_edit)
        workspace_form.addRow(BodyLabel("Python 解释器:"), python_row)

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
        self.output_name_edit.textChanged.connect(
            lambda v: set_str("presentation_output_name", v)
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
        self.pptx_check.stateChanged.connect(
            lambda: set_bool("presentation_pptx", self.pptx_check.isChecked())
        )
        self.pdf_check.stateChanged.connect(
            lambda: set_bool("presentation_pdf", self.pdf_check.isChecked())
        )
        self.html_check.stateChanged.connect(
            lambda: set_bool("presentation_html", self.html_check.isChecked())
        )
        format_row.addWidget(self.pptx_check)
        format_row.addWidget(self.pdf_check)
        format_row.addWidget(self.html_check)
        format_row.addStretch()
        option_layout.addLayout(format_row)

        layout.addWidget(option_card)

        # ---- 操作 ----
        btn_row = QHBoxLayout()
        self.validate_btn = PushButton("校验配置")
        self.inspect_btn = PushButton("提取项目上下文")
        self.build_btn = PrimaryPushButton("生成汇报")
        self.open_btn = PushButton("打开生成结果")
        self.open_btn.setEnabled(False)

        self.validate_btn.clicked.connect(self._on_validate)
        self.inspect_btn.clicked.connect(self._on_inspect)
        self.build_btn.clicked.connect(self._on_build)
        self.open_btn.clicked.connect(self._open_result)

        btn_row.addWidget(self.validate_btn)
        btn_row.addWidget(self.inspect_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.build_btn)
        layout.addLayout(btn_row)

        self.status_label = BodyLabel("请选择工作区")
        layout.addWidget(self.status_label)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addWidget(StrongBodyLabel("执行日志"))
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(120)
        self.log_edit.setPlaceholderText("校验、提取上下文和生成结果会显示在这里。")
        layout.addWidget(self.log_edit)

        layout.addStretch()

    # ---- 操作入口 ----

    def _on_validate(self) -> None:
        self._run_operation("validate")

    def _on_inspect(self) -> None:
        self._run_operation("inspect")

    def _on_build(self) -> None:
        self._run_operation("build")

    def _run_operation(self, operation: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            warning("任务进行中", "请等待当前任务结束。", self)
            return

        try:
            request = self._build_request(operation)
        except ValueError as exc:
            error("参数错误", str(exc), self)
            return

        action_names = {
            "validate": "校验配置",
            "inspect": "提取项目上下文",
            "build": "生成汇报",
        }
        action_name = action_names[operation]

        self._set_inputs_enabled(False)
        self.status_label.setText(f"正在{action_name}...")
        self.log_edit.setPlainText(
            f"操作：{action_name}\n工作区：{request.workspace}\n"
        )
        self.progress.setVisible(True)
        self.progress.setValue(0)

        self._worker = PresentationWorker(operation, request)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ---- Worker 回调 ----

    def _on_done(self, result: PresentationResult) -> None:
        self._set_inputs_enabled(True)
        self.progress.setVisible(False)
        self._last_outputs = result.outputs

        payload_text = json.dumps(result.payload, ensure_ascii=False, indent=2)
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

        # build
        self.status_label.setText("✅ 汇报生成完成")
        self.open_btn.setEnabled(bool(self._last_outputs))
        outputs = "\n".join(
            f"{key.upper()}: {value}"
            for key, value in self._last_outputs.items()
        )
        info("生成完成", outputs or "汇报文件已生成。", self)

    def _on_error(self, err: str) -> None:
        self._set_inputs_enabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("❌ 操作失败")
        self.log_edit.setPlainText(err)

        lines = [line.strip() for line in err.splitlines() if line.strip()]
        message = lines[-1] if lines else err
        error("PPT 操作失败", message, self)

    # ---- 请求构建 ----

    def _build_request(self, operation: str) -> PresentationRequest:
        workspace_text = self.workspace_browser.path.strip()
        if not workspace_text:
            raise ValueError("请选择工作区目录。")

        workspace = Path(workspace_text)
        ready, reason = self._service.is_workspace_ready(workspace)
        if not ready:
            raise ValueError(f"工作区尚未准备好：{reason}")

        profile = self.profile_combo.currentText().strip()
        theme = self.theme_combo.currentText().strip() if self.theme_combo.count() else ""
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
            theme=theme,
            deck=deck,
            output_dir=output_dir,
            output_name=self.output_name_edit.text().strip(),
            formats=formats or ("pptx",),
            python_executable=self.python_edit.text().strip(),
            timeout=600,
        )

    # ---- 工作区刷新 ----

    def _on_workspace_changed(self, _: str) -> None:
        self._refresh_workspace()

    def _refresh_workspace(self) -> None:
        workspace_text = self.workspace_browser.path.strip()
        previous_profile = (
            self.profile_combo.currentText()
            if self.profile_combo.count()
            else get_str("presentation_profile", "技术分享")
        )
        previous_theme = (
            self.theme_combo.currentText()
            if self.theme_combo.count()
            else get_str("presentation_theme", "")
        )

        self.profile_combo.clear()
        self.theme_combo.clear()

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
            set_str("presentation_profile", preferred)

        # ---- 主题发现 ----
        themes = self._service.discover_themes(workspace)
        if themes:
            theme_names = [t["name"] for t in themes]
            self.theme_combo.addItems(theme_names)

            preferred_theme = previous_theme if previous_theme in theme_names else ""
            if not preferred_theme:
                saved = get_str("presentation_theme", "")
                preferred_theme = saved if saved in theme_names else ""
            if not preferred_theme and theme_names:
                # 优先选 signal，其次选 workspace-default
                for fav in ("signal", "workspace-default"):
                    if fav in theme_names:
                        preferred_theme = fav
                        break
                if not preferred_theme:
                    preferred_theme = theme_names[0]
            if preferred_theme:
                self.theme_combo.setCurrentText(preferred_theme)
                set_str("presentation_theme", preferred_theme)

        self.status_label.setText(
            f"✅ 工作区就绪，共发现 {len(profiles)} 个汇报类型、{len(themes)} 个主题"
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

    # ---- 打开结果 ----

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

    # ---- 输入锁 & 持久化 ----

    def _set_inputs_enabled(self, enabled: bool) -> None:
        """异步操作期间禁用所有输入控件"""
        self.workspace_browser.setEnabled(enabled)
        self.profile_combo.setEnabled(enabled)
        self.theme_combo.setEnabled(enabled)
        self.refresh_btn.setEnabled(enabled)
        self.python_edit.setEnabled(enabled)
        self.deck_browser.setEnabled(enabled)
        self.latest_deck_btn.setEnabled(enabled)
        self.output_name_edit.setEnabled(enabled)
        self.output_dir_browser.setEnabled(enabled)
        self.pptx_check.setEnabled(enabled)
        self.pdf_check.setEnabled(enabled)
        self.html_check.setEnabled(enabled)
        self.validate_btn.setEnabled(enabled)
        self.inspect_btn.setEnabled(enabled)
        self.build_btn.setEnabled(enabled)

    def _load_settings(self) -> None:
        self.workspace_browser.path = get_str("presentation_workspace")
        self.deck_browser.path = get_str("presentation_deck")
        self.output_dir_browser.path = get_str("presentation_output_dir")
        self.output_name_edit.setText(get_str("presentation_output_name"))
        self.python_edit.setText(get_str("presentation_python"))

        self.pptx_check.setChecked(get_bool("presentation_pptx", True))
        self.pdf_check.setChecked(get_bool("presentation_pdf", True))
        self.html_check.setChecked(get_bool("presentation_html", False))
