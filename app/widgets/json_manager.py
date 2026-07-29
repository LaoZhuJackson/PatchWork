"""JSON 标注管理面板：按帧间隔删除/生成"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QHeaderView, QTableWidget, QTableWidgetItem, QButtonGroup,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, ProgressBar,
    BodyLabel, StrongBodyLabel, SubtitleLabel,
    CardWidget, SpinBox, ComboBox, LineEdit, RadioButton, TableWidget,
)

from app.services.json_manager import (
    scan_directory, delete_by_interval, generate_empty_json,
)
from app.utils.config import get_str, set_str, get_int, set_int
from app.utils.message import info, error, confirm
from app.utils.worker import Worker
from app.widgets.path_browser import PathBrowser
from app.utils.logger import get_logger

logger = get_logger(__name__)


class JsonManagerPanel(QWidget):
    """JSON 标注管理"""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("json_manager_panel")
        self._prefix_data: list[dict] = []  # scan_directory 的返回
        self._interval_spins: list[SpinBox] = []  # 每行一个 SpinBox
        self._worker: Worker | None = None

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("JSON 标注管理"))

        # ---- 目录 ----
        layout.addWidget(StrongBodyLabel("目标目录"))
        dir_card = CardWidget()
        dir_layout = QVBoxLayout(dir_card)

        dir_row = QHBoxLayout()
        self.dir_browser = PathBrowser(
            label="", mode="dir",
            placeholder="选择包含图片和 JSON 的目录...",
            config_key="jm_target_dir",
        )
        dir_row.addWidget(self.dir_browser, 1)
        self.scan_btn = PushButton("扫描")
        self.scan_btn.clicked.connect(self._on_scan)
        dir_row.addWidget(self.scan_btn)
        dir_layout.addLayout(dir_row)

        layout.addWidget(dir_card)

        # ---- 前缀表格 ----
        layout.addWidget(StrongBodyLabel("前缀配置"))
        self.prefix_table = TableWidget(self)
        self.prefix_table.setWordWrap(False)
        self.prefix_table.setBorderVisible(True)
        self.prefix_table.setBorderRadius(8)
        self.prefix_table.setRowCount(0)
        self.prefix_table.setColumnCount(3)
        self.prefix_table.verticalHeader().hide()
        self.prefix_table.setHorizontalHeaderLabels(["前缀", "图片数/JSON数", "间隔(帧)"])
        self.prefix_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.prefix_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Fixed
        )
        self.prefix_table.setColumnWidth(1, 200)
        self.prefix_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Fixed
        )
        self.prefix_table.setColumnWidth(2, 200)
        self.prefix_table.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        self.prefix_table.setAlternatingRowColors(True)
        layout.addWidget(self.prefix_table, 1)

        # ---- 操作 ----
        layout.addWidget(StrongBodyLabel("操作"))
        action_card = CardWidget()
        action_layout = QVBoxLayout(action_card)

        mode_row = QHBoxLayout()
        mode_row.addWidget(BodyLabel("模式:"))
        self.delete_radio = RadioButton("删除多余 JSON")
        self.create_radio = RadioButton("生成空 JSON")
        self.delete_radio.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.delete_radio, 0)
        self._mode_group.addButton(self.create_radio, 1)
        self._mode_group.buttonClicked.connect(self._on_mode_changed)
        mode_row.addWidget(self.delete_radio)
        mode_row.addWidget(self.create_radio)
        mode_row.addStretch()
        action_layout.addLayout(mode_row)

        ver_row = QHBoxLayout()
        ver_row.addWidget(BodyLabel("Version:"))
        self.version_edit = LineEdit()
        self.version_edit.setText("4.0.0-beta.13")
        self.version_edit.setPlaceholderText("X-AnyLabeling 版本号")
        self.version_edit.setEnabled(False)
        self.version_edit.textChanged.connect(
            lambda v: set_str("jm_version", v)
        )
        ver_row.addWidget(self.version_edit, 1)
        ver_row.addStretch()
        action_layout.addLayout(ver_row)

        btn_row = QHBoxLayout()
        self.status_label = BodyLabel("")
        btn_row.addWidget(self.status_label, 1)
        self.dry_btn = PushButton("干运行")
        self.dry_btn.clicked.connect(lambda: self._on_execute(apply=False))
        btn_row.addWidget(self.dry_btn)
        self.apply_btn = PrimaryPushButton("执行")
        self.apply_btn.clicked.connect(lambda: self._on_execute(apply=True))
        btn_row.addWidget(self.apply_btn)
        action_layout.addLayout(btn_row)

        layout.addWidget(action_card)

        # ---- 进度 ----
        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

    # ---- 扫描 ----

    def _on_scan(self) -> None:
        target = Path(self.dir_browser.path)
        if not target.is_dir():
            error("路径错误", "请选择有效的目录", self)
            return

        self._prefix_data = scan_directory(target)
        self._interval_spins.clear()
        self.prefix_table.setRowCount(0)

        # 读取已保存的间隔配置
        saved = {}
        raw = get_str("jm_intervals", "{}")
        try:
            saved = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        for i, info in enumerate(self._prefix_data):
            prefix = info["prefix"]
            self.prefix_table.insertRow(i)

            # 前缀名
            item = QTableWidgetItem(prefix)
            item.setToolTip(prefix)
            self.prefix_table.setItem(i, 0, item)

            # 图片/JSON 数
            stats = f"{info['image_count']}图/{info['json_count']}JSON"
            item2 = QTableWidgetItem(stats)
            item2.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.prefix_table.setItem(i, 1, item2)

            # 间隔 SpinBox
            spin = SpinBox()
            spin.setRange(1, 9999)
            default = saved.get(prefix, 5)
            spin.setValue(default)
            spin.valueChanged.connect(self._save_intervals)
            self._interval_spins.append(spin)
            self.prefix_table.setCellWidget(i, 2, spin)

        self.status_label.setText(f"扫描完成: {len(self._prefix_data)} 个前缀")

    # ---- 持久化前缀间隔 ----

    def _save_intervals(self) -> None:
        data = {}
        for info, spin in zip(self._prefix_data, self._interval_spins):
            data[info["prefix"]] = spin.value()
        set_str("jm_intervals", json.dumps(data, ensure_ascii=False))

    # ---- 执行 ----

    def _on_mode_changed(self) -> None:
        is_create = self._mode_group.checkedId() == 1
        self.version_edit.setEnabled(is_create)

    def _get_prefix_intervals(self) -> dict[str, int]:
        result = {}
        for info, spin in zip(self._prefix_data, self._interval_spins):
            result[info["prefix"]] = spin.value()
        return result

    def _on_execute(self, apply: bool) -> None:
        target = Path(self.dir_browser.path)
        if not target.is_dir():
            error("路径错误", "请选择有效的目录", self)
            return
        if not self._prefix_data:
            error("未扫描", "请先点击「扫描」", self)
            return

        intervals = self._get_prefix_intervals()
        is_create = self._mode_group.checkedId() == 1
        label = "生成" if is_create else "删除"
        action = "实际执行" if apply else "干运行"

        if apply and is_create:
            pass  # 生成模式不需要确认
        elif apply:
            if not confirm(
                    f"确认{label}",
                    f"即将{label}多余 JSON 文件。\n此操作不可撤销，是否继续？",
                    self,
            ):
                return

        self.dry_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText(f"{label}中 ({action})...")

        try:
            if is_create:
                result = generate_empty_json(
                    target, intervals,
                    version=self.version_edit.text().strip() or "4.0.0-beta.13",
                )
            else:
                result = delete_by_interval(target, intervals, apply=apply)

            self._show_result(result, label, action, is_create)
        except Exception as e:
            error(f"{label}失败", str(e), self)
        finally:
            self.dry_btn.setEnabled(True)
            self.apply_btn.setEnabled(True)
            self.progress.setVisible(False)

    def _show_result(
            self, result: dict, label: str, action: str, is_create: bool,
    ) -> None:
        parts = [f"{label}结果 ({action})\n"]
        total_op = 0
        for prefix, s in result.items():
            prefix_short = prefix[:60] + "..." if len(prefix) > 63 else prefix
            if is_create:
                parts.append(
                    f"  {prefix_short}: "
                    f"新建 {s['created']} / 跳过 {s['skipped']}"
                )
                total_op += s["created"]
            else:
                parts.append(
                    f"  {prefix_short}: "
                    f"保留 {s['kept']} / {'删除' if action == '实际执行' else '将删除'}{s['deleted']}"
                )
                total_op += s["deleted"]
        parts.append(f"\n共{'生成' if is_create else '处理'} {total_op} 个文件")
        self.status_label.setText(f"{label}完成: {total_op} 个")
        info(f"{label}结果", "\n".join(parts), self)

    # ---- 持久化 ----

    def _load_settings(self) -> None:
        self.dir_browser.path = get_str("jm_target_dir")
        saved_ver = get_str("jm_version", "4.0.0-beta.13")
        if saved_ver:
            self.version_edit.setText(saved_ver)
