"""回调捕获面板：设置端口/路由启动监听，收到请求后列表 + 详情展示"""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    SpinBox,
    SubtitleLabel,
)

from app.services.callback_dump import CallbackDumper
from app.utils.config import get_int, get_str, set_int, set_str
from app.utils.message import info, warning


CFG_PORT, CFG_ROUTE, CFG_DUMP = "cb_dump_port", "cb_dump_route", "cb_dump_file"

class CallbackDumpPanel(QWidget):
    record_signal = Signal(object)  # 监听线程 -> GUI 线程投递

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("callback_dump_panel")

        self._dumper = CallbackDumper(on_record=self._on_record_bg)
        self._records: list[dict] = []
        self.record_signal.connect(self._append_record)

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("回调捕获"))

        # ── 控制卡片 ──
        toolbar_card = CardWidget()
        toolbar = QVBoxLayout(toolbar_card)
        toolbar.setContentsMargins(12, 12, 12, 12)
        toolbar.setSpacing(10)

        # 端口 / 路由 / 落盘
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(BodyLabel("端口:"))
        self.port_spin = SpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(9999)
        self.port_spin.setFixedWidth(90)
        row.addWidget(self.port_spin)

        row.addWidget(BodyLabel("监听路由:"))
        self.route_edit = LineEdit()
        self.route_edit.setPlaceholderText("留空接收全部，如 /cb")
        self.route_edit.setFixedWidth(180)
        row.addWidget(self.route_edit)

        row.addWidget(BodyLabel("落盘文件:"))
        self.dump_edit = LineEdit()
        self.dump_edit.setPlaceholderText("可留空（仅内存），如 cb.log")
        self.dump_edit.setFixedWidth(180)
        row.addWidget(self.dump_edit)
        row.addStretch()
        toolbar.addLayout(row)

        # 状态 + 按钮
        btn_row = QHBoxLayout()
        self.status_label = CaptionLabel("未监听")
        btn_row.addWidget(self.status_label)
        btn_row.addStretch()

        self.stop_btn = PushButton("停止")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.stop_btn)

        self.copy_btn = PushButton("复制地址")
        btn_row.addWidget(self.copy_btn)

        self.clear_btn = PushButton("清空")
        btn_row.addWidget(self.clear_btn)

        self.start_btn = PrimaryPushButton("启动")
        btn_row.addWidget(self.start_btn)
        toolbar.addLayout(btn_row)

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.copy_btn.clicked.connect(self._on_copy)
        self.clear_btn.clicked.connect(self._on_clear)
        layout.addWidget(toolbar_card)

        # ── 状态行 ──
        self.line_label = BodyLabel("共 0 条请求")
        layout.addWidget(self.line_label)

        # ── 请求列表卡片 ──
        list_card = CardWidget()
        list_lay = QVBoxLayout(list_card)
        list_lay.setContentsMargins(12, 8, 12, 12)
        list_lay.addWidget(BodyLabel("收到的请求"))
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._show_detail)
        list_lay.addWidget(self.list_widget)
        layout.addWidget(list_card, 1)

        # ── 详情卡片 ──
        detail_card = CardWidget()
        detail_lay = QVBoxLayout(detail_card)
        detail_lay.setContentsMargins(12, 8, 12, 12)
        detail_lay.addWidget(BodyLabel("请求详情"))
        self.tabs = QTabWidget()
        self.headers_view = QTextEdit()
        self.headers_view.setReadOnly(True)
        self.body_view = QTextEdit()
        self.body_view.setReadOnly(True)
        self.tabs.addTab(self.headers_view, "Headers")
        self.tabs.addTab(self.body_view, "Body")
        detail_lay.addWidget(self.tabs)
        detail_card.setFixedHeight(240)
        layout.addWidget(detail_card)

    # ── 持久化 ────────────────────────────────────

    def _load_settings(self) -> None:
        self.port_spin.setValue(get_int(CFG_PORT, 9999))
        self.route_edit.setText(get_str(CFG_ROUTE, ""))
        self.dump_edit.setText(get_str(CFG_DUMP, ""))

    def _save_settings(self) -> None:
        set_int(CFG_PORT, self.port_spin.value())
        set_str(CFG_ROUTE, self.route_edit.text().strip())
        set_str(CFG_DUMP, self.dump_edit.text().strip())

    # ── 动作 ──────────────────────────────────────
    def _on_start(self) -> None:
        ok, msg = self._dumper.start(
            port=self.port_spin.value(),
            route=self.route_edit.text().strip(),
        )
        if ok:
            self._save_settings()
            self.status_label.setText(msg)
            self._set_running_ui(True)
            url = f"http://host.docker.internal:{self.port_spin.value()}"
            route = self.route_edit.text().strip()
            info("启动成功", f"{msg}\n推送 address：{url}{route or '/'}", self)
        else:
            warning("启动失败",msg,self)

    def _on_stop(self) -> None:
        self._dumper.stop()
        self.status_label.setText("未监听")
        self._set_running_ui(False)

    def _on_copy(self) -> None:
        url = f"http://127.0.0.1:{self.port_spin.value()}"
        route = self.route_edit.text().strip()
        QGuiApplication.clipboard().setText(url + (route or "/"))
        info("已复制", f"回调地址：{url}{route or '/'}", self)

    def _on_clear(self) -> None:
        self._records.clear()
        self.list_widget.clear()
        self.headers_view.clear()
        self.body_view.clear()
        self.line_label.setText("共 0 条请求")

    def _set_running_ui(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.port_spin.setEnabled(not running)
        self.route_edit.setEnabled(not running)

    # ── 接收 ──────────────────────────────────────

    def _on_record_bg(self, rec: dict) -> None:
        self.record_signal.emit(rec)  # Signal 跨线程自动 QueuedConnection

    def _append_record(self, rec: dict) -> None:
        self._records.append(rec)
        item = QListWidgetItem(
            f"#{rec['index']}  {rec['ts']}  {rec['method']} {rec['path']}{rec['query'] or ''}  from {rec['ip']}"
        )
        item.setData(Qt.ItemDataRole.UserRole, rec)
        self.list_widget.addItem(item)
        self.list_widget.setCurrentItem(item)
        self.line_label.setText(f"共 {len(self._records)} 条请求")

    def _show_detail(self, current: QListWidgetItem, previous) -> None:
        if current is None:
            return
        rec = current.data(Qt.ItemDataRole.UserRole)
        self.headers_view.setPlainText(rec["headers"])
        self.body_view.setPlainText(rec["pretty"])


