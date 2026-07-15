"""F6: 远程 GPU 监控面板"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, LineEdit, ProgressBar,
    BodyLabel, StrongBodyLabel, SubtitleLabel,
    CardWidget, PasswordLineEdit, SpinBox,
)

from app.services.gpu_client import fetch_gpu_info, GPUInfo
from app.utils.config import get_str, set_str, get_int, set_int
from app.utils.message import error, info
from app.utils.worker import Worker


class GPUWorker(Worker):
    """后台 SSH 获取 GPU 信息"""

    def __init__(self, host: str, port: int, username: str, password: str, key_path: str) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_path = key_path

    def dp_work(self) -> list[GPUInfo]:
        return fetch_gpu_info(
            self.host, self.port, self.username, self.password, self.key_path
        )


class GPUMonitorPanel(QWidget):
    """远程 GPU 监控面板"""
    status_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("gpu_monitor_panel")
        self._worker: GPUWorker | None = None
        self._session_password: str = ""  # 仅会话内存活

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # ---- 标题 ----
        layout.addWidget(SubtitleLabel("GPU 监控"))

        # ---- SSH 配置 ----
        layout.addWidget(StrongBodyLabel("SSH 连接设置"))
        conn_card = CardWidget()
        conn_form = QFormLayout(conn_card)

        # Host
        self.host_edit = LineEdit()
        self.host_edit.setPlaceholderText("服务器地址")
        self.host_edit.textChanged.connect(
            lambda v: set_str("gpu_host", v)
        )
        conn_form.addRow(BodyLabel("Host:"), self.host_edit)

        # Port
        self.port_spin = SpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        self.port_spin.valueChanged.connect(
            lambda v: set_int("gpu_port", v)
        )
        conn_form.addRow(BodyLabel("Port:"), self.port_spin)

        # User
        self.user_edit = LineEdit()
        self.user_edit.setPlaceholderText("root")
        self.user_edit.textChanged.connect(
            lambda v: set_str("gpu_username", v)
        )
        conn_form.addRow(BodyLabel("User:"), self.user_edit)

        # Key path
        self.key_edit = LineEdit()
        self.key_edit.setPlaceholderText("留空则使用密码登录")
        self.key_edit.textChanged.connect(
            lambda v: set_str("gpu_key_path", v)
        )
        conn_form.addRow(BodyLabel("密钥路径:"), self.key_edit)

        # Password（仅会话持有，不持久化）
        self.pwd_edit = PasswordLineEdit()
        self.pwd_edit.setPlaceholderText("输入 SSH 密码（仅本次会话有效）")
        conn_form.addRow(BodyLabel("密码:"), self.pwd_edit)

        layout.addWidget(conn_card)

        # ---- 刷新按钮 ----
        btn_row = QHBoxLayout()
        self.status_label = BodyLabel("")
        btn_row.addWidget(self.status_label)
        btn_row.addStretch()
        self.refresh_btn = PrimaryPushButton("刷新 GPU 状态")
        self.refresh_btn.clicked.connect(self._on_refresh)
        btn_row.addWidget(self.refresh_btn)

        layout.addLayout(btn_row)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # ---- GPU 卡片区 ----
        self.cards_widget = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_widget)
        self.cards_layout.setSpacing(8)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.cards_widget, 1)

        layout.addStretch()

    # ---- 事件 ----
    def _on_refresh(self) -> None:
        host = self.host_edit.text().strip()
        if not host:
            error("错误", "请填写 Host 地址", self)
            return

        port = self.port_spin.value()
        username = self.user_edit.text().strip()
        key_path = self.key_edit.text().strip()
        password = self.pwd_edit.text()

        # 保存会话密码
        if password:
            self._session_password = password

        if not username:
            error("错误", "请填写用户名", self)
            return

        if not key_path and not self._session_password:
            error("错误", "请填写密码或指定密钥路径", self)
            return

        self.refresh_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("正在连接...")

        self._worker = GPUWorker(
            host, port, username, self._session_password, key_path
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, gpus: list[GPUInfo]) -> None:
        self.refresh_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(
            f"共 {len(gpus)} 张 GPU，刷新成功"
        )
        self._build_cards(gpus)

    def _on_error(self, err_msg: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("❌ 连接失败")
        error("GPU 监控出错", err_msg, self)

    def _build_cards(self, gpus: list[GPUInfo]) -> None:
        """根据 GPU 数据重建卡片"""
        # 清空旧卡片
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for gpu in gpus:
            card = CardWidget()
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(6)
            # 标题行: GPU 名称 + 利用率
            title_row = QHBoxLayout()
            title = BodyLabel(f"GPU {gpu.index}: {gpu.name}")
            title_row.addWidget(title)
            title_row.addStretch()
            title_row.addWidget(
                BodyLabel(f"利用率 {gpu.utilization}%")
            )
            card_layout.addLayout(title_row)

            # 利用率进度条
            util_bar = QProgressBar()
            util_bar.setRange(0, 100)
            util_bar.setValue(gpu.utilization)
            util_bar.setTextVisible(True)
            util_bar.setFormat(f"{gpu.utilization}%")
            util_bar.setMaximumHeight(22)
            card_layout.addWidget(util_bar)

            # 显存信息
            mem_label = BodyLabel(
                f"显存: {gpu.memory_used} / {gpu.memory_total} MB  ({gpu.memory_percent}%)"
            )
            card_layout.addWidget(mem_label)

            card_layout.addWidget(mem_label)
            mem_bar = QProgressBar()

            mem_bar.setRange(0, 100)
            mem_bar.setValue(int(gpu.memory_percent))
            mem_bar.setTextVisible(True)
            mem_bar.setFormat(f"{gpu.memory_percent}%")
            mem_bar.setMaximumHeight(22)
            card_layout.addWidget(mem_bar)

            # 进程列表
            if gpu.processes:
                proc_text = "\n".join(
                    f"  PID {p['pid']}  {p['name']}  ({p['memory']} MB)"
                    for p in gpu.processes[:10]
                )
                proc_label = QLabel(proc_text)
                proc_label.setStyleSheet("color: #888; font-size: 12px;")
                card_layout.addWidget(proc_label)
            else:
                card_layout.addWidget(
                    BodyLabel("  (无运行进程)")
                )

            self.cards_layout.addWidget(card)

    # ---- 持久化 ----

    def _load_settings(self) -> None:
        self.host_edit.setText(get_str("gpu_host", ""))
        self.port_spin.setValue(get_int("gpu_port", 22))
        self.user_edit.setText(get_str("gpu_username", ""))
        self.key_edit.setText(get_str("gpu_key_path", ""))
        self.pwd_edit.setText(self._session_password)
