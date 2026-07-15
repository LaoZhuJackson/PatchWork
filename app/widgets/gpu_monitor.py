"""F6: 远程 GPU 监控面板（nvidia-smi / gpustat / HTTP 三种模式）"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
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
    CardWidget, PasswordLineEdit, SpinBox, RadioButton,
)

from app.services.gpu_client import (
    fetch_via_nvidia_smi,
    fetch_via_gpustat,
    fetch_via_http,
    GPUInfo,
)
from app.utils.config import get_str, set_str, get_int, set_int
from app.utils.message import error
from app.utils.worker import Worker


# ============================================================
# Workers — 每种模式一个
# ============================================================

class NvidiaSmiWorker(Worker):
    def __init__(self, host, port, username, password, key_path, gpu_cmd, proc_cmd):
        super().__init__()
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_path = key_path
        self.gpu_cmd = gpu_cmd
        self.proc_cmd = proc_cmd

    def do_work(self) -> list[GPUInfo]:
        return fetch_via_nvidia_smi(
            self.host, self.port, self.username,
            self.password, self.key_path,
            gpu_cmd=self.gpu_cmd, proc_cmd=self.proc_cmd,
        )


class GpustatWorker(Worker):
    def __init__(self, host, port, username, password, key_path, gpustat_cmd, conda_path, conda_env):
        super().__init__()
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_path = key_path
        self.gpustat_cmd = gpustat_cmd
        self.conda_path = conda_path
        self.conda_env = conda_env

    def do_work(self) -> list[GPUInfo]:
        return fetch_via_gpustat(
            self.host, self.port, self.username,
            self.password, self.key_path,
            gpustat_cmd=self.gpustat_cmd,
            conda_path=self.conda_path,
            conda_env=self.conda_env,
        )


class HttpWorker(Worker):
    def __init__(self, api_url, token):
        super().__init__()
        self.api_url = api_url
        self.token = token

    def do_work(self) -> list[GPUInfo]:
        return fetch_via_http(self.api_url, self.token)


# ============================================================
# 面板
# ============================================================

class GPUMonitorPanel(QWidget):
    """远程 GPU 监控面板"""

    status_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("gpu_monitor_panel")
        self._worker: Worker | None = None
        self._session_password: str = ""

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(SubtitleLabel("GPU 监控"))

        # ============================================================
        # 模式选择
        # ============================================================
        mode_row = QHBoxLayout()
        mode_row.addWidget(BodyLabel("获取方式:"))

        self.smi_radio = RadioButton("nvidia-smi")
        self.gpustat_radio = RadioButton("gpustat")
        self.http_radio = RadioButton("HTTP")

        self.smi_radio.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.smi_radio, 0)
        self._mode_group.addButton(self.gpustat_radio, 1)
        self._mode_group.addButton(self.http_radio, 2)
        self._mode_group.buttonClicked.connect(self._on_mode_changed)

        mode_row.addWidget(self.smi_radio)
        mode_row.addWidget(self.gpustat_radio)
        mode_row.addWidget(self.http_radio)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ============================================================
        # SSH 设置区（nvidia-smi 和 gpustat 共用）
        # ============================================================
        self.ssh_card = CardWidget()
        ssh_form = QFormLayout(self.ssh_card)

        self.host_edit = LineEdit()
        self.host_edit.setPlaceholderText("服务器地址")
        self.host_edit.textChanged.connect(lambda v: set_str("gpu_host", v))
        ssh_form.addRow(BodyLabel("Host:"), self.host_edit)

        self.port_spin = SpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        self.port_spin.valueChanged.connect(lambda v: set_int("gpu_port", v))
        ssh_form.addRow(BodyLabel("Port:"), self.port_spin)

        self.user_edit = LineEdit()
        self.user_edit.setPlaceholderText("用户名")
        self.user_edit.textChanged.connect(lambda v: set_str("gpu_username", v))
        ssh_form.addRow(BodyLabel("User:"), self.user_edit)

        self.key_edit = LineEdit()
        self.key_edit.setPlaceholderText("留空则使用密码登录")
        self.key_edit.textChanged.connect(lambda v: set_str("gpu_key_path", v))
        ssh_form.addRow(BodyLabel("密钥路径:"), self.key_edit)

        self.pwd_edit = PasswordLineEdit()
        self.pwd_edit.setPlaceholderText("SSH 密码（仅本次会话有效，不写盘）")
        ssh_form.addRow(BodyLabel("密码:"), self.pwd_edit)

        # 命令 — 仅 nvidia-smi / gpustat 可见
        self.nvsmi_cmd_edit = LineEdit()
        self.nvsmi_cmd_edit.setPlaceholderText(
            "nvidia-smi --query-gpu=index,name,utilization.gpu,"
            "memory.used,memory.total --format=csv,noheader,nounits"
        )
        self.nvsmi_cmd_edit.textChanged.connect(lambda v: set_str("gpu_nvsmi_cmd", v))
        self.nvsmi_cmd_label = BodyLabel("nvidia-smi 命令:")
        ssh_form.addRow(self.nvsmi_cmd_label, self.nvsmi_cmd_edit)

        # Conda 路径
        self.conda_path_edit = LineEdit()
        self.conda_path_edit.setPlaceholderText("/opt/anaconda3/bin/conda")
        self.conda_path_edit.textChanged.connect(lambda v: set_str("gpu_conda_path", v))
        self.conda_path_label = BodyLabel("Conda 路径:")
        ssh_form.addRow(self.conda_path_label, self.conda_path_edit)
        
        self.conda_env_edit = LineEdit()
        self.conda_env_edit.setPlaceholderText("例如: yolo_env")
        self.conda_env_edit.textChanged.connect(lambda v: set_str("gpu_conda_env", v))
        self.conda_env_label = BodyLabel("Conda 环境:")
        ssh_form.addRow(self.conda_env_label, self.conda_env_edit)
        
        self.gpustat_cmd_edit = LineEdit()
        self.gpustat_cmd_edit.setPlaceholderText("gpustat --json")
        self.gpustat_cmd_edit.textChanged.connect(lambda v: set_str("gpu_gpustat_cmd", v))
        self.gpustat_cmd_label = BodyLabel("gpustat 命令:")
        ssh_form.addRow(self.gpustat_cmd_label, self.gpustat_cmd_edit)
        

        layout.addWidget(self.ssh_card)

        # ============================================================
        # HTTP 设置区
        # ============================================================
        self.http_card = CardWidget()
        http_form = QFormLayout(self.http_card)

        self.url_edit = LineEdit()
        self.url_edit.setPlaceholderText("http://服务器地址:5000/api/gpu")
        self.url_edit.textChanged.connect(lambda v: set_str("gpu_api_url", v))
        http_form.addRow(BodyLabel("API 地址:"), self.url_edit)

        self.token_edit = LineEdit()
        self.token_edit.setPlaceholderText("可选，留空则不验证")
        self.token_edit.textChanged.connect(lambda v: set_str("gpu_token", v))
        http_form.addRow(BodyLabel("Token:"), self.token_edit)

        self.http_card.setVisible(False)
        layout.addWidget(self.http_card)

        # ============================================================
        # 操作栏
        # ============================================================
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

        # ============================================================
        # GPU 卡片区
        # ============================================================
        self.cards_widget = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_widget)
        self.cards_layout.setSpacing(8)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.cards_widget, 1)

    # ============================================================
    # 模式切换
    # ============================================================

    def _on_mode_changed(self) -> None:
        mode = self._current_mode()
        set_str("gpu_mode", mode)

        is_ssh = mode in ("nvidia-smi", "gpustat")
        is_smi = mode == "nvidia-smi"
        is_gpustat = mode == "gpustat"

        self.ssh_card.setVisible(is_ssh)
        self.http_card.setVisible(not is_ssh)

        # nvidia-smi 专属
        self.nvsmi_cmd_label.setVisible(is_smi)
        self.nvsmi_cmd_edit.setVisible(is_smi)
        # gpustat 专属
        self.gpustat_cmd_label.setVisible(is_gpustat)
        self.gpustat_cmd_edit.setVisible(is_gpustat)
        self.conda_path_label.setVisible(is_gpustat)
        self.conda_path_edit.setVisible(is_gpustat)
        self.conda_env_label.setVisible(is_gpustat)
        self.conda_env_edit.setVisible(is_gpustat)

    def _current_mode(self) -> str:
        idx = self._mode_group.checkedId()
        if idx == 0:
            return "nvidia-smi"
        elif idx == 1:
            return "gpustat"
        return "http"

    # ============================================================
    # 刷新
    # ============================================================

    def _on_refresh(self) -> None:
        mode = self._current_mode()

        if mode in ("nvidia-smi", "gpustat"):
            host = self.host_edit.text().strip()
            if not host:
                error("错误", "请填写 Host 地址", self)
                return
            username = self.user_edit.text().strip()
            if not username:
                error("错误", "请填写用户名", self)
                return
            password = self.pwd_edit.text()
            if password:
                self._session_password = password
            key_path = self.key_edit.text().strip()
            if not key_path and not self._session_password:
                error("错误", "请填写密码或指定密钥路径", self)
                return

            if mode == "nvidia-smi":
                gpu_cmd = self.nvsmi_cmd_edit.text().strip()
                proc_cmd = (
                    "nvidia-smi --query-compute-apps=pid,gpu_index,process_name,"
                    "used_gpu_memory --format=csv,noheader,nounits 2>/dev/null"
                )
                self._worker = NvidiaSmiWorker(
                    host, self.port_spin.value(), username,
                    self._session_password, key_path,
                    gpu_cmd, proc_cmd,
                )
            else:  # gpustat
                gpustat_cmd = self.gpustat_cmd_edit.text().strip() or "gpustat --json"
                conda_env = self.conda_env_edit.text().strip()
                conda_path = self.conda_path_edit.text().strip()
                self._worker = GpustatWorker(
                    host, self.port_spin.value(), username,
                    self._session_password, key_path, gpustat_cmd, conda_path, conda_env,
                )

        else:  # http
            api_url = self.url_edit.text().strip()
            if not api_url:
                error("错误", "请填写 API 地址", self)
                return
            self._worker = HttpWorker(api_url, self.token_edit.text().strip())

        self.refresh_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("正在连接...")

        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, gpus: list[GPUInfo]) -> None:
        self.refresh_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(f"共 {len(gpus)} 张 GPU")
        self._build_cards(gpus)

    def _on_error(self, err_msg: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("❌ 连接失败")
        error("GPU 监控出错", err_msg, self)

    # ============================================================
    # 卡片构建
    # ============================================================

    def _build_cards(self, gpus: list[GPUInfo]) -> None:
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for gpu in gpus:
            card = CardWidget()
            cl = QVBoxLayout(card)
            cl.setSpacing(6)

            # 标题行
            title_row = QHBoxLayout()
            title_row.addWidget(BodyLabel(f"GPU {gpu.index}: {gpu.name}"))
            title_row.addStretch()
            title_row.addWidget(BodyLabel(f"利用率 {gpu.utilization}%"))
            cl.addLayout(title_row)

            # 利用率进度条
            ub = QProgressBar()
            ub.setRange(0, 100)
            ub.setValue(gpu.utilization)
            ub.setTextVisible(True)
            ub.setFormat(f"{gpu.utilization}%")
            ub.setMaximumHeight(22)
            cl.addWidget(ub)

            # 显存
            cl.addWidget(BodyLabel(
                f"显存: {gpu.memory_used} / {gpu.memory_total} MB  ({gpu.memory_percent}%)"
            ))
            mb = QProgressBar()
            mb.setRange(0, 100)
            mb.setValue(int(gpu.memory_percent))
            mb.setTextVisible(True)
            mb.setFormat(f"{gpu.memory_percent}%")
            mb.setMaximumHeight(22)
            cl.addWidget(mb)

            # 进程
            if gpu.processes:
                proc_text = "\n".join(
                    f"  PID {p['pid']}  {p['name']}  ({p['memory']} MB)"
                    for p in gpu.processes[:10]
                )
                pl = QLabel(proc_text)
                pl.setStyleSheet("color: #888; font-size: 12px;")
                cl.addWidget(pl)
            else:
                cl.addWidget(BodyLabel("  (无运行进程)"))

            self.cards_layout.addWidget(card)

    # ============================================================
    # 持久化
    # ============================================================

    def _load_settings(self) -> None:
        # SSH 通用
        self.host_edit.setText(get_str("gpu_host", "服务器地址"))
        self.port_spin.setValue(get_int("gpu_port", 22))
        self.user_edit.setText(get_str("gpu_username", "root"))
        self.key_edit.setText(get_str("gpu_key_path", ""))
        self.pwd_edit.setText(self._session_password)

        # nvidia-smi
        self.nvsmi_cmd_edit.setText(get_str("gpu_nvsmi_cmd", ""))

        # gpustat
        self.gpustat_cmd_edit.setText(get_str("gpu_gpustat_cmd", ""))
        self.conda_path_edit.setText(get_str("gpu_conda_path", ""))
        self.conda_env_edit.setText(get_str("gpu_conda_env", ""))

        # HTTP
        self.url_edit.setText(get_str("gpu_api_url", ""))
        self.token_edit.setText(get_str("gpu_token", ""))

        # 模式
        mode = get_str("gpu_mode", "nvidia-smi")
        if mode == "gpustat":
            self.gpustat_radio.setChecked(True)
        elif mode == "http":
            self.http_radio.setChecked(True)
        else:
            self.smi_radio.setChecked(True)
        self._on_mode_changed()
