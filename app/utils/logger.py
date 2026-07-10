"""日志系统：文件记录 + WARNING 以上弹窗提醒"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal


# ---- Qt 信号桥接 ----

class LogSignal(QObject):
    """logging → Qt GUI 的桥梁"""
    message = Signal(str, str)  # level, text


class QtLogHandler(logging.Handler):
    """将日志通过 Qt 信号发出，供主窗口弹窗"""

    def __init__(self, level: int = logging.WARNING) -> None:
        super().__init__(level)
        self.signal = LogSignal()

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.signal.message.emit(record.levelname, msg)


# ---- 全局 logger ----
_logger: logging.Logger | None = None
_qt_handler: QtLogHandler | None = None


def _get_app_root() -> Path:
    """应用根目录：开发时=项目根，打包后=exe 同级"""
    if getattr(sys, "frozen", False):
        # PyInstaller 打包后运行
        return Path(sys.executable).parent
    # 开发环境：从 app/utils/logger.py 往上三层到项目根
    return Path(__file__).resolve().parent.parent.parent


def setup_logging(log_dir: Path | None = None) -> QtLogHandler:
    """初始化日志系统，返回 QtLogHandler 供主窗口连接弹窗。

    - 文件日志: <log_dir>/patchwork_YYYY-MM-DD.log (保留所有级别)
    - Qt 信号: WARNING 以上触发弹窗
    - 控制台: INFO 以上输出
    """
    global _logger, _qt_handler

    if _logger is not None and _qt_handler is not None:
        return _qt_handler  # 已经初始化过，直接返回
    if log_dir is None:
        log_dir = _get_app_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    _logger = logging.getLogger("patchwork")
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件 handler（全量）
    filename = f"patchwork_{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)

    # 控制台 handler（开发用）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    _logger.addHandler(console_handler)

    # Qt handler(弹窗用)
    _qt_handler = QtLogHandler(level=logging.WARNING)
    _qt_handler.setFormatter(fmt)
    _logger.addHandler(_qt_handler)

    _logger.info(f"日志系统已就绪，日志目录: {log_dir}")
    return _qt_handler


def get_logger(name: str | None = None) -> logging.Logger:
    """获取 logger 实例"""
    if _logger is None:
        setup_logging()
    return logging.getLogger(f"patchwork.{name}" if name else "patchwork")
