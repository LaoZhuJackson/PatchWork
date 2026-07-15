"""全局异常捕获：Python 异常 + Qt 消息 + segfault 堆栈"""
from __future__ import annotations

import faulthandler
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import qInstallMessageHandler, QtMsgType

from app.utils.logger import get_logger, get_app_root

logger = get_logger("crash")

def _excepthook(exc_type, exc_value, exc_tb):
    """捕获未处理的 Python 异常"""
    if exc_type is KeyboardInterrupt:
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.critical(f"未捕获的异常:\n{tb_str}")

def _qt_message_handler(msg_type, context, message):
    """捕获 Qt 内部的调试/警告/错误消息"""
    level_map = {
        QtMsgType.QtDebugMsg: "DEBUG",
        QtMsgType.QtInfoMsg: "INFO",
        QtMsgType.QtWarningMsg: "WARNING",
        QtMsgType.QtCriticalMsg: "CRITICAL",
        QtMsgType.QtFatalMsg: "FATAL",
    }
    level = level_map.get(msg_type, "UNKNOWN")

    # 过滤掉无关紧要的警告
    if msg_type == QtMsgType.QtWarningMsg:
        msg_lower = message.lower()
        skip_keywords = [
            "libpng warning", "iccp",
            "known incorrect srgb",
            "qt.svg.link",
        ]
        if any(k in msg_lower for k in skip_keywords):
            return

    logger.warning(f"Qt {level}: {message}")

def setup_crash_handler(log_dir: Path | None = None) -> None:
    """安装全局异常捕获，所有错误写入日志文件"""
    # 1. Python 未捕获异常
    sys.excepthook = _excepthook
    # 2. Qt 内部消息
    qInstallMessageHandler(_qt_message_handler)
    # 3. segfault 时输出 Python 调用栈
    if log_dir is None:
        log_dir = get_app_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    faulthandler.enable(
        file=open(log_dir / "crash_dump.log", "a", encoding="utf-8"),
        all_threads=True,
    )

    logger.info("异常捕获已安装")