"""PatchWork 入口"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication
from qfluentwidgets import setTheme, Theme

from app.main_window import MainWindow
from app.utils.config import get_str
from app.utils.crash_handler import setup_crash_handler
from app.utils.logger import setup_logging
from app.utils.message import error, warning


def main() -> None:
    setup_crash_handler()

    app = QApplication(sys.argv)
    app.setApplicationName("PatchWork")

    # 恢复主题
    saved = get_str("app_theme", "light")
    setTheme(Theme.DARK if saved == "dark" else Theme.LIGHT)

    # 初始化日志
    qt_handler = setup_logging()
    # 保存引用防止被GC回收
    app.setProperty("log_signal", qt_handler.signal)

    window = MainWindow()

    # WARNING 以上弹窗提醒
    def show_log_popup(level: str, msg: str) -> None:
        if level == "CRITICAL":
            error("严重错误", msg, window)
        elif level == "ERROR":
            error("错误", msg, window)
        elif level == "WARNING":
            warning("警告", msg, window)

    qt_handler.signal.message.connect(show_log_popup)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
