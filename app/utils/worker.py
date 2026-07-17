"""后台 Worker 基类"""
from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

class Worker(QThread):
    """后台线程基类，耗时操作放在 run() 里，结果通过信号传递"""
    finished = Signal(object) # 结果数据
    error = Signal(str) # 错误信息
    progress = Signal(int) # 进度 0-100

    def __init__(self, parent=None):
        super().__init__(parent)

    def do_work(self):
        """子类必须重写此方法，返回结果数据"""
        raise NotImplementedError

    def run(self) -> None:
        """子类重写此方法，不要直接调用——用 self.start() 启动"""
        try:
            result = self.do_work()
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())
            