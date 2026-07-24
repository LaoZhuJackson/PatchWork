"""PatchWork 入口"""
from __future__ import annotations

import sys
import types
from pathlib import Path

from PySide6.QtWidgets import QApplication
from qfluentwidgets import setTheme, Theme

# ---- albumentations 1.x → 2.x 兼容 ----
# 2.0+ 把 transforms 从 albumentations.augmentations.transforms 移走了，
# 但旧模型 checkpoint 里 pickle 的类路径还在旧位置，torch.load 时会找不到。
# 这里注入一个 shim 模块，把新位置的类映射回旧路径。
def _setup_albumentations_shim() -> None:
    _OLD = "albumentations.augmentations.transforms"
    if _OLD in sys.modules:
        return
    try:
        __import__(_OLD)
    except ImportError:
        import albumentations as A
        # 注册父模块
        _parent = "albumentations.augmentations"
        if _parent not in sys.modules:
            sys.modules[_parent] = types.ModuleType(_parent)
        # 创建 shim：从新版导入路径收集所有类名，挂到旧路径
        _shim = types.ModuleType(_OLD)
        _shim.__all__ = []
        for _src in (A, A.core):
            for _name in dir(_src):
                if _name and _name[0].isupper():
                    setattr(_shim, _name, getattr(_src, _name))
                    _shim.__all__.append(_name)
        sys.modules[_OLD] = _shim

_setup_albumentations_shim()

from app.main_window import MainWindow
from app.utils.config import get_str
from app.utils.crash_handler import setup_crash_handler
from app.utils.logger import setup_logging
from app.utils.message import error, warning

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    setup_crash_handler()

    # ---- ultralytics 下载目录统一到项目 models/ ----
    _models_dir = PROJECT_ROOT / "models"
    _models_dir.mkdir(exist_ok=True)
    from ultralytics import settings as ultralytics_settings
    ultralytics_settings.update(weights_dir=str(_models_dir))

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

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
