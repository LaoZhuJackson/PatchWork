"""X-AnyLabeling 启动器"""
from __future__ import annotations

import subprocess
from pathlib import Path


def launch(executable: str | Path, folder: str | Path) -> subprocess.Popen:
    """用 X-AnyLabeling 打开指定文件夹。

    Args:
      executable: X-AnyLabeling 的 exe 路径
      folder: 要加载的数据集文件夹

    Returns:
      subprocess.Popen 对象

    Raises:
      FileNotFoundError: exe 不存在
      OSError: 启动失败
    """
    exe = Path(executable)
    if not exe.is_file():
        raise FileNotFoundError(f"找不到 X-AnyLabeling: {executable}")

    return subprocess.Popen(
        [str(exe), str(folder)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
