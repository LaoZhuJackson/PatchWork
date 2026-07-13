"""QSettings 配置持久化封装"""
from __future__ import annotations
from typing import Any
from PySide6.QtCore import QSettings

SETTINGS = QSettings("PatchWork", "PatchWork")


def _get(key: str, default: Any = "") -> Any:
    return SETTINGS.value(key, default)


def get_str(key: str, default: str = "") -> str:
    return str(_get(key, default))


def set_str(key: str, value: str) -> None:
    SETTINGS.setValue(key, value)


def get_int(key: str, default: int = 0) -> int:
    return int(_get(key, default))


def set_int(key: str, value: int) -> None:
    SETTINGS.setValue(key, value)


def get_bool(key: str, default: bool = False) -> bool:
    val = _get(key, default)
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


def set_bool(key: str, value: bool) -> None:
    SETTINGS.setValue(key, value)


def get_float(key: str, default: float = 0.0) -> float:
    return float(_get(key, default))


def set_float(key: str, value: float) -> None:
    SETTINGS.setValue(key, value)
