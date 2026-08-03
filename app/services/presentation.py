"""PPT 汇报生成服务。

PatchWork 不实现 Markdown/Marp 生成逻辑，只调用外部工作区中的：
    <workspace>/utils/make_slides.py

这样 Claude、命令行和 PatchWork 共用同一套生成核心。
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.utils.logger import get_logger

logger = get_logger(__name__)


class PresentationServiceError(RuntimeError):
    """汇报服务执行失败。"""

    def __init__(self, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {}


@dataclass(slots=True)
class PresentationRequest:
    """一次 PPT 工具调用的统一参数。"""

    workspace: Path
    profile: str = ""
    theme: str = ""
    deck: Path | None = None
    output_dir: Path | None = None
    output_name: str = ""
    formats: tuple[str, ...] = ("pptx",)
    projects: tuple[str, ...] = ()
    context_output: Path | None = None
    python_executable: str = ""
    timeout: int = 600

    def normalized(self) -> "PresentationRequest":
        formats = tuple(
            item.lower()
            for item in self.formats
            if item.lower() in {"pptx", "pdf", "html"}
        )
        return PresentationRequest(
            workspace=self.workspace.expanduser().resolve(),
            profile=self.profile.strip(),
            theme=self.theme.strip(),
            deck=self.deck.expanduser().resolve() if self.deck else None,
            output_dir=self.output_dir.expanduser().resolve() if self.output_dir else None,
            output_name=self.output_name.strip(),
            formats=formats or ("pptx",),
            projects=tuple(item.strip() for item in self.projects if item.strip()),
            context_output=(
                self.context_output.expanduser().resolve()
                if self.context_output
                else None
            ),
            python_executable=self.python_executable.strip(),
            timeout=max(30, int(self.timeout)),
        )


@dataclass(slots=True)
class PresentationResult:
    """make_slides.py 返回结果。"""

    operation: str
    success: bool
    payload: dict[str, Any]
    command: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""

    @property
    def outputs(self) -> dict[str, str]:
        raw = self.payload.get("outputs", {})
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): str(value)
            for key, value in raw.items()
            if value not in (None, "")
        }

    @property
    def primary_output(self) -> Path | None:
        for key in ("pptx", "pdf", "html", "markdown", "manifest"):
            value = self.outputs.get(key)
            if value:
                return Path(value)
        return None


class PresentationService:
    """调用工作区 PPT CLI 的无 UI 服务。"""

    def validate(self, request: PresentationRequest) -> PresentationResult:
        req = request.normalized()
        self._validate_workspace(req.workspace)

        args = ["validate", "--workspace", str(req.workspace)]
        if req.profile:
            args += ["--profile", req.profile]
        if req.theme:
            args += ["--theme", req.theme]

        return self._run("validate", req, args)

    def inspect(self, request: PresentationRequest) -> PresentationResult:
        req = request.normalized()
        self._validate_workspace(req.workspace)

        output = req.context_output
        if output is None:
            profile_name = self._safe_filename(req.profile or "技术分享")
            output = req.workspace / ".slides" / "runtime" / f"{profile_name}_context.json"

        args = [
            "inspect",
            "--workspace",
            str(req.workspace),
            "--output",
            str(output),
        ]
        if req.profile:
            args += ["--profile", req.profile]
        if req.theme:
            args += ["--theme", req.theme]
        if req.projects:
            args += ["--projects", *req.projects]

        return self._run("inspect", req, args)

    def build(self, request: PresentationRequest) -> PresentationResult:
        req = request.normalized()
        self._validate_workspace(req.workspace)

        if req.deck is None:
            raise PresentationServiceError("生成汇报前必须选择 deck.yaml。")
        if not req.deck.is_file():
            raise PresentationServiceError(f"deck 文件不存在：{req.deck}")

        args = [
            "build",
            "--workspace",
            str(req.workspace),
            "--deck",
            str(req.deck),
        ]
        if req.profile:
            args += ["--profile", req.profile]
        if req.theme:
            args += ["--theme", req.theme]
        if req.output_name:
            args += ["--name", req.output_name]
        if req.output_dir:
            args += ["--output-dir", str(req.output_dir)]

        for fmt in req.formats:
            args.append(f"--{fmt}")

        return self._run("build", req, args)

    @staticmethod
    def discover_themes(workspace: str | Path) -> list[dict[str, str]]:
        """扫描 .slides/themes/*.css，提取 @theme 名称。

        Returns:
            [{"name": "workspace-default", "file": "default.css"}, ...]
        """
        themes_dir = Path(workspace).expanduser() / ".slides" / "themes"
        if not themes_dir.is_dir():
            return []

        results: list[dict[str, str]] = []
        for path in sorted(themes_dir.glob("*.css"), key=lambda p: p.name.casefold()):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # 提取 /* @theme xxx */ 注释中的主题名
            import re
            match = re.search(r"/\*\s*@theme\s+(\S+)\s*\*/", text)
            if match:
                results.append({"name": match.group(1), "file": path.name})
            else:
                # 回退：用文件名（去掉 .css）作为主题名
                results.append({"name": path.stem, "file": path.name})
        return results

    @staticmethod
    def discover_profiles(workspace: str | Path) -> list[str]:
        profiles_dir = Path(workspace).expanduser() / ".slides" / "profiles"
        if not profiles_dir.is_dir():
            return []
        return sorted(
            {
                path.stem
                for path in profiles_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
            },
            key=str.casefold,
        )

    @staticmethod
    def discover_decks(workspace: str | Path) -> list[Path]:
        """查找常见 deck 文件，最近修改的排在最前。"""

        base = Path(workspace).expanduser()
        candidate_dirs = [
            base / ".slides" / "runtime",
            base / ".slides" / "examples",
            base / ".slides" / "example",
            base / "汇报记录",
        ]

        found: dict[Path, float] = {}
        for directory in candidate_dirs:
            if not directory.is_dir():
                continue
            for pattern in ("*.yaml", "*.yml"):
                for path in directory.glob(pattern):
                    name = path.name.casefold()
                    # profiles/config 不是 deck；候选目录中也尽量只保留 deck 命名文件。
                    if "deck" not in name and directory.name not in {"example", "examples"}:
                        continue
                    resolved = path.resolve()
                    try:
                        found[resolved] = resolved.stat().st_mtime
                    except OSError:
                        found[resolved] = 0.0

        return [
            path
            for path, _ in sorted(
                found.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

    @staticmethod
    def is_workspace_ready(workspace: str | Path) -> tuple[bool, str]:
        base = Path(workspace).expanduser()
        if not base.is_dir():
            return False, "工作区目录不存在"
        if not (base / "utils" / "make_slides.py").is_file():
            return False, "缺少 utils/make_slides.py"
        if not (base / ".slides" / "config.yaml").is_file():
            return False, "缺少 .slides/config.yaml"
        return True, ""

    def _run(
        self,
        operation: str,
        request: PresentationRequest,
        cli_args: Sequence[str],
    ) -> PresentationResult:
        python_command = self._resolve_python_command(request.python_executable)
        script = request.workspace / "utils" / "make_slides.py"
        command = [*python_command, str(script), *cli_args]

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        logger.info("运行 PPT 命令: %s", subprocess.list2cmdline(command))

        try:
            completed = subprocess.run(
                command,
                cwd=str(request.workspace),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=request.timeout,
                creationflags=creationflags,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PresentationServiceError(
                f"PPT 操作超时（{request.timeout} 秒）。"
            ) from exc
        except OSError as exc:
            raise PresentationServiceError(f"无法启动 PPT 工具：{exc}") from exc

        payload = self._parse_payload(completed.stdout)
        success = bool(payload.get("success")) and completed.returncode == 0

        result = PresentationResult(
            operation=operation,
            success=success,
            payload=payload,
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

        if not success:
            message = str(payload.get("error") or "").strip()
            if not message:
                message = completed.stderr.strip() or completed.stdout.strip()
            if not message:
                message = f"PPT 工具退出码：{completed.returncode}"
            raise PresentationServiceError(message, payload=payload)

        return result

    @staticmethod
    def _validate_workspace(workspace: Path) -> None:
        ready, reason = PresentationService.is_workspace_ready(workspace)
        if not ready:
            raise PresentationServiceError(f"工作区尚未准备好：{reason}")

    @staticmethod
    def _resolve_python_command(explicit: str) -> list[str]:
        """找到可用于运行工作区脚本的 Python。

        开发模式优先使用当前解释器。PyInstaller 打包后 sys.executable
        指向 PatchWork.exe，因此改为查找系统 python / py。
        """

        if explicit:
            parts = shlex.split(explicit, posix=os.name != "nt")
            if not parts:
                raise PresentationServiceError("Python 解释器配置为空。")
            executable = Path(parts[0]).expanduser()
            if executable.is_file():
                return [str(executable.resolve()), *parts[1:]]
            resolved = shutil.which(parts[0])
            if resolved:
                return [resolved, *parts[1:]]
            raise PresentationServiceError(f"找不到 Python 解释器：{parts[0]}")

        if not getattr(sys, "frozen", False):
            return [sys.executable]

        python = shutil.which("python")
        if python:
            return [python]

        py_launcher = shutil.which("py")
        if py_launcher:
            return [py_launcher, "-3"]

        raise PresentationServiceError(
            "打包版 PatchWork 找不到 Python。请设置 Python 解释器路径。"
        )

    @staticmethod
    def _parse_payload(stdout: str) -> dict[str, Any]:
        text = stdout.lstrip("\ufeff").strip()
        if not text:
            return {}

        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {"result": payload}
        except json.JSONDecodeError:
            # 兼容 CLI 前后混入少量日志的情况。
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    payload = json.loads(text[start : end + 1])
                    return payload if isinstance(payload, dict) else {"result": payload}
                except json.JSONDecodeError:
                    pass
        return {"success": False, "error": f"无法解析 PPT 工具输出：\n{text}"}

    @staticmethod
    def _safe_filename(value: str) -> str:
        invalid = '<>:"/\\|?*'
        result = "".join("_" if char in invalid else char for char in value)
        return result.strip().strip(".") or "slides"
