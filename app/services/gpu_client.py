"""GPU 监控：nvidia-smi / gpustat / HTTP 三种获取方式"""
from __future__ import annotations

import json
import ssl
from dataclasses import dataclass, field
from urllib.request import Request, urlopen
from urllib.error import URLError

import paramiko


@dataclass
class GPUInfo:
    index: int
    name: str
    utilization: int         # 0-100
    memory_used: int         # MB
    memory_total: int        # MB
    processes: list[dict] = field(default_factory=list)

    @property
    def memory_percent(self) -> float:
        if self.memory_total > 0:
            return round(self.memory_used / self.memory_total * 100, 1)
        return 0.0


# ============================================================
# 解析器
# ============================================================

def _parse_nvidia_smi_csv(raw: str) -> list[GPUInfo]:
    """解析 nvidia-smi CSV 输出"""
    gpus: dict[int, GPUInfo] = {}
    for line in raw.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
        idx = int(parts[0])
        gpus[idx] = GPUInfo(
            index=idx,
            name=parts[1],
            utilization=int(parts[2]),
            memory_used=int(parts[3]),
            memory_total=int(parts[4]),
        )
    return list(gpus.values())


def _parse_nvidia_smi_processes(raw: str, gpus: dict[int, GPUInfo]) -> None:
    """解析 nvidia-smi 进程 CSV 输出，填充到已有 gpus 中"""
    for line in raw.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_idx = int(parts[1])
        if gpu_idx in gpus:
            gpus[gpu_idx].processes.append({
                "pid": parts[0],
                "name": parts[2],
                "memory": parts[3],
            })


def _parse_gpustat_json(data: dict) -> list[GPUInfo]:
    """解析 gpustat --json 输出"""
    gpus: list[GPUInfo] = []
    for raw in data.get("gpus", []):
        gpus.append(GPUInfo(
            index=raw.get("index", 0),
            name=raw.get("name", "?"),
            utilization=int(raw.get("utilization.gpu", 0)),
            memory_used=int(raw.get("memory.used", 0)),
            memory_total=int(raw.get("memory.total", 0)),
            processes=[
                {
                    "pid": str(p.get("pid", "")),
                    "name": p.get("command", p.get("full_command", "?")),
                    "memory": str(p.get("gpu_memory_usage", 0)),
                }
                for p in raw.get("processes", [])
            ],
        ))
    return gpus


# ============================================================
# SSH 连接辅助
# ============================================================

def _ssh_exec(
    host: str,
    port: int,
    username: str,
    password: str,
    key_path: str,
    commands: list[str],
    timeout: int = 10,
) -> list[str]:
    """SSH 连接服务器，顺序执行多条命令，返回每条的 stdout 内容。

    Raises:
        paramiko.AuthenticationException
        paramiko.SSHException
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        kwargs: dict = {
            "hostname": host, "port": port,
            "username": username, "timeout": timeout,
        }
        if key_path:
            kwargs["key_filename"] = key_path
        else:
            kwargs["password"] = password

        client.connect(**kwargs)

        results: list[str] = []
        for cmd in commands:
            _, stdout, stderr = client.exec_command(cmd)
            out = stdout.read().decode("utf-8", errors="replace").strip()
            err = stderr.read().decode("utf-8", errors="replace").strip()
            if err and not out:
                raise RuntimeError(f"命令执行失败: {cmd}\n{err}")
            results.append(out)

        return results

    finally:
        client.close()


# ============================================================
# 三种获取方式
# ============================================================

def fetch_via_nvidia_smi(
    host: str,
    port: int = 22,
    username: str = "",
    password: str = "",
    key_path: str = "",
    gpu_cmd: str = (
        "nvidia-smi --query-gpu=index,name,utilization.gpu,"
        "memory.used,memory.total --format=csv,noheader,nounits"
    ),
    proc_cmd: str = (
        "nvidia-smi --query-compute-apps=pid,gpu_index,process_name,"
        "used_gpu_memory --format=csv,noheader,nounits 2>/dev/null"
    ),
    timeout: int = 10,
) -> list[GPUInfo]:
    """SSH + nvidia-smi CSV 解析。

    Raises:
        paramiko.AuthenticationException
        RuntimeError: nvidia-smi 不可用
    """
    results = _ssh_exec(
        host, port, username, password, key_path,
        [gpu_cmd, proc_cmd], timeout,
    )

    gpus_dict: dict[int, GPUInfo] = {}
    for g in _parse_nvidia_smi_csv(results[0]):
        gpus_dict[g.index] = g
    _parse_nvidia_smi_processes(results[1], gpus_dict)

    return list(gpus_dict.values())


def fetch_via_gpustat(
    host: str,
    port: int = 22,
    username: str = "",
    password: str = "",
    key_path: str = "",
    gpustat_cmd: str = "gpustat --json",
    conda_path: str = "",
    conda_env: str = "",
    timeout: int = 10,
) -> list[GPUInfo]:
    """SSH + gpustat --json 解析。

    Args:
        conda_env: 可选，conda 环境名。不为空时自动拼接激活命令。
    """
    # 拼接 conda 激活前缀
    if conda_path and conda_env:
        if "/" in conda_env or "\\" in conda_env:
            # 完整路径 → --prefix
            cmd = f"{conda_path} run --prefix {conda_env} {gpustat_cmd}"
        else:
            # 环境名 → -n
            cmd = f"{conda_path} run -n {conda_env} {gpustat_cmd}"
    elif conda_path:
        cmd = f"{conda_path} run {gpustat_cmd}"
    else:
        cmd = gpustat_cmd

    results = _ssh_exec(
        host, port, username, password, key_path,
        [cmd], timeout,
    )

    try:
        data = json.loads(results[0])
    except json.JSONDecodeError:
        # 可能 conda 路径不对，尝试直接跑
        results = _ssh_exec(
            host, port, username, password, key_path,
            [gpustat_cmd],
            timeout,
        )
        data = json.loads(results[0])

    return _parse_gpustat_json(data)


def fetch_via_http(
    api_url: str,
    token: str = "",
    timeout: int = 10,
) -> list[GPUInfo]:
    """HTTP API 获取 GPU 状态。

    Raises:
        URLError: 网络不通
        RuntimeError: API 返回异常
    """
    req = Request(api_url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        resp = urlopen(req, timeout=timeout, context=ctx)
    except URLError as e:
        raise URLError(f"无法连接到 {api_url}: {e.reason}")

    body = json.loads(resp.read().decode("utf-8"))

    if not body.get("ok"):
        raise RuntimeError(body.get("error", "未知错误"))

    return _parse_gpustat_json(body.get("data", {}))
