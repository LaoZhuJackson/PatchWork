"""GPU 监控：SSH + nvidia-smi 解析"""
from __future__ import annotations

from dataclasses import dataclass, field

import paramiko

@dataclass
class GPUInfo:
    index:int
    name: str
    utilization: int
    memory_used: int
    memory_total: int
    processes: list[dict] = field(default_factory=list)

    @property
    def memory_percent(self) -> float:
        if self.memory_total > 0:
            return round(self.memory_used / self.memory_total * 100, 1)
        return 0.0

def fetch_gpu_info(host: str, port: int = 22, username: str = "", password: str = "", key_path: str = "") -> list[GPUInfo]:
    """SSH 连接远程服务器，解析 nvidia-smi 输出。

    Raises:
      paramiko.AuthenticationException: 认证失败
      paramiko.SSHException: 连接失败
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # 连接
        connect_kwargs: dict = {
            "hostname": host,
            "port": port,
            "username": username,
            "timeout": 10,
        }
        if key_path:
            connect_kwargs["key_filename"] = key_path
        else:
            connect_kwargs["password"] = password

        client.connect(**connect_kwargs)

        # 查询GPU信息
        gpu_cmd = (
            "nvidia-smi --query-gpu=index,name,utilization.gpu,"
            "memory.used,memory.total --format=csv,noheader,nounits"
        )
        _, stdout, stderr = client.exec_command(gpu_cmd)
        err = stderr.read().decode().strip()
        if err and "error" in err.lower():
            raise RuntimeError(f"nvidia-smi 执行失败: {err}")

        gpus: dict[int, GPUInfo] = {}
        for line in stdout.read().decode().strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 5:
                continue
            idx = int(parts[0].strip())
            gpus[idx] = GPUInfo(
                index=idx,
                name=parts[1].strip(),
                utilization=int(parts[2].strip()),
                memory_used=int(parts[3].strip()),
                memory_total=int(parts[4].strip()),
            )

        # 查询进程信息
        proc_cmd = (
            "nvidia-smi --query-compute-apps=pid,gpu_index,process_name,"
            "used_gpu_memory --format=csv,noheader,nounits 2>/dev/null"
        )
        _, stdout2, _ = client.exec_command(proc_cmd)
        for line in stdout2.read().decode().strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 4:
                continue
            gpu_idx = int(parts[1].strip())
            if gpu_idx in gpus:
                gpus[gpu_idx].processes.append({
                    "pid": parts[0].strip(),
                    "name": parts[2].strip(),
                    "memory": parts[3].strip(),
                })
    finally:
        client.close()
    return list(gpus.values())

