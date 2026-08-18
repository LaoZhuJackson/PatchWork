"""Callback 回调捕获服务：本地接收算法平台/实例 HTTP 回调并记录，可设监听路由。

独立运行：
    python -m app.services.callback_dump --port 9999 --route /cb --dump cb.log
GUI 集成（app.widgets.callback_dump）：
    dumper = CallbackDumper(on_record=cb)
    ok, msg = dumper.start(port=9999, route="/cb")   # cb 在监听线程被调用，注意跨线程
    dumper.stop()
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

_RECORD = dict  # {index,ts,ip,method,path,query,headers,body,pretty}


class _Handler(BaseHTTPRequestHandler):
    dumper: "CallbackDumper | None" = None

    def _handle(self) -> None:
        dumper = self.dumper
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length).decode("utf-8", "replace")  # 当编码错误时使用 � 替换无效字节
        split = urlsplit(self.path)
        if not dumper.route_match(split.path):
            self.send_error(404, "route not matched")
            return
        rec = dumper._on_received(
            ip=self.client_address[0], method=self.command,
            path=split.path, query=split.query,
            headers=dict(self.headers), body=body,
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(f'{{"code":200,"msg":"received #{rec["index"]}"}}'.encode("utf-8"))

    do_POST = _handle
    do_GET = _handle
    do_PUT = _handle

    def log_message(self, *args) -> None:
        pass


class CallbackDumper:
    """宿主侧 HTTP 回调监听器（自带线程，可随时 start/stop/改路由）。"""

    def __init__(self, dump_path: str | Path | None = None, on_record=None) -> None:
        self._dump_path = Path(dump_path) if dump_path else None
        self._on_record = on_record
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._host = ""
        self._port: int | None = None
        self._route = ""
        self._records: list[_RECORD] = []
        self._lock = threading.Lock()

    # ---- 状态 ----
    @property
    def is_running(self) -> bool:
        return self._server is not None

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def route(self) -> str:
        return self._route

    @property
    def records(self) -> list[_RECORD]:
        with self._lock:
            return list(self._records)

    # ---- 控制 ----
    def start(self, port: int = 9999, host: str = "0.0.0.0", route: str = "") -> tuple[bool, str]:
        if self.is_running:
            return False, f"已在监听 {self._host}:{self._port} route={self._route}"

        try:
            server = HTTPServer((host, port), _Handler)
        except OSError as e:
            return False, f"启动失败（端口 {port} 被占/被防火墙拒绝）: {e}"
        _Handler.dumper = server.dumper = self  # noqa
        self._server = server
        self._host = host
        self._port = port
        self._route = route.lstrip() or ""
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)  # 守护线程：主线程结束后它会自动结束
        self._thread.start()
        logger.info("callback dumper listening on %s:%d route=%r", host, port, self._route)
        return True, f"监听中 {host}:{port} route={self._route or '(全部)'}"

    def stop(self) -> None:
        if self._server is None:
            return
        server = self._server
        self._server = None
        server.shutdown()
        server.server_close()
        self._thread = None
        self._port = None
        logger.info("callback dumper stopped")

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def route_match(self, path:str) -> bool:
        """路由匹配：空路由接受全部；否则 path 等于 route 或以其前缀开头。"""
        if not self._route:
            return True
        r = self._route.rstrip("/") + "/"
        return path == self._route or path.startswith(r)

    # ---- 内部 ----
    def _on_received(self, ip:str, method:str, path:str, query:str, headers:dict, body: str) -> _RECORD:
        try:
            pretty = json.dumps(json.loads(body), ensure_ascii=False, indent=2)
        except Exception:
            pretty = body
        rec: _RECORD = {
            "index": len(self._records) + 1,
            "ts": datetime.now().strftime("%H:%M:%S"),
            "ip": ip,
            "method": method,
            "path": path,
            "query": query,
            "headers": json.dumps(headers, ensure_ascii=False, indent=2),
            "body": body,
            "pretty": pretty,
        }
        if self._dump_path:
            self._dump_path.parent.mkdir(parents=True, exist_ok=True)
            with self._dump_path.open("a", encoding="utf-8") as f:
                f.write(f"\n==== {rec['ts']} {method} {path}?{query} from {ip} ====\n{pretty}\n")
        with self._lock:
            self._records.append(rec)
        logger.info("[cb] #%d %s %s%s len=%d", rec["index"], method, path, query, len(body))
        if self._on_record:
            try:
                self._on_record(rec)
            except Exception as e:
                logger.warning("on_record 回调异常: %s", e)
        return rec

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="本地回调捕获服务")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--route", default="", help="监听路由，如 /cb，留空接受全部")
    ap.add_argument("--dump", default="", help="落盘文件路径")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dumper = CallbackDumper(dump_path=args.dump or None)
    ok, msg = dumper.start(args.port, route=args.route)
    print(msg)
    if not ok:
        raise SystemExit(1)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        dumper.stop()
