"""Loopback-only broker process, independent of individual Codex MCP sessions."""
import argparse
import json
import msvcrt
import os
from pathlib import Path
import secrets
import threading
import psutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from broker import Broker, LIVE
from manager_control import verify_continuation, DISABLED_MESSAGE

ROOT = Path(__file__).resolve().parent
VERSION = "1.2.0"
READ_METHODS = {"status", "history", "learning_status", "health_report", "app_memory_status", "control_status"}
WRITE_METHODS = {"request_model", "lease_status", "release", "release_idle", "attach_worker",
                 "prepare_work", "complete_work", "feedback", "rollback_policy", "register_model"}


def dispatch(broker, method, params, shutdown=None):
    """Serialize shutdown with mutations so no lease can enter after its check."""
    if not isinstance(params, dict):
        raise ValueError("RPC params must be an object")
    if method == "health":
        return {"service": "local-model-manager", "version": VERSION, "pid": os.getpid(),
                "stopping": broker.stop_event.is_set(),
                "control": broker.control_status() if hasattr(broker, 'control_status') else None}
    if method == "shutdown":
        with broker.lock:
            if (broker.lifecycle_action or
                    any(x["state"] in LIVE or x["state"] == "queued" for x in broker.state["leases"]) or
                    any(x["state"] in {"queued", "running"} for x in broker.state["cleanup"])):
                raise ValueError("Pending or active work exists; release/cancel it before shutdown")
            broker.stop_event.set()
            if hasattr(broker, 'monitor_wake'):
                broker.monitor_wake.set()
        if shutdown:
            threading.Thread(target=shutdown, daemon=True).start()
        return {"state": "stopping", "models": "retained"}
    if method in READ_METHODS:
        return getattr(broker, method)(**params)
    if method in WRITE_METHODS:
        # prepare/register may refresh. Match the monitor's lock order to avoid
        # holding state while waiting for a sampler that needs the state lock.
        with broker.refresh_lock, broker.lock:
            if broker.stop_event.is_set():
                raise ValueError("Manager is stopping; retry after shutdown completes")
            if broker.control_status()['disabled']:
                if method in {'release_idle', 'register_model', 'rollback_policy'}:
                    raise ValueError(DISABLED_MESSAGE)
                if method != 'request_model':
                    allowed = set(broker.control_status().get('preserve_lease_ids', []))
                    if params.get('lease_id') not in allowed:
                        raise ValueError(DISABLED_MESSAGE)
            return getattr(broker, method)(**params)
    raise ValueError("Unknown method")


def serve(root=ROOT, listen_port=None, continue_existing_only=False):
    verify_continuation(root / 'work/state', explicit_continuation=continue_existing_only)
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (psutil.Error, AttributeError):
        pass
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config["host"] != "127.0.0.1":
        raise ValueError("Public network binding is not supported")
    # A local maintenance bridge may temporarily route clients through another
    # loopback port while a replacement broker resumes the same durable state.
    if listen_port is not None:
        if type(listen_port) is not int or not 1024 <= listen_port <= 65535:
            raise ValueError("Invalid loopback maintenance port")
        config['port'] = listen_port
    data = root / "work/state"
    data.mkdir(parents=True, exist_ok=True)
    lock_file = (data / "service.lock").open("a+b")
    lock_file.seek(0)
    if not lock_file.read(1):
        lock_file.write(b"0")
        lock_file.flush()
    lock_file.seek(0)
    try:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        return  # Another chat already started the singleton.
    token_file = data / "api-token"
    if not token_file.exists():
        fd = os.open(token_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as f:
            f.write(secrets.token_urlsafe(48))
    api_token = token_file.read_text(encoding="ascii").strip()
    broker = Broker(data, config, config_path=root / "config.json")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log credentials or lease tokens.

        def send_json(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            expected = "Bearer " + api_token
            if self.headers.get("Origin") or not secrets.compare_digest(self.headers.get("Authorization", ""), expected):
                self.send_json(403, {"error": "Loopback authentication required"})
                return
            if self.path != "/rpc":
                self.send_json(404, {"error": "Not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                method, params = body["method"], body.get("params", {})
                result = dispatch(broker, method, params, shutdown=server.shutdown)
                self.send_json(200, {"result": result})
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(400, {"error": str(exc)[:300]})
            except Exception as exc:
                self.send_json(500, {"error": type(exc).__name__ + ": operation failed; inspect broker state"})

    server = ThreadingHTTPServer((config["host"], config["port"]), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=broker.run, daemon=True, name="resource-monitor")
    worker.start()
    app_worker = threading.Thread(target=broker.app_reclaimer.run, args=(broker.stop_event,),
                                  daemon=True, name="idle-app-memory")
    app_worker.start()
    def stop_after_existing_tasks():
        broker.stop_event.wait()
        server.shutdown()
    threading.Thread(target=stop_after_existing_tasks, daemon=True, name='disabled-task-drain').start()
    try:
        server.serve_forever(poll_interval=.3)
    finally:
        broker.stop_event.set()
        worker.join(timeout=10)
        app_worker.join(timeout=5)
        server.server_close()
        lock_file.close()


if __name__ == "__main__":
    serve()
