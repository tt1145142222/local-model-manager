import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib import request, error
from manager_control import disabled_state, DISABLED_MESSAGE

ROOT = Path(__file__).resolve().parent


def raw_call(method, params=None, timeout=25):
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    token = (ROOT / "work/state/api-token").read_text(encoding="ascii").strip()
    req = request.Request(f"http://127.0.0.1:{config['port']}/rpc",
                          data=json.dumps({"method": method, "params": params or {}}).encode("utf-8"),
                          headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    try:
        with request.build_opener(request.ProxyHandler({})).open(req, timeout=timeout) as r:
            result = json.load(r)
    except error.HTTPError as exc:
        try:
            message = json.loads(exc.read()).get("error", "request rejected")
        except Exception:
            message = "HTTP request rejected"
        raise RuntimeError(message) from None
    return result["result"]


def ensure_service():
    try:
        h = raw_call("health", timeout=2)
        if h.get("service") == "local-model-manager":
            return
        raise RuntimeError("Port belongs to another service")
    except (FileNotFoundError, error.URLError, TimeoutError):
        pass
    if disabled_state(ROOT / 'work/state')['disabled']:
        raise RuntimeError(DISABLED_MESSAGE)
    logs = ROOT / "work/logs"
    logs.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with (logs / "service.out.log").open("ab") as out, (logs / "service.err.log").open("ab") as err:
        subprocess.Popen([sys.executable, str(ROOT / "service.py")], cwd=ROOT, stdout=out, stderr=err,
                         stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    for _ in range(40):
        try:
            if raw_call("health", timeout=1).get("service") == "local-model-manager":
                return
        except (FileNotFoundError, error.URLError, TimeoutError):
            pass
        time.sleep(.25)
    raise RuntimeError("Manager did not start. Inspect work/logs/service.err.log")


def call(method, params=None):
    ensure_service()
    return raw_call(method, params)
