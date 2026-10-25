"""Host measurements. Unknown values are never treated as free resources."""
import csv
import ctypes
import io
import os
import subprocess
import time
import psutil
from collections import deque
import threading

GIB = 1024 ** 3
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_probe = None


class ResponsivenessProbe:
    """Bounded timer drift proxy, not a claim of measured UI/input latency."""
    def __init__(self):
        self.delays = deque(maxlen=40)
        self.stop = threading.Event()
        threading.Thread(target=self.run, daemon=True, name="desktop-load-probe").start()

    def run(self):
        while not self.stop.is_set():
            before = time.monotonic()
            if self.stop.wait(.25):
                return
            self.delays.append(max(0, (time.monotonic()-before-.25)*1000))


def snapshot():
    global _probe
    if _probe is None:
        _probe = ResponsivenessProbe()
    v = psutil.virtual_memory()
    result = {"time": time.time(), "ram_total_gib": v.total / GIB,
              "ram_available_gib": v.available / GIB, "commit_free_gib": None,
              "commit_used_gib": None, "commit_limit_gib": None, "gpus": [], "errors": []}
    result.update(cpu_percent=psutil.cpu_percent(interval=None), cpu_count=psutil.cpu_count() or 1,
                  scheduler_delay_ms=max(list(_probe.delays), default=0),
                  manager_rss_mib=psutil.Process().memory_info().rss/1024**2)
    if os.name == "nt":
        class PERFORMANCE_INFORMATION(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong)] + [(x, ctypes.c_size_t) for x in
                ("CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
                 "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
                (x, ctypes.c_ulong) for x in ("HandleCount", "ProcessCount", "ThreadCount")]
        p = PERFORMANCE_INFORMATION()
        p.cb = ctypes.sizeof(p)
        if ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb):
            result.update(commit_used_gib=p.CommitTotal * p.PageSize / GIB,
                          commit_limit_gib=p.CommitLimit * p.PageSize / GIB,
                          commit_free_gib=(p.CommitLimit-p.CommitTotal)*p.PageSize/GIB)
        else:
            result["errors"].append("GetPerformanceInfo failed")
    else:
        result["errors"].append("Windows commit metrics unavailable on this OS")
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              timeout=4, creationflags=NO_WINDOW, check=True)
        for row in csv.reader(io.StringIO(proc.stdout)):
            if len(row) == 8:
                result["gpus"].append({"index": int(row[0]), "uuid": row[1].strip(), "name": row[2].strip(),
                                      "total_gib": float(row[3])/1024, "used_gib": float(row[4])/1024,
                                      "free_gib": float(row[5])/1024, "utilization_percent": float(row[6]),
                                      "temperature_c": float(row[7])})
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["errors"].append("GPU telemetry unavailable: " + type(exc).__name__)
    return result


def pressure_reasons(s, config, model=None, cold=False):
    limits = {"ram_available_gib": config["min_available_ram_gib"],
              "commit_free_gib": config["min_commit_free_gib"]}
    gpu_limit = config["min_free_vram_gib"]
    if cold and model:
        limits["ram_available_gib"] = max(limits["ram_available_gib"], model["startup_free_ram_gib"])
        limits["commit_free_gib"] = max(limits["commit_free_gib"], model["startup_commit_free_gib"])
        gpu_limit = max(gpu_limit, model["startup_free_vram_gib"])
    reasons = []
    for key, minimum in limits.items():
        value = s.get(key)
        if value is None:
            reasons.append(key + " unavailable")
        elif value < minimum:
            reasons.append(f"{key} {value:.1f} < {minimum:.1f} GiB")
    gpu = next((x for x in s.get("gpus", []) if x["index"] == 0), None)
    if gpu is None:
        reasons.append("GPU 0 telemetry unavailable")
    elif gpu["free_gib"] < gpu_limit:
        reasons.append(f"free_vram {gpu['free_gib']:.1f} < {gpu_limit:.1f} GiB")
    return reasons
