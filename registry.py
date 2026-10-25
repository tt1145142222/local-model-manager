"""Declarative registration for verified local ComfyUI and llama.cpp backends.

Registration validates configuration only; it neither starts nor adopts a model.
No plugin modules, Python expressions, shell commands, or credentials are loaded.

Manifest examples (replace placeholders with actual installed absolute paths):

ComfyUI/video:
    {"label": "Another video backend", "kind": "comfyui",
     "workload_kind": "video", "root": "D:/AI/AnotherVideo",
     "exe": "D:/AI/AnotherVideo/python/python.exe",
     "url": "http://127.0.0.1:8190",
     "start_argv": ["D:/AI/AnotherVideo/python/python.exe", "-s",
                    "D:/AI/AnotherVideo/ComfyUI/main.py", "--listen",
                    "127.0.0.1", "--port", "8190"],
     "startup_free_ram_gib": 20, "startup_commit_free_gib": 36,
     "startup_free_vram_gib": 24}

llama.cpp/LLM:
    {"label": "Another local LLM", "kind": "llamacpp",
     "workload_kind": "llm", "root": "D:/AI/AnotherLLM",
     "exe": "D:/AI/AnotherLLM/llama-server.exe",
     "url": "http://127.0.0.1:8090", "context_capacity": 32768,
     "start_argv": ["D:/AI/AnotherLLM/llama-server.exe", "-m",
                    "D:/AI/AnotherLLM/model.gguf", "-c", "32768",
                    "-np", "1", "--host", "127.0.0.1", "--port", "8090", "--slots"],
     "startup_free_ram_gib": 16, "startup_commit_free_gib": 26,
     "startup_free_vram_gib": 22}

All resource estimates are nonnegative GiB, supplied conservatively for this
model. start_cwd defaults to root and cannot escape it. Optional llama.cpp
api_key_file must already exist inside root; its contents are never read here.
Optional agent_url is a separate fixed-loopback OpenCode status API. llama.cpp
must expose /health and /slots; ComfyUI must expose /queue. Other protocols need
a separately implemented and verified adapter, not a manifest naming arbitrary
Python code. Registration itself never changes generation quality settings.
For llama.cpp the sole -c/--ctx-size value must equal context_capacity, and an
api_key_file must match an explicit --api-key-file argument. Plaintext --api-key
arguments are rejected so registration/status output cannot expose credentials.
An explicit -np/--parallel value of 1 is required until per-slot context budgeting
is supported; context must not be implicitly divided between concurrent slots.
"""
import copy
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


MODEL_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
RESERVED_IDS = frozenset({"h3", "muse"})
KINDS = {"comfyui": "video", "llamacpp": "llm"}
STARTUP_FIELDS = ("startup_free_ram_gib", "startup_commit_free_gib", "startup_free_vram_gib")
REQUIRED_FIELDS = frozenset({"label", "kind", "workload_kind", "root", "exe", "url", "start_argv", *STARTUP_FIELDS})
OPTIONAL_FIELDS = frozenset({"start_cwd", "context_capacity", "agent_url", "api_key_file"})
SHELL_NAMES = frozenset({"cmd", "command", "powershell", "pwsh", "bash", "sh", "dash", "zsh", "fish",
                         "wscript", "cscript", "mshta", "rundll32", "regsvr32"})


def _model_id(value):
    if not isinstance(value, str) or MODEL_ID.fullmatch(value) is None:
        raise ValueError("model_id must match [a-z][a-z0-9_-]{0,63}")
    if value in RESERVED_IDS:
        raise ValueError("h3 and muse are reserved installed adapters; do not replace them by registration")
    return value


def _text(value, name, limit=4096):
    if (not isinstance(value, str) or not value.strip() or len(value) > limit
            or any(ord(c) < 32 for c in value)):
        raise ValueError(name + " must be a bounded string without control characters")
    return value


def _path(value, name, directory=False):
    _text(value, name)
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(name + " must be an absolute path")
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(name + " does not resolve to an existing path") from exc
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(name + (" must be an existing directory" if directory else " must be an existing file"))
    return path


def _same_path(left, right):
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _loopback(value, name):
    _text(value, name, 256)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(name + " must be a fixed 127.0.0.1 HTTP URL with a valid port") from exc
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or port is None or not 1 <= port <= 65535
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
            or parsed.netloc != f"127.0.0.1:{port}"):
        raise ValueError(name + " must be http://127.0.0.1:<port> without credentials, path or query")
    return f"http://127.0.0.1:{port}", port


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(field + " must be finite nonnegative GiB")
    return float(value)


def _option(argv, flag):
    # Deliberately require a single explicit long-form flag and its next token.
    # The adapter can then identify the exact listener without shell parsing.
    if any(item.startswith(flag + "=") for item in argv) or argv.count(flag) != 1:
        raise ValueError("start_argv requires exactly one " + flag + " followed by its value")
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError("missing " + flag + " value")
    return argv[index + 1]


def validate_manifest(model_id, manifest):
    """Validate and normalize one compatible backend manifest without side effects."""
    _model_id(model_id)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a mapping")
    unknown, missing = set(manifest) - REQUIRED_FIELDS - OPTIONAL_FIELDS, REQUIRED_FIELDS - set(manifest)
    if unknown or missing:
        raise ValueError("manifest fields invalid: missing=" + ",".join(sorted(missing))
                         + "; unknown=" + ",".join(sorted(str(x) for x in unknown)))
    label = _text(manifest["label"], "label", 128)
    kind = manifest["kind"]
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError("supported backend kinds are comfyui and llamacpp")
    if manifest["workload_kind"] != KINDS[kind]:
        raise ValueError("workload_kind must match the registered backend")
    root = _path(manifest["root"], "root", directory=True)
    exe = _path(manifest["exe"], "exe")
    stem = exe.stem.lower()
    if stem in SHELL_NAMES or exe.suffix.lower() in (".cmd", ".bat", ".ps1", ".sh", ".vbs", ".js"):
        raise ValueError("exe must be the model server executable, not a shell or script launcher")
    cwd = _path(manifest.get("start_cwd", str(root)), "start_cwd", directory=True)
    if not _same_path(cwd, root):
        raise ValueError("start_cwd must resolve to root")
    url, port = _loopback(manifest["url"], "url")
    original_argv = manifest["start_argv"]
    if not isinstance(original_argv, list) or not 2 <= len(original_argv) <= 256:
        raise ValueError("start_argv must be an argument list of length 2..256")
    argv = [_text(item, "start_argv item", 8192) for item in original_argv]
    argv_exe = _path(argv[0], "start_argv[0]")
    if not _same_path(argv_exe, exe):
        raise ValueError("start_argv[0] must be the exact registered exe")
    argv[0] = str(exe)
    if _option(argv, "--port") != str(port):
        raise ValueError("start_argv --port must match url")
    binding_flag = "--listen" if kind == "comfyui" else "--host"
    if _option(argv, binding_flag) != "127.0.0.1":
        raise ValueError("start_argv must bind explicitly to 127.0.0.1")
    other_binding = "--host" if binding_flag == "--listen" else "--listen"
    if any(item == other_binding or item.startswith(other_binding + "=") for item in argv):
        raise ValueError("ambiguous alternate listener flag")
    if "python" in stem and any(item in ("-c", "--command", "-m") for item in argv[1:]):
        raise ValueError("Python inline code or module launch is unsupported; register an existing server script")
    if any(item == "--api-key" or item.startswith("--api-key=") for item in argv):
        raise ValueError("plaintext --api-key is unsupported; use api_key_file and --api-key-file")

    result = {"label": label, "kind": kind, "workload_kind": KINDS[kind], "root": str(root),
              "exe": str(exe), "url": url, "start_argv": argv, "start_cwd": str(root)}
    for field in STARTUP_FIELDS:
        result[field] = _number(manifest[field], field)
    if kind == "llamacpp":
        capacity = manifest.get("context_capacity")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 1048576:
            raise ValueError("llamacpp context_capacity must be an integer in 1..1048576")
        context_flags = [flag for flag in ("-c", "--ctx-size")
                         if flag in argv or any(item.startswith(flag + "=") for item in argv)]
        if len(context_flags) != 1 or _option(argv, context_flags[0]) != str(capacity):
            raise ValueError("exactly one -c/--ctx-size value must equal context_capacity")
        parallel_flags = [flag for flag in ("-np", "--parallel")
                          if flag in argv or any(item.startswith(flag + "=") for item in argv)]
        if len(parallel_flags) != 1 or _option(argv, parallel_flags[0]) != "1":
            raise ValueError("registered llama.cpp requires exactly one -np/--parallel value of 1")
        result["context_capacity"] = capacity
        if "agent_url" in manifest:
            agent_url, agent_port = _loopback(manifest["agent_url"], "agent_url")
            if agent_port == port:
                raise ValueError("agent_url must differ from the model server URL")
            result["agent_url"] = agent_url
        if "api_key_file" in manifest:
            key = _path(manifest["api_key_file"], "api_key_file")
            try:
                key.relative_to(root)
            except ValueError as exc:
                raise ValueError("api_key_file must resolve inside root") from exc
            argv_key = _path(_option(argv, "--api-key-file"), "start_argv --api-key-file")
            if not _same_path(argv_key, key):
                raise ValueError("start_argv --api-key-file must match api_key_file")
            result["api_key_file"] = str(key)
        elif any(item == "--api-key-file" or item.startswith("--api-key-file=") for item in argv):
            raise ValueError("start_argv --api-key-file requires the matching manifest api_key_file")
    elif set(manifest) & {"context_capacity", "agent_url", "api_key_file"}:
        raise ValueError("context_capacity, agent_url and api_key_file apply only to llamacpp")
    return result


def register_config(config, model_id, manifest, replace=False):
    """Return a validated copy; caller serializes/reloads under its own lock.

    Existing model IDs require explicit replace=True. A live or queued adapter
    must not be replaced until the broker confirms there is no outstanding use.
    """
    if not isinstance(config, dict) or not isinstance(config.get("models"), dict):
        raise ValueError("config must contain a models mapping")
    if type(replace) is not bool:
        raise ValueError("replace must be boolean")
    _model_id(model_id)
    if model_id in config["models"] and not replace:
        raise ValueError("model already registered; explicit replace=True is required")
    validated = validate_manifest(model_id, manifest)
    for other_id, other in config["models"].items():
        if other_id != model_id and other.get("url", "").rstrip("/") == validated["url"]:
            raise ValueError("model URL already belongs to another registered adapter")
    result = copy.deepcopy(config)
    result["models"][model_id] = validated
    return result
