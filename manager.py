"""CLI and one-command resource wrapper for Local Model Manager."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
from client import call

OUTCOMES = ("completed", "oom", "failed", "cancelled")
QUALITIES = ("passed", "failed", "unknown")


def json_object_file(path):
    if not path:
        return None
    with Path(path).open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("JSON metadata file exceeds 64 KiB")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("JSON metadata must contain an object")
    return value


def qa_result(path, started_ns, returncode):
    """Accept only fresh child-provided QA; process success alone leaves it unknown."""
    result = {"outcome": "completed" if returncode == 0 else "failed", "quality": "unknown",
              "desktop_responsive": None}
    if not path:
        return result
    try:
        qa_path = Path(path)
        if qa_path.stat().st_mtime_ns < started_ns:
            raise ValueError("QA file predates this command")
        record = json_object_file(qa_path)
        quality = record.get("quality", "unknown")
        responsive = record.get("desktop_responsive")
        outcome = record.get("outcome", result["outcome"])
        if quality not in QUALITIES or outcome not in OUTCOMES:
            raise ValueError("Unrecognized QA outcome or quality")
        if responsive is not None and type(responsive) is not bool:
            raise ValueError("desktop_responsive must be boolean or null")
        result.update(outcome=outcome, quality=quality, desktop_responsive=responsive)
        if returncode != 0 and outcome == "completed":
            result.update(outcome="failed", quality="unknown")
        if result["outcome"] != "completed" and result["quality"] == "passed":
            result["quality"] = "unknown"
    except (OSError, ValueError, TypeError, UnicodeError):
        print("[local-model-manager] QA file missing, stale or invalid; quality remains unknown.", file=sys.stderr)
    return result


def run_managed(args):
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Provide a command after --")
    task = json_object_file(getattr(args, "task_file", None))
    lease = call("request_model", {"model": args.model, "owner": args.owner, "request_key": str(uuid.uuid4()),
                                    "ttl_seconds": 3600, "keep_loaded": args.keep_loaded, "task": task})
    cred = {"lease_id": lease["id"], "token": lease["token"]}
    work_id = str(uuid.uuid4())
    child, prepared, started_ns = None, False, None
    try:
        deadline = time.monotonic() + args.wait_seconds
        last_reason = None
        while time.monotonic() < deadline:
            state = call("lease_status", cred)
            if state["state"] in {"failed", "expired", "cancelled", "released"}:
                raise RuntimeError(state.get("reason", "Lease ended before launch"))
            reason = state.get("reason", "Waiting for resources")
            if state["state"] == "active" and state["proceed"]:
                admission = call("prepare_work", {**cred, "work_id": work_id, "task": task})
                if admission.get("proceed") is True:
                    prepared = True
                    break
                reason = admission.get("reason", "Waiting for task-specific resources")
                if admission.get('retryable') is False:
                    raise RuntimeError(reason)
            if reason != last_reason:
                print("[local-model-manager] " + str(reason), file=sys.stderr)
                last_reason = reason
            time.sleep(5)
        else:
            raise RuntimeError("Resource wait timed out")
        started_ns = time.time_ns()
        child = subprocess.Popen(command)  # argv only; never a shell string.
        try:
            call("attach_worker", {**cred, "pid": child.pid})
        except RuntimeError:
            if child.poll() is None:
                raise
        last_proceed = True
        while child.poll() is None:
            time.sleep(5)
            state = call("lease_status", cred)
            if not state["proceed"] and last_proceed:
                print("[local-model-manager] Resource pressure: subsequent submissions require a fresh prepare check. In-flight process retained.", file=sys.stderr)
            last_proceed = bool(state["proceed"])
        return child.returncode
    finally:
        # This wrapper gates the command's launch. A script with repeated model
        # submissions must use prepare/complete around each internal work item.
        # A detached live child keeps its lease until the broker verifies it idle.
        if child is None or child.poll() is not None:
            try:
                if prepared:
                    result = (qa_result(getattr(args, "qa_file", None), started_ns, child.returncode)
                              if child is not None else
                              {"outcome": "cancelled", "quality": "unknown", "desktop_responsive": None})
                    call("complete_work", {**cred, "work_id": work_id, **result})
            except (RuntimeError, ValueError, OSError) as exc:
                print("[local-model-manager] Completion recording failed: " + str(exc), file=sys.stderr)
            finally:
                call("release", cred)


def add_credential_args(parser):
    parser.add_argument("lease_id")
    parser.add_argument("token")


def add_outcome_args(parser):
    parser.add_argument("--outcome", choices=OUTCOMES, default="completed")
    parser.add_argument("--quality", choices=QUALITIES, default="unknown")
    parser.add_argument("--desktop-responsive", choices=["true", "false", "unknown"], default="unknown")


def main(argv=None):
    p = argparse.ArgumentParser(description="本地模型管家 / Local Model Manager")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("status")
    sub.add_parser("history")
    sub.add_parser("health-report")
    sub.add_parser("app-memory")
    sub.add_parser("shutdown")
    q = sub.add_parser("learning")
    q.add_argument("--model")
    q = sub.add_parser("register")
    q.add_argument("--model", required=True)
    q.add_argument("--manifest", required=True)
    q.add_argument("--replace", action="store_true")
    q = sub.add_parser("request")
    q.add_argument("model")
    q.add_argument("--owner", required=True)
    q.add_argument("--key", required=True)
    q.add_argument("--ttl", type=int, default=3600)
    q.add_argument("--keep-loaded", action="store_true")
    q.add_argument("--task-file")
    for name in ["lease", "release", "prepare", "complete", "feedback", "rollback"]:
        q = sub.add_parser(name)
        add_credential_args(q)
        if name not in {"lease", "release"}:
            q.add_argument("work_id")
        if name == "prepare":
            q.add_argument("--task-file")
        elif name in {"complete", "feedback"}:
            add_outcome_args(q)
            if name == "feedback":
                q.add_argument("--feedback-id", required=True)
        elif name == "rollback":
            q.add_argument("--target-revision", type=int, required=True)
            q.add_argument("--reason", required=True)
    q = sub.add_parser("release-idle")
    q.add_argument("model")
    q.add_argument("--allow-external", action="store_true")
    q = sub.add_parser("run")
    q.add_argument("--model", required=True)
    q.add_argument("--owner", required=True)
    q.add_argument("--wait-seconds", type=int, default=1800)
    q.add_argument("--keep-loaded", action="store_true")
    q.add_argument("--task-file")
    q.add_argument("--qa-file")
    q.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    if args.action == "run":
        return run_managed(args)
    if args.action == "register":
        result = call("register_model", dict(model=args.model, manifest=json_object_file(args.manifest), replace=args.replace))
    elif args.action == "request":
        result = call("request_model", dict(model=args.model, owner=args.owner, request_key=args.key,
                                            ttl_seconds=args.ttl, keep_loaded=args.keep_loaded,
                                            task=json_object_file(args.task_file)))
    elif args.action in {"lease", "release", "prepare", "complete", "feedback", "rollback"}:
        params = dict(lease_id=args.lease_id, token=args.token)
        if args.action not in {"lease", "release"}:
            params["work_id"] = args.work_id
        if args.action == "prepare":
            params["task"] = json_object_file(args.task_file)
        elif args.action in {"complete", "feedback"}:
            params.update(outcome=args.outcome, quality=args.quality,
                          desktop_responsive={"true": True, "false": False, "unknown": None}[args.desktop_responsive])
            if args.action == "feedback":
                params["feedback_id"] = args.feedback_id
        elif args.action == "rollback":
            params.update(target_revision=args.target_revision, reason=args.reason)
        method = {"lease": "lease_status", "prepare": "prepare_work", "complete": "complete_work",
                  "rollback": "rollback_policy"}.get(args.action, args.action)
        result = call(method, params)
    elif args.action == "health-report":
        result = call('health_report')
    elif args.action == "app-memory":
        result = call('app_memory_status')
    elif args.action == "learning":
        result = call("learning_status", dict(model=args.model))
    elif args.action == "release-idle":
        result = call("release_idle", dict(model=args.model, allow_external=args.allow_external))
    else:
        result = call(args.action)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
