"""Use the existing idle Muse, prove queue exclusion, and preserve its process."""
import json
import time
import uuid
from pathlib import Path
from client import call, raw_call
from adapters import Adapter, http_json


def wait_state(cred, wanted, timeout=30):
    end = time.time()+timeout
    while time.time() < end:
        value = call("lease_status", cred)
        if value["state"] in wanted:
            return value
        time.sleep(1)
    raise RuntimeError("Timed out waiting for lease state")


def main():
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    adapter = Adapter("muse", config["models"]["muse"])
    before = adapter.inspect()
    assert before["state"] == "ready" and before["busy"] is False, "Muse must already be idle and ready"
    key = "live-smoke-" + str(uuid.uuid4())
    muse = call("request_model", {"model": "muse", "owner": "local-model-manager-install-verification", "request_key": key})
    mc = {"lease_id": muse["id"], "token": muse["token"]}
    h3c = None
    try:
        granted = wait_state(mc, {"active", "failed"})
        assert granted["state"] == "active" and granted["proceed"]
        reply = http_json(config["models"]["muse"]["url"] + "/v1/chat/completions",
                          {"model": "muse-glimmer", "messages": [{"role": "user", "content": "Reply with OK only."}],
                           "max_tokens": 16, "temperature": 0, "stream": False}, adapter.headers(), timeout=60)
        assert reply.get("choices"), "No completion received"
        h3 = call("request_model", {"model": "h3", "owner": "local-model-manager-second-chat-test", "request_key": key+"-h3"})
        h3c = {"lease_id": h3["id"], "token": h3["token"]}
        time.sleep(6)
        waiting = call("lease_status", h3c)
        assert waiting["state"] == "queued" and not waiting["proceed"]
        call("release", mc)
        released = wait_state(mc, {"released", "failed"})
        assert released["state"] == "released"
        deadline = time.time()+30
        while time.time() < deadline:
            waiting_external = call("lease_status", h3c)
            if waiting_external["state"] != "queued" or "muse" in waiting_external["reason"]:
                break
            time.sleep(1)
        assert waiting_external["state"] == "queued" and "muse" in waiting_external["reason"]
        assert adapter.inspect()["identity"] == before["identity"]
        call("release", h3c)
        after = adapter.inspect()
        result = {"muse_completion_passed": True, "muse_completion_tokens": reply.get("usage", {}).get("completion_tokens"),
                  "cross_chat_exclusion_passed": True, "external_service_retained": after["identity"] == before["identity"],
                  "h3_blocked_by_external_muse": True, "lease_release_passed": True,
                  "timestamp": time.time(), "muse_pid_preserved": after["identity"]["pid"]}
        target = Path(__file__).parent / "work/verification/live-muse.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        call("release", mc)
        if h3c:
            call("release", h3c)


if __name__ == "__main__":
    main()
