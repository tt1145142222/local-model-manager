"""Codex-facing stdio tools; all sessions share the independent broker."""
import asyncio
from typing import Literal
from lite_mcp import FastMCP
from client import call

mcp = FastMCP("local_models_mcp", instructions="Manage registered local model GPU resources, initially H3 and Muse. Request a lease with task metadata; wait for active AND proceed=true. Before each discrete inference, call prepare and wait for proceed=true. Complete each work item and report only actually observed QA/desktop feedback. Preserve model and generation quality settings. Release with its credential after work finishes. Externally started services need specific user authorization before release_idle(allow_external=true). Register additional models only for explicit integration requests; registration does not run inference.")
READ = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
WRITE = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}


@mcp.tool(annotations=READ)
async def local_models_status() -> dict:
    """Inspect RAM, Windows commit, GPU 0, service busy/ownership, queue and monitoring health. Starts only the lightweight manager if absent."""
    return await asyncio.to_thread(call, "status")


@mcp.tool(annotations=WRITE)
async def local_models_request(model: str, owner: str, request_key: str,
                               ttl_seconds: int = 3600, keep_loaded: bool = False,
                               task: dict | None = None) -> dict:
    """Queue an exclusive GPU lease, starting the model when safe. owner identifies chat/task. Reuse request_key on retry. Returns id/token: retain both privately. Renew within TTL; never use model until state=active and proceed=true. keep_loaded allows up to 120s warm reuse after release."""
    return await asyncio.to_thread(call, "request_model", dict(model=model, owner=owner, request_key=request_key,
                                  ttl_seconds=ttl_seconds, keep_loaded=keep_loaded, task=task))


@mcp.tool(annotations=WRITE)
async def local_models_prepare(lease_id: str, token: str, work_id: str, task: dict | None = None) -> dict:
    """Check task-specific resource admission immediately before ONE inference submission. Reuse work_id when retrying; submit only when proceed=true. If retryable=false/policy_infeasible, stop polling, release this lease and report the budget conflict; never lower quality or clamp budgets to force admission. Metadata describes the existing task. Recheck before each later submission even within one lease. A permit does not guarantee smoothness during an indivisible GPU kernel."""
    return await asyncio.to_thread(call, "prepare_work", dict(lease_id=lease_id, token=token, work_id=work_id, task=task))


@mcp.tool(annotations=WRITE)
async def local_models_complete(lease_id: str, token: str, work_id: str,
                                outcome: Literal["completed", "oom", "failed", "cancelled"] = "completed",
                                quality: Literal["passed", "failed", "unknown"] = "unknown",
                                desktop_responsive: bool | None = None) -> dict:
    """Mark one prepared inference finished and store measured resource observations. A zero process exit or API response does not prove quality: leave quality unknown until the original workflow QA checks pass. desktop_responsive is observed feedback, not a guess from resource counters. This does not release the lease."""
    return await asyncio.to_thread(call, "complete_work", dict(lease_id=lease_id, token=token, work_id=work_id,
                                    outcome=outcome, quality=quality, desktop_responsive=desktop_responsive))


@mcp.tool(annotations=WRITE)
async def local_models_feedback(lease_id: str, token: str, work_id: str, feedback_id: str,
                                outcome: Literal["completed", "oom", "failed", "cancelled"] = "completed",
                                quality: Literal["passed", "failed", "unknown"] = "unknown",
                                desktop_responsive: bool | None = None) -> dict:
    """Attach later QA/user responsiveness feedback to your work item using its lease credential. Reuse feedback_id on retry. Report passed only for completed QA, and desktop responsiveness only from real observation. Feedback can tighten or roll back task-specific resource policy; model quality settings remain unchanged."""
    return await asyncio.to_thread(call, "feedback", dict(lease_id=lease_id, token=token, work_id=work_id,
                                    feedback_id=feedback_id, outcome=outcome, quality=quality,
                                    desktop_responsive=desktop_responsive))


@mcp.tool(annotations=READ)
async def local_models_learning(model: str | None = None) -> dict:
    """Read task-profile observations, versioned resource policies and feedback. Resource estimates are conservative admission aids, not quality claims or guaranteed optimal settings. Does not expose prompts, model API keys or lease credentials."""
    return await asyncio.to_thread(call, "learning_status", dict(model=model))


@mcp.tool(annotations=WRITE)
async def local_models_lease(lease_id: str, token: str, renew: bool = True) -> dict:
    """Inspect your lease and optionally renew TTL. Poll every 5-15s while queued; renew at least every 5 minutes during long jobs. proceed=false means pause new submissions, not cancel in-flight inference."""
    return await asyncio.to_thread(call, "lease_status", dict(lease_id=lease_id, token=token, renew=renew))


@mcp.tool(annotations=WRITE)
async def local_models_release(lease_id: str, token: str) -> dict:
    """Finish your resource use or cancel your queued request. Waits for verified idle. Stops manager-owned service by default; externally started services are retained. Recheck status until terminal."""
    return await asyncio.to_thread(call, "release", dict(lease_id=lease_id, token=token))


@mcp.tool(annotations={**WRITE, "destructiveHint": True})
async def local_models_release_idle(model: str, allow_external: bool = False) -> dict:
    """Request stop of a verified idle model service, requiring no active leases. For an externally started service, allow_external=true requires explicit human authorization for stopping that model. Does not stop Muse's OpenCode UI. Returns operation id; inspect status for completion."""
    return await asyncio.to_thread(call, "release_idle", dict(model=model, allow_external=allow_external))


@mcp.tool(annotations=READ)
async def local_models_history(limit: int = 20) -> dict:
    """Read recent lease outcomes, resource peaks and lifecycle events. Device VRAM peaks include other programs. Tokens, prompts and API keys are omitted."""
    return await asyncio.to_thread(call, "history", dict(limit=limit))


@mcp.tool(annotations=WRITE)
async def local_models_register(model: str, manifest: dict, replace: bool = False) -> dict:
    """Register an additional local ComfyUI or llama.cpp model only when the user requests model integration. Manifest pins an existing local executable, loopback endpoint, exact startup argv and resource floors; preserve model/generation quality settings. Registration requires no active/queued work, writes local configuration with a backup, and does not start or infer with the model. Replace only the explicitly selected registered model."""
    return await asyncio.to_thread(call, "register_model", dict(model=model, manifest=manifest, replace=replace))


if __name__ == "__main__":
    mcp.run(transport="stdio")
