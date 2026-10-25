"""Read-only MCP transport validation; no model start or stop."""
import asyncio
import json
from pathlib import Path
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    async with stdio_client(StdioServerParameters(command=sys.executable, args=[str(Path(__file__).with_name("mcp_server.py"))])) as (r, w):
        async with ClientSession(r, w) as session:
            info = await session.initialize()
            catalog = await session.list_tools()
            status = await session.call_tool("local_models_status", {})
            assert not status.isError, status
            data = status.structuredContent
            if not data:
                data = json.loads(status.content[0].text)
            assert data["version"] == "1.1.1"
            assert {"h3", "muse"} <= set(data["models"])
            history = await session.call_tool("local_models_history", {"limit": 5})
            assert not history.isError
            print(json.dumps({"protocol": info.protocolVersion, "tools": [x.name for x in catalog.tools],
                              "resources_measured": bool(data["resources"].get("gpus")),
                              "models": {k: v["state"] for k, v in data["models"].items()},
                              "status_call_passed": True, "history_call_passed": True}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
