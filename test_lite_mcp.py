"""Stdlib transport contract tests; SDK interop test is read-only and opt-in."""
import asyncio
import io
import json
import os
from pathlib import Path
import sys
from typing import Literal, Optional
import unittest

from lite_mcp import FastMCP, MAX_LINE_BYTES, PROTOCOLS, decode


class LiteProtocol(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = FastMCP('test', instructions='test instructions')
        self.calls = []
        @self.server.tool(annotations={'readOnlyHint': True})
        async def example(name: str, count: int = 1, task: Optional[dict] = None,
                          quality: Literal['passed', 'failed', 'unknown'] = 'unknown',
                          responsive: bool | None = None) -> dict:
            self.calls.append(name)
            return {'name': name, 'count': count, 'task': task, 'quality': quality, 'responsive': responsive}
        @self.server.tool()
        async def failure(token: str) -> dict:
            raise OSError('unexpected secret content ' + token)
        @self.server.tool()
        async def rejected(token: str) -> dict:
            raise ValueError('request rejected for ' + token)
        await self.request('initialize', {'protocolVersion': PROTOCOLS[0]})

    async def request(self, method, params=None, request_id=1):
        return await self.server.handle({'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params or {}})

    async def test_tool_schema_optional_defaults_and_annotations(self):
        result = await self.request('tools/list')
        tool = result['result']['tools'][0]
        schema = tool['inputSchema']
        self.assertEqual(schema['required'], ['name'])
        self.assertEqual(schema['properties']['task']['anyOf'],
                         [{'type': 'object', 'additionalProperties': True}, {'type': 'null'}])
        self.assertEqual(schema['properties']['quality']['default'], 'unknown')
        self.assertTrue(tool['annotations']['readOnlyHint'])

    async def test_calls_produce_structured_and_equivalent_text_with_defaults(self):
        reply = await self.request('tools/call', {'name': 'example', 'arguments': {'name': '中文'}})
        result = reply['result']
        self.assertFalse(result['isError'])
        self.assertEqual(json.loads(result['content'][0]['text']), result['structuredContent'])
        self.assertEqual(result['structuredContent']['quality'], 'unknown')
        self.assertIsNone(result['structuredContent']['task'])

    async def test_invalid_types_missing_unknown_args_do_not_call_tool(self):
        for arguments in ({'name': 'x', 'count': True}, {'name': 'x', 'quality': 'invented'},
                          {'name': 'x', 'responsive': 'yes'}, {'name': 'x', 'extra': 2}, {},
                          {'name': 'x', 'task': {'bad': float('nan')}}):
            with self.subTest(arguments=arguments):
                result = await self.request('tools/call', {'name': 'example', 'arguments': arguments})
                self.assertEqual(result['error']['code'], -32602)
        self.assertEqual(self.calls, [])

    async def test_notifications_never_invoke_tools_or_respond(self):
        for method in ('notifications/initialized', 'notifications/cancelled', 'tools/call'):
            response = await self.server.handle({'jsonrpc': '2.0', 'method': method,
                                                'params': {'name': 'example', 'arguments': {'name': 'x'}}})
            self.assertIsNone(response)
        self.assertEqual(self.calls, [])

    async def test_unknown_method_and_tool_and_ping(self):
        self.assertEqual((await self.request('resources/list'))['error']['code'], -32601)
        self.assertEqual((await self.request('tools/call', {'name': 'unknown'}))['error']['code'], -32602)
        self.assertEqual((await self.request('ping'))['result'], {})

    async def test_unexpected_errors_redacted_expected_errors_actionable(self):
        secret = 'synthetic-secret-token'
        unexpected = await self.request('tools/call', {'name': 'failure', 'arguments': {'token': secret}})
        self.assertTrue(unexpected['result']['isError'])
        self.assertNotIn(secret, json.dumps(unexpected))
        self.assertNotIn('unexpected secret content', json.dumps(unexpected))
        expected = await self.request('tools/call', {'name': 'rejected', 'arguments': {'token': secret}})
        self.assertIn('request rejected', expected['result']['content'][0]['text'])
        self.assertNotIn(secret, json.dumps(expected))

    async def test_long_credential_is_redacted_before_error_truncation(self):
        secret = 'long-credential-' + 'sensitive-prefix-' * 40
        reply = await self.request('tools/call', {'name': 'rejected', 'arguments': {'token': secret}})
        message = reply['result']['content'][0]['text']
        self.assertIn('[redacted]', message)
        self.assertNotIn('long-credential', message)
        self.assertNotIn('sensitive-prefix', message)
        self.assertLessEqual(len(message), 300)

    async def test_lone_surrogate_request_id_does_not_break_stdio(self):
        raw = (b'{"jsonrpc":"2.0","id":"\\ud800","method":"ping"}\n'
               b'{"jsonrpc":"2.0","id":10,"method":"ping"}\n')
        output = io.BytesIO()
        await self.server.serve(io.BytesIO(raw), output)
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(replies[0]['id'], '\ud800')
        self.assertEqual(replies[0]['result'], {})
        self.assertEqual(replies[1], {'jsonrpc': '2.0', 'id': 10, 'result': {}})

    async def test_supported_versions_negotiate_tools_only(self):
        for version in PROTOCOLS:
            server = FastMCP('version-test')
            response = await server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                                            'params': {'protocolVersion': version}})
            self.assertEqual(response['result']['protocolVersion'], version)
            self.assertEqual(set(response['result']['capabilities']), {'tools'})

    async def test_bounded_frame_and_bad_json_recover_for_next_request(self):
        ping = b'{"jsonrpc":"2.0","id":9,"method":"ping"}\n'
        raw = (b'x' * (MAX_LINE_BYTES + 50) + b'\n' + b'{"bad":NaN}\n' + b'{"bad":1e999}\n' +
               b'[' * 2000 + b']' * 2000 + b'\n' + ping)
        out = io.BytesIO()
        await self.server.serve(io.BytesIO(raw), out)
        rows = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(rows[0]['error']['code'], -32600)
        self.assertTrue(all(row['error']['code'] == -32700 for row in rows[1:4]))
        self.assertEqual(rows[4], {'jsonrpc': '2.0', 'id': 9, 'result': {}})

    async def test_duplicate_keys_and_nonfinite_numbers_rejected(self):
        for raw in ('{"a":1,"a":2}', '{"a":Infinity}', '{"a":1e1000}'):
            with self.assertRaises(ValueError):
                decode(raw)


@unittest.skipUnless(os.environ.get('LOCAL_MODEL_LITE_INTEROP') == '1', 'opt-in read-only real MCP client test')
class RealSDKInterop(unittest.TestCase):
    def test_staged_server_via_sdk_and_connection_rss(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        import psutil
        from mcp.shared.exceptions import McpError
        stage = Path(__file__).resolve().parent
        # Pin the read-only proxy to the already-running installed broker. Never
        # call ensure_service: a missing broker must fail rather than start one.
        program = ("import sys;from pathlib import Path;sys.path.insert(0," + repr(str(stage)) + ");"
                   "import client;client.call=client.raw_call;"
                   "import mcp_server;mcp_server.mcp.run()")

        async def exercise():
            async with stdio_client(StdioServerParameters(command=sys.executable, args=['-c', program])) as (r, w):
                async with ClientSession(r, w) as session:
                    info = await session.initialize()
                    catalog = await session.list_tools()
                    self.assertEqual(info.protocolVersion, '2025-11-25')
                    self.assertEqual(len(catalog.tools), 11)
                    request = next(x for x in catalog.tools if x.name == 'local_models_request')
                    complete = next(x for x in catalog.tools if x.name == 'local_models_complete')
                    self.assertNotIn('task', request.inputSchema['required'])
                    self.assertEqual(complete.inputSchema['properties']['quality']['default'], 'unknown')
                    result = await session.call_tool('local_models_status', {})
                    self.assertFalse(result.isError)
                    from service import VERSION
                    self.assertEqual(result.structuredContent['version'], VERSION)
                    self.assertEqual(json.loads(result.content[0].text), result.structuredContent)
                    with self.assertRaises(McpError):
                        await session.call_tool('local_models_history', {'limit': True})
                    with self.assertRaises(McpError):
                        await session.call_tool('unknown-tool', {})
                    children = [p for p in psutil.Process().children(recursive=True)
                                if '-c' in p.cmdline() and program in p.cmdline()]
                    self.assertTrue(children)
                    processes = [{'pid': child.pid, 'rss_mib': round(child.memory_info().rss / 1024**2, 2)}
                                 for child in children]
                    rss = sum(child.memory_info().rss for child in children) / 1024**2
                    print(json.dumps({'lite_mcp_rss_mib': round(rss, 2), 'tools': len(catalog.tools),
                                      'processes': processes, 'protocol': info.protocolVersion, 'status_readonly_passed': True}))
                    # Includes Windows' venv redirector and actual interpreter;
                    # measured ~33 MiB, with room for minor runtime variation.
                    self.assertLess(rss, 45.)
        asyncio.run(exercise())


if __name__ == '__main__':
    unittest.main()
