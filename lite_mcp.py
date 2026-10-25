"""Small stdlib-only MCP stdio transport for this manager's fixed JSON tools.

Only initialize, ping, notifications and tools are supported and advertised.
Requests are serialized. Cancellation never interrupts model lifecycle work.
This is intentionally not a general replacement for the MCP Python SDK.
"""
import asyncio
import inspect
import json
import math
import sys
import types
from typing import Literal, Union, get_args, get_origin, get_type_hints

MAX_LINE_BYTES = 1024 * 1024
PROTOCOLS = ('2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05')


def encode(value):
    # Escaping also keeps legal JSON containing lone surrogate escapes writable
    # as UTF-8; clients still decode ordinary Chinese text without any loss.
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':'))


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite JSON number')
    return number


def _constant(value):
    raise ValueError('Nonfinite JSON constant')


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key')
        result[key] = value
    return result


def decode(value):
    depth, quoted, escaped = 0, False, False
    for char in value:
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '[{':
            depth += 1
            if depth > 64:
                raise ValueError('JSON nesting limit exceeded')
        elif char in ']}':
            depth -= 1
    return json.loads(value, parse_float=_finite_float, parse_constant=_constant, object_pairs_hook=_object)


def schema(annotation):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        return {'anyOf': [schema(item) for item in args]}
    if origin is Literal:
        result = {'enum': list(args)}
        kinds = {type(value) for value in args}
        if len(kinds) == 1:
            result.update(schema(next(iter(kinds))))
        return result
    if annotation is type(None):
        return {'type': 'null'}
    if annotation is dict or origin is dict:
        return {'type': 'object', 'additionalProperties': True}
    kind = {str: 'string', int: 'integer', bool: 'boolean', float: 'number'}.get(annotation)
    if kind:
        return {'type': kind}
    raise TypeError('Unsupported MCP tool annotation')


def matches(value, annotation):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        return any(matches(value, item) for item in args)
    if origin is Literal:
        return any(type(value) is type(item) and value == item for item in args)
    if annotation is dict or origin is dict:
        return type(value) is dict
    if annotation is float:
        return type(value) in (int, float) and math.isfinite(value)
    return type(value) is annotation


def error(request_id, code, message):
    return {'jsonrpc': '2.0', 'id': request_id, 'error': {'code': code, 'message': message}}


def rejected_message(exc, arguments):
    message = str(exc)
    # Broker validation errors are actionable. Still redact credentials even if
    # a future expected error accidentally incorporates one of these arguments.
    for key in ('token', 'api_key', 'authorization', 'password'):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            message = message.replace(value, '[redacted]')
    return message[:300] or 'Model manager rejected the operation'


class FastMCP:
    def __init__(self, name, instructions=''):
        self.name, self.instructions = name, instructions
        self.tools = {}
        self.initialized = False

    def tool(self, *, annotations=None):
        def register(function):
            name = function.__name__
            if name in self.tools:
                raise ValueError('Duplicate MCP tool name')
            signature = inspect.signature(function)
            hints = get_type_hints(function)
            properties, required = {}, []
            for key, parameter in signature.parameters.items():
                if parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
                    raise TypeError('MCP tools require named parameters')
                item = schema(hints[key])
                if parameter.default is inspect.Parameter.empty:
                    required.append(key)
                else:
                    item['default'] = parameter.default
                properties[key] = item
            spec = {'name': name, 'description': inspect.getdoc(function) or '',
                    'inputSchema': {'type': 'object', 'properties': properties, 'required': required,
                                    'additionalProperties': False},
                    'outputSchema': {'type': 'object', 'additionalProperties': True},
                    'annotations': dict(annotations or {})}
            self.tools[name] = (function, signature, hints, spec)
            return function
        return register

    async def handle(self, message):
        if (not isinstance(message, dict) or message.get('jsonrpc') != '2.0' or
                not isinstance(message.get('method'), str)):
            return error(None, -32600, 'Invalid JSON-RPC request')
        if 'id' not in message:
            # Notifications produce no response and never invoke a mutating tool.
            return None
        request_id = message['id']
        if type(request_id) not in (int, str):
            return error(None, -32600, 'Invalid request id')
        method, params = message['method'], message.get('params', {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return error(request_id, -32602, 'Params must be an object')
        if method == 'initialize':
            requested = params.get('protocolVersion')
            if not isinstance(requested, str):
                return error(request_id, -32602, 'protocolVersion must be a string')
            if self.initialized:
                return error(request_id, -32600, 'Already initialized')
            self.initialized = True
            result = {'protocolVersion': requested if requested in PROTOCOLS else PROTOCOLS[0],
                      'capabilities': {'tools': {'listChanged': False}},
                      'serverInfo': {'name': self.name, 'version': '1.1.1'}, 'instructions': self.instructions}
        elif method == 'ping':
            result = {}
        elif not self.initialized:
            return error(request_id, -32000, 'Initialize before requesting tools')
        elif method == 'tools/list':
            if params.get('cursor') is not None:
                return error(request_id, -32602, 'This tool list is not paginated')
            result = {'tools': [item[3] for item in self.tools.values()]}
        elif method == 'tools/call':
            name, arguments = params.get('name'), params.get('arguments', {})
            if not isinstance(name, str) or name not in self.tools:
                return error(request_id, -32602, 'Unknown tool')
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return error(request_id, -32602, 'Tool arguments must be an object')
            function, signature, hints, _ = self.tools[name]
            try:
                encode(arguments)  # Also protect programmatic handle() callers.
                bound = signature.bind(**arguments)
                bound.apply_defaults()
                for key, value in bound.arguments.items():
                    if not matches(value, hints[key]):
                        return error(request_id, -32602, 'Invalid argument type or enum: ' + key)
            except (TypeError, ValueError, OverflowError):
                return error(request_id, -32602, 'Missing, unknown or invalid tool arguments')
            try:
                value = function(**bound.arguments)
                if inspect.isawaitable(value):
                    value = await value
                if not isinstance(value, dict):
                    raise TypeError('Tool output must be an object')
                rendered = encode(value)
                result = {'content': [{'type': 'text', 'text': rendered}],
                          'structuredContent': value, 'isError': False}
            except (RuntimeError, ValueError) as exc:
                result = {'content': [{'type': 'text', 'text': rejected_message(exc, bound.arguments)}],
                          'isError': True}
            except Exception:
                # Never echo exception strings, arguments, tokens, or API keys.
                result = {'content': [{'type': 'text', 'text': 'Model manager operation failed or was rejected. Inspect status and request parameters.'}],
                          'isError': True}
        else:
            return error(request_id, -32601, 'Method not found')
        return {'jsonrpc': '2.0', 'id': request_id, 'result': result}

    async def serve(self, stdin=None, stdout=None):
        reader = stdin if stdin is not None else sys.stdin.buffer
        writer = stdout if stdout is not None else sys.stdout.buffer
        while True:
            line = await asyncio.to_thread(reader.readline, MAX_LINE_BYTES + 1)
            if not line:
                return
            if len(line) > MAX_LINE_BYTES:
                # Drain the frame in bounded chunks; never buffer the whole input.
                while line and not line.endswith(b'\n'):
                    line = await asyncio.to_thread(reader.readline, MAX_LINE_BYTES + 1)
                response = error(None, -32600, 'Request exceeds 1 MiB limit')
            else:
                try:
                    message = decode(line.decode('utf-8'))
                except (ValueError, UnicodeError, RecursionError):
                    response = error(None, -32700, 'Invalid JSON')
                else:
                    try:
                        response = await self.handle(message)
                    except Exception:
                        response = error(message.get('id') if isinstance(message, dict) else None,
                                         -32603, 'Internal protocol error')
            if response is not None:
                writer.write((encode(response) + '\n').encode('utf-8'))
                writer.flush()

    def run(self, transport='stdio'):
        if transport != 'stdio':
            raise ValueError('Only local stdio MCP is supported')
        try:
            asyncio.run(self.serve())
        except (BrokenPipeError, KeyboardInterrupt):
            pass
