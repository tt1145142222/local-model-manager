"""Read-only API boundary and concurrent-client singleton checks."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib import request, error
from client import call, raw_call


def main():
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    token = (Path(__file__).parent/'work/state/api-token').read_text().strip()
    url = f"http://127.0.0.1:{cfg['port']}/rpc"
    checks = []
    for name, headers, body, expected in [
        ('missing_auth', {}, b'{"method":"status"}', 403),
        ('cross_origin', {'Authorization': 'Bearer '+token, 'Origin': 'https://example.com'}, b'{"method":"status"}', 403),
        ('unknown_method', {'Authorization': 'Bearer '+token}, b'{"method":"execute_shell"}', 400),
        ('malformed_json', {'Authorization': 'Bearer '+token}, b'{', 400),
    ]:
        try:
            request.build_opener(request.ProxyHandler({})).open(request.Request(url, data=body, headers=headers), timeout=5)
            raise AssertionError(name+' unexpectedly accepted')
        except error.HTTPError as exc:
            assert exc.code == expected, (name, exc.code)
            checks.append(name)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: call('health'), range(8)))
    assert len({x['pid'] for x in results}) == 1
    result = {'rejection_checks_passed': checks, 'concurrent_clients': 8, 'singleton_pid': results[0]['pid'],
              'single_shared_manager': True}
    p = Path(__file__).parent/'work/verification/api-boundaries.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
