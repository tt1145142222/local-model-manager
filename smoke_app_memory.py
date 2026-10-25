"""Native smoke test: trim only a test child created here, then verify its data."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
from app_memory import WindowsApps, MIB


def child():
    # This temporary test process has no user document and never opens a window.
    data = bytearray(b'Z' * (128*MIB))
    import hashlib
    print(hashlib.sha256(data).hexdigest(), flush=True)
    for line in sys.stdin:
        if line.strip() == 'check':
            print(hashlib.sha256(data).hexdigest(), flush=True)
        elif line.strip() == 'quit':
            return


def main():
    backend = WindowsApps()
    with subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--child'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW) as process:
        try:
            expected = process.stdout.readline().strip()
            assert len(expected) == 64, 'Test child did not initialize'
            launcher = psutil.Process(process.pid)
            # Windows venv python.exe is a small redirector. The allocation is
            # held by its interpreter child; restrict selection to this test tree.
            p = max([launcher] + launcher.children(recursive=True), key=lambda item: item.memory_info().rss)
            identity = {'pid': p.pid, 'created': p.create_time(), 'exe': p.exe()}
            before = p.memory_info()
            # Identity mismatch must be rejected on the actual opened OS handle.
            try:
                backend.trim_one(dict(identity, created=identity['created']-100), lambda: True)
                raise AssertionError('Stale identity unexpectedly accepted')
            except OSError:
                pass
            assert backend.trim_one(identity, lambda: False) is None
            result = backend.trim_one(identity, lambda: True)
            after = p.memory_info()
            assert process.poll() is None, 'Child exited after trim'
            assert result['working_set_reduction_mib'] > 64, f'Native trim result: {result}; before={before}; after={after}'
            assert abs(after.private-before.private) < MIB, 'Unexpected private commit change'
            process.stdin.write('check\n')
            process.stdin.flush()
            actual = process.stdout.readline().strip()
            assert actual == expected, 'Data changed after reclaim and reload'
            assert process.poll() is None
            output = {'test_child_only': True, 'pid': p.pid, 'rss_before_mib': round(before.rss/MIB, 2),
                      'rss_after_mib': round(after.rss/MIB, 2),
                      'private_commit_before_mib': round(before.private/MIB, 2),
                      'private_commit_after_mib': round(after.private/MIB, 2),
                      'data_hash_unchanged': True, 'child_alive_after_reclaim_and_reload': True,
                      'stale_identity_rejected': True, 'cancelled_permission_skipped': True}
            print(json.dumps(output, ensure_ascii=False))
        finally:
            if process.poll() is None:
                process.stdin.write('quit\n')
                process.stdin.flush()
                process.wait(timeout=10)


if __name__ == '__main__':
    child() if '--child' in sys.argv else main()
