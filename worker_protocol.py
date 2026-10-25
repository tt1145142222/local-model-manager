"""Local file queue protocol. No model imports or installed-manager mutations."""
import hashlib
import json
import os
from pathlib import Path
import uuid
import time
from contextlib import contextmanager


def read(path):
    data = Path(path).read_bytes()
    if len(data) > 2 * 1024 * 1024:
        raise ValueError('Protocol JSON exceeds size limit')
    return json.loads(data.decode('utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@contextmanager
def queue_lock(session):
    """OS releases this advisory lock if its process exits; no stale lock cleanup."""
    path = Path(session) / 'queue.lock'
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0'); stream.flush()
        deadline = time.monotonic() + 5
        while True:
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Queue protocol lock timed out')
                time.sleep(.01)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
