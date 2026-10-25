"""Proposed adapter: short-lived private worker, no network endpoint or shell."""
import os
from pathlib import Path
import subprocess
import time
import uuid
import psutil
from adapters import Adapter, process_identity, same_process
from resources import NO_WINDOW
from worker_protocol import read, write, sha, digest, queue_lock


def make_adapter(model_id, config):
    return TransformersWorkerAdapter(model_id, config) if config['kind'] == 'transformers_worker' else Adapter(model_id, config)


class TransformersWorkerAdapter(Adapter):
    def __init__(self, model_id, config, *, fake_cpu=False):
        super().__init__(model_id, config)
        self.fake_cpu = fake_cpu  # Test-only constructor, never part of registered manifests.
        self.state_root = self.root / 'work' / 'qwen2-audio'
        self.pointer = self.state_root / 'active-session.json'
        self._child = None

    def _session(self):
        record = read(self.pointer)
        session = Path(record['session']).resolve()
        if not session.is_relative_to(self.state_root.resolve()):
            raise ValueError('Invalid session directory')
        return record, session

    def inspect(self):
        if not self.pointer.exists():
            return {'state': 'stopped', 'busy': False, 'identity': None}
        try:
            record, session = self._session()
            identity = record['identity']
            if not same_process(identity):
                return {'state': 'stopped', 'busy': False, 'identity': None}
            p = psutil.Process(identity['pid'])
            argv = p.cmdline()
            if (str(Path(self.c['worker_path']).resolve()) not in argv or '--session' not in argv or
                    Path(argv[argv.index('--session') + 1]).resolve() != session or
                    record['config_sha256'] != digest(self.c) or
                    sha(self.c['worker_path']) != self.c['worker_sha256'] or
                    sha(Path(self.c['worker_path']).parent / 'worker_protocol.py') != self.c['protocol_sha256'] or
                    sha(self.c['model_receipt']) != self.c['model_receipt_sha256']):
                raise ValueError('Process/config/script/receipt identity mismatch')
            state = read(session / 'status.json')
            if state['nonce'] != record['nonce'] or state['identity'] != identity:
                raise ValueError('Worker session identity mismatch')
            if not 0 <= time.time() - state['heartbeat'] < 10:
                raise ValueError('Worker heartbeat is stale')
            if state['state'] != 'ready' or type(state['busy']) is not bool:
                raise ValueError('Worker is not verified ready')
            if state['config_sha256'] != record['config_sha256']:
                raise ValueError('Worker loaded another config')
            pending = len(list((session / 'jobs').glob('*.json')))
            result = {'state': 'ready', 'busy': state['busy'] or bool(pending), 'identity': identity,
                      'rss_gib': p.memory_info().rss / 1024**3,
                      'private_commit_gib': getattr(p.memory_info(), 'private', None),
                      'running': state.get('work_id'), 'transport': 'file_queue',
                      'pending': pending, 'phase': state.get('phase')}
            if result['private_commit_gib'] is not None:
                result['private_commit_gib'] /= 1024**3
            result.update(self._process_telemetry(p, identity))
            return result
        except Exception as exc:
            return {'state': 'unknown', 'busy': None, 'identity': locals().get('identity'),
                    'reason': type(exc).__name__ + ': worker state could not be verified'}

    def start(self):
        if self.inspect()['state'] != 'stopped':
            raise RuntimeError('Start requires verified stopped worker')
        # Do not execute modified code even if the prior process has exited.
        if (sha(self.c['worker_path']) != self.c['worker_sha256'] or sha(self.c['model_receipt']) != self.c['model_receipt_sha256']
                or sha(Path(self.c['worker_path']).parent / 'worker_protocol.py') != self.c['protocol_sha256']):
            raise RuntimeError('Pinned worker/receipt changed')
        nonce = uuid.uuid4().hex
        session = (self.state_root / nonce).resolve()
        session.mkdir(parents=True)
        write(session / 'config.json', self.c)
        argv = [self.c['exe'], '-B', self.c['worker_path'], '--session', str(session), '--nonce', nonce]
        if self.fake_cpu:
            argv.append('--fake-cpu')
        else:
            argv.append('--gpu-execute')
        with (session / 'stdout.log').open('ab') as out, (session / 'stderr.log').open('ab') as err:
            child = subprocess.Popen(argv, cwd=self.root, stdout=out, stderr=err,
                                     stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
        self._child = child
        # Windows venv python.exe may be a redirector. Verify the actual worker
        # is this exact launch or its child; never adopt a matching process name.
        boot_deadline = time.monotonic() + 10
        while not (session / 'boot.json').exists() and time.monotonic() < boot_deadline:
            if child.poll() is not None:
                raise RuntimeError('Worker bootstrap exited before identity handshake')
            time.sleep(.02)
        boot = read(session / 'boot.json')
        process = psutil.Process(boot['pid'])
        if (boot.get('nonce') != nonce or not same_process(boot) or
                (process.pid != child.pid and child.pid not in [p.pid for p in process.parents()]) or
                self.c['worker_path'] not in process.cmdline()):
            raise RuntimeError('Worker bootstrap does not belong to this launch')
        identity = process_identity(process)
        identity.update(nonce=nonce, worker_sha256=self.c['worker_sha256'],
                        model_receipt_sha256=self.c['model_receipt_sha256'])
        record = {'identity': identity, 'nonce': nonce, 'session': str(session), 'config_sha256': digest(self.c)}
        write(session / 'launch.json', record)
        write(self.pointer, record)
        deadline = time.monotonic() + (10 if self.fake_cpu else 240)
        while time.monotonic() < deadline:
            state = self.inspect()
            if state['state'] == 'ready' and state['identity'] == identity:
                return identity
            if child.poll() is not None:
                raise RuntimeError('Worker exited before ready; retained local logs')
            time.sleep(.05 if self.fake_cpu else .5)
        raise RuntimeError('Worker startup timed out; retained for identity-safe inspection')

    def stop(self, identity):
        for _ in range(2):
            state = self.inspect()
            if state['state'] == 'stopped':
                return state
            if state['state'] != 'ready' or state['busy'] is not False or state['identity'] != identity:
                raise RuntimeError('Worker changed or busy/unknown; stop refused')
            time.sleep(.05)
        if not same_process(identity):
            raise RuntimeError('Process identity changed')
        record, session = self._session()
        # Worker acknowledges this only while idle. No terminate/kill fallback.
        request_id = uuid.uuid4().hex
        with queue_lock(session):
            state = read(session / 'status.json')
            if state.get('phase') != 'idle' or any((session / 'jobs').glob('*.json')):
                raise RuntimeError('Worker has claimed/pending work; stop refused')
            write(session / 'stop.json', {'nonce': record['nonce'], 'request_id': request_id})
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if not same_process(identity):
                state = self.inspect()
                if state['state'] != 'stopped':
                    raise RuntimeError('Worker stop verification failed')
                if self._child is not None:
                    self._child.wait(timeout=5)
                    self._child = None
                return state
            if ((session / 'stop-refused.json').exists()
                    and read(session / 'stop-refused.json').get('request_id') == request_id):
                raise RuntimeError('Worker became busy; stop refused without interrupting work')
            time.sleep(.05)
        raise RuntimeError('Worker did not stop; retained without force termination')
