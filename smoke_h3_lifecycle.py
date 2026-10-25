"""H3 server-only startup/stop smoke; no model weights or video are submitted."""
import json
from pathlib import Path
import time
from adapters import Adapter
from client import call


def main():
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    h3 = Adapter('h3', config['models']['h3'])
    muse = Adapter('muse', config['models']['muse'])
    before_muse = muse.inspect()
    assert not call('status')['leases'], 'Another task owns the manager'
    assert h3.inspect()['state'] == 'stopped', 'H3 already running; do not disturb it'
    start = time.time()
    identity = h3.start()
    started_seconds = time.time()-start
    try:
        healthy = h3.inspect()
        assert healthy['state'] == 'ready' and healthy['busy'] is False
        assert muse.inspect()['identity'] == before_muse['identity']
    finally:
        h3.stop(identity)
    result = {'h3_start_ready': True, 'h3_idle_queue_verified': True, 'h3_identity_stop_verified': True,
              'h3_after': h3.inspect()['state'], 'startup_seconds': round(started_seconds, 2),
              'muse_process_preserved': muse.inspect()['identity'] == before_muse['identity'],
              'video_submitted': False, 'timestamp': time.time()}
    p = Path(__file__).parent/'work/verification/live-h3.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
