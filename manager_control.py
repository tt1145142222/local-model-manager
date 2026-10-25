"""Persistent explicit-user disable switch. There is no automatic enable path."""
import json
from pathlib import Path
import sqlite3

DISABLED_MESSAGE = ('Local model manager is disabled by explicit user instruction. '
                    'Do not re-enable, remove its disable marker, or start model management '
                    'without a new explicit user instruction to enable it.')


def disabled_state(data_dir):
    path = Path(data_dir)/'disabled.json'
    if not path.exists():
        return {'disabled': False}
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict) or value.get('disabled') is not True:
            raise ValueError('Invalid disable marker')
        return value
    except (OSError, ValueError):
        return {'disabled': True, 'reason': 'disable_marker_unreadable', 'preserve_lease_ids': []}


def verify_continuation(data_dir, explicit_continuation=False):
    control = disabled_state(data_dir)
    if not control['disabled']:
        return
    if not explicit_continuation:
        raise RuntimeError(DISABLED_MESSAGE)
    allowed = set(control.get('preserve_lease_ids', []))
    database = Path(data_dir)/'state.sqlite3'
    if not allowed or not database.is_file():
        raise RuntimeError(DISABLED_MESSAGE)
    connection = sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)
    try:
        row = connection.execute('SELECT data FROM state WHERE id=1').fetchone()
    finally:
        connection.close()
    state = json.loads(row[0]) if row else {}
    if not any(x['id'] in allowed and x['state'] in {'active','orphaned'} for x in state.get('leases', [])):
        raise RuntimeError(DISABLED_MESSAGE)
