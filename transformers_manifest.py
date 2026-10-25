"""Strict opt-in registration for this fixed Qwen2-Audio file-worker protocol."""
import math
import re
from pathlib import Path
from worker_protocol import read, sha

FIELDS = {'label', 'kind', 'workload_kind', 'transport', 'root', 'exe', 'worker_path',
          'worker_sha256', 'protocol_sha256', 'model_dir', 'model_receipt', 'model_receipt_sha256',
          'context_capacity', 'startup_free_ram_gib', 'startup_commit_free_gib',
          'startup_free_vram_gib'}
REVISION = '0a095220c30b7b31434169c3086508ef3ea5bf0a'


def validate_transformers_manifest(model_id, manifest):
    if not re.fullmatch('[a-z][a-z0-9_-]{0,63}', model_id) or model_id in {'h3', 'muse'}:
        raise ValueError('Invalid or reserved model ID')
    if set(manifest) != FIELDS:
        raise ValueError('Unexpected/missing transformers worker fields')
    m = dict(manifest)
    if (m['kind'], m['workload_kind'], m['transport']) != ('transformers_worker', 'llm', 'file_queue'):
        raise ValueError('Incorrect worker protocol')
    if not isinstance(m['label'], str) or not m['label'].strip() or len(m['label']) > 128:
        raise ValueError('Invalid label')
    for field in ['root', 'model_dir', 'exe', 'worker_path', 'model_receipt']:
        p = Path(m[field])
        if not p.is_absolute():
            raise ValueError(field + ' must be absolute')
        p = p.resolve(strict=True)
        if not (p.is_dir() if field in {'root', 'model_dir'} else p.is_file()):
            raise ValueError(field + ' has wrong file type')
        m[field] = str(p)
    if Path(m['exe']).name.lower() not in {'python.exe', 'python', 'python3'}:
        raise ValueError('Only a pinned Python interpreter is supported')
    if not Path(m['worker_path']).is_relative_to(Path(m['root'])):
        raise ValueError('Worker must be installed under its registered root')
    if (sha(m['worker_path']) != m['worker_sha256'] or sha(m['model_receipt']) != m['model_receipt_sha256']
            or sha(Path(m['worker_path']).parent / 'worker_protocol.py') != m['protocol_sha256']):
        raise ValueError('Worker or model receipt identity changed')
    receipt = read(m['model_receipt'])
    if receipt.get('revision') != REVISION or receipt.get('status') != 'downloaded_and_hash_verified':
        raise ValueError('Pinned verified Qwen2-Audio download receipt required')
    for item in receipt['files']:
        target = (Path(m['model_dir']) / item['name']).resolve()
        if not target.is_relative_to(Path(m['model_dir'])) or not target.is_file():
            raise ValueError('Model file missing/escapes model directory')
        if target.stat().st_size != item['size'] or not re.fullmatch('[a-f0-9]{64}', item['sha256']):
            raise ValueError('Model file identity metadata invalid')
    config = read(Path(m['model_dir']) / 'config.json')
    capacity = config.get('text_config', {}).get('max_position_embeddings')
    if type(m['context_capacity']) is not int or m['context_capacity'] != capacity or capacity != 8192:
        raise ValueError('Context capacity must match installed Qwen2-Audio config')
    for field in ['startup_free_ram_gib', 'startup_commit_free_gib', 'startup_free_vram_gib']:
        value = m[field]
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid conservative startup budget')
    return m
