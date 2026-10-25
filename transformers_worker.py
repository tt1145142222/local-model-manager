"""PROPOSAL: one batch-owned Qwen2-Audio worker; GPU requires explicit flag.

An explicit fake CPU mode exists only for isolated lifecycle tests. Production
jobs verify the real manager lease and perform prepare/complete per inference.
"""
import argparse
import gc
import hashlib
import math
import os
from pathlib import Path
import threading
import time
from worker_protocol import read, write, sha, digest, queue_lock


class QwenAudioEngine:
    def __init__(self, config):
        # Full model-byte identity check happens before any model load.
        receipt = read(config['model_receipt'])
        for item in receipt['files']:
            if sha(Path(config['model_dir']) / item['name']) != item['sha256']:
                raise RuntimeError('Model bytes do not match pinned download receipt')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        torch.set_num_threads(6)
        torch.manual_seed(20261019)
        self.processor = AutoProcessor.from_pretrained(config['model_dir'], local_files_only=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            config['model_dir'], device_map='cuda:0', dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True).eval()
        self.config = config

    def prepare(self, job):
        import numpy as np
        import soundfile as sf
        import torch
        from scipy.signal import resample_poly
        if not 1 <= len(job['audios']) <= 2 or not isinstance(job['question'], str) or not job['question'].strip():
            raise ValueError('One or two waveform windows and a concrete question required')
        content, waves, evidence = [], [], []
        for index, source in enumerate(job['audios']):
            info = sf.info(source['path'])
            start, end = float(source.get('start', 0)), float(source.get('end', info.duration))
            if not all(math.isfinite(x) for x in [start, end]) or not 0 <= start < end <= info.duration + .001 or end-start > 25.0001:
                raise ValueError('Invalid audio window; limit is 25 seconds')
            wave, sr = sf.read(source['path'], dtype='float32', always_2d=True,
                               start=round(start * info.samplerate), stop=round(end * info.samplerate))
            wave = wave.mean(axis=1)
            if sr != 16000:
                divisor = math.gcd(sr, 16000)
                wave = resample_poly(wave, 16000//divisor, sr//divisor).astype(np.float32)
            if not len(wave) or not np.isfinite(wave).all():
                raise ValueError('Invalid waveform')
            content += [{'type': 'text', 'text': 'Audio ' + str(index + 1) + ':'},
                        {'type': 'audio', 'audio_url': source['path']}]
            waves.append(wave)
            evidence.append({'source': source, 'source_sha256': sha(source['path']),
                             'samples': len(wave), 'sample_rate': 16000,
                             'waveform_sha256': hashlib.sha256(wave.tobytes()).hexdigest()})
        content.append({'type': 'text', 'text': job['question']})
        conversation = [{'role': 'system', 'content': 'Analyze only audible evidence. If uncertain say uncertain. Do not infer from filenames.'},
                        {'role': 'user', 'content': content}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt, audio=waves, sampling_rate=16000, return_tensors='pt', padding=True)
        if 'input_features' not in inputs or 'feature_attention_mask' not in inputs:
            raise RuntimeError('Refuse text-only fallback')
        features = inputs.input_features
        tokens = int((inputs.input_ids == self.processor.audio_token_id).sum())
        if features.shape[0] != len(waves) or not torch.isfinite(features).all() or tokens <= len(waves) or not bool(inputs.feature_attention_mask.any()):
            raise RuntimeError('Audio feature/token validation failed')
        count = int(inputs.input_ids.shape[1])
        if count + 384 > self.config['context_capacity']:
            raise ValueError('Context overflow; never truncate the audio silently')
        task = {'input_tokens': count, 'max_output_tokens': 384,
                'context_tokens': self.config['context_capacity'],
                'workflow_signature': digest({'engine': 'qwen2-audio', 'receipt': self.config['model_receipt_sha256'],
                                             'dtype': 'bfloat16', 'attention': 'sdpa', 'analysis_rate': 16000,
                                             'audio_shapes': [len(w) for w in waves], 'sample': False, 'output': 384})}
        proof = {'waveforms': evidence, 'features_shape': list(features.shape), 'expanded_audio_tokens': tokens,
                 'feature_sha256': hashlib.sha256(features.numpy().tobytes()).hexdigest(), 'input_tokens': count}
        return {'inputs': inputs, 'task': task, 'evidence': proof}

    def infer(self, prepared):
        import torch
        inputs = prepared['inputs'].to('cuda:0')
        try:
            with torch.inference_mode():
                ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=384)
            answer = self.processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0]
            return {'raw_answer': answer, 'input_evidence': prepared['evidence'],
                    'human_listening': False, 'quality': 'unknown'}
        finally:
            prepared['inputs'] = None

    def close(self):
        import torch
        self.model = self.processor = None
        gc.collect()
        torch.cuda.empty_cache()


class FakeCpuEngine:
    def prepare(self, job):
        time.sleep(job.get('fake_prepare_delay', 0))
        return {'task': {'input_tokens': 12, 'max_output_tokens': 384, 'context_tokens': 8192,
                         'workflow_signature': 'fake-cpu-lifecycle-only'}, 'job': job}

    def infer(self, prepared):
        gate = prepared['job'].get('fake_wait_file')
        deadline = time.monotonic() + 10
        while gate and not Path(gate).exists() and time.monotonic() < deadline:
            time.sleep(.02)
        time.sleep(prepared['job'].get('fake_delay', .05))
        if prepared['job'].get('fake_fail'):
            raise RuntimeError('Injected CPU worker error')
        return {'fake_cpu': True, 'quality': 'unknown', 'human_listening': False}

    def close(self):
        pass


def execute_item(engine, job, manager, publish_busy):
    """Single inference contract; manager is injectable for CPU tests."""
    cred = job['credentials']
    work_id = job['id']
    prepared = engine.prepare(job)
    admission = manager('prepare_work', {**cred, 'work_id': work_id, 'task': prepared['task']})
    if admission.get('proceed') is not True:
        return {'state': 'not_submitted', 'reason': admission.get('reason', 'No admission'),
                'retryable': admission.get('retryable', True)}
    publish_busy(True, work_id)
    outcome, result = 'completed', None
    try:
        result = engine.infer(prepared)
    except Exception as exc:
        outcome = 'oom' if 'out of memory' in str(exc).lower() else 'failed'
        result = {'error_type': type(exc).__name__, 'error': str(exc)[:500]}
    finally:
        publish_busy(False, None)  # complete re-inspects native idle state.
    completion = manager('complete_work', {**cred, 'work_id': work_id, 'outcome': outcome,
                                          'quality': 'unknown', 'desktop_responsive': None})
    return {'state': 'completed' if outcome == 'completed' else 'failed', 'outcome': outcome,
            'quality': 'unknown', 'result': result, 'completion_recorded': bool(completion)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--nonce', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fake-cpu', action='store_true')
    mode.add_argument('--gpu-execute', action='store_true')
    args = parser.parse_args()
    session = args.session.resolve()
    config = read(session / 'config.json')
    import psutil
    process = psutil.Process()
    write(session / 'boot.json', {'pid': process.pid, 'created': process.create_time(),
                                 'exe': str(Path(process.exe()).resolve()), 'nonce': args.nonce})
    deadline = time.monotonic() + 5
    while not (session / 'launch.json').exists() and time.monotonic() < deadline:
        time.sleep(.01)
    launch = read(session / 'launch.json')
    if launch['nonce'] != args.nonce or launch['identity']['pid'] != os.getpid() or launch['config_sha256'] != digest(config):
        raise ValueError('Launch identity mismatch')
    lock = threading.Lock()
    state = {'state': 'loading', 'busy': True, 'phase': 'loading', 'accepting': False,
             'work_id': None, 'nonce': args.nonce,
             'identity': launch['identity'], 'config_sha256': digest(config), 'heartbeat': time.time()}
    stopping = threading.Event()

    def publish(busy=None, work_id=None):
        with lock:
            if busy is not None:
                state.update(busy=busy, work_id=work_id, phase='inference' if busy else 'idle')
            state['heartbeat'] = time.time()
            write(session / 'status.json', state)

    def heartbeat():
        while not stopping.wait(.2):
            publish()
    pulse = threading.Thread(target=heartbeat, daemon=True)
    pulse.start()
    engine = None
    try:
        engine = FakeCpuEngine() if args.fake_cpu else QwenAudioEngine(config)
        state['state'] = 'ready'
        state['accepting'] = True
        publish(False, None)
        if args.fake_cpu:
            def manager(method, params):
                return {'proceed': True, 'test_only': True}
        else:
            from client import raw_call  # Installed-manager import; never auto-start another broker.
            manager = raw_call
        (session / 'jobs').mkdir(exist_ok=True)
        while True:
            path = None
            with queue_lock(session):
                pending = sorted((session / 'jobs').glob('*.json'))
                if (session / 'stop.json').exists():
                    request = read(session / 'stop.json')
                    if request.get('nonce') != args.nonce:
                        raise ValueError('Stop request nonce mismatch')
                    if pending:
                        write(session / 'stop-refused.json', request)
                        (session / 'stop.json').unlink()
                    else:
                        state.update(accepting=False, phase='stopping')
                        publish()
                        break
                if pending:
                    path = pending[0]
                    claimed = session / ('claimed-' + path.name)
                    path.replace(claimed)
                    path = claimed
                    state['phase'] = 'admitting'
                    publish()
            if path is None:
                time.sleep(.05)
                continue
            job = read(path)
            result_path = session / 'results' / path.name.removeprefix('claimed-')
            if result_path.exists():
                path.unlink()  # Idempotent receipt, never repeat inference.
                publish(False, None)
                continue
            if job.get('nonce') != args.nonce:
                raise ValueError('Job nonce mismatch')
            try:
                if not args.fake_cpu:
                    lease = manager('lease_status', job['credentials'])
                    if lease.get('identity') != launch['identity'] or not lease.get('proceed'):
                        raise ValueError('Lease no longer owns this ready worker')
                result = execute_item(engine, job, manager, publish)
                publish(False, None)
                write(result_path, {'id': job['id'], 'job_sha256': digest({k:v for k,v in job.items() if k!='credentials'}), **result})
            except Exception as exc:
                publish(False, None)
                write(result_path, {'id': job.get('id'), 'state': 'unknown', 'error_type': type(exc).__name__,
                                    'error': str(exc)[:500], 'retry': 'inspect existing work before resubmission'})
            path.unlink()
    finally:
        if engine:
            engine.close()
        state['state'] = 'stopping'
        publish(False, None)
        stopping.set()
        pulse.join(timeout=1)


if __name__ == '__main__':
    main()
