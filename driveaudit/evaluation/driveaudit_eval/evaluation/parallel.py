"""Bounded, process-isolated evaluation stages using the existing evaluators."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from driveaudit_eval.utils.io import jsonable, write_json

PRIMARY = ('video', 'trajectory', 'consistency', 'agent')
CPU_CONSISTENCY = {'mtpc', 'tcvv', 'tcvv_bench', 'tcvv_h', 'tcvv_o'}
DEPENDENCY_OK = {'completed', 'complete', 'valid', 'available', 'partial', 'ready'}


def _resources(options):
    workers, threads = options.get('cpu_workers', 2), options.get('threads_per_worker', 2)
    for name, value in [('cpu_workers', workers), ('threads_per_worker', threads)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('execution.' + name + ' must be a positive integer')
    devices = options.get('gpu_devices', [])
    if not isinstance(devices, list):
        raise ValueError('execution.gpu_devices must be a list')
    devices = [str(value) for value in devices]
    if len(set(devices)) != len(devices) or any(not d or ',' in d or d == '-1' for d in devices):
        raise ValueError('execution.gpu_devices must contain unique individual GPU IDs')
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices and inherited is not None:
        allowed = {v.strip() for v in inherited.split(',') if v.strip() and v.strip() != '-1'}
        if not set(devices).issubset(allowed):
            raise ValueError('execution.gpu_devices must stay within inherited CUDA_VISIBLE_DEVICES')
    # No explicit split: one exclusive slot preserves the inherited allocation.
    return workers, threads, devices or [inherited]


def _plan(config, section):
    if section not in (*PRIMARY, 'all'):
        raise ValueError('Unknown evaluation section: ' + section)
    settings = config['evaluation']
    plan, disabled = {}, {}
    for name in PRIMARY:
        cfg = settings.get(name, {'enabled': False})
        if not isinstance(cfg, dict):
            raise ValueError(name + ' settings must be a mapping')
        if not cfg.get('enabled', True) or section not in ('all', name):
            disabled[name] = dict(status='disabled', reason='disabled by config or section selection')
            continue
        gpu = name in ('video', 'agent')
        if name == 'consistency':
            active = {key for key, value in cfg.get('backends', {}).items() if value.get('enabled', False)}
            gpu = not (cfg.get('mode') == 'input_gt' and active <= CPU_CONSISTENCY)
            scene_workers = cfg.get('scene_workers', 1)
            if isinstance(scene_workers, bool) or not isinstance(scene_workers, int) or scene_workers < 1:
                raise ValueError('consistency.scene_workers must be a positive integer')
            if gpu and scene_workers > 1:
                raise ValueError('GPU consistency requires scene_workers=1')
            if not gpu and scene_workers > config.get('execution', {}).get('cpu_workers', 2):
                raise ValueError('consistency.scene_workers exceeds execution.cpu_workers')
        cpu_slots = cfg.get('scene_workers', 1) if name == 'consistency' and not gpu else int(not gpu)
        plan[name] = dict(resource='gpu' if gpu else 'cpu', cpu_slots=cpu_slots, dependencies=[])
    if section in ('all', 'trajectory', 'agent') and config.get('traffic_compliance', {}).get('enabled', False):
        deps = [name for name in ('trajectory', 'agent') if name in plan]
        plan['traffic_and_responsibility'] = dict(resource='cpu', cpu_slots=1, dependencies=deps)
    if section in ('all', 'agent') and config.get('episode_rollout', {}).get('enabled', False):
        plan['episode_rollout'] = dict(resource='cpu', cpu_slots=1, dependencies=['agent'])
    core = settings.get('core_metrics', {})
    if (section in ('all', 'trajectory') and core.get('enabled', settings.get('profile', 'legacy') != 'legacy')
            and core.get('trs', {}).get('fresh_population', False)):
        plan['population_trs'] = dict(resource='cpu', cpu_slots=1, dependencies=['trajectory'])
    return plan, disabled


def _command(job):
    return [sys.executable, '-m', 'driveaudit_eval.evaluation.parallel', '--worker', str(job)]


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)


def _stop(processes):
    """Terminate only sessions created by this scheduler, including descendants."""
    groups = [proc.pid for proc in processes]
    for pid in groups:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 2
    while groups and time.monotonic() < deadline:
        live = []
        for pid in groups:
            try:
                os.killpg(pid, 0)
                live.append(pid)
            except ProcessLookupError:
                pass
        groups = live
        if groups:
            time.sleep(.05)
    for pid in groups:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for proc in processes:
        proc.wait()


def _publish(work, output, inputs):
    reserved = {'job.json', 'result.json', 'stage.log', 'resolved_config.yaml', *inputs}
    artifacts = [path for path in sorted(work.iterdir()) if path.name not in reserved]
    for path in artifacts:
        destination = output/path.name
        if destination.exists() or destination.is_symlink():
            raise ValueError('Stage artifact would overwrite another stage: ' + str(destination))
    for path in artifacts:
        (output/path.name).symlink_to(os.path.relpath(path, output), target_is_directory=path.is_dir())


def run_stages(config, context, section='all', dry_run=False):
    """Overlap ready stages. SIGTERM cleanup requires a main-thread caller."""
    options = config.get('execution', {})
    cpu_workers, threads, devices = _resources(options)
    plan, summary = _plan(config, section)
    output, root = Path(context['output_dir']).resolve(), Path(context['root']).resolve()
    logger = context.get('logger') or logging.getLogger(__name__)
    config_hash = hashlib.sha256(json.dumps(jsonable(config), sort_keys=True).encode()).hexdigest()
    timeline = dict(schema='driveaudit_eval_stage_execution_v1', config_sha256=config_hash,
                    cpu_workers=cpu_workers, threads_per_worker=threads, gpu_devices=devices,
                    plan=plan, stages=[], started_at_unix_s=time.time(), status='running')
    timeline_path = output/'execution_timeline.json'
    output.mkdir(parents=True, exist_ok=True)
    if dry_run:
        timeline['status'] = 'dry_run'
        write_json(timeline_path, timeline)
        return dict(summary, **{name: dict(status='dry_run', **spec) for name, spec in plan.items()})
    for name, result in summary.items():
        write_json(output/(name+'.json'), result)
    pending, running = dict(plan), {}
    main_thread = threading.current_thread() is threading.main_thread()
    previous_sigterm = signal.getsignal(signal.SIGTERM) if main_thread else None
    timeline['sigterm_cleanup_installed'] = main_thread
    write_json(timeline_path, timeline)
    try:
        if main_thread:
            signal.signal(signal.SIGTERM, _terminate)
        while pending or running:
            progressed = False
            for name, state in list(running.items()):
                proc = state['process']
                if proc.poll() is None:
                    continue
                # A returned worker cannot leave a model service owning its slot.
                _stop([proc])
                state['log'].close()
                receipt, work = state['receipt'], state['work']
                try:
                    if proc.returncode:
                        raise RuntimeError('Worker exited with code ' + str(proc.returncode))
                    result_file = work/'result.json'
                    result = json.loads(result_file.read_text())
                    if not isinstance(result, dict) or not isinstance(result.get('status'), str):
                        raise ValueError('Worker did not return an evaluator result with status')
                    result['status'] = result['status'].lower()
                    receipt['result_sha256'] = hashlib.sha256(result_file.read_bytes()).hexdigest()
                    _publish(work, output, state['inputs'])
                except Exception as exc:
                    result = dict(status='failed', reason=f'{type(exc).__name__}: {exc}',
                                  log=str(work/'stage.log'))
                receipt.update(finished_at_unix_s=time.time(), status=result['status'],
                               duration_s=time.monotonic()-state['started'], returncode=proc.returncode)
                summary[name] = result
                write_json(output/(name+'.json'), result)
                logger.info('[%s] %s: %s', name.upper(), result['status'], result.get('reason', ''))
                del running[name]
                progressed = True
            for name, spec in list(pending.items()):
                deps = spec['dependencies']
                if any(dep not in summary for dep in deps):
                    continue
                failed = [dep for dep in deps if str(summary[dep].get('status', '')).lower() not in DEPENDENCY_OK]
                if failed:
                    result = dict(status='blocked', reason='Dependency unavailable: '+', '.join(failed),
                                  dependencies={dep: summary[dep].get('status') for dep in deps})
                    summary[name] = result
                    write_json(output/(name+'.json'), result)
                    timeline['stages'].append(dict(name=name, **spec, status='blocked', reason=result['reason']))
                    del pending[name]
                    progressed = True
                    continue
                if spec['resource'] == 'cpu':
                    if sum(state['cpu_slots'] for state in running.values()) + spec['cpu_slots'] > cpu_workers:
                        continue
                    slot, mask = None, ''
                else:
                    used = {state['slot'] for state in running.values() if state['resource'] == 'gpu'}
                    available = [index for index in range(len(devices)) if index not in used]
                    if not available:
                        continue
                    slot = available[0]
                    mask = devices[slot]
                work = output/'_stages'/name
                work.mkdir(parents=True, exist_ok=False)
                inputs = []
                if name in ('traffic_and_responsibility', 'episode_rollout') and (output/'real_agents').exists():
                    (work/'real_agents').symlink_to(os.path.relpath(output/'real_agents', work), target_is_directory=True)
                    inputs.append('real_agents')
                if (output/'resolved_config.yaml').is_file():
                    (work/'resolved_config.yaml').symlink_to(os.path.relpath(output/'resolved_config.yaml', work))
                job = work/'job.json'
                write_json(job, dict(name=name, config=config, summary={dep: summary[dep] for dep in deps},
                    context=dict(root=root, output_dir=work, dry_run=False)))
                env = os.environ.copy()
                for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                    env[key] = str(threads)
                if mask is not None:
                    env['CUDA_VISIBLE_DEVICES'] = mask
                env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(Path(__file__).resolve().parents[2]), env.get('PYTHONPATH'))))
                stream = (work/'stage.log').open('w', encoding='utf-8')
                try:
                    proc = subprocess.Popen(_command(job), cwd=root, env=env, stdout=stream,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close()
                    raise
                receipt = dict(name=name, **spec, pid=proc.pid, started_at_unix_s=time.time(),
                               gpu_slot=slot, cuda_visible_devices=mask, status='running',
                               config_sha256=config_hash, log=str(work/'stage.log'))
                timeline['stages'].append(receipt)
                running[name] = dict(process=proc, log=stream, work=work, inputs=inputs, receipt=receipt,
                    resource=spec['resource'], cpu_slots=spec['cpu_slots'], slot=slot, started=time.monotonic())
                logger.info('[%s] Running in PID %s (%s%s)', name.upper(), proc.pid, spec['resource'],
                            ' '+str(mask) if spec['resource']=='gpu' else '')
                del pending[name]
                progressed = True
            if progressed:
                write_json(timeline_path, timeline)
            elif not running and pending:
                raise RuntimeError('Unresolved stage dependencies: '+', '.join(pending))
            elif running:
                time.sleep(.05)
        timeline['status'] = ('failed' if any(str(r.get('status', '')).lower() in ('failed', 'blocked', 'skipped')
                                             for r in summary.values()) else 'completed')
        return summary
    except BaseException as exc:
        timeline['status'] = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        timeline['reason'] = f'{type(exc).__name__}: {exc}'
        _stop([state['process'] for state in running.values()])
        for name, state in running.items():
            state['receipt'].update(status='interrupted', finished_at_unix_s=time.time())
            state['log'].close()
        raise
    finally:
        try:
            timeline['finished_at_unix_s'] = time.time()
            write_json(timeline_path, timeline)
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, previous_sigterm)


def _worker(job_path):
    job = json.loads(Path(job_path).read_text())
    context = job['context']
    context.update(root=Path(context['root']), output_dir=Path(context['output_dir']))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    context['logger'] = logging.getLogger('evaluation.stage.'+job['name'])
    from driveaudit_eval.evaluation.runner import evaluate_stage
    result = evaluate_stage(job['name'], job['config'], context, job['summary'])
    write_json(context['output_dir']/'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True, type=Path)
    _worker(parser.parse_args().worker)
