"""Run the bundled driving services on real input-gt observations, then score plans."""
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from driveaudit_eval.agents.agent_input_adapter import assert_no_evaluation_leakage, build_http_request, planning_to_unified_ego, select_vad_plan
from driveaudit_eval.agents.core import WORLD_DREAMER_CAMERA_ORDER, derive_command, derive_kinematics, quat_rotation, quat_to_yaw, write_json, write_csv
from driveaudit_eval.agents.agent_input_adapter import parse_output

from driveaudit_eval.agents.registry import AGENTS, dependencies_root, service_path
from driveaudit_eval.agents.base import AgentAdapter


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def model_input_signature(manifest):
    frames={s['scene_id']:s['frames'] for s in manifest['scenes']}
    if len(frames)!=len(manifest['scenes']):
        raise ValueError('Duplicate scene in agent input manifest')
    return hashlib.sha256(json.dumps(frames,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def matrix(record):
    out = np.eye(4)
    out[:3, :3] = quat_rotation(record['rotation'])
    out[:3, 3] = record['translation']
    return out


def prepare(config, context, output, previous_manifest=None):
    from driveaudit_eval.data.scene_input import load_scene_input
    from driveaudit_eval.agents.input_gt import prediction_windows
    from driveaudit_eval.agents.registry import REQUIREMENTS as FORMAL_REQUIREMENTS, requirements_for
    root=Path(context['root'])
    windows=prediction_windows(config,context)
    scenes=[]; sources=[]; hashes={}; lidar_index={}; statuses=[]
    selected=config.get('agents',list(FORMAL_REQUIREMENTS))
    sensor_mode=config.get('sensor_protocol','camera_only')
    REQUIREMENTS={n:requirements_for(n,sensor_mode) for n in selected}
    if sensor_mode=='camera_only' and config.get('allow_real_lidar',False):
        raise RuntimeError('Real LiDAR is forbidden in camera-only leaderboard mode')
    for value in config['scene_paths']:
        scene=load_scene_input(root/value,agent_only=True) if sensor_mode=='camera_only' else load_scene_input(root/value)
        required=[r['frame_id'] for r in windows if r['scene_id']==scene.scene_id]
        if not required: raise ValueError('No scoreable decision frames for '+scene.scene_id)
        for paths in scene.cameras.values():
            for path in paths: hashes[path]=digest(path)
        frames=[AgentAdapter.prepare_input(scene,i) for i in range(max(required)+1)]
        lidar=None
        if any(REQUIREMENTS[n].lidar_required for n in selected):
            from driveaudit_eval.research.sensors.preparation import prepare_scene_lidar
            lidar,scene_index=prepare_scene_lidar(scene,required,config,output)
            lidar_index.update(scene_index)
        caps=dict(scene.capabilities,lidar=bool(lidar and all(r is not None for r in lidar['records'])))
        for name in selected:
            missing=REQUIREMENTS[name].missing(caps)
            reason='required_lidar_not_available' if 'lidar' in missing else ('missing_required_modalities' if missing else None)
            if 'lidar' in missing and lidar:
                reasons={r.get('reason') for r in lidar.get('missing',[])}
                if 'invalid_sensor_alignment' in reasons: reason='invalid_sensor_alignment'
                elif 'invalid_sensor_data' in reasons: reason='invalid_sensor_data'
            statuses.append(dict(dataset=scene.dataset_name,scene=scene.scene_id,agent=name,requirements=REQUIREMENTS[name].as_dict(),
                available_modalities=dict(caps),missing_modalities=missing,input_status='unavailable' if missing else 'ready',
                inference_status='unavailable' if missing else 'not_run',evaluation_status='not_run',failure_reason=reason,
                sensor_alignment=lidar.get('missing',[]) if lidar else []))
        scenes.append(dict(scene_id=scene.scene_id,scene_token=scene.metadata['original_scene_id'],dataset=scene.dataset_name,
            scene_path=str(scene.scene_dir),frames=frames,capabilities=caps))
        sources.append(dict(scene_id=scene.scene_id,metadata={k:v for k,v in scene.metadata.items() if k!='history_camera_poses'},
            source_sha256={str(p):digest(p) for p in [scene.scene_dir/'timeline.json',scene.scene_dir/'calibration/cameras.json',
                scene.scene_dir/'calibration/reference_ego_poses.json',Path(scene.metadata['source_provenance_path'])]},
            lidar_audit=lidar))
    manifest=dict(scenes=scenes,common_windows=windows,sources=sources,image_sha256=hashes,scene_status=statuses,
        protocol='oracle-conditioned open_loop_proxy; canonical input-gt; no dataset-native core input dependency',
        camera_intrinsic_basis='challenge K scaled to restored 1600x900 runtime canvas',
        camera_time_policy='native asynchronous exposures; bounded exported pose interpolation; latest exposure availability',
        sensor_protocol=sensor_mode,
        sensor_difference='RGB-only perception; fusion checkpoints use fixed zero LiDAR tensors' if sensor_mode=='camera_only' else 'GoalFlow and DiffusionDriveV2 require measured causal LiDAR')
    if previous_manifest is not None:
        if (any(previous_manifest.get(key) != manifest[key] for key in
                ('image_sha256','common_windows','sensor_protocol'))
                or model_input_signature(previous_manifest) != model_input_signature(manifest)):
            raise ValueError('Same-run resume inputs differ from the recorded manifest')
        # Keep the original bytes bound to completed inference_status files.
    else:
        write_json(output/'input_manifest.json',manifest)
    if sensor_mode!='camera_only':
        from driveaudit_eval.research.sensors.preparation import write_lidar_indexes
        write_lidar_indexes(output,lidar_index)
    write_json(output/'scene_status.json',statuses)
    return manifest


def request(url, payload=None, timeout=30):
    req = Request(url, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    with urlopen(req, timeout=timeout) as response: return response.read().decode()


def _wait_output(endpoint, timeout=120):
    """A successful HTTP 204 is pending inference, not a JSON prediction."""
    deadline = time.monotonic()+timeout
    while True:
        try:
            body = request(endpoint+'/driver-get/', timeout=15)
            if body: return body
        except URLError:
            pass
        if time.monotonic()>deadline: raise RuntimeError('Inference timeout')
        time.sleep(.1)


def _check_gpu_memory(python, env):
    """Probe logical CUDA 0 in the service environment, including UUID/MIG masks."""
    probe = '''import json, torch
torch.cuda.set_device(0)
if hasattr(torch.cuda, 'mem_get_info'):
    free, total = torch.cuda.mem_get_info(0)
else:
    # Frozen UniAD/VAD use PyTorch 1.9, before the public memory query API.
    import ctypes
    from pathlib import Path
    libraries = sorted((Path(torch.__file__).parent/'lib').glob('libcudart*.so*'))
    if not libraries:
        raise RuntimeError('Cannot locate the frozen PyTorch CUDA runtime')
    runtime = ctypes.CDLL(str(libraries[0]))
    query = runtime.cudaMemGetInfo
    query.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    query.restype = ctypes.c_int
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    status = query(ctypes.byref(free), ctypes.byref(total))
    if status:
        raise RuntimeError('cudaMemGetInfo failed: '+str(status))
    free, total = free.value, total.value
print(json.dumps(dict(free_bytes=free, total_bytes=total, name=torch.cuda.get_device_name(0))))
'''
    result = subprocess.run([str(python), '-c', probe], env=env, check=True,
                            capture_output=True, text=True, timeout=60)
    memory = json.loads(result.stdout)
    free, total = memory['free_bytes'], memory['total_bytes']
    if type(free) is not int or type(total) is not int or not 0 <= free <= total:
        raise RuntimeError('Invalid CUDA memory probe result')
    memory.update(cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'), logical_device=0)
    if free < 12000 * 1024**2:
        raise RuntimeError('Insufficient free GPU memory on selected CUDA device: '+json.dumps(memory))
    return memory


def infer(item, manifest, output, context):
    name, directory, environment, extra = item
    sensor_mode=manifest.get('sensor_protocol','camera_only')
    adapter=AgentAdapter(item,sensor_mode)
    agent=adapter.service_name
    root = Path(context['root']); target = output/name; target.mkdir(exist_ok=True)
    # Concurrent evaluations on one host must use distinct port ranges.
    try:
        port_base = int(os.environ.get('DRIVEAUDIT_AGENT_PORT_BASE', '12209'))
    except ValueError as exc:
        raise ValueError('DRIVEAUDIT_AGENT_PORT_BASE must be an integer') from exc
    max_port_base = 65535 - (len(AGENTS) - 1)
    if not 1024 <= port_base <= max_port_base:
        raise ValueError(f'DRIVEAUDIT_AGENT_PORT_BASE must be between 1024 and {max_port_base}')
    port = port_base + [a[0] for a in AGENTS].index(name)
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1',port))
    service,cwd,env=adapter.launch(root,output,port)
    write_json(target/'gpu_memory.json', _check_gpu_memory(service[0], env))
    (target/'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    write_json(target/'command.json', dict(service=service,cwd=str(cwd),environment=environment,
        implementation_sha256={str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('base.py'),Path(__file__).with_name('core.py'),Path(service[1]))},
        camera_intrinsic_policy='actual image K for DrivoR; native K for other services'))
    endpoint = f'http://127.0.0.1:{port}'; rows=[]; input_rows=[]; started=time.monotonic()
    required = {(r['scene_id'],r['frame_id']) for r in manifest['common_windows']}
    from driveaudit_eval.agents.provenance import model_identity
    from driveaudit_eval.metrics.testing_effectiveness import content_hash
    identity = model_identity(name)
    write_json(target/'model_identity.json', identity)
    with (target/'server.log').open('w') as log, (target/'predictions.jsonl').open('w') as pred, (target/'raw_outputs.jsonl').open('w') as raw_file:
        process = subprocess.Popen(service,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(300):
                if process.poll() is not None: raise RuntimeError('Service exited; see '+str(target/'server.log'))
                try:
                    health=request(endpoint+adapter.health_path,timeout=2)
                    (target/'health.json').write_text(health); break
                except OSError: time.sleep(1)
            else: raise RuntimeError('Service startup timeout')
            for scene in manifest['scenes']:
                request(endpoint+'/driver-clean/')
                for frame in scene['frames']:
                    fid=frame['frame_id']; sid=scene['scene_id']
                    if not adapter.temporal and (sid,fid) not in required: continue
                    start=time.monotonic(); inp=frame['model_input']; assert_no_evaluation_leakage(inp)
                    payload=adapter.payload(frame,sid)
                    request(endpoint+'/driver-api/',payload,timeout=120)
                    body = _wait_output(endpoint)
                    traj, mode, score, raw = AgentAdapter.normalize_output(agent, body)
                    row=dict(scene_id=sid,frame_id=fid,agent=name,timestamp=frame['timestamp'],planning_traj=traj,
                        coordinate_system='ego_x_forward_y_left',time_offsets=[.5,1,1.5,2,2.5,3],inference_success=True,latency_s=time.monotonic()-start)
                    pred.write(json.dumps(row)+'\n'); pred.flush()
                    raw_file.write(json.dumps(dict(scene_id=sid,frame_id=fid,output=raw))+'\n'); raw_file.flush()
                    if sensor_mode=='camera_only':
                        input_rows.append(dict(agent=name,dataset=scene['dataset'],scene=sid,decision_frame=fid,
                            scored_decision=(sid,fid) in required,rgb_used=True,camera_count=adapter.requirements.required_camera_count,
                            ego_state_used=True,command_used=True,lidar_used=False,real_lidar_accessed=False,real_lidar_used=False,
                            gt_future_used=False,gt_boxes_used=False,
                            lidar_mode='zero' if name in ('goalflow','diffusiondrivev2') else 'not_requested'))
                    if (sid,fid) in required: rows.append(row)
                    if fid%10==9: context['logger'].info('[AGENT] %s %s frame %d',name,sid,fid)
            if len(rows)!=len(required): raise ValueError('Missing common prediction windows')
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
    if sensor_mode=='camera_only':
        audit=[json.loads(line) for line in (target/'sensor_access_audit.jsonl').read_text().splitlines()]
        if not any(r['event']=='guard_installed' for r in audit) or any(r['event']=='blocked' for r in audit):
            raise RuntimeError('Sensor leakage detected or guard missing')
        write_json(target/'agent_input_manifest.json',dict(protocol='camera_only',sensor_leakage_detected=False,
            guard='process-wide Python open audit installed before model imports',decisions=input_rows))
    if model_identity(name) != identity:
        raise ValueError('Model resources changed during inference')
    evidence = {filename:digest(target/filename) for filename in ('predictions.jsonl','raw_outputs.jsonl',
        'agent_input_manifest.json','sensor_access_audit.jsonl') if (target/filename).is_file()}
    write_json(target/'inference_status.json',dict(model_identity_hash=content_hash(identity), evidence_sha256=evidence, status='COMPLETED',agent=name,common_predictions=len(rows),sensor_protocol=sensor_mode,
        run_directory=str(output.resolve()),
        total_inference_frames=sum(1 for _ in (target/'predictions.jsonl').open()),elapsed_s=time.monotonic()-started,
        predictions_sha256=digest(target/'predictions.jsonl'),manifest_sha256=digest(output/'input_manifest.json')))
    return rows


def _same_run_predictions(output, name, manifest):
    """Only accept completed, unchanged evidence recorded in this run directory."""
    from driveaudit_eval.agents.provenance import model_identity
    from driveaudit_eval.metrics.testing_effectiveness import content_hash
    target = output/name
    if target.is_symlink():
        raise ValueError('Same-run predictions cannot use an external model directory')
    status_path = target/'inference_status.json'
    if status_path.is_symlink():
        raise ValueError('Same-run inference status cannot be an external link')
    if not status_path.is_file():
        return None
    status = json.loads(status_path.read_text())
    if status.get('status') != 'COMPLETED':
        return None
    if (status.get('run_directory') != str(output.resolve()) or status.get('agent') != name
            or status.get('sensor_protocol') != manifest['sensor_protocol']
            or status.get('manifest_sha256') != digest(output/'input_manifest.json')
            or status.get('common_predictions') != len(manifest['common_windows'])):
        raise ValueError('Same-run inference does not match this run and its inputs')
    identity_path = target/'model_identity.json'
    if identity_path.is_symlink():
        raise ValueError('Same-run model identity cannot be an external link')
    identity = json.loads(identity_path.read_text())
    if status.get('model_identity_hash') != content_hash(identity) or identity != model_identity(name):
        raise ValueError('Same-run model identity changed')
    evidence = status.get('evidence_sha256', {})
    required = {'predictions.jsonl','raw_outputs.jsonl'}
    if manifest['sensor_protocol'] == 'camera_only':
        required.update(('agent_input_manifest.json','sensor_access_audit.jsonl'))
    if not required.issubset(evidence):
        raise ValueError('Same-run inference evidence is incomplete')
    for filename, expected in evidence.items():
        path = target/filename
        if Path(filename).name != filename or path.is_symlink() or digest(path) != expected:
            raise ValueError('Same-run inference evidence changed: '+filename)
    if status.get('predictions_sha256') != evidence['predictions.jsonl']:
        raise ValueError('Same-run prediction hash is inconsistent')
    return [json.loads(line) for line in (target/'predictions.jsonl').read_text().splitlines() if line.strip()]


def evaluate(config, context):
    from driveaudit_eval.agents.input_gt import evaluate_predictions
    selected=config.get('agents',[a[0] for a in AGENTS])
    if len(set(selected))!=len(selected) or set(selected)-{a[0] for a in AGENTS}: raise ValueError('Invalid agent names')
    if context.get('dry_run'):
        from driveaudit_eval.data.scene_input import load_scene_input
        scenes=[load_scene_input(Path(context['root'])/p) for p in config['scene_paths']]
        return dict(status='READY',agents=selected,model_inference_executed=False,scene_count=len(scenes),
                    limitation='Canonical core inputs validated; model and extra sensor readiness require smoke tests')
    # Release completed in-process visual stages before isolated model services.
    import gc
    import sys
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    resume = config.get('same_run_resume', False)
    if resume and config.get('reuse_predictions_from'):
        raise ValueError('Same-run resume cannot use legacy prediction sources')
    output=Path(context['output_dir'])/'real_agents'
    previous_manifest = None
    if resume and output.exists():
        if output.is_symlink() or (output/'input_manifest.json').is_symlink():
            raise ValueError('Same-run input directory and manifest must be local files')
        previous_manifest = json.loads((output/'input_manifest.json').read_text())
    output.mkdir(exist_ok=resume)
    context['logger'].info('[AGENT] Preparing inputs under %s',config.get('sensor_protocol','camera_only'))
    package = Path(__file__).resolve().parents[1]
    code_paths=[p for p in package.rglob('*.py') if 'research' not in p.relative_to(package).parts]
    if config.get('sensor_protocol','camera_only')=='original_multimodal':
        code_paths+=list((package/'research/sensors').rglob('*.py'))
    manifest=prepare(config,context,output,previous_manifest)
    write_json(output/'implementation_sha256.json',{str(p):digest(p) for p in code_paths})
    results={}; comparisons=[]
    scene_status=manifest['scene_status']
    for item in AGENTS:
        name=item[0]
        if name not in selected: continue
        states=[r for r in scene_status if r['agent']==name]
        ready={r['scene'] for r in states if r['input_status']=='ready'}
        current=dict(manifest,scenes=[s for s in manifest['scenes'] if s['scene_id'] in ready],
                     common_windows=[w for w in manifest['common_windows'] if w['scene_id'] in ready])
        current_config=dict(config,scene_paths=[s['scene_path'] for s in current['scenes']])
        if not ready:
            results[name]=dict(status='UNAVAILABLE',reason='required_inputs_unavailable',per_scene=[],scene_status=states,model_inference_executed=False)
            comparisons.append(dict(agent=name,status='UNAVAILABLE',reason='required_inputs_unavailable'))
            write_json(output/'results.json',results); write_csv(output/'comparison.csv',comparisons)
            continue
        try:
            context['logger'].info('[AGENT] Starting %s',name)
            reuse = config.get('reuse_predictions_from')
            rows = _same_run_predictions(output,name,current) if resume else None
            resumed = rows is not None
            if resumed:
                context['logger'].info('[AGENT] Resuming verified same-run predictions for %s',name)
            elif reuse:
                source = Path(context['root'])/reuse/name
                prediction_path = source/'predictions.jsonl'
                inference_status = json.loads((source/'inference_status.json').read_text())
                if inference_status.get('status') != 'COMPLETED' or inference_status.get('predictions_sha256') != digest(prediction_path):
                    raise ValueError('Reused prediction source is incomplete or its hash changed')
                rows = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
                old_manifest = json.loads((source.parent/'input_manifest.json').read_text())
                if old_manifest.get('sensor_protocol','original_multimodal')!=manifest['sensor_protocol']:
                    raise ValueError('Reused prediction sensor protocol mismatch')
                if (old_manifest['image_sha256'] != manifest['image_sha256'] or old_manifest['common_windows'] != manifest['common_windows']
                    or model_input_signature(old_manifest)!=model_input_signature(manifest)):
                    raise ValueError('Reused predictions do not match current images and scoring windows')
                if manifest['sensor_protocol']=='camera_only':
                    from .sensor_guard import validate_camera_only_run
                    validate_camera_only_run(source.parent, agents=[name])
                filenames = ['predictions.jsonl', 'inference_status.json']
                if manifest['sensor_protocol']=='camera_only':
                    filenames += ['agent_input_manifest.json','sensor_access_audit.jsonl','raw_outputs.jsonl','model_identity.json']
                source_artifacts = {}
                from driveaudit_eval.metrics.testing_effectiveness import content_hash
                for filename in filenames:
                    path = source/filename
                    if filename == 'predictions.jsonl':
                        if digest(path) != inference_status['predictions_sha256']:
                            raise ValueError('Reused predictions changed during verification')
                    elif filename == 'model_identity.json':
                        if content_hash(json.loads(path.read_text())) != inference_status.get('model_identity_hash'):
                            raise ValueError('Reused model identity hash mismatch')
                    elif filename not in ('predictions.jsonl', 'inference_status.json'):
                        if digest(path) != inference_status.get('evidence_sha256', {}).get(filename):
                            raise ValueError('Reused inference evidence hash mismatch: '+filename)
                    source_artifacts[filename] = dict(path=str(path.resolve()), sha256=digest(path))
                (output/name).mkdir(exist_ok=True)
                for filename, artifact in source_artifacts.items():
                    (output/name/filename).symlink_to(artifact['path'])
                write_json(output/name/'prediction_reuse.json', dict(source=str(prediction_path), sha256=digest(prediction_path),
                    source_artifacts=source_artifacts, same_run_resume_supported=False,
                    current_model_inference_executed=False, description='Verified external real model predictions referenced without copying; freshly rescored'))
            else:
                rows=AgentAdapter(item).load(context).predict(current, output)
            result=evaluate_predictions(dict(current_config,agent=name),dict(context,output_dir=output/name),rows)
            if reuse:
                for artifact in source_artifacts.values():
                    if digest(artifact['path']) != artifact['sha256']:
                        raise ValueError('Reused inference evidence changed during scoring: '+artifact['path'])
            result['inference']=json.loads((output/name/'inference_status.json').read_text())
            result['model_inference_executed']=not bool(reuse) and not resumed
            result['real_prediction_reused']=bool(reuse)
            result['same_run_resumed']=resumed
            for state in states:
                if state['scene'] in ready:
                    row=next(r for r in result['per_scene'] if r['scene_id']==state['scene'])
                    missing_score=row.get('PDMS_open_loop') is None
                    state.update(inference_status='completed',evaluation_status='partial' if missing_score else 'completed',
                        failure_reason='insufficient_drivable_area_map_support' if not state['available_modalities']['drivable_area'] else None)
            result['scene_status']=states
            if len(ready)<len(states): result['status']='PARTIAL'
            results[name]=result
            comparison=dict(agent=name,status=result['status'],common_windows=result['valid_windows'])
            for key in ('NC','DAC','TTC_score','Comfort','EP','EP_current_speed_proxy','PDMS_open_loop','RC_open_loop','Driving_Score_open_loop','ADE_3s','FDE_3s'):
                vals=[s.get(key) for s in result['per_scene']]
                comparison[key]=float(np.mean(vals)) if all(v is not None for v in vals) else None
            comparisons.append(comparison)
        except Exception as exc:
            context['logger'].exception('[AGENT] Failed %s',name)
            for state in states:
                if state['scene'] in ready: state.update(inference_status='failed',failure_reason=f'{type(exc).__name__}: {exc}')
            results[name]=dict(status='FAILED',reason=f'{type(exc).__name__}: {exc}',scene_status=states)
            comparisons.append(dict(agent=name,status='FAILED',reason=str(exc)))
        write_json(output/'results.json',results); write_csv(output/'comparison.csv',comparisons)
        write_json(output/'scene_status.json',scene_status)
    from driveaudit_eval.agents.registry import write_support_matrix
    write_support_matrix(scene_status,output)
    from driveaudit_eval.metrics.leaderboard import build_cross_dataset_leaderboards
    build_cross_dataset_leaderboards(results,output,expected_agents=selected)
    if any(digest(p)!=value for p,value in manifest['image_sha256'].items()): raise ValueError('Source RGB changed during inference')
    complete=all(r['status'].upper()=='COMPLETED' for r in results.values()) and len(results)==len(selected)
    executed=any(r.get('model_inference_executed', False) for r in results.values())
    usable=any(r.get('per_scene') for r in results.values())
    result=dict(status='COMPLETED' if complete else 'FAILED' if not usable and any(r['status']=='FAILED' for r in results.values()) else 'PARTIAL',model_inference_executed=executed,
        same_run_resumed=any(r.get('same_run_resumed',False) for r in results.values()),agents=results,comparison=comparisons,
        common_windows=len(manifest['common_windows']),scene_count=len(manifest['scenes']),metric_type='oracle-conditioned open_loop_proxy',
        aggregation='equal_scene_mean',artifacts=[str(output/'comparison.csv'),str(output/'results.json'),str(output/'input_manifest.json')])
    write_json(output/'completion.json',dict(status=result['status'],agents=selected,common_windows=result['common_windows']))
    return result
