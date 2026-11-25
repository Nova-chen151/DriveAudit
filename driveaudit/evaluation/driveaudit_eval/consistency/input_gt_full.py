"""Export real input-gt conditions and run the existing complete CPU backend chain."""
from copy import deepcopy
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from driveaudit_eval.consistency.input_schema import _read, _rigid
from driveaudit_eval.consistency.input_schema import write_json
from driveaudit_eval.video.utils import read_image_bgr
from driveaudit_eval.consistency.config import save_snapshot


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def _quat(matrix):
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return q[[3, 0, 1, 2]].tolist()


def _native_records(path):
    # nuScenes pretty JSON tables have one top-level record per brace-delimited block.
    lines = []
    for line in Path(path).open():
        if line.strip() == '{':
            if lines: raise ValueError('Unexpected nested nuScenes table record')
            lines = [line]
        elif lines:
            lines.append(line)
            if line.strip() in {'}', '},'}:
                yield json.loads(''.join(lines).rstrip(' ,\n'))
                lines = []
    if lines: raise ValueError('Truncated nuScenes metadata table')


def _native_camera_poses(scene, native_root, frames, manifest, reference_poses, output):
    cache = output/'native_camera_poses.json'
    table = Path(native_root)/'v1.0-trainval'
    source_key={'source_root':str(Path(native_root).resolve()),'scene':str(scene.resolve()),
        'input_hashes':{str(p.relative_to(scene)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [scene/'timeline.json',scene/'future/rgb_manifest.json',scene/'calibration/reference_ego_poses.json']},
        'native_table_stat':{name:[(table/name).stat().st_size,(table/name).stat().st_mtime_ns] for name in ['scene.json','sample.json','sample_data.json','ego_pose.json','calibrated_sensor.json']}}
    if cache.is_file():
        cached=_read(cache)
        selected=cached.get('selected_source_records',{})
        digest=hashlib.sha256(json.dumps(selected,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if cached.get('source_key')==source_key and digest==cached.get('selected_source_records_sha256'):
            return cached
    record = next(r for r in _read(table/'scene.json') if r['name'] == manifest['source_scene'])
    samples = {r['token'] for r in _read(table/'sample.json') if r['scene_token'] == record['token']}
    data = [r for r in _native_records(table/'sample_data.json') if r['sample_token'] in samples and '/CAM_' in r['filename']]
    by_camera = {}
    for row in data:
        camera = row['filename'].split('/')[1]
        by_camera.setdefault(camera, {})[row['timestamp']] = row
    future = {(r['frame_index'], r['camera']):r for r in manifest['images']}
    exposures = {(i,c): (frames[i]['cameras'][c] if i<10 else future[(i,c)])['exposure_time_us']
                 for i in range(90) for c in by_camera}
    candidates = [stamp-exposures[(0,'CAM_FRONT')] for stamp in by_camera['CAM_FRONT']]
    candidates = [base for base in candidates if all(base+offset in by_camera[c] for (i,c),offset in exposures.items())]
    if len(candidates)!=1: raise ValueError(f'Native exposure timeline must match exactly once, found {len(candidates)}')
    base = candidates[0]
    selected = {(i,c):by_camera[c][base+offset] for (i,c),offset in exposures.items()}
    tokens = {r['ego_pose_token'] for r in selected.values()}
    ego = {r['token']:r for r in _native_records(table/'ego_pose.json') if r['token'] in tokens}
    calibrated = {r['token']:r for r in _read(table/'calibrated_sensor.json')}
    def matrix(r):
        m=np.eye(4);q=np.asarray(r['rotation']);m[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();m[:3,3]=r['translation'];return m
    anchor = matrix(ego[selected[(9,'CAM_FRONT')]['ego_pose_token']]); inv = np.linalg.inv(anchor)
    rows=[]
    for (i,c),r in selected.items():
        relative = inv@matrix(ego[r['ego_pose_token']])
        if c=='CAM_FRONT' and not np.allclose(relative, reference_poses[i],atol=1e-4):
            raise ValueError(f'Native source ego pose disagrees with export at frame {i}')
        rows.append({'frame_index':i,'camera':c,'camera_to_anchor':(relative@matrix(calibrated[r['calibrated_sensor_token']])).tolist(),
                     'sample_data_token':r['token'],'absolute_timestamp_us':r['timestamp']})
    selected_records={'sample_data':list(selected.values()),'ego_pose':list(ego.values()),
                      'calibrated_sensor':[calibrated[token] for token in sorted({r['calibrated_sensor_token'] for r in selected.values()})]}
    result={'source_root':str(native_root),'source_key':source_key,'absolute_time_base_us':base,'rows':rows,'all540timestamps_exact':len(rows)==540,
            'selected_source_records':selected_records,
            'selected_source_records_sha256':hashlib.sha256(json.dumps(selected_records,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'hash_scope':'selected native records, not the complete native metadata tables'}
    write_json(cache,result);return result


def prepare(scene, output, native_root=None, candidate_role='ground_truth'):
    """Keep actual ego motion, anchor-coordinate boxes, and measured exposures explicit."""
    if candidate_role not in {'ground_truth', 'generated'}:
        raise ValueError('candidate_role must be ground_truth or generated')
    cameras = _read(scene/'calibration/cameras.json')
    frames = _read(scene/'timeline.json')['frames']
    pose_data = _read(scene/'calibration/reference_ego_poses.json')
    poses = [_rigid(p, f'pose {i}') for i, p in enumerate(pose_data['ego_to_anchor'])]
    if len(poses) != 90 or [r['frame_index'] for r in frames] != list(range(90)):
        raise ValueError('Complete 90-frame pose/timeline required')
    if pose_data['coordinate_frame'] not in {'history_frame_9_ego', 'history_last_ego'}:
        raise ValueError('Unsupported source coordinate frame')
    future_manifest = _read(scene/'future/rgb_manifest.json')
    future = {(r['frame_index'], r['camera']): r for r in future_manifest['images']}
    native = _native_camera_poses(scene,native_root,frames,future_manifest,poses,output) if native_root else None
    native_poses = {(r['frame_index'],r['camera']):np.asarray(r['camera_to_anchor']) for r in native['rows']} if native else {}
    condition = output/'conditions'
    pose_rows, calibration, sources = [], [], []
    for frame, pose in enumerate(poses):
        pose_rows.append({'frame_index':frame,'ego2global_translation':pose[:3,3].tolist(),
            'ego2global_rotation':_quat(pose),'lidar2ego_translation':[0,0,0],'lidar2ego_rotation':[1,0,0,0],
            'lidar_basis_note':'derived input is native ego coordinates; no LiDAR measurement is asserted'})
        for camera in cameras['cameras']:
            name=camera['camera']; extrinsic=_rigid(camera['camera_to_ego_4x4'], name)
            if native:
                extrinsic = np.linalg.inv(pose) @ native_poses[(frame,name)]
            metadata=frames[frame]['cameras'].get(name) if frame<10 else future.get((frame,name))
            if not metadata or 'exposure_time_us' not in metadata:raise ValueError(f'Missing exposure {frame}/{name}')
            rgb=scene/('history' if frame<10 else 'future')/'rgb'/name/f'{frame:06d}.jpg'
            image=read_image_bgr(rgb)
            if list(image.shape[1::-1])!=camera['challenge_image_size']:raise ValueError(f'RGB size mismatch {rgb}')
            digest=hashlib.sha256(rgb.read_bytes()).hexdigest()
            if frame>=10 and digest!=metadata['sha256']:raise ValueError(f'RGB hash mismatch {rgb}')
            intrinsics=np.asarray(camera['challenge_intrinsics'],float)
            if intrinsics.shape!=(3,3) or not np.isfinite(intrinsics).all() or min(intrinsics[0,0],intrinsics[1,1])<=0:raise ValueError('Invalid intrinsic')
            calibration.append({'frame_index':frame,'camera':name,'timestamp':metadata['exposure_time_us'],
                'camera_intrinsics':intrinsics.tolist(),'image_width':image.shape[1],'image_height':image.shape[0],
                'sensor2ego_translation':extrinsic[:3,3].tolist(),'sensor2ego_rotation':_quat(extrinsic),
                'ego2global_translation':pose[:3,3].tolist(),'ego2global_rotation':_quat(pose),
                'sensor2lidar_rotation':extrinsic[:3,:3].tolist(),'sensor2lidar_translation':extrinsic[:3,3].tolist(),
                'pose_time_us':frames[frame]['reference_time_us'],
                'pose_time_protocol':'native_camera_exposure_exact' if native else 'exported_reference_pose_with_exposure_offset'})
            sources.append({'frame_index':frame,'camera':name,'file':str(rgb),'sha256':digest,
                'exposure_time_us':metadata['exposure_time_us'],'reference_time_us':frames[frame]['reference_time_us'],
                'reused_source_frame':metadata.get('reused_source_frame',False)})
    boxes=[]
    for rel in ['history/agents.csv','future/future_trajectories.csv']:
        for r in csv.DictReader((scene/rel).open()):
            frame=int(r['frame_index']);pose=poses[frame]
            center=np.asarray([float(r[k]) for k in ('x','y','z')]);local=pose[:3,:3].T@(center-pose[:3,3])
            yaw=float(r['yaw']);ego_yaw=np.arctan2(pose[1,0],pose[0,0])
            # Source future velocity is absent. It is unused by the visual observer and stays NaN.
            velocity=np.array([float(r.get('vx','nan')),float(r.get('vy','nan')),0.]);local_velocity=pose[:3,:3].T@velocity
            boxes.append({'frame_index':frame,'obj_id':r['agent_id'],'class_name':r['category'],
                'valid_flag':r['valid'],'x':local[0],'y':local[1],'z':local[2],
                'width':r['width'],'length':r['length'],'height':r['height'],'yaw':yaw-ego_yaw,
                'vx':local_velocity[0],'vy':local_velocity[1],
                'global_x':r['x'],'global_y':r['y'],'global_z':r['z'],'global_yaw':r['yaw'],
                'box_coordinate_system':'ego_export_with_exact_anchor_global_fields'})
    boxes.sort(key=lambda r:r['frame_index'])
    _csv(condition/'frames.csv',[{'frame_index':r['frame_index'],'timestamp':r['reference_time_us']} for r in frames])
    _csv(condition/'ego_pose.csv',pose_rows);_csv(condition/'camera_calibration.csv',calibration);_csv(condition/'boxes.csv',boxes)
    pair_id=scene.parent.parent.name+'_'+scene.name
    pair={'pair_id':pair_id,'camera_order':cameras['camera_order'],'history_metadata_path':str(scene/'history/agents.csv'),
        'image_sequence':{'scene_id':pair_id,'history_root':str(scene/'history/rgb'),
            'generated_future_root':str(scene/'future/rgb'),'conditions_root':str(condition)},
        'trajectory_sequence':{'scene_id':pair_id,'trajectory_path':str(scene/'future/future_trajectories.csv')},
        'calibration':{'path':str(condition/'camera_calibration.csv')},'ego_pose':{'path':str(condition/'ego_pose.csv')},
        'timeline':{'image_history_frames':10,'image_future_frames':80},
        'candidate_role':candidate_role,
        'source_semantics':('GT self-evaluation: generated_future_root is the backend field name for real future RGB'
                            if candidate_role == 'ground_truth' else
                            'Generated candidate evaluation: future RGB and submitted trajectories come from the candidate scene')}
    write_json(output/'source_manifest.json',{'scene':str(scene),'images':sources,
        'conditions_derived_without_source_modification':True,'camera_size_explicit':True,'candidate_role':candidate_role,
        'reference_frame_trajectory_protocol':True,'native_camera_pose_alignment':bool(native),'trajectory_time_protocol':'original TCVV frame-reference expectations',
        'maximum_exposure_offset_us':max(abs(r['exposure_time_us']-r['reference_time_us']) for r in sources)})
    return pair


def evaluate_scene(scene, config, output, logger):
    import cv2
    from driveaudit_eval.consistency.mtpc.aggregate import run as run_mtpc
    if config.get('profile') != 'core':
        from driveaudit_eval.research.consistency.trajectory_verification.aggregate import run as run_tcvv
        from driveaudit_eval.research.consistency.verification_benchmark.aggregate import Runner as Bench
        from driveaudit_eval.research.consistency.hypothesis_verification.aggregate import run as run_h
        from driveaudit_eval.research.consistency.observable_verification.aggregate import Runner as O
    cv2.setNumThreads(1)
    selected = ('mtpc',) if config.get('profile') == 'core' else ('mtpc','tcvv','tcvv_bench','tcvv_h','tcvv_o')
    disabled=[name for name in selected if not config['backends'][name].get('enabled',False)]
    if disabled:raise ValueError(f'Full consistency chain requires enabled backends: {disabled}')
    candidate_role=config.get('candidate_role','ground_truth')
    pair=prepare(scene,output/'derived',config.get('native_nuscenes_root'),candidate_role); manifest={'pairs':[pair]}
    cfg={name:deepcopy(config['backends'][name]) for name in selected}
    for name,c in cfg.items():
        c['retain_intermediates']=config.get('retain_intermediates',True)
        c['output']=str(output/name);c['_config_source']=str(output/'resolved_backend_config.json')
        if 'pairing' in c:c['pairing']={'manifest':manifest,'pair_id':pair['pair_id']}
        if 'inputs' in c:
            c['inputs'].update({'pair_manifest':manifest,'tcvv_config':cfg['tcvv'],'tcvv_results':str(output/'tcvv'),
                'tcvv_bench_results':str(output/'tcvv_bench'),'tcvv_h_results':str(output/'tcvv_h')})
        if 'holdout' in c:c['holdout']['scene_id']=pair['pair_id']
    cfg['mtpc']['research_diagnostics'] = config.get('profile') != 'core'
    if config.get('profile') != 'core':
        cfg['tcvv_o']['inputs']['tcvv_h_config']=cfg['tcvv_h']
        cfg['tcvv_bench']['benchmark']['current_holdout_scene']=pair['pair_id']
        cfg['tcvv_h']['scene_discovery']['roots']=[str(output)]
        cfg['tcvv_h']['inputs']['pair_manifest_source']=str(output/'resolved_backend_config.json')
    write_json(output/'resolved_backend_config.json',cfg)
    results={}
    for name,call in [('mtpc',lambda:run_mtpc(cfg['mtpc'],'full')),('tcvv',lambda:run_tcvv(cfg['tcvv'],'full')),
                      ('tcvv_bench',lambda:Bench(cfg['tcvv_bench']).validate()),('tcvv_h',lambda:run_h(cfg['tcvv_h'],'full')),
                      ('tcvv_o',lambda:O(cfg['tcvv_o']).full())]:
        if name not in selected:
            continue
        logger.info('[CONSISTENCY] Original backend %s: %s',name,scene.name);start=time.time()
        required={'mtpc':['final_metrics.json','extracted_visual_tracks/per_agent_tracks.json'],
                  'tcvv':['final_metrics.json','local_visual_evidence/per_window.json'],
                  'tcvv_bench':['final_metrics.json','final_report.md','negative_pairs/negative_features.json'],
                  'tcvv_h':['final_metrics.json','history_positive/history_hypothesis_rows.json','current_holdout/holdout_hypothesis_rows.json'],
                  'tcvv_o':['final_metrics.json','history_validation/model_gate.json','holdout/observable_motion_rows.json']}[name]
        result=call()
        blockers=[str(result.get(k,'')) for k in ('status','implementation_status') if str(result.get(k,'')).lower().startswith(('blocked','skipped','not_run','failed'))]
        if blockers:raise RuntimeError(f'{name} execution did not finish: {blockers}')
        missing=[path for path in required if not (output/name/path).is_file()]
        if missing:raise RuntimeError(f'{name} terminal artifacts missing: {missing}')
        results[name]={'execution_status':'completed','verified_artifacts':required,'runtime_seconds':time.time()-start,'result':result}
        if name == 'tcvv_bench':
            baseline=output/name/'baselines/tcvv_bench_v1_rule_based'
            baseline.mkdir(parents=True,exist_ok=True)
            mapping={'final_metrics.json':'final_metrics.json',
                     'calibrated_verifier/discrimination_metrics.json':'discrimination_metrics.json',
                     'calibrated_verifier/per_negative_family.csv':'per_negative_family.csv',
                     'false_support_analysis/summary.json':'false_support_summary.json',
                     'statistics/severity_monotonicity.json':'severity_monotonicity.json'}
            hashes={}
            for source,destination in mapping.items():
                source=output/name/source;target=baseline/destination
                save_snapshot(source,target,config.get('retain_intermediates',True))
                hashes[destination]=hashlib.sha256(source.read_bytes()).hexdigest()
                if hashlib.sha256(target.read_bytes()).hexdigest()!=hashes[destination]:raise IOError('Baseline copy hash mismatch')
            save_snapshot(output/'resolved_backend_config.json',baseline/'config_snapshot.yaml',config.get('retain_intermediates',True))
            write_json(baseline/'provenance.json',{'source':'same-run completed TCVV-bench v1 rule baseline','source_sha256':hashes})
        write_json(output/'chain_progress.json',results)
    scene_result = {'status':'completed','execution_status':'completed','scientific_acceptance':'see unchanged per-backend gates',
        'scene':str(scene),'protocol':'input_gt_original_mtpc_tcvv_bench_h_o_chain','backends':results,
        'candidate_role':candidate_role,
        'ground_truth_self_evaluation':candidate_role=='ground_truth','future_rgb_observer_is_real_pixel_processing':True,
        'native_camera_pose_alignment':bool(config.get('native_nuscenes_root')),
        'native_trajectory_exposure_alignment_claimed':False,
        'timing_limit':('Native source camera poses match measured exposure timestamps. ' if config.get('native_nuscenes_root') else 'Camera poses use exported reference timestamps with measured exposure offsets. ')
            +'Original TCVV frame-reference trajectory expectations are retained; actor boxes are not asserted to be native-exposure interpolated.'}
    from driveaudit_eval.evaluation.detail_retention import enabled, compact_consistency_result
    if enabled(config):
        scene_result = compact_consistency_result(scene_result, output, output.parents[3])
        cleanup = config.get('metric_detail_cleanup')
        if cleanup:
            from tools.cleanup_output_gt_scratch import compact_terminal_scene
            receipt = compact_terminal_scene(output, cleanup['run_root'], output.parents[3],
                                             phase=cleanup['phase'], profile=config.get('profile','research'))
            if receipt['status'] in ('cleaned', 'already_cleaned'):
                scene_result['_native_metric_retention'] = dict(schema=receipt['schema'],
                    manifest=str((output/'retained_details/native_metrics_manifest.json').relative_to(output.parents[3])),
                    archive=receipt['archive'], archive_sha256=receipt['archive_sha256'])
    return scene_result


def _evaluate_scene_task(scene, config, output, dry_run=False, logger=None):
    logger=logger or logging.getLogger(__name__)
    try:
        if dry_run:
            required=[scene/rel for rel in ['calibration/cameras.json','calibration/reference_ego_poses.json','calibration/history_camera_poses.json','timeline.json','future/rgb_manifest.json','history/agents.csv','future/future_trajectories.csv']]
            if config.get('native_nuscenes_root'):required.extend(Path(config['native_nuscenes_root'])/'v1.0-trainval'/name for name in ['scene.json','sample.json','sample_data.json','ego_pose.json','calibrated_sensor.json'])
            missing=[str(p) for p in required if not p.is_file()]
            selected = ('mtpc',) if config.get('profile') == 'core' else ('mtpc','tcvv','tcvv_bench','tcvv_h','tcvv_o')
            disabled=[n for n in selected if not config.get('backends',{}).get(n,{}).get('enabled',False)]
            return {'status':'skipped' if missing or disabled else 'dry_run','scene':str(scene),'missing_files':missing,'disabled_required_backends':disabled}
        return evaluate_scene(scene,config,output,logger)
    except Exception as error:
        logger.exception('[CONSISTENCY] Full chain failed: %s',scene)
        backends={}
        try:
            from tools.cleanup_output_gt_scratch import read_native_json
            progress=read_native_json(Path(output)/'chain_progress.json')
            if isinstance(progress,dict):
                backends={name:row for name,row in progress.items()
                          if isinstance(row,dict) and row.get('execution_status')=='completed'}
        except (OSError,ValueError):
            pass
        return {'status':'failed','execution_status':'failed','scene':str(scene),
                'reason':f'{type(error).__name__}: {error}','backends':backends}


def _init_scene_worker():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def evaluate(config,context):
    workers=config.get('scene_workers',1)
    if type(workers) is not int or workers<1:
        raise ValueError('consistency.scene_workers must be a positive integer')
    logger=context.get('logger') or logging.getLogger(__name__)
    scenes=[Path(context['root'])/name for name in config['scene_paths']]
    outputs=[Path(context['output_dir'])/'consistency'/scene.parent.parent.name/scene.name/'full_chain' for scene in scenes]
    if len({p.resolve() for p in scenes})!=len(scenes) or len({p.resolve() for p in outputs})!=len(outputs):
        raise ValueError('Duplicate consistency scene path or output identity')
    if workers>1 and len(scenes)>1 and not context.get('dry_run'):
        from concurrent.futures import ProcessPoolExecutor
        from itertools import repeat
        from multiprocessing import get_context
        # Children inherit the scheduler's thread budget before importing NumPy.
        previous={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}
        for key in previous:
            os.environ.setdefault(key,'1')
        try:
            with ProcessPoolExecutor(max_workers=min(workers,len(scenes)), mp_context=get_context('spawn'), initializer=_init_scene_worker) as pool:
                results=list(pool.map(_evaluate_scene_task, scenes, repeat(config), outputs))
        finally:
            for key,value in previous.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
    else:
        results=[_evaluate_scene_task(scene,config,output,context.get('dry_run',False),logger) for scene,output in zip(scenes,outputs)]
    return {'status':'failed' if any(r['status']=='failed' for r in results) else 'skipped' if any(r['status']=='skipped' for r in results) else 'dry_run' if context.get('dry_run') else 'completed',
            'execution_scope':'all original enabled consistency backend stages','scenes':results}
