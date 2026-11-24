from driveaudit_eval.consistency.input_schema import _read, _rigid
"""Read the exported input-gt schema and measure RGB motion with existing kernels."""
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from driveaudit_eval.research.consistency.observed_flow.calibration import calibrate
from driveaudit_eval.research.consistency.observed_flow.gt_expected_motion import build as expected_motion
from driveaudit_eval.research.consistency.observed_flow.gt_instance_region import build as build_regions
from driveaudit_eval.research.consistency.observed_flow.image_motion import extract
from driveaudit_eval.research.consistency.observed_flow.projection import project_box
from driveaudit_eval.research.consistency.observed_flow.schema import write_json, write_csv
from driveaudit_eval.research.consistency.observed_flow.time_alignment import interpolate_observations
from driveaudit_eval.research.consistency.observed_flow.visual_motion_consistency import metric_rows, apply_support, summarize
from driveaudit_eval.research.consistency.trajectory_verification.visibility import apply_visibility






def _calibration(matrix, intrinsics):
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {'camera_intrinsics': intrinsics, 'sensor2ego_rotation': quaternion[[3, 0, 1, 2]].tolist(),
            'sensor2ego_translation': matrix[:3, 3].tolist(),
            'ego2global_rotation': [1, 0, 0, 0], 'ego2global_translation': [0, 0, 0]}


def _load_scene(scene, output, cfg):
    import cv2
    cameras = _read(scene / 'calibration/cameras.json')
    timeline = _read(scene / 'timeline.json')
    pose_data = _read(scene / 'calibration/reference_ego_poses.json')
    history = _read(scene / 'calibration/history_camera_poses.json')
    manifest = _read(scene / 'future/rgb_manifest.json')
    if pose_data['coordinate_frame'] not in {'history_frame_9_ego', 'history_last_ego'} or history['coordinate_frame'] not in {'history_frame_9_ego', 'history_last_ego'}:
        raise ValueError('Unsupported pose coordinate frame')
    frames = timeline['frames']
    if [row['frame_index'] for row in frames] != list(range(90)):
        raise ValueError('Expected complete 90-frame exported timeline')
    times = np.asarray([row['reference_time_us'] for row in frames], float) / 1e6
    if not np.all(np.diff(times) > 0):
        raise ValueError('Reference timestamps must increase strictly')
    poses = np.asarray([_rigid(value, f'ego pose {i}') for i, value in enumerate(pose_data['ego_to_anchor'])])
    if len(poses) != len(times):
        raise ValueError('Pose count does not match timeline')
    slerp = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))
    camera_rows = {row['camera']: row for row in cameras['cameras']}
    if set(camera_rows) != set(cameras['camera_order']) or len(camera_rows) != 6:
        raise ValueError('Six unique named cameras required')
    hposes = {(row['frame_index'], row['camera']): _rigid(row['camera_to_anchor_4x4'], 'history camera pose') for row in history['rows']}
    future = {(row['frame_index'], row['camera']): row for row in manifest['images']}
    tracks = defaultdict(list)
    for relative in ('history/agents.csv', 'future/future_trajectories.csv'):
        with (scene / relative).open() as stream:
            for row in csv.DictReader(stream):
                if row['valid'].lower() not in {'true', '1'}:
                    continue
                box = {key: float(row[key]) for key in ('x', 'y', 'z', 'width', 'length', 'height', 'yaw')}
                if not all(np.isfinite(list(box.values()))) or min(box[key] for key in ('width', 'length', 'height')) <= 0:
                    raise ValueError('Invalid source cuboid')
                frame = int(row['frame_index'])
                tracks[row['agent_id']].append({'frame_index': frame, 'timestamp': float(times[frame]),
                    'box_bev': box, 'velocity': [float('nan'), float('nan')], 'category': row['category']})
    # The reused interpolator requires velocity; NaN explicitly marks it unavailable and it is never scored.
    aligned, exclusions, source_rows = [], Counter(), []
    for camera in cameras['camera_order']:
        camera_cfg = camera_rows[camera]
        extrinsic = _rigid(camera_cfg['camera_to_ego_4x4'], camera)
        intrinsics = np.asarray(camera_cfg['challenge_intrinsics'], float)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0:
            raise ValueError(f'{camera}: invalid intrinsics')
        size = tuple(camera_cfg['challenge_image_size'])
        if size != tuple(cfg['image']['formal_size']):
            raise ValueError('Image size differs from configured metric resolution')
        for frame in range(90):
            metadata = frames[frame]['cameras'].get(camera) if frame < 10 else future.get((frame, camera))
            if not metadata or 'exposure_time_us' not in metadata:
                raise ValueError(f'Missing measured exposure: {frame}/{camera}')
            exposure = float(metadata['exposure_time_us']) / 1e6
            image = scene / ('history' if frame < 10 else 'future') / 'rgb' / camera / f'{frame:06d}.jpg'
            if not image.is_file():
                raise FileNotFoundError(image)
            decoded = cv2.imread(str(image))
            if decoded is None or decoded.shape[1::-1] != size:
                raise ValueError(f'RGB decode/size mismatch: {image}')
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if frame >= 10 and digest != metadata['sha256']:
                raise ValueError(f'Future RGB hash mismatch: {image}')
            target = output / 'rgb' / camera / image.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(image.resolve())
            source_rows.append({'frame_index': frame, 'camera': camera, 'file': str(image), 'sha256': digest,
                'exposure_time_s': exposure, 'reference_time_s': float(times[frame]),
                'reused_source_frame': bool(metadata.get('reused_source_frame', False)),
                'source_camera': camera_cfg.get('source_camera', camera)})
            if frame < 10:
                camera_pose = hposes[(frame, camera)]
            elif times[0] <= exposure <= times[-1]:
                ego = np.eye(4)
                ego[:3, :3] = slerp(exposure).as_matrix()
                ego[:3, 3] = [np.interp(exposure, times, poses[:, axis, 3]) for axis in range(3)]
                camera_pose = ego @ extrinsic
            else:
                exclusions['exposure_outside_ego_pose_range'] += 1
                continue
            cal = _calibration(camera_pose, intrinsics.tolist())
            for agent, observations in tracks.items():
                if frame not in {row['frame_index'] for row in observations}:
                    continue
                sample, protocol, error = interpolate_observations(observations, exposure, 0.)
                if sample is None:
                    exclusions['box_exposure_outside_contiguous_track'] += 1
                    continue
                projected = project_box(sample['box_bev'], cal, size, size, float(cfg['projection']['minimum_depth_m']))
                aligned.append({'agent_id': agent, 'agent_type': 'traffic_participant', 'category': sample['category'],
                    'frame_index': frame, 'camera_name': camera, 'timestamp': exposure,
                    'box_reference': sample['box_bev'], 'alignment_protocol': protocol,
                    'timestamp_error_s': error, **projected})
    apply_visibility(aligned, cfg['projection'])
    for row in aligned:
        row['visibility_reason'] = row['observability_status']
    return aligned, source_rows, dict(exclusions), cameras


def evaluate_scene(scene, config, output):
    import cv2
    cv2.setNumThreads(1)
    started = time.time()
    cfg = deepcopy(config['backends']['gt_vtc'])
    cfg['image']['formal_size'] = list(cfg['image']['low_resolution_size'])
    cfg['inputs'] = {'gt_images': str(output / 'rgb')}
    cfg['output'] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    projections, sources, exclusions, cameras = _load_scene(scene, output, cfg)
    write_json(output / 'source_manifest.json', {'scene': str(scene), 'images': sources, 'exclusions': exclusions})
    regions = build_regions(projections, cfg, output)
    expectations = expected_motion(projections, list(cfg['motion']['candidate_deltas']))
    zero_delta = sum(row['delta_seconds'] <= 0 for row in expectations)
    expectations = [row for row in expectations if row['delta_seconds'] > 0]
    history = metric_rows(extract(cfg, projections, regions, expectations, output / 'history', (0, 10)), expectations)
    primary = int(cfg['motion']['window_delta'])
    frozen = calibrate([row for row in history if row['delta_frames'] == primary], cfg, output)
    apply_support(history, frozen)
    future = metric_rows(extract(cfg, projections, regions, expectations, output / 'future', (10, 90)), expectations)
    apply_support(future, frozen)
    hm = summarize([row for row in history if row['delta_frames'] == primary], output, 'history')
    fm = summarize([row for row in future if row['delta_frames'] == primary], output, 'future')
    region_count = sum(row['region_scorable'] for row in regions)
    result = {'status': 'diagnostic_only_source_visibility_unavailable', 'execution_status': 'completed',
        'scene': str(scene), 'dataset': scene.parent.parent.name, 'protocol': 'input_gt_rgb_motion_self_consistency',
        'observed_motion_source': 'existing_gt_vtc_rgb_lk_farneback_orb', 'future_rgb_role': 'evaluation_reference',
        'projected_flow_substituted_for_observation': False, 'generation_or_agent_inference_run': False,
        'history': hm, 'future': fm, 'history_frozen_thresholds': frozen,
        'effective_config': cfg,
        'metric_resolution': cfg['image']['formal_size'],
        'future_window_coverage': {
            'projectable_primary_windows': sum(r['frame_index'] >= 10 and r['delta_frames'] == primary for r in expectations),
            'geometrically_visible_primary_windows': sum(r['frame_index'] >= 10 and r['delta_frames'] == primary and r['expected_visible'] for r in expectations),
            'rgb_extracted_primary_windows': fm['expected_window_count'],
            'rgb_scorable_primary_windows': fm['scorable_window_count'],
            'denominator_note': 'Original backend rates condition on region-scorable extracted windows; excluded geometry/region windows are not successful evidence.'},
        'projection_count': len(projections), 'geometrically_visible_count': sum(row['expected_visible'] for row in projections),
        'scorable_region_count': region_count, 'source_rgb_count': len(sources),
        'nonpositive_exposure_interval_windows_excluded': zero_delta, 'alignment_exclusions': exclusions,
        'reused_source_rgb_count': sum(row['reused_source_frame'] for row in sources),
        'maximum_exposure_offset_s': max(abs(row['exposure_time_s']-row['reference_time_s']) for row in sources),
        'camera_mapping': [{key: row.get(key) for key in ('camera', 'source_camera', 'direction_error_deg', 'distortion_processing')} for row in cameras['cameras']],
        'gates': {'source_visibility': {'passed': False, 'status': 'not_available_export_has_no_source_visibility_tokens'},
                  'rgb_evidence': {'passed': fm['scorable_window_count'] > 0, 'status': 'measured' if fm['scorable_window_count'] else 'unscorable'}},
        'limitations': ['TCVV geometric visibility used without fabricating GT visibility tokens; original GT upper-bound acceptance is unavailable.',
                       'Future camera poses interpolate reference ego rigid poses at measured exposure times; native camera ego poses are not exported.',
                       'Boxes interpolate contiguous frame samples at measured exposure times; out-of-range samples are excluded.',
                       'Original GT-VTC v1 weak-support rule is preserved; weak support is not a contradiction detector.',
                       'Projected cuboid cores are approximate object regions, not independent segmentation truth.'],
        'runtime_seconds': time.time()-started}
    write_json(output / 'metrics.json', result)
    return result


def evaluate(config, context):
    root = Path(context['root'])
    paths = config.get('scene_paths')
    if not paths:
        raise ValueError('input_gt consistency requires explicit scene_paths')
    results = []
    for value in paths:
        scene = Path(value)
        if not scene.is_absolute():
            scene = root / scene
        output = Path(context['output_dir']) / 'consistency' / scene.parent.parent.name / scene.name
        if context.get('dry_run'):
            results.append({'scene': str(scene), 'status': 'dry_run' if scene.is_dir() else 'skipped'})
            continue
        context['logger'].info('[CONSISTENCY] RGB motion %s', scene)
        try:
            result = evaluate_scene(scene, config, output)
        except (ImportError, FileNotFoundError) as error:
            result = {'scene': str(scene), 'status': 'skipped', 'reason': str(error)}
        except Exception as error:
            context['logger'].exception('[CONSISTENCY] %s', scene)
            result = {'scene': str(scene), 'status': 'failed', 'reason': f'{type(error).__name__}: {error}'}
        results.append(result)
    status = 'failed' if any(row['status'] == 'failed' for row in results) else 'partial'
    return {'status': status, 'protocol': 'input_gt_rgb_motion_self_consistency', 'scenes': results}
