"""Lossless frame/mask adapter for input-gt participant self-reference checks."""
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from driveaudit_eval.trajectory.io import derive_scalar_motion
from driveaudit_eval.trajectory.schema import Trajectory
from driveaudit_eval.utils.io import write_json

CATEGORY_TYPES = {'vehicle': 'vehicle', 'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle', 'trailer': 'vehicle',
                  'motorcycle': 'vehicle', 'bicycle': 'bicycle', 'pedestrian': 'pedestrian'}


def load_scene(scene_path):
    scene_path = Path(scene_path)
    scene_id = scene_path.parents[1].name + '/' + scene_path.name
    timeline = json.loads((scene_path / 'timeline.json').read_text())
    if timeline.get('unit') != 'microseconds_relative_to_frame_0':
        raise ValueError('input-gt trajectory timeline must declare microseconds_relative_to_frame_0')
    frames = timeline['frames']
    frame_ids = [int(item['frame_index']) for item in frames]
    if frame_ids != list(range(len(frames))):
        raise ValueError('input-gt requires ordered consecutive frame indices')
    timestamps = np.asarray([item['reference_time_us'] for item in frames], float) / 1e6
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError('input-gt reference timestamps must strictly increase')
    generation = json.loads((scene_path / 'generation_input.json').read_text())
    history_count = int(generation['reference_frame']) + 1
    rows = defaultdict(dict)
    excluded = defaultdict(set)
    source_files = [scene_path / 'timeline.json', scene_path / 'generation_input.json']
    for relative in ('history/agents.csv', 'future/future_trajectories.csv'):
        path = scene_path / relative
        source_files.append(path)
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                category, agent_id = row['category'], row['agent_id']
                if category not in CATEGORY_TYPES:
                    excluded[category].add(agent_id)
                    continue
                frame = int(row['frame_index'])
                if frame not in frame_ids or frame in rows[agent_id]:
                    raise ValueError(f'invalid or duplicate frame for {agent_id}: {frame}')
                rows[agent_id][frame] = row
    tracks = []
    for agent_id, records in sorted(rows.items()):
        n = len(timestamps)
        positions, headings, valid = np.zeros((n, 2)), np.zeros(n), np.zeros(n, bool)
        first = next(iter(records.values()))
        if any(row['category'] != first['category'] for row in records.values()):
            raise ValueError(f'category changes for {agent_id}')
        for frame, row in records.items():
            valid[frame] = row['valid'].lower() in ('true', '1')
            if valid[frame]:
                positions[frame] = [float(row['x']), float(row['y'])]
                headings[frame] = float(row['yaw'])
        speed, acceleration = derive_scalar_motion(positions, valid, timestamps)
        track = Trajectory(scene_id=scene_id, agent_id=agent_id, agent_type=CATEGORY_TYPES[first['category']],
                           source='ground_truth', coordinate_system='history_last_ego', timestamps=timestamps.copy(),
                           positions=positions, headings=headings, velocities=speed, accelerations=acceleration,
                           valid_mask=valid, shape={'length': float(first['length']), 'width': float(first['width'])},
                           metadata={'dt': float(np.median(np.diff(timestamps))), 'heading_unit': 'rad',
                                     'history_frames': history_count, 'source_category': first['category'],
                                     'predict': bool(np.any(valid[history_count:])),
                                     'coordinate_frame': f'fixed_history_frame_{history_count - 1}_ego_x_forward_y_left',
                                     'missing_point_policy': 'zero inactive storage placeholders; valid_mask false; no interpolation',
                                     'shape_policy': 'constant first available annotation dimensions required by existing scalar-shape metric schema',
                                     'derived_fields': ['velocity and acceleration from adjacent valid positions and native reference timestamps']})
        errors = track.validate()
        if errors:
            raise ValueError(f'{agent_id}: {errors}')
        tracks.append(track)
    source_files += [p for p in (scene_path / 'map').glob('*') if p.is_file()]
    source_files += [scene_path / name for name in ('generation_provenance.json', 'scene_context.json') if (scene_path / name).is_file()]
    provenance = {'mode': 'input_gt_self_reference', 'scene_path': str(scene_path.resolve()),
                  'scene_id': scene_id, 'coordinate_system': 'history_last_ego',
                  'timestamps': 'actual timeline reference_time_us / 1e6; no nominal FPS replacement',
                  'category_mapping': CATEGORY_TYPES, 'excluded_unsupported_categories': {k: sorted(v) for k, v in excluded.items()},
                  'ego_scope': 'ego evaluated in agent module; ego CSV has no participant shape',
                  'files': {str(p.relative_to(scene_path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    return tracks, provenance


def prepare(config, context):
    scenes = []
    settings = {key: value for key, value in config.items() if key not in ('scene_paths', 'resume', 'retain_intermediates')}
    source_paths = list(Path(__file__).parent.glob('*.py')) + list((Path(__file__).parents[1] / 'utils/map_provider').glob('*.py'))
    source_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(source_paths))).hexdigest()
    folder = Path(context['output_dir']) / 'trajectory_inputs'
    for value in config.get('scene_paths', []):
        path = Path(context['root']) / value
        tracks, provenance = load_scene(path)
        native = config.get('native_map') or {}
        if native.get('dataroot') and (path / 'scene_context.json').is_file():
            scene_context = json.loads((path / 'scene_context.json').read_text())
            if scene_context.get('dataset') == 'nuscenes':
                native_file = Path(native['dataroot']) / 'maps/expansion' / (scene_context['location'] + '.json')
                provenance['native_map_source'] = {'path': str(native_file), 'sha256': hashlib.sha256(native_file.read_bytes()).hexdigest() if native_file.is_file() else None}
        signature = hashlib.sha256(json.dumps([settings, provenance, source_hash], sort_keys=True).encode()).hexdigest()
        scene = {'input_gt_signature': signature, 'scene_id': provenance['scene_id'],
                 'input_gt_scene': str(path), 'map_file': str(path / 'map/semantic_map.npz'),
                 'input_gt_provenance': provenance}
        if config.get('retain_intermediates', True):
            output = folder / (path.parents[1].name + '__' + path.name + '.json')
            payload = [dict(track.to_dict(), source='generated') for track in tracks]
            write_json(output, {'trajectories': payload, 'input_gt_provenance': provenance})
            scene.update(submission_file=str(output), gt_file=str(output))
        else:
            # The file adapter labels both self-reference copies as generated.
            scene['_tracks'] = [replace(track, source='generated') for track in tracks]
        scenes.append(scene)
    return scenes
