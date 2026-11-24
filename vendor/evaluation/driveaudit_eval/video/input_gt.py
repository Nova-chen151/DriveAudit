"""Direct source RGB references with native exposure provenance; no image copies."""
from copy import deepcopy
import json
from pathlib import Path


def prepare(config, context):
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    from driveaudit_eval.utils.io import write_json
    root = Path(context['root'])
    output = Path(context['output_dir']) / 'input_gt_video'
    paths = [root / p for p in config['scene_paths']]
    references = [root / p for p in config.get('reference_scene_paths', config['scene_paths'])]
    if len(paths) != len(references):
        raise ValueError('Generated/GT video scene count mismatch')
    provenance = []
    scene_sources = []
    overlap_pairs = {}
    validation = []
    for scene, reference in zip(paths, references):
        key = scene.parent.parent.name + '__' + scene.name
        if any(row['scene_id'] == key for row in scene_sources):
            raise ValueError('Duplicate video scene identity: '+key)
        if key != reference.parent.parent.name + '__' + reference.name:
            raise ValueError('Generated/GT video scene identity mismatch')
        timeline = json.loads((scene / 'timeline.json').read_text())
        camera_data = json.loads((scene / 'calibration/cameras.json').read_text())['cameras']
        records = json.loads((scene / 'future/rgb_manifest.json').read_text())
        if config.get('universal_protocol'):
            from PIL import Image
            from driveaudit_eval.video.geometry import overlapping_camera_pairs
            if len(camera_data) != 6:
                raise ValueError(f'{key}: expected six cameras')
            decoded = 0
            for camera in camera_data:
                for kind, indices in (('history', range(10)), ('future', range(10,90))):
                    actual = {int(p.stem): p for p in (scene/kind/'rgb'/camera['camera']).glob('*.jpg')}
                    if set(actual) != set(indices):
                        raise ValueError(f'{key}/{camera["camera"]}/{kind}: incomplete video')
                    for path in actual.values():
                        with Image.open(path) as im:
                            im.load()
                            if list(im.size) != camera['challenge_image_size']:
                                raise ValueError(f'{path}: image size differs from calibration')
                        decoded += 1
            pairs, graph = overlapping_camera_pairs(camera_data)
            if not pairs:
                raise ValueError(f'{key}: no overlapping camera frusta')
            overlap_pairs[key] = pairs
            validation.append(dict(scene_id=key, status='VALID', decoded_frames=decoded, overlap_graph=graph))
        image_records = {(int(row['frame_index']), row['camera']): row for row in records['images']}
        times = np.array([row['reference_time_us'] for row in timeline['frames']], float) / 1e6
        poses = np.asarray(json.loads((scene / 'calibration/reference_ego_poses.json').read_text())['ego_to_anchor'])
        if len(times) != len(poses) or np.any(np.diff(times) <= 0):
            raise ValueError(f'{key}: invalid pose timestamps')
        rotations = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))
        frames = []
        excluded = []
        for frame in timeline['frames']:
            index = int(frame['frame_index'])
            metadata = {}
            for camera in camera_data:
                name = camera['camera']
                record = image_records.get((index, name), {})
                timing = record or frame.get('cameras', {}).get(name, {})
                if 'exposure_time_us' not in timing:
                    excluded.append({'frame_index': index, 'camera': name, 'reason': 'exposure_time_unavailable'})
                    continue
                exposure = timing['exposure_time_us'] / 1e6
                synchronized = record.get('synchronized_within_75ms', True)
                if not synchronized or exposure < times[0] or exposure > times[-1]:
                    excluded.append({'frame_index': index, 'camera': name, 'reason': 'unsynchronized_or_pose_extrapolation_required'})
                    continue
                transform = np.eye(4)
                transform[:3, :3] = rotations([exposure]).as_matrix()[0]
                transform[:3, 3] = [np.interp(exposure, times, poses[:, axis, 3]) for axis in range(3)]
                transform = transform @ np.asarray(camera['camera_to_ego_4x4'], float)
                metadata[name] = {'K': camera['challenge_intrinsics'], 'R': transform[:3, :3].tolist(),
                                  't': transform[:3, 3].tolist(), 'image_size': camera['challenge_image_size']}
            frames.append({'frame_index': index, 'cameras': metadata})
        metadata_path = output / 'metadata' / (key + '.json')
        write_json(metadata_path, {'frames': frames})
        source_record = dict(scene_id=key, metadata_path=str(metadata_path.resolve()))
        for kind, source in [('future_images', scene / 'future/rgb'), ('real_reference_images', reference / 'future/rgb'), ('history_images', scene / 'history/rgb')]:
            source_record[kind] = {}
            for camera in camera_data:
                name = camera['camera']
                selected = []
                for image in sorted((source / name).glob('*.jpg')):
                    record = image_records.get((int(image.stem), name), {}) if kind != 'history_images' else {}
                    # Preserve the source order; do not invent an exposure for exceptions.
                    if record.get('synchronized_within_75ms', True) is False and not config.get('universal_protocol'):
                        continue
                    # Keep the source frame name even when its RGB is an existing symlink.
                    selected.append(str(image.absolute()))
                source_record[kind][name] = selected
        scene_sources.append(source_record)
        provenance.append({'scene_id': key, 'source_scene': str(scene), 'reference_scene':str(reference),
                           'comparison': 'same_GT_RGB_in_both_domains' if scene.resolve()==reference.resolve() else 'generated_RGB_vs_paired_GT_RGB',
                           'timeline': timeline, 'rgb_manifest': records, 'geometry_exclusions': excluded,
                           'geometry_frame': 'camera_exposure_pose_in_history_frame_9_ego',
                           'pose_interpolation': 'linear_translation_and_SO3_slerp_no_extrapolation'})
    write_json(output / 'provenance.json', {'source': 'ground_truth_oracle', 'scenes': provenance,
               'time_policy': 'native_order', 'fixed_uniform_fps_claim': False,
               'fvd_scope': 'ordered stored exposures; nominal FPS is not a measured exposure rate'})
    cfg = deepcopy(config)
    if config.get('universal_protocol'):
        write_json(output/'generated_video_validation_gate.json', {'status':'VALID', 'scenes':validation})
        cfg.setdefault('metrics', {}).setdefault('loftr', {}).update(
            auto_overlap_pairs=True, camera_pairs_by_scene=overlap_pairs)
    if not paths:
        raise ValueError('Video scene selection must not be empty')
    cfg['scene_sources'] = scene_sources
    cfg['paths'] = {**cfg.get('paths', {}), 'dataset_root': str(output), 'generated_root': str(paths[0] / 'future/rgb'),
                    'real_reference_root': str(references[0] / 'future/rgb'), 'history_root': str(paths[0] / 'history/rgb'),
                    'metadata_root': str(output / 'metadata'), 'evaluation_manifest': None}
    cfg.setdefault('geometry', {})['extrinsic_convention'] = 'sensor_to_ego'
    cfg.setdefault('sampling', {})['time_policy'] = 'native_order'
    cfg['input_source'] = 'ground_truth_oracle' if all(a.resolve()==b.resolve() for a,b in zip(paths,references)) else 'paired_generated_RGB'
    return cfg, {'source': cfg['input_source'], 'model_generation_executed': False, 'scene_count': len(paths),
                 'provenance': str(output / 'provenance.json'), 'time_policy': 'native_order_no_uniform_fps_claim'}
