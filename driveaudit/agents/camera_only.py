"""Inference-only boundaries for legacy MMDet templates; no sensor data loading."""
import hashlib

META_KEYS = {'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction',
             'img_norm_cfg', 'img_nrom_cfg', 'can_bus', 'scene_token', 'sample_idx',
             'lidar2img', 'box_type_3d', 'box_mode_3d'}
INPUT_KEYS = {
    'uniad': {'img', 'img_metas', 'l2g_t', 'l2g_r_mat', 'timestamp', 'command'},
    'vad': {'img', 'img_metas', 'ego_fut_cmd', 'ego_lcf_feat'},
}


def inference_inputs(data, agent):
    """Discard evaluation templates, retaining calibration matrices (not LiDAR)."""
    def metadata(value):
        if isinstance(value, dict):
            clean = {k: v for k, v in value.items() if k in META_KEYS}
            if agent == 'uniad':
                # Native panseg_head uses this legacy name only to label output;
                # never retain a source point-cloud path or token from a template.
                identity = str(clean.get('scene_token', 'scene')) + ':' + str(clean.get('sample_idx', 0))
                clean['pts_filename'] = 'rgb_panoptic_' + hashlib.sha256(identity.encode()).hexdigest()[:20] + '.png'
            return clean
        if isinstance(value, list):
            return [metadata(v) for v in value]
        raise TypeError('Expected unpacked image metadata')
    out = {key: value for key, value in data.items() if key in INPUT_KEYS[agent]}
    out['img_metas'] = metadata(out['img_metas'])
    return out


def stable_request_seed(scene_id, frame_id):
    """Stable scene/frame RNG identity, independent of Python hash or scene order."""
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError('Deterministic request requires a nonempty scene_id')
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError('Deterministic request requires a nonnegative integer frame_id')
    # Fixed protocol salt preserves previously issued scene/frame seeds.
    salt = bytes.fromhex('44726976654172656e612d74656d706f72616c2d53522d763100')
    identity = salt + f'{scene_id}\0{frame_id}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], 'big') % (2**63)


def seed_temporal_request(parameters):
    import os
    if os.getenv('DRIVEAUDIT_SENSOR_PROTOCOL') != 'camera_only' and os.getenv('DRIVEAUDIT_DETERMINISTIC_AGENT') != '1':
        return None
    import torch
    seed = stable_request_seed(parameters.get('scene_id'), parameters.get('frame_id'))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
