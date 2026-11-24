"""Panacea native camera stripes and legal GT2 autoregressive conditioning."""
import numpy as np
import torch
from PIL import Image
from genbench_gt2.constants import IMAGE_SIZE
from genbench_gt2.geometry import scaled_calibration
from sgm.data.genbench_gt2_native import render_frame_conditions, render_rays

# Released dataset final_dim=(256,512); intra/cross-view attention requires 2:1 views.
MODEL_CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT')


def split_panacea_views(image):
    if image.ndim != 3 or image.shape[0] != 3 or image.shape[-1] != 3072:
        raise ValueError('Expected [3,H,3072] Panacea panorama')
    return dict(zip(MODEL_CAMERAS, image.split(512, dim=-1)))


def build_panacea_window(inputs, window, condition_image=None):
    frames = (window.condition_frame,) + window.control_frames
    if condition_image is None:
        if window.condition_frame != 9:
            raise ValueError('Generated preceding window is required after history frame 9')
        stripes = []
        for camera in MODEL_CAMERAS:
            with Image.open(inputs.history_rgb[camera][9]) as image:
                stripes.append(np.asarray(image.convert('RGB').resize((512, 256)), dtype=np.float32))
        condition_image = torch.from_numpy(np.concatenate(stripes, axis=1).transpose(2, 0, 1).copy()) / 127.5 - 1
    if tuple(condition_image.shape) != (3, 256, 3072):
        raise ValueError('Condition must be [3,256,3072]')
    jpg = torch.full((8, 3, 256, 3072), -1.0)
    jpg[0] = condition_image.detach().cpu()
    final_cond_zero = torch.zeros_like(jpg)
    final_cond_zero[0] = jpg[0]
    controls = torch.empty((8, 19, 256, 3072), dtype=torch.float32)
    calibrations = {name: scaled_calibration(inputs.calibrations[name], IMAGE_SIZE, (512, 256)) for name in MODEL_CAMERAS}
    rays = render_rays(calibrations, (512, 256))
    # Fill stripes directly, avoiding a second 600MB temporary panorama stack.
    for ti, frame in enumerate(frames):
        conditions = render_frame_conditions(inputs, frame, MODEL_CAMERAS, calibrations=calibrations, rays=rays)
        for ci, camera in enumerate(MODEL_CAMERAS):
            c = conditions[camera]
            arr = np.concatenate((c.box_rgb, c.depth_by_class, c.hdmap_rgb, c.ray_rgb), axis=-1)
            controls[ti, :, :, ci * 512:(ci + 1) * 512] = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float() / 255.0
    return {'jpg': jpg, 'cond_img': controls, 'final_cond_zero': final_cond_zero,
            'txt': inputs.prompt, 'control_frame_indices': frames,
            'filenames': [[f'{inputs.scene.dataset}/{inputs.scene.scene_id}/{camera}/{frame:06d}.jpg' for camera in MODEL_CAMERAS] for frame in frames]}


TEMPORAL_ADAPTER_VERSION = 'last-reference-reversed-v2'


def reference_windows(reference_mode):
    """Keep reference geometry tied to the image actually supplied to the model."""
    from dataclasses import replace
    from genbench_gt2.temporal import panacea_plan
    windows = panacea_plan().windows
    if reference_mode == 'history-anchor':
        return tuple(replace(window, condition_frame=9) for window in windows)
    if reference_mode == 'autoregressive':
        return windows
    raise ValueError('Unknown reference mode: {}'.format(reference_mode))


def prepare_native_panacea_batch(batch, device):
    """Put the known frame in the released checkpoint's last reference slot.

    GT2 remains chronological outside this boundary. Internally all eight frames
    are reversed, including their geometry: future 16..10 followed by history 9.
    This is a temporal-reversal adaptation of last-frame-conditioned inference,
    not an assertion that the checkpoint natively supports first-frame inputs.
    """
    result = {key: batch[key].flip(0).unsqueeze(0).to(device)
              for key in ('jpg', 'cond_img', 'final_cond_zero')}
    result['txt'] = [batch['txt']]
    return result


def restore_panacea_timeline(samples):
    """Return native reverse-order samples to [reference, future1, ..., future7]."""
    return samples.flip(0)
