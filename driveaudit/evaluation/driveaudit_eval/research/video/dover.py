from pathlib import Path
import numpy as np
from driveaudit_eval.video.visual_quality import unavailable

def evaluate_dover(samples, cfg, state):
    """Optional official DOVER model/view sampler, kept outside startup imports."""
    import importlib
    import sys
    import tempfile
    source, weight, settings = [Path(cfg.get(key, '')).expanduser() for key in ('source_dir', 'weights_path', 'config_path')]
    for path in (source/'dover', weight, settings):
        if not path.exists() or path == Path('.'):
            return unavailable('DOVER source/checkpoint/config missing: '+str(path))
    import torch
    import cv2
    import yaml
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(source))
        datasets = importlib.import_module('dover.datasets')
        models = importlib.import_module('dover.models')
    finally:
        sys.path[:] = old_path
    options = yaml.safe_load(settings.read_text())
    device = state.device if torch.cuda.is_available() else 'cpu'
    model = models.DOVER(**options['model']['args']).to(device).eval()
    model.load_state_dict(torch.load(weight, map_location='cpu'), strict=True)
    dopt = options['data'][cfg.get('dataset_profile', 'val-l1080p')]['args']
    samplers = {}
    for key, value in dopt['sample_types'].items():
        samplers[key] = (datasets.UnifiedFrameSampler(value['clip_len']//value['t_frag'], value['t_frag'],
                value['frame_interval'], value['num_clips']) if 't_frag' in value else
                datasets.UnifiedFrameSampler(value['clip_len'], value['num_clips'], value['frame_interval']))
    mean = torch.tensor([123.675, 116.28, 103.53])
    std = torch.tensor([58.395, 57.12, 57.375])
    rows = []
    for scene in samples:
        for camera, paths in scene.future_images.items():
            if not paths:
                continue
            with tempfile.TemporaryDirectory(prefix='driveaudit_eval_dover_') as directory:
                video = Path(directory)/'ordered_frames.avi'
                first = cv2.imread(str(paths[0]))
                height, width = first.shape[:2]
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'),
                                        float(state.cfg['sampling']['source_fps']), (width, height))
                if not writer.isOpened():
                    raise RuntimeError('lossless FFV1 video writer unavailable')
                try:
                    for path in paths:
                        frame = cv2.imread(str(path))
                        if frame is None or frame.shape[:2] != (height, width):
                            raise ValueError('DOVER sequence contains unreadable/differently sized frames')
                        writer.write(frame)
                finally:
                    writer.release()
                views, frame_indices = datasets.spatial_temporal_view_decomposition(str(video), dopt['sample_types'], samplers)
                for key, value in views.items():
                    clips = dopt['sample_types'][key].get('num_clips', 1)
                    views[key] = ((value.permute(1, 2, 3, 0)-mean)/std).permute(3, 0, 1, 2).reshape(
                        value.shape[0], clips, -1, *value.shape[2:]).transpose(0, 1).to(device)
                with torch.inference_mode():
                    technical, aesthetic = [float(x.mean()) for x in model(views)]
                fusion = (technical-.1107)/.07355*.6104+(aesthetic+.08285)/.03774*.3896
                overall = float(1/(1+np.exp(-np.clip(fusion, -700, 700))))
                rows.append(dict(scene_id=scene.scene_id, camera=camera, dover_technical_score=technical,
                    dover_aesthetic_score=aesthetic, dover_overall_score=overall, status='valid',
                    source_frame_count=len(paths), sampled_frame_indices=frame_indices,
                    timing_policy='ordered_native_frames_in_lossless_FFV1_container; nominal_FPS_not_measured_exposure_rate'))
    return dict(status='valid' if rows else 'insufficient_samples', reason=None if rows else 'no_videos',
                scene_camera_metrics=rows, checkpoint=str(weight), config_path=str(settings),
                official_adapter='VQAssessment/DOVER evaluate_one_video.py model/view sampler and fusion')

