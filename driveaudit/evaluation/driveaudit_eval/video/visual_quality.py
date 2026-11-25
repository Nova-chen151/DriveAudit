"""Optional additions to the existing video graph, with per-input provenance."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json

import numpy as np

from driveaudit_eval.video.metrics.distribution_quality import cmmd
from driveaudit_eval.video.sampling import ImageSample
from driveaudit_eval.video.utils import read_image


def unavailable(reason, status='unavailable'):
    return dict(status=status, reason=reason, coverage=0.0)


def statistics(values, prefix):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return unavailable('no_valid_samples', 'insufficient_samples')
    if not np.isfinite(a).all():
        raise ValueError('non-finite quality scores')
    return {prefix+'_'+key: float(value) for key, value in zip(
        ('mean', 'std', 'p10', 'p50', 'p90', 'min', 'max'),
        (a.mean(), a.std(), *np.percentile(a, [10, 50, 90]), a.min(), a.max()))}


def image_samples(samples, domain):
    items, seen = [], set()
    for scene in samples:
        groups = scene.future_images if domain == 'fake' else scene.real_reference_images
        for camera, paths in sorted(groups.items()):
            for position, path in enumerate(paths):
                resolved = str(path.resolve())
                if domain == 'real' and resolved in seen:
                    continue
                seen.add(resolved)
                items.append(ImageSample(domain, scene.scene_id, camera, position,
                                         int(path.stem) if path.stem.isdigit() else position, path))
    return items


def aggregate_frames(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        if row.get(metric) is None:
            continue
        for level, scene, camera in [('scene_camera', row['scene_id'], row['camera']),
                                      ('scene', row['scene_id'], None),
                                      ('camera', None, row['camera']), ('overall', None, None)]:
            groups[level, scene, camera].append(row[metric])
    return [dict(level=level, scene_id=scene, camera=camera, sample_count=len(values),
                 aggregation='frame_weighted', status='valid', **statistics(values, metric))
            for (level, scene, camera), values in groups.items()]


def infer_batches(model, items, transform, device, batch_size, *, scalar=False):
    """Keep one model instance; halve OOM batches, then retry on CPU."""
    import torch
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    outputs, offset = [], 0
    while offset < len(items):
        count = min(batch_size, len(items)-offset)
        try:
            batch = torch.stack([transform(item.path) for item in items[offset:offset+count]]).to(device)
            with torch.inference_mode():
                result = model(batch)
            if scalar:
                if result.numel() != count:
                    raise ValueError('IQA must return exactly one score per image')
                result = result.reshape(count, 1)
            if result.ndim != 2 or result.shape[0] != count or not torch.isfinite(result).all():
                raise ValueError('invalid model output shape or non-finite output')
            outputs.append(result.detach().cpu().numpy())
            offset += count
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            if 'batch' in locals():
                del batch
            torch.cuda.empty_cache()
            if count > 1:
                batch_size = max(1, count//2)
            elif str(device).startswith('cuda'):
                model.to('cpu')
                if hasattr(model, 'device'):
                    model.device = torch.device('cpu')
                device = 'cpu'
            else:
                raise
    return np.concatenate(outputs), dict(final_batch_size=batch_size, final_device=str(device), forward_executed=True)


def evaluate(state, binding):
    options = state.cfg.get('visual_quality', {})
    import sys
    if options.get('python') and Path(options['python']).resolve() != Path(sys.executable).resolve():
        from copy import deepcopy
        from types import SimpleNamespace
        import subprocess
        import tempfile
        from driveaudit_eval.utils.io import write_json
        # Keep KID on the shared canonical Inception; isolate the other models.
        local_cfg = deepcopy(state.cfg)
        local_cfg['visual_quality'] = {'kid': options.get('kid', {})}
        local = evaluate(SimpleNamespace(**dict(vars(state), cfg=local_cfg)), binding)
        child_cfg = deepcopy(state.cfg)
        child_cfg['visual_quality'].pop('python')
        child_cfg['visual_quality']['kid'] = {'enabled': False}
        scenes = [dict(scene_id=s.scene_id, cameras=s.cameras, history_images=s.history_images,
                       future_images=s.future_images, real_reference_images=s.real_reference_images) for s in state.samples]
        with tempfile.TemporaryDirectory(prefix='driveaudit_eval_visual_models_') as directory:
            request, output = Path(directory)/'request.json', Path(directory)/'result.json'
            write_json(request, dict(cfg=child_cfg, samples=scenes, cache_dir=state.cache_dir,
                                     seed=state.seed, device=state.device, results=state.results))
            command = [options['python'], '-c',
                'import sys; from driveaudit_eval.video.visual_quality import _visual_worker; _visual_worker(sys.argv[1], sys.argv[2])',
                str(request), str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=options.get('timeout_s', 1800))
            if completed.returncode:
                reason = 'visual subprocess failed: '+completed.stderr[-2000:]
                for name in ('clipiqa', 'musiq', 'dino_temporal', 'dover'):
                    if options.get(name, {}).get('enabled', False):
                        local['metrics'][name] = unavailable(reason, 'unscorable')
                if options.get('cmmd', {}).get('enabled', False):
                    local['distribution']['cmmd'] = unavailable(reason, 'unscorable')
                return local
            result = json.loads(output.read_text())
            result['distribution'].update(local['distribution'])
            result['isolated_model_python'] = options['python']
            return result
    fake, real = image_samples(state.samples, 'fake'), image_samples(state.samples, 'real')
    result = dict(metrics={}, frame_metrics=[], aggregate_metrics=[], distribution={},
                  protocol='independent_unpaired_distribution; native_image_IQA')
    rows = [dict(scene_id=x.scene_id, camera=x.camera, frame_id=x.frame_id,
                 frame_position=x.frame_position, source_path=str(x.path.resolve())) for x in fake]
    for name, registered in [('clipiqa', 'clipiqa+'), ('musiq', 'musiq')]:
        cfg = options.get(name, {})
        if not cfg.get('enabled', False):
            continue
        try:
            import torch
            import pyiqa
            from PIL import Image
            if not fake:
                raise ValueError('no_generated_images')
            hub = Path(torch.hub.get_dir())
            default_weight = hub/'pyiqa'/('CLIP-IQA%2B_learned_prompts-603f3273.pth' if name == 'clipiqa' else 'musiq_koniq_ckpt-e95806b9.pth')
            weight = Path(cfg.get('weights_path', default_weight)).expanduser()
            assets = [weight] + ([hub/'clip/RN50.pt'] if name == 'clipiqa' else [])
            missing = [str(p) for p in assets if not p.is_file()]
            if missing:
                raise FileNotFoundError('checkpoint_missing: '+', '.join(missing))
            device = state.cfg['runtime'].get('device', 'cpu')
            if str(device).startswith('cuda') and not torch.cuda.is_available():
                device = 'cpu'
            kwargs = {'pretrained_model_path': str(weight)} if name == 'musiq' else {'pretrained': False}
            model = pyiqa.create_metric(registered, device=device, **kwargs)
            if name == 'clipiqa':
                prompts = torch.load(weight, map_location='cpu')
                expected = model.net.prompt_learner.ctx.shape
                if prompts.shape not in (expected, expected[1:]) or not torch.isfinite(prompts).all():
                    raise ValueError('CLIP-IQA+ learned prompt shape mismatch')
                with torch.no_grad():
                    # Official shared prompt checkpoint is 16x512; the learner expands it across classes.
                    model.net.prompt_learner.ctx.data = prompts.to(model.net.prompt_learner.ctx)
            def transform(path):
                return torch.from_numpy(np.array(read_image(path), copy=True)).permute(2, 0, 1).float()/255
            # Native image dimensions may vary: batch each shape separately.
            groups = defaultdict(list)
            for index, item in enumerate(fake):
                with Image.open(item.path) as im:
                    groups[im.size].append(index)
            audits = []
            for indices in groups.values():
                values, audit = infer_batches(model, [fake[i] for i in indices], transform, device,
                                              int(cfg.get('batch_size', 4)), scalar=True)
                audits.append(audit)
                device = audit['final_device']
                for index, value in zip(indices, values[:, 0]):
                    rows[index][name] = float(value)
            result['metrics'][name] = dict(status='valid', reason=None, coverage=1.0,
                model=registered, pyiqa_version=pyiqa.__version__, batches=audits,
                checkpoint_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in assets},
                **statistics([r[name] for r in rows], name))
            result['aggregate_metrics'].extend(aggregate_frames(rows, name))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, FileNotFoundError) as exc:
            result['metrics'][name] = unavailable(f'{type(exc).__name__}: {exc}')
        except Exception as exc:
            result['metrics'][name] = unavailable(f'{type(exc).__name__}: {exc}', 'unscorable')
        for row in rows:
            row[name+'_status'] = result['metrics'][name]['status']
            row[name+'_reason'] = result['metrics'][name]['reason']
    cfg = options.get('kid', {})
    if cfg.get('enabled', False):
        try:
            if not real or not fake:
                result['distribution']['kid'] = unavailable('empty_distribution', 'insufficient_samples')
            else:
                # Separate domain labels preserve the original FID execution audit.
                def forward(items, domain):
                    import torch
                    while True:
                        try:
                            return binding.run_fid_features(items, domain)
                        except RuntimeError as exc:
                            if 'out of memory' not in str(exc).lower():
                                raise
                            executor = binding._fid()
                            torch.cuda.empty_cache()
                            if executor.batch_size > 1:
                                executor.batch_size = max(1, executor.batch_size//2)
                            elif str(executor.device).startswith('cuda'):
                                executor.device = 'cpu'
                                executor.model.to('cpu')
                            else:
                                raise
                x, xa = forward(real, 'kid_real')
                y, ya = forward(fake, 'kid_generated')
                from driveaudit_eval.research.video.distribution import kid
                result['distribution']['kid'] = dict(kid(x, y, **{k: cfg[k] for k in
                    ('subset_size', 'num_subsets', 'seed') if k in cfg}),
                    backend=xa['provenance']['backend'],
                    feature_protocol=xa['provenance']['protocol'],
                    checkpoint_sha256=xa['provenance']['checkpoint_sha256'],
                    real_forward=xa, generated_forward=ya)
        except (ImportError, FileNotFoundError) as exc:
            result['distribution']['kid'] = unavailable(str(exc))
        except Exception as exc:
            result['distribution']['kid'] = unavailable(f'{type(exc).__name__}: {exc}', 'unscorable')
    for name, backbone in [('cmmd', 'clip'), ('dino_temporal', 'dino')]:
        cfg = options.get(name, {})
        if not cfg.get('enabled', False):
            continue
        try:
            pool = real+fake if name == 'cmmd' else fake
            if not pool:
                raise ValueError('no_images')
            features, audit = embedding_features(pool, cfg, state, backbone)
            if name == 'cmmd':
                metric = dict(cmmd(features[:len(real)], features[len(real):], **{k: cfg[k] for k in
                    ('sigma', 'scale', 'block_size', 'seed') if k in cfg}), embedding_audit=audit)
                result['distribution'][name] = metric
            else:
                result['metrics'][name] = dict(temporal_scores(fake, features, cfg, state), embedding_audit=audit)
        except (ImportError, FileNotFoundError, KeyError) as exc:
            target = result['distribution'] if name == 'cmmd' else result['metrics']
            target[name] = unavailable(f'{type(exc).__name__}: {exc}')
        except Exception as exc:
            target = result['distribution'] if name == 'cmmd' else result['metrics']
            target[name] = unavailable(f'{type(exc).__name__}: {exc}', 'unscorable')
    if options.get('dover', {}).get('enabled', False):
        try:
            from driveaudit_eval.research.video.dover import evaluate_dover
            result['metrics']['dover'] = evaluate_dover(state.samples, options['dover'], state)
        except (ImportError, FileNotFoundError) as exc:
            result['metrics']['dover'] = unavailable(f'{type(exc).__name__}: {exc}')
        except Exception as exc:
            result['metrics']['dover'] = unavailable(f'{type(exc).__name__}: {exc}', 'unscorable')
    result['frame_metrics'] = rows
    return result


def _visual_worker(request_path, output_path):
    from types import SimpleNamespace
    from driveaudit_eval.video.io import SceneSample
    from driveaudit_eval.video.runtime_state import set_reproducibility
    from driveaudit_eval.utils.io import write_json
    data = json.loads(Path(request_path).read_text())
    set_reproducibility(int(data['seed']))
    scenes = []
    for row in data['samples']:
        for key in ('history_images', 'future_images', 'real_reference_images'):
            row[key] = {camera: [Path(p) for p in paths] for camera, paths in row[key].items()}
        scenes.append(SceneSample(**row))
    data['samples'], data['cache_dir'] = scenes, Path(data['cache_dir'])
    write_json(Path(output_path), evaluate(SimpleNamespace(**data), None))


def embedding_features(items, cfg, state, backbone):
    import sys
    output = getattr(state, 'cfg', {}).get('output', {})
    cache_enabled = output.get('retention') != 'metrics' and output.get('save_cache', cfg.get('save_cache', True))
    if cfg.get('python') and Path(cfg['python']).resolve() != Path(sys.executable).resolve():
        import subprocess
        import tempfile
        from dataclasses import asdict
        from driveaudit_eval.utils.io import write_json
        with tempfile.TemporaryDirectory(prefix='driveaudit_eval_embedding_') as directory:
            request, output = Path(directory)/'request.json', Path(directory)/'result.npz'
            child_cfg = {k: v for k, v in cfg.items() if k != 'python'}
            child_cfg['save_cache'] = cache_enabled
            write_json(request, dict(items=[asdict(x) for x in items], cfg=child_cfg,
                backbone=backbone, state=dict(device=state.device, seed=state.seed, cache_dir=str(state.cache_dir))))
            command = [cfg['python'], '-c',
                'import sys; from driveaudit_eval.video.visual_quality import _embedding_worker; _embedding_worker(sys.argv[1], sys.argv[2])',
                str(request), str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=cfg.get('timeout_s', 1800))
            if completed.returncode:
                raise RuntimeError('embedding subprocess failed: '+completed.stderr[-2000:])
            with np.load(output, allow_pickle=False) as data:
                return data['features'].copy(), dict(json.loads(str(data['audit'].item())), python=cfg['python'])
    import torch
    from PIL import Image
    from driveaudit_eval.video.cache import DiskCache
    weight = Path(cfg['weights_path']).expanduser()
    if not weight.is_file():
        raise FileNotFoundError('checkpoint_missing: '+str(weight))
    h = hashlib.sha256()
    with weight.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    if backbone == 'clip' and h.hexdigest() != '3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02':
        raise ValueError('CMMD requires the official OpenAI ViT-L/14@336px checkpoint')
    preprocessing = ('center_square_then_bicubic336_clip_normalization_L2_fp32_v2' if backbone == 'clip'
                     else 'resize_short256_bicubic_crop224_imagenet_cls_L2_fp32_v2')
    provenance = dict(backbone=backbone, checkpoint=str(weight), checkpoint_sha256=h.hexdigest(),
                      preprocessing=preprocessing, seed=cfg.get('seed', state.seed))
    signatures = [(str(x.path.resolve()), x.path.stat().st_size, x.path.stat().st_mtime_ns) for x in items]
    key = [provenance, signatures]
    cache = DiskCache(Path(cfg.get('cache_dir', state.cache_dir/'visual_embeddings')), enabled=cache_enabled)
    cached = cache.load_npz('features', key)
    if cached is not None:
        values = cached['features']
        if values.shape != (len(items), 768 if backbone == 'clip' else 384) or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-4):
            raise ValueError('invalid feature cache')
        return values, dict(provenance, cache_hit=True, forward_executed=False)
    device = state.device if torch.cuda.is_available() else 'cpu'
    if backbone == 'clip':
        import clip
        model, _ = clip.load(str(weight), device=device, jit=False)
        if getattr(model.visual, 'input_resolution', None) != 336 or model.visual.output_dim != 768:
            raise ValueError('CMMD requires OpenAI ViT-L/14@336px, 768 projected dimensions')
        mean, std = (.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711)
        def transform(path):
            im = read_image(path)
            width, height = im.size
            side = min(width, height)
            left, top = (width-side)//2, (height-side)//2
            im = im.crop((left, top, left+side, top+side)).resize((336, 336), Image.Resampling.BICUBIC)
            x = torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1).float()/255
            return (x-torch.tensor(mean)[:, None, None])/torch.tensor(std)[:, None, None]
        class Encoder(torch.nn.Module):
            def __init__(self, network):
                super().__init__()
                self.network = network
            def forward(self, x):
                return self.network.encode_image(x)
        model = Encoder(model)
    else:
        from torchvision import transforms
        source = Path(cfg['source_dir']).expanduser()
        if not (source/'hubconf.py').is_file():
            raise FileNotFoundError('DINO source missing: '+str(source))
        model = torch.hub.load(str(source), 'dinov2_vits14', source='local', pretrained=False)
        model.load_state_dict(torch.load(weight, map_location='cpu'), strict=True)
        pipeline = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
        def transform(path):
            return pipeline(read_image(path))
    model.eval().to(device)
    features, audit = infer_batches(model, items, transform, device, int(cfg.get('batch_size', 8)))
    features = features.astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('zero embedding norm')
    features = features/norms
    cache.save_npz('features', key, features=features)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features, dict(provenance, **audit, cache_hit=False)


def _embedding_worker(request_path, output_path):
    from types import SimpleNamespace
    from driveaudit_eval.video.runtime_state import set_reproducibility
    data = json.loads(Path(request_path).read_text())
    set_reproducibility(int(data['state']['seed']))
    data['state']['cache_dir'] = Path(data['state']['cache_dir'])
    items = [ImageSample(**dict(row, path=Path(row['path']))) for row in data['items']]
    features, audit = embedding_features(items, data['cfg'], SimpleNamespace(**data['state']), data['backbone'])
    np.savez(output_path, features=features, audit=json.dumps(audit))


def temporal_scores(items, features, cfg, state):
    timestamps = {}
    if state.cfg.get('mode') == 'input_gt':
        source = Path(state.cfg['paths']['dataset_root'])/'provenance.json'
        for scene in json.loads(source.read_text())['scenes']:
            for row in scene['rgb_manifest']['images']:
                if row.get('exposure_time_us') is not None:
                    timestamps[scene['scene_id'], row['camera'], int(row['frame_index'])] = row['exposure_time_us']/1e6
        time_policy = 'native_camera_exposure'
    else:
        if cfg.get('require_native_timestamps'):
            return unavailable('native_exposure_timestamps_required')
        fps = cfg.get('fps', state.cfg.get('sampling', {}).get('source_fps'))
        if not fps or fps <= 0:
            return unavailable('timestamps_or_positive_fps_required')
        timestamps = {(x.scene_id, x.camera, x.frame_id): x.frame_id/fps for x in items}
        time_policy = 'configured_fps'
    groups = defaultdict(list)
    for index, item in enumerate(items):
        t = timestamps.get((item.scene_id, item.camera, item.frame_id))
        if t is not None:
            groups[item.scene_id, item.camera].append((t, index))
    rows = []
    intervals = cfg.get('interval_seconds', {'short': .3, 'medium': 1., 'long': 2.5})
    tolerance = cfg.get('time_tolerance_s', .08)
    for (scene, camera), group in groups.items():
        group.sort()
        if cfg.get('unique_exposures'):
            group = list({t: (t, index) for t, index in reversed(group)}.values())
            group.sort()
        times = np.array([x[0] for x in group])
        for interval, delta in intervals.items():
            if delta <= 0 or tolerance < 0:
                raise ValueError('invalid temporal interval/tolerance')
            for offset, (t, index) in enumerate(group):
                j = int(np.argmin(abs(times-(t+delta))))
                if j <= offset or abs(times[j]-(t+delta)) > tolerance:
                    continue
                other = group[j][1]
                rows.append(dict(scene_id=scene, camera=camera, frame_id=items[index].frame_id,
                    next_frame_id=items[other].frame_id, interval=interval, actual_delta_s=float(times[j]-t),
                    requested_delta_s=delta, cosine_similarity=float(np.clip(features[index]@features[other], -1, 1))))
    metrics = {}
    for interval in intervals:
        values = [r['cosine_similarity'] for r in rows if r['interval'] == interval]
        metrics[interval] = statistics(values, 'dino_temporal_'+interval)
    return dict(status='valid' if rows else 'insufficient_samples', reason=None if rows else 'no_valid_time_pairs',
                time_policy=time_policy, pairs=rows, metrics=metrics,
                temporal_motion_diagnostics=state.results.get('temporal_consistency', {}),
                interpretation='High similarity may indicate frozen video; inspect motion collapse and flow magnitude.')
