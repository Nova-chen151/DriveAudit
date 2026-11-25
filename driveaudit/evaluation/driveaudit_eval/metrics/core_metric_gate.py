"""Fail-closed acceptance of measured dataset x agent x metric evidence."""
import math
import csv
import json
from pathlib import Path

from driveaudit_eval.agents.registry import FORMAL_AGENTS
from driveaudit_eval.metrics.registry import CORE_IDS, METRICS
from driveaudit_eval.utils.io import write_csv
from driveaudit_eval.utils.io import write_json

DATASETS = ('nuscenes', 'nuplan', 'av2', 'pandaset')
# Frozen before the stability runs: covariance needs >=2 clips; other minima
# require repeated observations, in addition to the independent 10-scene gate.
REQUIREMENTS = {
    'fvd': (2, 1.), 'cmmd': (100, 1.), 'clip_iqa': (100, 1.),
    'dino_t': (30, .8), 'epi_3px': (100, .8), 'tvc': (100, .8),
    'trs': (10, 1.), 'rpdms': (10, .8), 'route_completion_ratio': (10, 1.),
}


def validate_cell(record):
    minimum, threshold = REQUIREMENTS[record['metric']]
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    value, coverage, count = (record.get(k) for k in ('value', 'coverage', 'sample_count'))
    lower, upper = METRICS[record['metric']]['range']
    valid = (record.get('status') in ('valid', 'CORE_VALID') and finite(value)
             and (lower is None or value >= lower) and (upper is None or value <= upper)
             and finite(coverage) and threshold <= coverage <= 1
             and isinstance(count, int) and not isinstance(count, bool) and count >= minimum)
    return dict(record, gate_status='CORE_VALID' if valid else 'INVALID',
                minimum_samples=minimum, minimum_coverage=threshold)


def evaluate_gate(records, inference, output_dir):
    """Inference rows must contain measured completed/expected scene+decision counts."""
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    expected = {(d, a, m) for d in DATASETS for a in FORMAL_AGENTS for m in CORE_IDS}
    cells = {}
    for record in records:
        key = tuple(record[k] for k in ('dataset', 'agent', 'metric'))
        if key not in expected or key in cells:
            raise ValueError('Unexpected or duplicate core cell: '+str(key))
        cells[key] = validate_cell(record)
    inference_keys = set()
    inference_valid = {}
    for row in inference:
        key = row['dataset'], row['agent']
        if key in inference_keys or key not in {(d,a) for d,a,_ in expected}:
            raise ValueError('Unexpected or duplicate inference cell')
        inference_keys.add(key)
        counts_ok = all(isinstance(row.get(k), int) and not isinstance(row.get(k), bool) and row[k]>=0
                        for k in ('expected_scenes','completed_scenes','expected_decisions','completed_decisions'))
        inference_valid[key] = (counts_ok and row.get('expected_scenes',0) >= 10 and row.get('completed_scenes') == row.get('expected_scenes')
            and row.get('expected_decisions', 0) >= 10*row.get('expected_scenes',0)
            and row.get('completed_decisions') == row.get('expected_decisions')
            and row.get('input_validation') == 'VALID')
    matrix, values = [], []
    for dataset in DATASETS:
        for agent in FORMAL_AGENTS:
            row, value_row = dict(dataset=dataset, agent=agent), dict(dataset=dataset, agent=agent)
            for metric in CORE_IDS:
                cell = cells.get((dataset, agent, metric), {})
                row[metric] = 'VALID' if cell.get('gate_status') == 'CORE_VALID' else 'INVALID'
                value_row[metric] = cell.get('value')
                if metric in ('tvc', 'trs'):
                    value_row[metric+'_coverage'] = cell.get('coverage')
            row['inference'] = 'VALID' if inference_valid.get((dataset,agent)) else 'INVALID'
            matrix.append(row); values.append(value_row)
    passed = len(cells) == len(expected) and all(all(v == 'VALID' for k,v in row.items()
                 if k not in ('dataset','agent')) for row in matrix)
    write_csv(directory/'CORE_METRIC_UNIVERSALITY_MATRIX.csv', matrix)
    write_csv(directory/'core_metric_evidence.csv', list(cells.values()))
    result = dict(status='PASS' if passed else 'FAIL', expected_cells=len(expected),
                  measured_cells=len(cells), valid_cells=sum(c['gate_status']=='CORE_VALID' for c in cells.values()),
                  dataset_status={d: 'LEADERBOARD_READY' if all(all(v=='VALID' for k,v in r.items()
                    if k not in ('dataset','agent')) for r in matrix if r['dataset']==d) else 'NOT_READY' for d in DATASETS})
    write_json(directory/'core_universality_gate.json', result)
    if passed:
        write_csv(directory/'UNIVERSAL_CORE_METRICS.csv', values)
        from driveaudit_eval.metrics.leaderboard import rank_rows
        from driveaudit_eval.utils.aggregation import average_ranks
        ranked = {d:rank_rows([dict(r,model=r['agent']) for r in values if r['dataset']==d]) for d in DATASETS}
        averages = []
        for agent in FORMAL_AGENTS:
            scores = {d:next(r['avg_rank'] for r in ranked[d] if r['agent']==agent) for d in DATASETS}
            averages.append(dict(agent=agent,**{d+'_avg_rank':v for d,v in scores.items()},avg_rank=sum(scores.values())/4))
        for row, rank in zip(averages,average_ranks([r['avg_rank'] for r in averages])):
            row['overall_rank']=rank
        write_csv(directory/'UNIVERSAL_AGENT_AVG_RANK.csv', averages)
    elif (directory/'UNIVERSAL_CORE_METRICS.csv').exists():
        raise ValueError('A failed gate cannot coexist with a prior accepted table; use a fresh output directory')
    return result


def audit_video_sources(validation, provenance, vq, epi_rows, expected):
    """Bind measured support to frozen image identities and native exposure pairs."""
    from bisect import bisect_left
    from driveaudit_eval.metrics.core import number
    identity = lambda value: str(value).replace('__', '/')
    timestamps, image_keys, groups = {}, set(), {}
    provenance_ids = [identity(s['scene_id']) for s in provenance['scenes']]
    manifest_ok = len(provenance_ids) == len(set(provenance_ids)) and set(provenance_ids) == expected
    for scene in provenance['scenes']:
        sid = identity(scene['scene_id'])
        rows = scene['rgb_manifest']['images']
        keys = [(sid, r['camera'], int(r['frame_index'])) for r in rows]
        cameras = {r['camera'] for r in rows}
        manifest_ok &= len(cameras) == 6 and len(keys) == len(set(keys)) and set(keys) == {
            (sid, camera, frame) for camera in cameras for frame in range(10, 90)}
        for row, key in zip(rows, keys):
            image_keys.add(key)
            value = number(row.get('exposure_time_us'))
            if value is not None:
                timestamps[key] = value/1e6
                groups.setdefault(key[:2], []).append((value/1e6, key[2]))
    frames = vq['frame_metrics']
    frame_keys = [(identity(r['scene_id']), r['camera'], int(r['frame_id'])) for r in frames]
    iqa_ok = manifest_ok and len(frame_keys) == len(set(frame_keys)) and set(frame_keys) <= image_keys
    image_count = sum(number(r.get('clipiqa')) is not None for r in frames)
    allowed = {}
    intervals = {'short': .3, 'medium': 1., 'long': 2.5}
    for (sid, camera), group in groups.items():
        # Match temporal_scores: keep the earliest canonical image per native exposure.
        unique = sorted({t: frame for t, frame in reversed(sorted(group))}.items())
        times = [t for t, _ in unique]
        for name, delta in intervals.items():
            for index, (time, frame) in enumerate(unique):
                pos = bisect_left(times, time+delta)
                candidates = [j for j in (pos-1, pos) if 0 <= j < len(times)]
                if not candidates:
                    continue
                other = min(candidates, key=lambda j: (abs(times[j]-(time+delta)), j))
                if other > index and abs(times[other]-(time+delta)) <= .08:
                    allowed[sid, camera, frame, unique[other][1], name] = times[other]-time
    pairs = vq['metrics']['dino_temporal'].get('pairs', [])
    pair_keys = [(identity(r['scene_id']), r['camera'], int(r['frame_id']), int(r['next_frame_id']), r['interval']) for r in pairs]
    dino_ok = manifest_ok and len(pair_keys) == len(set(pair_keys))
    for row, key in zip(pairs, pair_keys):
        actual, requested = number(row.get('actual_delta_s')), number(row.get('requested_delta_s'))
        dino_ok &= (key in allowed and actual is not None and requested is not None
                    and math.isclose(actual, allowed.get(key, 0), abs_tol=1e-9)
                    and math.isclose(requested, intervals.get(key[-1], -1), abs_tol=1e-12))
    expected_epi = {(identity(scene['scene_id']), frame, '__'.join(edge['pair']))
                    for scene in validation['scenes'] for edge in scene['overlap_graph'] if edge['valid']
                    for frame in range(80)}
    attempted, measured, epi_count, epi_ok = set(), set(), 0, bool(expected_epi)
    for row in epi_rows:
        key = identity(row['scene_id']), int(row['frame_id']), row['camera_pair']
        is_attempt = row.get('loftr_num_matches') not in ('', None)
        is_measured = number(row.get('epipolar_inlier_ratio_3px')) is not None
        if not (is_attempt or is_measured):
            continue
        epi_ok &= key in expected_epi and number(row.get('actual_frame_id')) == key[1]+10
        if is_attempt:
            epi_ok &= key not in attempted
            attempted.add(key)
        if is_measured:
            count = number(row.get('epipolar_valid_match_count'))
            epi_ok &= key not in measured and count is not None and count > 0 and count == int(count)
            measured.add(key)
            if count is not None and count > 0 and count == int(count):
                epi_count += int(count)
    epi_ok &= measured <= attempted
    return dict(source_valid={'clip_iqa': bool(iqa_ok), 'dino_t': bool(dino_ok), 'epi_3px': bool(epi_ok)},
                image_count=image_count, epi_count=epi_count, epi_coverage=len(measured & expected_epi)/len(expected_epi) if expected_epi else 0.,
                epi_expected_pairs=len(expected_epi), epi_attempted_pairs=len(attempted), epi_valid_pairs=len(measured))


def collect_evidence(dataset, summary, video_dir, expected_scenes, expected_frames):
    """Extract counts from native metric artifacts, never infer success from a value."""
    from driveaudit_eval.metrics.core import number, mean
    video_dir = Path(video_dir)
    core = summary['core_metrics']; model = core['model_row']; video = summary['video']
    validation = json.loads((video_dir/'input_gt_video/generated_video_validation_gate.json').read_text())
    validated = {r['scene_id'] for r in validation['scenes'] if r['status']=='VALID' and r['decoded_frames']==540}
    expected = {str(s).replace('__','/') for s in expected_scenes}
    def identity(value):
        value = str(value).replace('__','/')
        return Path(value).parents[1].name+'/'+Path(value).name if '/scenes/' in value else value
    def exact_scenes(rows, key):
        ids = [identity(r.get(key,'')) for r in rows]
        return len(ids)==len(set(ids)) and set(ids)==expected
    input_valid = validation['status']=='VALID' and {s.replace('__','/') for s in validated}==expected
    sampling = json.loads((video_dir/'video_details/sampling_manifest.json').read_text())
    vq = video['visual_quality']; fvd = video['fvd']; cmmd = vq['distribution']['cmmd']
    dino = vq['metrics']['dino_temporal']; pairs = dino.get('pairs', [])
    span_counts = {s:sum(r['interval']==s for r in pairs) for s in ('short','medium','long')}
    temporal_groups = {(r['scene_id'],r['camera'],r['interval']) for r in pairs}
    with (video_dir/'video_details/camera_pair_metrics.csv').open(newline='') as handle:
        epi_rows = list(csv.DictReader(handle))
    provenance = json.loads((video_dir/'input_gt_video/provenance.json').read_text())
    video_support = audit_video_sources(validation, provenance, vq, epi_rows, expected)
    tvc_count = 0
    tvc_diagnostics = []
    for scene in summary['consistency'].get('scenes', []):
        result = scene.get('backends',{}).get('mtpc',{}).get('result',{})
        primary = result.get('primary_metrics',{})
        visible = primary.get('presence',{}).get('expected_visible_agent_camera_frames',0)
        tvc_count += visible
        tvc_diagnostics.append(dict(scene=scene.get('scene'), expected_visible=visible,
            matched_geometry=primary.get('projection_geometry',{}).get('matched_projection_samples'),
            matched_motion=primary.get('image_motion',{}).get('matched_motion_samples')))
    fvd_count = min(fvd.get('num_real_clips',0),fvd.get('num_gen_clips',0))
    fvd_coverage = min(fvd.get('num_real_clips',0)/sampling['num_real_clips'],
                       fvd.get('num_gen_clips',0)/sampling['num_fake_clips'])
    # VQ consumes all decoded future images, independently of the FVD image sampler.
    expected_images = len(expected)*6*80
    shared = {
        'fvd': (fvd_count, fvd_coverage),
        'cmmd': (min(cmmd['real_sample_count'],cmmd['generated_sample_count']),
                 min(cmmd['real_sample_count'],cmmd['generated_sample_count'])/expected_images),
        'clip_iqa': (video_support['image_count'],video_support['image_count']/expected_images),
        'dino_t': (len(pairs),len(temporal_groups)/(len(expected)*6*3)),
        'epi_3px': (video_support['epi_count'],video_support['epi_coverage']),
        'tvc': (tvc_count,model['tvc_coverage']*len(core['tvc'])/len(expected)),
        'trs': (len(core['trs']),model['trs_coverage']*len(core['trs'])/len(expected)),
    }
    source_scenes = dict(tvc=exact_scenes(summary['consistency'].get('scenes',[]),'scene'),
                         trs=exact_scenes(core['trs'],'scene_id'),
                         fvd=set(sampling['scene_clip_starts'])==validated)
    source_scenes['trs'] &= all(set(r.get('sample_counts',{}))=={'validity','kinematic','interaction','collision'}
        and all(isinstance(n,int) and n>0 for n in r['sample_counts'].values()) for r in core['trs'])
    source_scenes.update(video_support['source_valid'])
    records, inference, route_diagnostics = [], [], []
    for agent in FORMAL_AGENTS:
        a = next(r for r in core['rpdms_agents'] if r['agent']==agent)
        episodes = [e for e in summary['episode_rollout'].get('episodes',[]) if (e.get('agent') or e.get('agent_name'))==agent]
        finite_episodes = [e for e in episodes if e.get('status')=='valid' and e.get('route_available')
                           and identity(e.get('scene_id','')) in expected and number(e.get('route_completion_ratio')) is not None]
        route_ok = exact_scenes(finite_episodes,'scene_id')
        route_diagnostics.extend(dict(agent=agent,scene_id=e.get('scene_id'),
            route_projection_coverage=e.get('route_projection_coverage'),route_completion_status=e.get('route_completion_status')) for e in episodes)
        common_count = sum(r['common_window_count'] for r in core['rpdms_scenes']
                           if r['agent']==agent and identity(r['scene_id']) in expected)
        evidence = dict(shared, rpdms=(common_count,common_count/(len(expected)*len(expected_frames))),
                        route_completion_ratio=(len(finite_episodes),len(finite_episodes)/len(expected)))
        for metric,(count,coverage) in evidence.items():
            value = a.get(metric) if metric in ('rpdms','route_completion_ratio') else model.get(metric)
            source_valid = core['metric_statuses'].get(metric)=='valid'
            source_valid &= source_scenes.get(metric,True)
            if metric=='route_completion_ratio': source_valid &= route_ok
            if metric=='rpdms':
                source_valid &= all(w.get('responsibility_coverage')==1. for w in core['rpdms_windows']
                                    if w['agent']==agent and w['NC']==0)
            if metric=='dino_t':
                source_valid &= dino.get('time_policy')=='native_camera_exposure' and min(span_counts.values())>=10
            if metric=='epi_3px':
                source_valid &= video['geometry_consistency'].get('comparison_validity')!='invalid'
            records.append(dict(dataset=dataset,agent=agent,metric=metric,value=value,
                status='valid' if source_valid and input_valid else 'invalid',sample_count=int(count),coverage=coverage,
                scope='agent_plan' if metric in ('rpdms','route_completion_ratio') else 'shared_GT_self_evaluation'))
        states = summary['agent']['agents'][agent].get('scene_status',[])
        complete_scenes = {r['scene'].replace('__','/') for r in states if r.get('inference_status')=='completed'}
        windows = core['rpdms_windows']
        completed = {(r['scene_id'],r['frame_id']) for r in windows if r['agent']==agent}
        expected_keys = {(sid,frame) for sid in expected for frame in expected_frames}
        inference.append(dict(dataset=dataset,agent=agent,expected_scenes=len(expected),
            completed_scenes=len(complete_scenes & expected),expected_decisions=len(expected_keys),
            completed_decisions=len(completed & expected_keys),input_validation='VALID' if input_valid else 'INVALID'))
    return records, inference, dict(dino_span_counts=span_counts,tvc_observability=tvc_diagnostics,
                                    epi_attempted_pairs=video_support['epi_attempted_pairs'],epi_valid_pairs=video_support['epi_valid_pairs'],
                                    epi_expected_pairs=video_support['epi_expected_pairs'],route_projection=route_diagnostics)


def run_universal(config, root):
    """Use the existing runner independently per dataset, then admit all or none."""
    from copy import deepcopy
    from driveaudit_eval.evaluation.runner import run, SECTIONS
    root = Path(root)
    output = root/config['output']['root']/config['output']['run_name']
    if output.exists():
        raise FileExistsError(output)
    groups = {d:[] for d in DATASETS}
    for value in config['evaluation']['agent']['scene_paths']:
        scene = root/value
        dataset = scene.parents[1].name.split('_challenger')[0].replace('nuscenes300','nuscenes')
        groups[dataset].append(value)
    if any(len(v)<10 or len(set(v))!=len(v) for v in groups.values()):
        raise ValueError('Universal gate requires at least 10 distinct frozen scenes per dataset')
    records, inference, diagnostics = [], [], {}
    for dataset, paths in groups.items():
        current = deepcopy(config)
        current['output'] = dict(root=str(output),run_name=dataset)
        current['evaluation'].setdefault('leaderboard',{})['generate_rank']=False
        for section in SECTIONS:
            current['evaluation'][section]['scene_paths']=paths
        summary, directory = run(current,root)
        if current['evaluation'].get('profile')=='leaderboard':
            from driveaudit_eval.agents.sensor_guard import validate_camera_only_run
            write_json(directory/'sensor_purity.json',validate_camera_only_run(directory/'real_agents'))
        expected = [Path(p).parents[1].name+'/'+Path(p).name for p in paths]
        rows, states, diagnostic = collect_evidence(dataset,summary,directory,expected,
                                                   current['evaluation']['agent']['score_frame_ids'])
        records.extend(rows);inference.extend(states);diagnostics[dataset]=diagnostic
    write_json(output/'observable_support.json',diagnostics)
    gate = evaluate_gate(records,inference,output)
    return {'universality':dict(gate,status='completed' if gate['status']=='PASS' else 'failed')}, output
