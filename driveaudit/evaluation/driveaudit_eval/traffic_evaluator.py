"""Traffic and collision additions invoked by the unified evaluation runner."""
import csv
import json
from pathlib import Path

import numpy as np

from driveaudit_eval.agents.core import EGO_LENGTH, EGO_WIDTH, ego_to_world
from driveaudit_eval.agents.scoring import overlap
from driveaudit_eval.agents.input_gt import load_scene as load_ego, sample
from driveaudit_eval.utils.map_provider.geometry import trajectory_headings
from driveaudit_eval.utils.map_provider.input_gt_provider import InputGTMapProvider
from driveaudit_eval.trajectory.input_gt import load_scene
from driveaudit_eval.trajectory.io import derive_scalar_motion
from driveaudit_eval.trajectory.schema import Trajectory
from driveaudit_eval.trajectory.collision_metrics import evaluate_collisions
from driveaudit_eval.traffic_compliance import MapContext, evaluate_scene_compliance, evaluate_actor_compliance, aggregate_actors
from driveaudit_eval.responsibility.cars_adapter import evaluate_collision_responsibility, _summary as collision_summary


def ego_track(scene_id, times, values, history_count, source='ground_truth'):
    mask = np.ones(len(times), bool)
    speed, acceleration = derive_scalar_motion(values[:, :2], mask, times)
    return Trajectory(scene_id, 'ego', 'ego', source, 'history_last_ego', times.copy(), values[:, :2].copy(),
        values[:, 2].copy(), speed, acceleration, mask, dict(length=EGO_LENGTH, width=EGO_WIDTH),
        dict(history_frames=history_count, dt=float(np.median(np.diff(times))), heading_unit='rad',
             shape_source='existing_evaluator_fixed_proxy_not_measured_source_dimensions'))


def evaluate(config, context, summary):
    cfg = config.get('traffic_compliance', {})
    responsibility_cfg = config.get('collision_responsibility', {})
    v2_cfg = config.get('responsibility_v2', {})
    v2_batches = []
    if v2_cfg.get('enabled', False):
        from driveaudit_eval.responsibility.ratios import raw_evidence_tracks, evaluate_batch, assemble
    augment_pdms = cfg.get('pdms_tc', {}).get('enabled', True)
    agent_cfg = config['evaluation'].get('agent', {})
    source_cfg = config['evaluation'].get('trajectory', {})
    paths = source_cfg.get('scene_paths', agent_cfg.get('scene_paths', []))
    native = agent_cfg.get('native_map', source_cfg.get('native_map'))
    scenes, models, all_actors, all_events = [], [], [], []
    for relative in paths:
        path = Path(context['root'])/relative
        context['logger'].info('[TRAFFIC] Evaluating %s', path.name)
        tracks, provenance = load_scene(path)
        timestamps, times, values, raw_actors, timeline = load_ego(path)
        scene_id = provenance['scene_id']
        history_count = int(tracks[0].metadata['history_frames']) if tracks else 10
        ego = ego_track(scene_id, times, values, history_count)
        tracks = [ego, *tracks]
        if v2_cfg.get('enabled', False):
            evidence_tracks = raw_evidence_tracks(scene_id, raw_actors, provenance)
            evidence_tracks.update({t.agent_id: t for t in tracks})
        from driveaudit_eval.traffic_compliance import load_map_context
        map_context = load_map_context(path, native)
        scene = evaluate_scene_compliance(tracks, map_context, cfg)
        scene['provenance'] = provenance
        scene['evaluation_scope'] = 'ground_truth_scenario_behavioral_realism_self_evaluation'
        if responsibility_cfg.get('enabled', False) or v2_cfg.get('enabled', False):
            collisions = evaluate_collisions(tracks, config['evaluation'].get('trajectory', {}).get('collision'))
            scene['collision_responsibility'] = evaluate_collision_responsibility(collisions['events'],
                {t.agent_id: t for t in tracks}, map_context,
                dict(responsibility_cfg, traffic_config=cfg), context)
            all_events.extend(scene['collision_responsibility']['events'])
            if v2_cfg.get('enabled', False):
                v2_batches.append(evaluate_batch(scene['collision_responsibility']['events'],
                    evidence_tracks, map_context, v2_cfg, cfg, source_cfg.get('kinematics', {})))
        scenes.append(scene)
        all_actors.extend(scene['actors'])
        canonical = path.parent.parent.name+'__'+path.name
        for agent in agent_cfg.get('agents', []):
            if not cfg.get('evaluate_ego', True):
                continue
            directory = Path(context['output_dir'])/'real_agents'/agent
            predictions_file = directory/'predictions.jsonl'
            metrics_file = directory/'agent_details'/agent/'per_frame_metrics.csv'
            if not predictions_file.is_file() or not metrics_file.is_file():
                continue
            predictions = {(r['scene_id'], int(r['frame_id'])): r for r in
                (json.loads(line) for line in predictions_file.read_text().splitlines() if line.strip())}
            with metrics_file.open() as handle:
                metric_rows = [r for r in csv.DictReader(handle) if r['scene_id'] == canonical]
            model_rows, model_events = [], []
            for original in metric_rows:
                frame = int(original['frame_id'])
                record = predictions[canonical, frame]
                start = timestamps[frame]
                origin, observed = sample(times, values, np.array([start]))
                if not observed.all():
                    raise ValueError('prediction decision pose unavailable')
                offsets = np.asarray(record['time_offsets'], float)
                world = ego_to_world(record['planning_traj'], [*origin[0, :2], 0.], origin[0, 2])
                headings = trajectory_headings(np.vstack([origin[0, :2], world]), origin[0, 2])[1:]
                past = times <= start
                combined_times = np.r_[times[past], start+offsets]
                combined_values = np.vstack([values[past], np.column_stack([world, headings])])
                prediction = ego_track(scene_id, combined_times, combined_values, int(past.sum())-1, 'generated')
                prediction.metadata.update(replay_max_gap_s=.5,
                    history_policy='observed_past_plus_one_independent_open_loop_plan; no_future_GT_ego')
                traffic = evaluate_actor_compliance('ego', prediction, [prediction, *tracks[1:]],
                                                     map_context, dict(cfg, max_gap_s=.5))
                score = traffic['profiles']['normalized']['traffic_compliance_score']
                pdms = float(original['PDMS_open_loop']) if original['PDMS_open_loop'] else None
                model_rows.append(dict(scene_id=scene_id, canonical_scene_id=canonical, agent=agent, frame_id=frame,
                    PDMS_open_loop=pdms, Traffic_Compliance_Score_ego=score,
                    PDMS_TC_open_loop=pdms*score if augment_pdms and pdms is not None and score is not None else None,
                    pdms_tc_status='disabled' if not augment_pdms else 'valid' if pdms is not None and score is not None else 'unavailable',
                    status='valid' if pdms is not None and score is not None else 'unavailable',
                    definition='Experimental PDMS augmented with traffic-rule compliance', traffic=traffic))
                if (responsibility_cfg.get('enabled', False) or v2_cfg.get('enabled', False)) and float(original.get('collision_rate') or 0) > 0:
                    events = []
                    known = {t.agent_id: t for t in tracks}
                    for ident, native_actor in raw_actors.items():
                        other = known.get(ident)
                        interpolated, mask = sample(native_actor[:, 0], native_actor[:, 1:], start+offsets)
                        hits = np.flatnonzero(mask & overlap(world, headings, interpolated[:, :2],
                            interpolated[:, 2], interpolated[:, 3], interpolated[:, 4])).tolist()
                        groups = []
                        for hit in hits:
                            if not groups or hit > groups[-1][-1]+1:
                                groups.append([])
                            groups[-1].append(hit)
                        for group in groups:
                            first = float(start+offsets[group[0]])
                            pre_start = max(first-responsibility_cfg.get('desired_pre_collision_s', 5.), combined_times[0], native_actor[0, 0])
                            pre_end = min(first+responsibility_cfg.get('post_collision_s', 1.), combined_times[-1], native_actor[-1, 0])
                            native_frame = min(timestamps, key=lambda index: abs(timestamps[index]-first))
                            events.append(dict(scene_id=scene_id, collision_pair=['ego', ident],
                                collision_event_id=f'{scene_id}/{agent}/decision_{frame}/ego_{ident}/step_{group[0]}',
                                first_collision_time=float(start+offsets[group[0]]),
                                first_collision_frame=native_frame, first_collision_frame_domain='nearest_native_scene_frame',
                                first_collision_frame_time_error_s=abs(timestamps[native_frame]-first),
                                first_prediction_step=group[0], collision_duration=float(len(group)*.5),
                                collision_type='ego-'+(other.agent_type if other else 'unsupported'), agent=agent, decision_frame_id=frame,
                                actor_categories={'ego': 'ego', ident: other.metadata['source_category'] if other else next(
                                    (category for category, ids in provenance['excluded_unsupported_categories'].items() if ident in ids), 'unknown')},
                                pre_collision_window=dict(requested_seconds=responsibility_cfg.get('desired_pre_collision_s', 5.),
                                    available_seconds=first-pre_start, coverage=(first-pre_start)/responsibility_cfg.get('desired_pre_collision_s', 5.),
                                    start_time=pre_start, end_time=pre_end, post_collision_seconds=max(0., pre_end-first)),
                                collision_sample_steps=group, geometry='original_agent_SAT_no_core_inset_native_sampled_dimensions'))
                    if not events:
                        raise ValueError('Original agent collision could not be reproduced for responsibility triggering')
                    diagnosed = evaluate_collision_responsibility(events, {t.agent_id: t for t in [prediction, *tracks[1:]]},
                        map_context, dict(responsibility_cfg, traffic_config=dict(cfg, max_gap_s=.5)), context)
                    model_events.extend(diagnosed['events'])
                    if v2_cfg.get('enabled', False):
                        v2_batches.append(evaluate_batch(diagnosed['events'], dict(evidence_tracks, ego=prediction),
                            map_context, v2_cfg, dict(cfg, max_gap_s=.5), source_cfg.get('kinematics', {})))
            scored = [r for r in model_rows if r['Traffic_Compliance_Score_ego'] is not None]
            tc = float(np.mean([r['Traffic_Compliance_Score_ego'] for r in scored])) if scored else None
            original_scores = [r['PDMS_open_loop'] for r in model_rows if r['PDMS_open_loop'] is not None]
            pdms = float(np.mean(original_scores)) if original_scores else None
            models.append(dict(scene_id=scene_id, agent=agent, windows=model_rows, collision_events=model_events,
                collision_summary=collision_summary(model_events),
                Traffic_Compliance_Score_ego=tc,
                Traffic_Compliance_Score_non_ego=scene['non_ego_traffic_compliance']['compliance_mean'],
                PDMS_open_loop=pdms, PDMS_TC_open_loop=pdms*tc if augment_pdms and tc is not None and pdms is not None else None,
                total_windows=len(model_rows), traffic_scorable_windows=len(scored),
                mean_window_PDMS_TC=float(np.mean([r['PDMS_TC_open_loop'] for r in scored])) if augment_pdms and scored else None,
                aggregation='scene-level product of means; window-product mean also reported separately'))
    overall_models = []
    for name in sorted({m['agent'] for m in models}):
        records = [m for m in models if m['agent'] == name]
        item = dict(agent=name, scene_count=len(records), aggregation='equal_scene_mean; overall experimental score is product of component means')
        for key in ('PDMS_open_loop', 'Traffic_Compliance_Score_ego', 'Traffic_Compliance_Score_non_ego'):
            vals = [r[key] for r in records if r[key] is not None]
            item[key] = float(np.mean(vals)) if vals else None
        pdms, tc = item['PDMS_open_loop'], item['Traffic_Compliance_Score_ego']
        item['PDMS_TC_open_loop'] = pdms*tc if augment_pdms and pdms is not None and tc is not None else None
        overall_models.append(item)
    result = dict(status='completed', scenes=scenes, agent_scenes=models, agent_overall=overall_models,
                ego_traffic_compliance=aggregate_actors([a for a in all_actors if a['actor_type']=='ego']),
                non_ego_traffic_compliance=aggregate_actors([a for a in all_actors if a['actor_type']!='ego']),
                all_actor_traffic_compliance=aggregate_actors(all_actors),
                scope='scenario_GT_and_independent_agent_open_loop_plans_are_reported_separately')

    if config.get('collision_responsibility_v3', {}).get('enabled', False) and not v2_cfg.get('enabled', False):
        raise ValueError('V3 requires enabled V2 evidence; it must not recompute a different ratio algorithm')
    if v2_cfg.get('enabled', False):
        result['responsibility_v2'] = assemble(v2_batches, models, [s['scene_id'] for s in scenes], v2_cfg)
    if config.get('collision_responsibility_v3', {}).get('enabled', False):
        from driveaudit_eval.responsibility.attribution import assemble as assemble_v3
        result['responsibility_v3'] = assemble_v3(result['responsibility_v2'], config['collision_responsibility_v3'])
    return result
