"""Single source of metric names, order, ranking and public contracts."""

TRS_POLICY = {
    'name': 'catk_common6_g2',
    'version': 'common6_g2_excess_q055_release_v1',
    'q': 0.55,
    'anchor_excess_fraction': 0.01,
    'feature_profile': 'common_six',
    'geometry': 'rounded_xy_g2',
    'c': 0.7,
    'dtype': 'float64',
    'overlap_rule': 'signed_distance < 0',
    'coverage_threshold': 1.0,
}

GROUP_COLUMNS = {'distribution': 'distribution_rank', 'visual_quality': 'visual_rank',
                 'consistency': 'consistency_rank', 'scene_fidelity': 'scene_fidelity_rank',
                 'driving_utility': 'driving_rank'}

def _metric(key, name, short, group, direction='higher', bounds=(0, 1),
            tier='core', aggregation='scene_equal_mean', reference=False,
            pairing=False, gt=False, level='dataset', description='', **extra):
    return dict(metric_id=key, name=name, display_name=name, short_name=short, group=group,
                direction=direction, higher_is_better=direction == 'higher',
                lower_is_better=direction == 'lower', range=list(bounds), unit='dimensionless',
                level=level, tier=tier, requires_reference=reference, requires_pairing=pairing,
                requires_gt=gt, aggregation=aggregation, coverage_field=key+'_coverage',
                status_field=key+'_status', coverage=key+'_coverage',
                display_order=None, description=description or name, **extra)

_CORE = [
    _metric('fvd', 'Fréchet Video Distance', 'FVD', 'distribution', 'lower', (0,None), aggregation='dataset_feature_distribution', reference=True),
    _metric('cmmd', 'CLIP Maximum Mean Discrepancy', 'CMMD', 'distribution', 'lower', (0,None), aggregation='dataset_feature_distribution', reference=True),
    _metric('clip_iqa', 'CLIP-IQA+', 'CLIP-IQA+', 'visual_quality', aggregation='frame_equal_mean'),
    _metric('dino_t', 'DINO Temporal Consistency', 'DINO-T', 'consistency', bounds=(-1,1), aggregation='equal_mean_short_medium_long', description='Mean of three native-time-span cosine means; self-consistency is not fidelity.'),
    _metric('epi_3px', 'Epipolar Inlier Ratio at 3 px', 'Epi@3px', 'consistency', pairing=True, aggregation='existing_epipolar_inlier_ratio_3px'),
    _metric('tvc', 'Trajectory–Visual Consistency Score', 'TVC', 'scene_fidelity', pairing=True, description='Available-weight mean of five MTPC dimensions; optical-flow motion is diagnostic evidence, not calibrated motion fidelity.'),
    _metric('trs', 'Traffic Realism Score', 'TRS', 'scene_fidelity',
            aggregation='population_D_times_G2_positive_excess_factor',
            version=TRS_POLICY['version'], reference=True, pairing=True, gt=True,
            q=TRS_POLICY['q'], anchor_excess_fraction=TRS_POLICY['anchor_excess_fraction'], scoring_policy='OWNER_APPROVED', input_root='challenge/input-catk',
            implementation='driveaudit_eval.trajectory.trs.compute_trs',
            description=f"Common-six native-time distribution D times {TRS_POLICY['q']}**(E_plus/{TRS_POLICY['anchor_excess_fraction']}); frozen rounded-XY G2 c=0.7, float64, signed_distance<0. Population metric; road-edge diagnostic only; geometric proxy, no safety guarantee."),
    _metric('rpdms', 'Responsibility-aware PDMS', 'RPDMS', 'driving_utility', gt=True, level='window/scene/agent/dataset', aggregation='window_mean_then_scene_equal_then_agent_equal', description='RNC*DAC*(5*TTC_score+2*Comfort+5*EP)/12; collision window V3 ratio only, no confidence weighting.'),
    _metric('route_completion_ratio', 'Route Completion Ratio', 'Route Comp.', 'driving_utility', gt=True, level='episode/agent/dataset', aggregation='episode_mean_then_agent_equal', description='Unchanged continuous accepted route projection progress / reference route length; offline proxy.'),
]
METRICS = {m['metric_id']: m for m in _CORE}
for order, metric in enumerate(_CORE, 1):
    metric['display_order'] = order
CORE_IDS = tuple(METRICS)
core_ids = CORE_IDS
for order, metric in enumerate([
    _metric('testing_effectiveness', 'Testing Effectiveness', 'TE', 'testing',
            pairing=True, gt=True, aggregation='agent_equal_then_scene_equal'),
    _metric('qtes', 'Quality-aware Testing Effectiveness Score', 'QTES', 'testing',
            bounds=(0,100), pairing=True, gt=True, aggregation='100 * V * TE * Q',
            final_ranking_metric=True, official_final_score=True, version='gt_relative_v1'),
], len(CORE_IDS)+1):
    metric['display_order'] = order
    METRICS[metric['metric_id']] = metric
OFFICIAL_IDS = CORE_IDS + ('testing_effectiveness', 'qtes')
METRICS['quality_absolute_score'] = _metric(
    'quality_absolute_score','Absolute Quality Score','Q_abs','quality',tier='diagnostic',
    aggregation='sqrt(geomean(CLIP-IQA,canonical DINO-T,Epi@3)*sqrt(TVC*TRS))')
METRICS['quality_retention_score'] = _metric(
    'quality_retention_score','Quality Retention Score','Q','quality',tier='core_internal',
    reference=True,pairing=True,gt=True,final_score_component=True,version='gt_relative_v1',
    aggregation='paired_scene_ratios; equal_group_geomean; scene_equal_mean',
    epsilons={key:1e-6 for key in ('clip_iqa','dino_t','epi_3px','tvc','trs')},
    reference_floor_policy='minimum_positive_coverage_qualified_GT_validation_score; audit trigger only',
    motion_guard='paired_exact_decoded_repeat_v1; DINO retention *= 1 - extra_repeat_ratio')

for key, name, group, direction, bounds in [
    ('fid','FID','distribution','lower',(0,None)), ('kid','KID','distribution','lower',(None,None)),
    ('musiq','MUSIQ','visual_quality','higher',(0,100)), ('dover','DOVER','visual_quality','higher',(None,None)),
    ('warp_error','Warp Error','consistency','lower',(0,None)), ('flow_magnitude','Flow Magnitude','consistency','none',(0,None)),
    ('motion_collapse','Motion Collapse','consistency','lower',(0,1)), ('loftr','LoFTR Matches','consistency','none',(0,None)),
    ('ransac','RANSAC Inlier Ratio','consistency','higher',(0,1)), ('epi_1px','Epi@1','consistency','higher',(0,1)),
    ('epi_5px','Epi@5','consistency','higher',(0,1)), ('history_future_boundary','History–Future Boundary','consistency','lower',(0,None)),
    ('trajectory_details','Trajectory Kinematics / Interaction / Reference / Distribution','scene_fidelity','none',(None,None)),
    ('valid_point_ratio','Valid Point Ratio','scene_fidelity','higher',(0,1)), ('offroad','Footprint Offroad Ratio','scene_fidelity','lower',(0,1)),
    ('wrong_way','Wrong-way Ratio','scene_fidelity','lower',(0,1)), ('min_ttc','Minimum TTC','driving_utility','higher',(0,None)),
    ('PDMS_open_loop','PDMS','driving_utility','higher',(0,1)), ('PDMS_TC_open_loop','PDMS-TC','driving_utility','higher',(0,1)),
    ('Driving_Score_open_loop','Driving Score','driving_utility','higher',(0,1)), ('task_success','Task Success','driving_utility','higher',(0,1)),
    ('task_success_rate','Task Success Rate','driving_utility','higher',(0,1)), ('episode_collision','Episode Collision','driving_utility','lower',(0,1)),
    ('TTC_score','TTC Score','driving_utility','higher',(0,1)), ('DAC','Drivable Area Compliance','driving_utility','higher',(0,1)),
    ('Comfort','Comfort','driving_utility','higher',(0,1)), ('RC_open_loop','Open-loop RC','driving_utility','higher',(0,1)),
    ('agent_details','ADE/FDE and Kinematic Details','driving_utility','none',(None,None)),
    ('traffic_compliance_score','Traffic Compliance','scene_fidelity','higher',(0,1)),
    ('traffic_compliance_coverage','Traffic Compliance Coverage','scene_fidelity','higher',(0,1)),
    ('traffic_rules','24 Rule Details','scene_fidelity','none',(None,None)),
    ('responsibility_v1','CARS Counterfactual Evidence','driving_utility','none',(None,None)),
    ('responsibility_v2','Responsibility Ratio Evidence','driving_utility','none',(None,None)),
    ('responsibility_v3','Responsibility V3 Attribution','driving_utility','none',(None,None)),
    ('route_projection','Route Projection Diagnostics','driving_utility','none',(None,None)),
    ('mtpc','MTPC','consistency','none',(None,None)), ('tcvv','TCVV','consistency','none',(None,None)),
    ('tcvv_bench','TCVV-Bench','consistency','none',(None,None)), ('tcvv_h','TCVV-H','consistency','none',(None,None)),
    ('tcvv_o','TCVV-O','consistency','none',(None,None)), ('tcvv_calibrated','TCVV-Calibrated','consistency','none',(None,None)),
    ('tcvv_h_c','TCVV-H-C','consistency','none',(None,None)),
]:
    METRICS[key] = _metric(key,name,name,group,direction,bounds,tier='diagnostic',
                          experimental=key in {'PDMS_TC_open_loop','tcvv_calibrated','tcvv_h_c'}, optional=key=='dover')
DEPRECATED_ALIASES = {
    'route_progress_ratio':'route_completion_ratio', 'proxy_route_progress_ratio':'route_completion_ratio',
    'proxy_route_completed':'task_success', 'route_completed_proxy':'task_success',
    'route_completion_rate':'task_success_rate', 'proxy_route_completion_rate':'task_success_rate',
}
for alias, replacement in DEPRECATED_ALIASES.items():
    METRICS[alias] = _metric(alias,alias,alias,'driving_utility',tier='legacy',deprecated_alias=replacement)

def get_registry():
    """Return an independent JSON-serializable registry."""
    from copy import deepcopy
    return deepcopy(METRICS)


def markdown_table():
    """Used to generate and verify the README's metric table."""
    lines = ['| Group | Metric | Direction | Aggregation |', '| --- | --- | --- | --- |']
    for key in OFFICIAL_IDS:
        metric = METRICS[key]
        lines.append('| ' + ' | '.join((metric['group'], metric['short_name'],
                     '↑' if metric['higher_is_better'] else '↓', metric['aggregation'])) + ' |')
    return '\n'.join(lines)
