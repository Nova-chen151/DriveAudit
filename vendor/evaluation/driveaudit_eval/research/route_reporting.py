"""Final route-distance and task-outcome exports; original scores are read-only."""
from collections import Counter
import json
from pathlib import Path
import statistics

from driveaudit_eval.utils.io import jsonable, write_json
from driveaudit_eval.research.episode_reporting import _finite, _stat
from driveaudit_eval.research.responsibility_reporting import _display, _table

CATALOG = '''
## Final route evaluation: distance versus endpoint success

- route_completion_ratio (线路完成率): distance-based Proxy Route Completion Ratio, max valid projected progress from episode start / remaining reference route length, clipped to [0,1]. Invalid projections do not add progress.
- task_success: binary Proxy Task Success, independently satisfied endpoint criterion without earlier collision.
- task_success_rate (任务成功率): successful episodes / valid episodes. This is not a route distance ratio.
- task_outcome: SUCCESS, COLLISION_FAILURE or INCOMPLETE. Invalid episodes remain explicit and are excluded from valid-episode rates.
- route_projection_coverage: valid projection frames / executed route frames. Consult route_completion_status before interpreting a low-coverage ratio.
- safe_route_completion_ratio: optional safety-conditioned diagnostic; it does not replace the primary distance ratio.
- route_progress_ratio and proxy_route_progress_ratio: legacy / compatibility aliases for route-distance output. proxy_route_completed: legacy / compatibility endpoint-success flag.
- Reference remains logged_ego_path_proxy; endpoint remains log_endpoint_proxy. Logged-state receding horizon plus terminal commit does not constitute true closed-loop simulation.
- SparseDriveV2 in the final comparison means SparseDriveV2 v2. Historical v1 artifacts remain unchanged.

See FINAL_ROUTE_EVALUATION_REPORT.md and final_route_agent_summary.csv. Original PDMS, PDMS_TC, collision, traffic and responsibility outputs remain separate.
'''


def _observed_table(rows):
    if not rows:
        return ['No observations attached.']
    return _table(rows,[(key,key) for key in dict.fromkeys(key for row in rows for key in row)
                        if not any(isinstance(row.get(key),(dict,list)) for row in rows)])


def summarize(result, summary=None):
    episodes = result.get('episodes', [])
    windows = result.get('windows', [])
    traffic = {row['agent']: row for row in (summary or {}).get('traffic_and_responsibility', {}).get('agent_overall', [])}
    agents = []
    for name in sorted({e['agent_name'] for e in episodes}):
        rows = [e for e in episodes if e.get('agent_name') == name]
        valid = [e for e in rows if e.get('termination_reason') not in ('invalid','invalid_state') and e.get('status') != 'invalid']
        n = len(valid)
        outcomes = Counter(e.get('task_outcome') for e in valid)
        available = [e for e in valid if _finite(e.get('route_completion_ratio'))]
        own_windows = [w for w in windows if w.get('agent_name') == name]
        collision_windows = sum(bool(w.get('open_loop_collision')) for w in own_windows)
        collisions = [e for e in valid if e.get('termination_reason') == 'collision']
        reference = traffic.get(name,{})
        row = dict(agent_name=name,episode_count=len(rows),valid_episode_count=n,invalid_episode_count=len(rows)-n,
            window_count=len(own_windows),open_loop_collision_window_count=collision_windows,
            open_loop_collision_window_rate=collision_windows/len(own_windows) if own_windows else None,
            episode_collision_count=len(collisions),episode_collision_rate=len(collisions)/n if n else None,
            mean_route_completion_ratio=_stat(available,'route_completion_ratio'),
            median_route_completion_ratio=_stat(available,'route_completion_ratio',statistics.median),
            min_route_completion_ratio=_stat(available,'route_completion_ratio',min),
            route_ratio_available_episode_count=len(available), task_success_count=outcomes['SUCCESS'],
            task_success_rate=outcomes['SUCCESS']/n if n else None,
            collision_failure_count=outcomes['COLLISION_FAILURE'],collision_failure_rate=outcomes['COLLISION_FAILURE']/n if n else None,
            incomplete_count=outcomes['INCOMPLETE'],incomplete_rate=outcomes['INCOMPLETE']/n if n else None,
            mean_route_projection_coverage=_stat(valid,'route_projection_coverage'),
            route_completion_status_counts=dict(Counter(e.get('route_completion_status') for e in rows)),
            mean_safe_route_completion_ratio=_stat(valid,'safe_route_completion_ratio'),
            mean_terminal_ego_responsibility=_stat(collisions,'terminal_ego_responsibility_ratio'),
            PDMS_open_loop=reference.get('PDMS_open_loop'),PDMS_TC_open_loop=reference.get('PDMS_TC_open_loop'),
            Traffic_Compliance_Score_ego=reference.get('Traffic_Compliance_Score_ego'),
            original_score_aggregation=reference.get('aggregation'),
            denominator_definition='outcome/collision rates: valid episodes; route means: finite episode distance ratios; open-loop rate: evaluated windows')
        agents.append(row)
    overall = dict(scene_count=len({e.get('scene_id') for e in episodes}),agent_count=len(agents),
        agent_names=[a['agent_name'] for a in agents], window_count=len(windows), episode_count=len(episodes),
        task_outcome_counts=dict(Counter(e.get('task_outcome') for e in episodes)),
        mean_route_completion_ratio=_stat(episodes,'route_completion_ratio'),
        mean_route_projection_coverage=_stat(episodes,'route_projection_coverage'),
        route_completion_status_counts=dict(Counter(e.get('route_completion_status') for e in episodes)),
        sparse_v1_present=any(e.get('agent_name')=='sparsedrivev2_v1' for e in episodes))
    return agents, overall


def _report(result, agents, overall):
    episodes = result.get('episodes', [])
    focused = [e for e in episodes if e.get('agent_name') in ('uniad','vad') and str(e.get('scene_id','')).endswith('adv_case_0052')]
    case_notes = []
    for episode in focused:
        ratio = episode.get('route_completion_ratio')
        if _finite(ratio):
            case_notes += [f"{episode['agent_name']} case0052: corrected distance ratio = {100*ratio:.8f}%; "
                f"task_success = {episode.get('task_success')}; outcome = {episode.get('task_outcome')}. "
                f"Terminal endpoint distance = {_display(episode.get('terminal_endpoint_distance'))} m; "
                f"raw terminal fraction = {_display(episode.get('terminal_projection_fraction_raw'))}; "
                f"terminal projection reason = {episode.get('terminal_projection_reason')}.", '']
    sensitivity_notes = []
    for episode in focused:
        observed = [row for row in result.get('sensitivity',[]) if row.get('episode_id') == episode.get('episode_id')
                    and _finite(row.get('route_completion_ratio'))]
        ratios = [row['route_completion_ratio'] for row in observed]
        if len(ratios) > 1:
            settings = ', '.join(f"{row.get('max_lateral_distance_m')} m: {100*row['route_completion_ratio']:.5f}%"
                                 for row in sorted(observed,key=lambda row:row.get('max_lateral_distance_m',0)))
            sensitivity_notes += [f"{episode['agent_name']} case0052 — {settings}; range = {100*(max(ratios)-min(ratios)):.4f} percentage points.", '']
    lines = ['# Final Route Evaluation', '',
        '## 1. Previous ambiguity', '',
        'A globally nearest polyline projection can clamp a distant vehicle to the final segment endpoint, '
        'creating apparent full distance progress without reaching the endpoint. In addition, a binary '
        'endpoint completion rate and a distance completion ratio answer different questions.', '',
        '## 2. Final definitions', '',
        'Route Completion Ratio / 线路完成率 is strictly a distance ratio: '
        '(max valid route_s − route_s_start) / (route_s_goal − route_s_start), clipped to [0,1]. '
        'Task Success / 任务成功 is the separately satisfied proxy endpoint criterion without earlier collision. '
        'Task Success Rate / 任务成功率 is successful episodes divided by valid episodes.', '',
        'A collision retains the distance completed before termination but has task_success = 0. '
        'Success within endpoint tolerance can have a distance ratio below one. Old route_progress_ratio, '
        'proxy_route_progress_ratio and proxy_route_completed remain legacy / compatibility fields. '
        'The optional safe route ratio is a safety-conditioned diagnostic, not the primary distance metric.', '',
        '## 3. Continuous route projection', '',
        'Each candidate segment exposes raw and clamped projection fractions, projected point, arc length '
        'and lateral error. Candidate search prioritizes nearby route segments and checks motion-based '
        'forward progress, backward tolerance and lateral distance. A global fallback is explicit. '
        'Invalid projections freeze the last trusted progress; they never create a full route ratio. '
        'Maximum valid progress is retained when the ego subsequently reverses.', '',
        'Resolved configuration and source audit:', '', '```json',
        json.dumps(jsonable(dict(config=result.get('config',{}),audit=result.get('audit',{}))),ensure_ascii=False,indent=2,allow_nan=False),
        '```', '',
        '## 4. Endpoint clamp correction', '',
        'A raw projection beyond the final segment endpoint is rejected outside the configured endpoint '
        'tolerance. Start clamping and lateral-distance limits are also explicit. The endpoint-success '
        'test is independent of the distance ratio. This correction does not move collision times or '
        'change the frozen termination and responsibility decisions.', '',
        '## 5. UniAD and VAD case0052: before and after', '',
        *_observed_table(result.get('before_comparison',[])), '', *case_notes,
        'A corrected ratio near 99.95% is retained when earlier valid projections support that distance. '
        'The fix rejects an invalid endpoint clamp; it does not impose an arbitrary 95% cap or lower distance '
        'to match a failure label. A near-full distance ratio is still not task success.', '',
        *_table(focused,[('episode_id','Episode'),('route_completion_ratio','Corrected distance ratio'),
            ('task_outcome','Task outcome'),('terminal_position_x','Terminal x'),('terminal_position_y','Terminal y'),
            ('terminal_endpoint_distance','Endpoint distance (m)'),('terminal_route_s','Retained route s'),
            ('terminal_route_projection_x','Projection x'),('terminal_route_projection_y','Projection y'),
            ('terminal_route_projection_distance','Projection distance (m)'),('terminal_projection_segment_index','Segment'),
            ('terminal_projection_fraction_raw','Raw fraction'),('terminal_projection_fraction_clamped','Clamped fraction'),
            ('terminal_projection_valid','Projection valid'),('terminal_projection_reason','Reason')]), '',
        'Retained progress reflects the last valid projection when terminal projection is invalid; '
        'the invalid terminal candidate is diagnostic evidence rather than credited progress.', '',
        '## 6. Final agent comparison', '',
        *_table([dict(metric=k,value=v) for k,v in overall.items()], [('metric','Metric'),('value','Observed value')]), '',
        'SparseDriveV2 v2 is the retained version; an unversioned SparseDriveV2 label in this final comparison '
        'means v2. Historical v1 outputs are not deleted. The actual included names and sample counts above '
        'are derived from this run.', '',
        *_table(agents,[('agent_name','Agent'),('episode_count','Episodes'),
            ('mean_route_completion_ratio','Route completion ratio ↑'),('task_success_rate','Task success rate ↑'),
            ('collision_failure_rate','Collision failure rate ↓'),('incomplete_rate','Incomplete rate ↓'),
            ('episode_collision_rate','Episode collision rate ↓'),('PDMS_open_loop','PDMS_open_loop ↑'),
            ('PDMS_TC_open_loop','PDMS_TC_open_loop ↑'),('open_loop_collision_window_rate','Open-loop collision rate'),
            ('Traffic_Compliance_Score_ego','Traffic compliance')]), '',
        'Prior scores are copied by agent key from the original traffic-and-responsibility summary; '
        'their original equal-scene aggregation remains unchanged. An unavailable score is not filled with zero.', '',
        '## 7. Route Completion Ratio', '',
        *_table(agents,[('agent_name','Agent'),('mean_route_completion_ratio','Mean distance ratio'),
            ('median_route_completion_ratio','Median'),('min_route_completion_ratio','Minimum'),
            ('mean_route_projection_coverage','Projection coverage'),('route_completion_status_counts','Statuses')]), '',
        *_table(episodes,[('episode_id','Episode'),('route_completion_ratio','Distance ratio'),
            ('route_projection_coverage','Projection coverage'),('route_completion_status','Status'),
            ('task_success','Task success'),('task_outcome','Outcome')]), '',
        '## 8. Task outcomes', '',
        'SUCCESS means the frozen proxy endpoint-success condition was met; COLLISION_FAILURE means collision '
        'termination; INCOMPLETE covers non-successful scene end, prediction exhaustion and timeout. '
        'Distance completed does not reclassify the endpoint outcome.', '',
        *_table(agents,[('agent_name','Agent'),('task_success_count','SUCCESS'),('task_success_rate','Success rate'),
            ('collision_failure_count','COLLISION_FAILURE'),('collision_failure_rate','Collision failure rate'),
            ('incomplete_count','INCOMPLETE'),('incomplete_rate','Incomplete rate')]), '',
        '## 9. PDMS and route distance measure different properties', '',
        'PDMS combines safety, comfort, drivable-area and progress components of independent plans. '
        'Route Completion Ratio measures credited reference-route distance until episode termination. '
        'The metrics need not have the same ranking; no route correction is used to force agreement.', '',
        '## 10. Projection sensitivity', '',
        'The configurable lateral gate is a route-proxy assumption, not a calibrated universal lane width. '
        'The requested 1.5 m, 2.5 m and 3.5 m checks are reported from actual rerun observations below.', '',
        *_observed_table(result.get('sensitivity',[])), '', *sensitivity_notes,
        'The observed threshold ranges must be interpreted as remaining sensitivity of route proxy projection. '
        'They do not support claims of threshold-invariant progress or fine-grained agent ranking stability.', '',
        'Compare finite distance ratios and coverage jointly: a stable number with little accepted projection '
        'evidence does not demonstrate route-matching reliability. Large threshold-dependent changes indicate '
        'that route proxy projection remains unstable.', '',
        '## 11. Regression and tests', '',
        'Regression evidence: '+_display(result.get('regression','Not attached; no PASS is inferred by this exporter.')), '',
        'Synthetic tests: '+_display(result.get('tests','Not attached; consult final validation evidence.')), '',
        'Collision times, termination reasons, responsibility and all original non-route scores must remain '
        'unchanged for the retained agents. The inclusion-list change reduces the new comparison only; '
        'historical artifacts remain separate.', '',
        '## 12. Limitations', '',
        'Formal interpretation is Proxy Route Completion Ratio and Proxy Task Success: the reference is '
        'logged_ego_path_proxy and the goal is log_endpoint_proxy. Logged-state receding horizon with a '
        'terminal saved-plan commit is not true closed-loop simulation. Missing projection evidence, route '
        'self-intersections and configurable gates remain limitations. Two scenes provide descriptive '
        'validation rather than a statistically representative ranking.', '']
    return '\n'.join(lines)


def export(result, directory, summary=None):
    from driveaudit_eval.utils.io import write_csv
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    agents,overall=summarize(result,summary)
    for name,rows,headers in [
        ('final_route_episode_metrics.csv',result.get('episodes',[]),['episode_id','scene_id','agent_name','route_completion_ratio','task_success','task_outcome']),
        ('final_route_agent_summary.csv',agents,['agent_name','episode_count','mean_route_completion_ratio','task_success_rate']),
        ('final_route_projection_sensitivity.csv',result.get('sensitivity',[]),['episode_id','max_lateral_distance_m','route_completion_ratio','route_projection_coverage']),
        ('final_route_before_after.csv',result.get('before_comparison',[]),['episode_id'])]:
        write_csv(directory/name,rows,headers)
    with (directory/'final_route_projection_frames.jsonl').open('w',encoding='utf-8') as handle:
        for row in result.get('route_timeseries',[]):
            handle.write(json.dumps(jsonable(row),ensure_ascii=False,allow_nan=False)+'\n')
    write_json(directory/'final_route_overall.json',overall)
    (directory/'FINAL_ROUTE_EVALUATION_REPORT.md').write_text(_report(result,agents,overall),encoding='utf-8')
    return CATALOG


if __name__=='__main__':
    import copy
    import tempfile
    result=dict(episodes=[dict(episode_id='s/a',agent_name='a',scene_id='s',status='valid',termination_reason='collision',
        route_completion_ratio=.7,task_success=0,task_outcome='COLLISION_FAILURE',route_projection_coverage=.9)],windows=[])
    summary=dict(traffic_and_responsibility=dict(agent_overall=[dict(agent='a',PDMS_open_loop=.2,PDMS_TC_open_loop=.18)]))
    before=copy.deepcopy(result)
    with tempfile.TemporaryDirectory() as temporary:
        export(result,temporary,summary)
        rows,overall=summarize(result,summary)
        assert rows[0]['mean_route_completion_ratio']==.7 and rows[0]['task_success_rate']==0
        assert rows[0]['collision_failure_rate']==1 and rows[0]['PDMS_TC_open_loop']==.18
        assert result==before
        assert (Path(temporary)/'FINAL_ROUTE_EVALUATION_REPORT.md').read_text().count('## ')==12
        print('final_route_reporting self-check passed')
