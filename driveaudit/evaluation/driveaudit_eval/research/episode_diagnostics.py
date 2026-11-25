"""Terminal-horizon-commit reporting, separate from strict-prefix exports."""
from collections import Counter
import json
from pathlib import Path

from driveaudit_eval.utils.io import jsonable, write_json
from driveaudit_eval.research.episode_reporting import _finite, _stat, _summary
from driveaudit_eval.research.responsibility_reporting import _display, _table

CATALOG = '''
## Episode Rollout V2: terminal horizon commit

The terminal-horizon-commit evaluation closes the artificial tail created by requiring a complete three-second plan at every common scoring window. It still does not constitute true closed-loop simulation because subsequent observations before the terminal commit remain logged-state conditioned.

- Ordinary decisions execute only to the next replanning time; only the last available saved plan is committed through its available horizon, bounded by native scene time and the first termination event.
- proxy_route_progress_ratio and proxy_route_completed use reference = logged ego path proxy, goal = log endpoint proxy. They do not establish mission success.
- temporal_episode_coverage = executed episode duration / (native scene end - episode start). Early collision, proxy endpoint or offroad termination can legitimately reduce coverage.
- scene_end_survival_rate = valid episodes ending at native scene_end / valid episodes. Proxy-goal termination is counted separately.
- Terminal Responsibility V3 retains the frozen formula and applies only to the executed terminal collision. Historical open-loop collisions are separate.
- nominal_prefix remains the strict-prefix baseline; horizon_commit outputs have separate filenames. See EPISODE_ROLLOUT_V2_REPORT.md and strict_vs_terminal_commit.csv.
'''


def _episodes(result):
    """Normalize aliases on copies for shared summaries; preserve source exports."""
    scene_ends = {row['scene_id']: row.get('scene_end_time') for row in result.get('audit', {}).get('scene_audit', [])}
    rows = []
    for original in result.get('episodes', []):
        row = dict(original)
        if row.get('termination_reason') == 'invalid':
            row['termination_reason'] = 'invalid_state'
        row['route_progress_ratio'] = row.get('proxy_route_progress_ratio', row.get('route_progress_ratio'))
        row['route_completed'] = row.get('proxy_route_completed', row.get('route_completed_proxy', row.get('route_completed')))
        native_end = row.get('native_scene_end_time', scene_ends.get(row.get('scene_id')))
        start = row.get('start_time')
        duration = row.get('executed_time', row.get('episode_duration', row.get('duration')))
        denominator = native_end-start if _finite(native_end) and _finite(start) else None
        row['temporal_episode_coverage'] = duration/denominator if _finite(duration) and denominator and denominator > 0 else None
        row['native_scene_end_time'] = native_end
        rows.append(row)
    return rows


def _summarize(episodes, windows):
    summary = _summary(episodes, windows)
    valid = [row for row in episodes if row.get('termination_reason') != 'invalid_state']
    reasons = Counter(row.get('termination_reason') for row in episodes)
    summary.update(proxy_route_progress_mean=summary['route_progress_mean'],
        proxy_route_completion_rate=summary['route_completion_rate'],
        proxy_route_completion_count=summary['route_completion_count'],
        scene_end_count=reasons['scene_end'], proxy_goal_count=reasons['goal_reached_proxy'],
        log_end_count=reasons['log_end'],
        scene_end_survival_rate=reasons['scene_end']/len(valid) if valid else None,
        mean_temporal_coverage=_stat(valid,'temporal_episode_coverage'))
    return summary


def summaries(result):
    episodes = _episodes(result)
    agents = [dict(agent_name=agent, **_summarize([e for e in episodes if e.get('agent_name') == agent],
        [w for w in result.get('windows', []) if w.get('agent_name') == agent]))
        for agent in sorted({e['agent_name'] for e in episodes})]
    return agents, _summarize(episodes, result.get('windows', []))


def _tables(result):
    agents, overall = summaries(result)
    strict_agents, strict_overall = summaries(result.get('strict_prefix', {}))
    strict_by_agent = {row['agent_name']: row for row in strict_agents}
    keys = ['episode_count', 'episode_collision_count', 'episode_collision_rate', 'proxy_route_progress_mean',
        'proxy_route_completion_rate', 'proxy_route_completion_count', 'termination_counts', 'mean_temporal_coverage']
    comparisons = []
    for row in agents:
        old = strict_by_agent.get(row['agent_name'], {})
        comparisons.append(dict(agent_name=row['agent_name'],
            **{'strict_'+key: old.get(key) for key in keys}, **{'commit_'+key: row.get(key) for key in keys}))
    coverage = []
    for mode, value in [('nominal_prefix', result.get('strict_prefix', {})), ('horizon_commit', result)]:
        for episode in _episodes(value):
            coverage.append(dict(episode_id=episode.get('episode_id'), scene_id=episode.get('scene_id'),
                agent_name=episode.get('agent_name'), mode=mode, start_time=episode.get('start_time'),
                end_time=episode.get('end_time'), native_scene_end_time=episode.get('native_scene_end_time'),
                executed_episode_duration=episode.get('executed_time', episode.get('episode_duration', episode.get('duration'))),
                temporal_episode_coverage=episode.get('temporal_episode_coverage'),
                proxy_route_progress_ratio=episode.get('route_progress_ratio'),
                proxy_route_completed=episode.get('route_completed'), termination_reason=episode.get('termination_reason'),
                route_reference=episode.get('route_kind')))
    tails = []
    for episode in result.get('episodes', []):
        windows = [w for w in result.get('windows', []) if w.get('episode_id') == episode.get('episode_id')]
        last = max(windows, key=lambda w: w['decision_time']) if windows else {}
        row = {key: value for key, value in episode.items() if key.startswith('terminal_') or key in
            ('episode_id','scene_id','agent_name','native_scene_end_time','termination_reason',
             'route_progress_before_terminal_commit','route_progress_after_terminal_commit','progress_gained_in_terminal_commit')}
        row.update(last_available_decision_frame=last.get('decision_frame_id'), last_available_decision_time=last.get('decision_time'),
            window_execution_mode=last.get('window_execution_mode'), window_execution_status=last.get('window_execution_status'),
            last_plan_open_loop_collision=last.get('open_loop_collision'),
            last_plan_open_loop_first_collision_time=last.get('open_loop_first_collision_time'),
            last_plan_executed_collision=last.get('executed_prefix_collision'),
            last_plan_executed_first_collision_time=last.get('executed_prefix_first_collision_time'),
            last_plan_actual_executed_duration=last.get('executed_prefix_duration'),
            historical_open_loop_collision_windows=sum(bool(w.get('open_loop_collision')) for w in windows))
        tails.append(row)
    return agents, overall, strict_overall, comparisons, coverage, tails


def _report(result, agents, overall, strict_overall, comparisons, coverage, tails):
    saturated_collisions = [e for e in result.get('episodes', []) if e.get('termination_reason') == 'collision'
        and e.get('proxy_route_progress_ratio', e.get('route_progress_ratio')) == 1.0 and not e.get('goal_reached')]
    proxy_goals = [e for e in result.get('episodes', []) if e.get('termination_reason') == 'goal_reached_proxy']
    goal_ratios = [e.get('proxy_route_progress_ratio', e.get('route_progress_ratio')) for e in proxy_goals]
    goal_ratios = [v for v in goal_ratios if _finite(v)]
    goal_config = result.get('config', {}).get('goal', {})
    goal_details = (f'Observed proxy-goal progress ratios span {min(goal_ratios):.9f} to {max(goal_ratios):.9f} '
        f'across {len(goal_ratios)} episodes.' if goal_ratios else 'No finite proxy-goal progress observations are available.')
    headline = ['episode_count','episode_collision_count','proxy_route_completion_count','proxy_goal_count',
        'scene_end_count','log_end_count','proxy_route_progress_mean','mean_temporal_coverage',
        'ego_dominant_terminal_collision_count','other_dominant_terminal_collision_count']
    lines = ['# Episode Rollout V2 — Terminal Horizon Commit', '',
        '## Execution change and scientific boundary', '',
        'The terminal-horizon-commit evaluation closes the artificial tail created by requiring a complete '
        'three-second plan at every common scoring window. It still does not constitute true closed-loop '
        'simulation because subsequent observations before the terminal commit remain logged-state conditioned.', '',
        'Ordinary windows keep the strict receding-horizon rule: execute only to the next valid decision. '
        'With no next decision, horizon_commit executes the last saved plan until the earliest collision, '
        'proxy endpoint, offroad termination if enabled, native scene end, or available prediction end. '
        'Collision wins an exact-time tie with the proxy endpoint. No new inference or observations are produced, '
        'and logged background actors do not respond to ego.', '',
        'nominal_prefix remains a separately reproduced baseline. Its zero completion must not be interpreted '
        'as all agents failing their routes: common-window filtering can exhaust available decisions about '
        'three seconds before the scene ends.', '',
        '## Two-mode results', '',
        *_table([dict(metric=key,strict_prefix=strict_overall.get(key),terminal_commit=overall.get(key)) for key in headline],
            [('metric','Metric'),('strict_prefix','Strict prefix'),('terminal_commit','Terminal commit')]), '',
        'Invalid episodes are excluded from valid-episode rates and counted separately. Proxy completion '
        'rates use valid route-observed episodes. Scene-end survival excludes proxy-goal termination; both '
        'outcomes are reported separately.', '',
        '## Agent comparison', '',
        *_table(comparisons,[('agent_name','Agent'),('strict_episode_collision_count','Strict collisions'),
            ('strict_proxy_route_progress_mean','Strict progress'),('strict_proxy_route_completion_rate','Strict completion'),
            ('strict_termination_counts','Strict termination'),('commit_episode_collision_count','Commit collisions'),
            ('commit_proxy_route_progress_mean','Commit progress'),('commit_proxy_route_completion_rate','Commit completion'),
            ('commit_termination_counts','Commit termination')]), '',
        *_table(agents,[('agent_name','Agent'),('episode_count','Episodes'),('episode_collision_rate','Collision rate'),
            ('scene_end_survival_rate','Scene-end survival'),('mean_temporal_coverage','Temporal coverage'),
            ('ego_dominant_terminal_collision_count','Ego-dominant terminals'),
            ('other_dominant_terminal_collision_count','Other-dominant terminals'),
            ('mean_terminal_ego_responsibility','Mean terminal ego ratio')]), '',
        '## Last available plans, including VAD, UniAD and DiffusionDriveV2', '',
        'The table evaluates the actual last available plan for every episode. A collision in a historical '
        'three-second horizon does not force a terminal collision in a later plan. Differences do not establish '
        'that feedback-driven replanning avoided a collision.', '',
        *_table(tails,[('episode_id','Episode'),('last_available_decision_frame','Last frame'),
            ('historical_open_loop_collision_windows','Historical collision windows'),
            ('last_plan_open_loop_collision','Last horizon collision'),
            ('last_plan_open_loop_first_collision_time','Last horizon collision time'),
            ('last_plan_actual_executed_duration','Executed last plan (s)'),
            ('last_plan_executed_collision','Executed last-plan collision'),('termination_reason','Termination')]), '',
        '## Terminal tail and route progress', '',
        'Route reference remains logged_ego_path_proxy and the endpoint remains log_endpoint_proxy. '
        'Proxy route progress is arc-length progress, not elapsed-time coverage or verified mission progress. '
        'Completion at this endpoint is reported as proxy completion. Progress freezes at the first termination '
        'event and is not forced to zero after collision.', '',
        f'{len(saturated_collisions)} collision-terminated episodes have projected progress 1.0 while goal_reached is false. '
        'Nearest-polyline projection can clamp to the final route point even when the ego is spatially far from '
        'that point. Therefore 100% projected arc-length progress does not itself satisfy the endpoint proximity '
        'criterion; these collisions retain proxy completion zero.', '',
        goal_details + ' The configured longitudinal endpoint tolerance is '
        + _display(goal_config.get('longitudinal_tolerance_m')) + ' m and Euclidean tolerance is '
        + _display(goal_config.get('euclidean_tolerance_m')) + ' m. A goal reached within tolerance may have '
        'an arc-length ratio below 1.0. The observed ratio is retained without forcing it to one, while proxy '
        'completion records the separately satisfied endpoint criterion.', '',
        *_table(tails,[('episode_id','Episode'),('terminal_decision_time','Terminal decision (s)'),
            ('native_scene_end_time','Native end (s)'),('terminal_plan_horizon_s','Plan horizon (s)'),
            ('terminal_scene_remaining_s','Scene remaining (s)'),('terminal_executed_duration_s','Executed (s)'),
            ('route_progress_before_terminal_commit','Progress before'),
            ('route_progress_after_terminal_commit','Progress after'),
            ('progress_gained_in_terminal_commit','Progress gain')]), '',
        '## Temporal coverage', '',
        'Temporal coverage = executed episode duration / (native scene end − episode start). '
        'Early collision, proxy endpoint or offroad termination can correctly produce coverage below one. '
        'Prediction exhaustion before native scene end remains explicit rather than claiming unseen execution.', '',
        *_table(coverage,[('episode_id','Episode'),('mode','Mode'),('end_time','Execution end (s)'),
            ('native_scene_end_time','Native end (s)'),('temporal_episode_coverage','Temporal coverage'),
            ('proxy_route_progress_ratio','Proxy route progress'),('proxy_route_completed','Proxy completion')]), '',
        '## Terminal Responsibility V3', '',
        'The frozen V3 evidence formula and fusion are reused only for executed terminal collisions. '
        'Ratios are model-based evidence proportions, not legal fault or calibrated probabilities. '
        'Full state arrays and nested evidence remain in episode_rollout.json; the terminal CSV contains scalar fields. '
        'Confidence is reliability information and does not gate finite attribution. CARS unavailable '
        'does not mean responsibility unavailable.', '']
    terminal = result.get('terminal_responsibility', [])
    lines += (_table(terminal,[('episode_id','Episode'),('ego_responsibility_ratio','Ego ratio'),
        ('other_responsibility_ratio','Other ratio'),('ego_responsibility_class','Direction'),
        ('responsibility_dominance_strength','Dominance'),('responsibility_confidence','Confidence'),
        ('cars_status','CARS')]) if terminal else ['No terminal responsibility rows are available; the CSV retains headers.'])
    lines += ['', '## Audit and resolved configuration', '', '```json',
        json.dumps(jsonable(dict(config=result.get('config',{}),audit=result.get('audit',{}))),ensure_ascii=False,indent=2,allow_nan=False),
        '```', '', '## Tests', '', _display(result.get('tests','Not attached to this export; consult the final validation report.')), '',
        '## Regression', '', _display(result.get('regression','Not attached to this export; no regression PASS is inferred.')), '',
        'The strict-prefix baseline, original open-loop metrics and Responsibility V1/V2/V3 remain separate. '
        'Only the last available plan execution policy changes.', '',
        '## Limitations', '',
        'Two scenes and seven agents provide descriptive validation, not a statistically representative ranking. '
        'Logged-state resets before the final commit are not continuous physical feedback. The final saved '
        'plan is executed open-loop against logged background trajectories. Sampled collision search and '
        'interpolation retain their documented resolution limits. A proxy endpoint is not a mission goal.', '']
    return '\n'.join(lines)


def export(result, directory):
    from driveaudit_eval.utils.io import write_csv
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    agents, overall, strict_overall, comparison, coverage, tails = _tables(result)
    for name,rows,headers in [
        ('episode_rollout_terminal_commit_events.csv',result.get('episodes',[]),['episode_id','scene_id','agent_name','termination_reason']),
        ('episode_rollout_terminal_commit_agent_summary.csv',agents,['agent_name','episode_count']),
        ('strict_vs_terminal_commit.csv',comparison,['agent_name']),
        ('terminal_tail_execution.csv',tails,['episode_id','scene_id','agent_name','terminal_commit_enabled']),
        ('episode_temporal_coverage.csv',coverage,['episode_id','mode','temporal_episode_coverage']),
        ('episode_terminal_responsibility_v3.csv',
         [{k:v for k,v in row.items() if not isinstance(v,(dict,list,tuple))} for row in result.get('terminal_responsibility',[])],
         ['episode_id','event_id','scene_id','agent_name','ego_responsibility_ratio','other_responsibility_ratio',
          'ego_responsibility_class','dominant_responsible_actor','responsibility_dominance_strength','responsibility_confidence','cars_status'])]:
        write_csv(directory/name,rows,headers)
    write_json(directory/'episode_rollout_terminal_commit_overall.json',dict(strict_prefix=strict_overall,terminal_commit=overall))
    (directory/'EPISODE_ROLLOUT_V2_REPORT.md').write_text(_report(result,agents,overall,strict_overall,comparison,coverage,tails),encoding='utf-8')
    return CATALOG


if __name__ == '__main__':
    import copy
    import csv
    import tempfile
    result = dict(episodes=[dict(episode_id='s/a',scene_id='s',agent_name='a',termination_reason='scene_end',
        route_available=True,proxy_route_progress_ratio=.9,proxy_route_completed=0,start_time=0.,
        end_time=10.,native_scene_end_time=10.,executed_time=10.)],windows=[],strict_prefix=dict(episodes=[],windows=[]))
    before=copy.deepcopy(result)
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        frozen=directory/'episode_rollout_events.csv'; frozen.write_text('baseline\n')
        export(result,directory)
        overall=json.loads((directory/'episode_rollout_terminal_commit_overall.json').read_text())['terminal_commit']
        assert overall['mean_temporal_coverage']==1 and overall['proxy_route_progress_mean']==.9
        assert overall['scene_end_count']==1 and overall['proxy_route_completion_count']==0
        assert list(csv.reader((directory/'episode_terminal_responsibility_v3.csv').open()))[0][0]=='episode_id'
        assert frozen.read_text()=='baseline\n' and result==before
        assert len(list(directory.glob('*.csv')))==7
        print('episode_reporting_v2 self-check passed')
