"""Episode execution exports, explicit denominators and descriptive diagnostics."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from driveaudit_eval.utils.io import jsonable, write_json
from driveaudit_eval.research.responsibility_reporting import _display, _table

CATALOG = '''
## Episode-level logged receding-horizon execution

Episode-level evaluation executes only the prefix of each logged-state plan until the next replanning instant. It does not model full environment feedback and therefore is not claimed as true closed-loop simulation.

- open_loop_collision_window_rate: Fraction of independently evaluated open-loop planning windows whose predicted horizon contains collision.
- episode_collision_rate: Episodes terminated by collision / valid episodes. Invalid-state episodes are reported separately.
- collision_beyond_execution_prefix_count: Evaluated windows with a predicted collision after their executed prefix; this does not prove replanning avoided a collision.
- route_progress_ratio: Maximum forward reference-polyline arc-length progress / remaining reference-route length, clipped to [0, 1]. Route source and availability remain explicit.
- route_completed: Declared reference endpoint reached before termination; logged-route proxy completion is not verified mission completion. Collision termination does not imply zero route progress.
- EP: Retained open_loop_progress_proxy; it is not full route completion.
- Terminal Responsibility V3: Frozen attribution applied only to episode terminal collisions. Original open-loop V1/V2/V3 outputs are preserved separately.

See EPISODE_ROLLOUT_REPORT.md, episode_rollout_window_execution.csv and route_progress_timeseries.jsonl for execution and route provenance. These are descriptive results on two scenes, not a statistically representative ranking.
'''


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _stat(rows, key, operation=statistics.mean):
    values = [row[key] for row in rows if _finite(row.get(key))]
    return operation(values) if values else None


def _summary(episodes, windows):
    valid = [row for row in episodes if row.get('termination_reason') not in ('invalid_state','invalid')]
    collisions = [row for row in valid if row.get('termination_reason') == 'collision']
    reasons = Counter(row.get('termination_reason') for row in episodes)
    n = len(valid)
    def rate(count):
        return count / n if n else None
    def direction(row):
        return row.get('terminal_ego_responsibility_class', row.get('terminal_responsibility_dominant'))
    ego_count = sum(direction(row) == 'ego_dominant' for row in collisions)
    other_count = sum(direction(row) == 'other_dominant' for row in collisions)
    observed_routes = [row for row in valid if row.get('route_available') and _finite(row.get('route_progress_ratio'))]
    completions = sum(bool(row.get('route_completed')) for row in observed_routes)
    horizon = sum(bool(row.get('open_loop_collision')) for row in windows)
    prefix = sum(bool(row.get('executed_prefix_collision')) for row in windows)
    beyond = sum(bool(row.get('collision_beyond_execution_prefix',
                 bool(row.get('open_loop_collision')) and not row.get('executed_prefix_collision')
                 and row.get('window_execution_status') == 'executed')) for row in windows)
    return dict(episode_count=len(episodes), valid_episode_count=n, invalid_episode_count=len(episodes)-n,
        open_loop_window_count=len(windows), open_loop_collision_window_count=horizon,
        open_loop_collision_window_rate=horizon/len(windows) if windows else None,
        executed_prefix_collision_count=prefix, collision_beyond_execution_prefix_count=beyond,
        episode_collision_count=len(collisions), episode_collision_rate=rate(len(collisions)),
        collision_free_episode_rate=rate(n-len(collisions)),
        route_available_episode_count=len(observed_routes), route_progress_mean=_stat(observed_routes,'route_progress_ratio'),
        route_progress_median=_stat(observed_routes,'route_progress_ratio',statistics.median),
        route_progress_min=_stat(observed_routes,'route_progress_ratio',min),
        route_completion_count=completions,
        route_completion_rate=completions/len(observed_routes) if observed_routes else None,
        mean_time_to_collision=_stat(collisions,'time_to_collision'),
        median_time_to_collision=_stat(collisions,'time_to_collision',statistics.median),
        mean_distance_before_collision=_stat(collisions,'distance_before_collision'),
        mean_route_progress_at_collision=_stat(collisions,'route_progress_ratio'),
        offroad_termination_rate=rate(reasons['offroad']), timeout_rate=rate(reasons['timeout']),
        log_end_rate=rate(reasons['log_end']), termination_counts=dict(reasons),
        mean_terminal_ego_responsibility=_stat(collisions,'terminal_ego_responsibility_ratio'),
        ego_dominant_terminal_collision_count=ego_count, other_dominant_terminal_collision_count=other_count,
        ego_dominant_terminal_collision_rate=ego_count/len(collisions) if collisions else None,
        other_dominant_terminal_collision_rate=other_count/len(collisions) if collisions else None,
        rate_denominators='episode rates: valid episodes; route completion: valid route-observed episodes; terminal direction rates: collision-terminated episodes',
        window_execution_status_counts=dict(Counter(row.get('window_execution_status') for row in windows)))


def _summaries(result):
    episodes, windows = result.get('episodes', []), result.get('windows', [])
    agents = [dict(agent_name=key, **_summary([e for e in episodes if e.get('agent_name') == key],
                  [w for w in windows if w.get('agent_name') == key]))
              for key in sorted({e['agent_name'] for e in episodes})]
    scenes = [dict(scene_id=key, **_summary([e for e in episodes if e.get('scene_id') == key],
                  [w for w in windows if w.get('scene_id') == key]))
              for key in sorted({e['scene_id'] for e in episodes})]
    columns = ['agent_name','open_loop_window_count','open_loop_collision_window_count',
        'open_loop_collision_window_rate','episode_count','valid_episode_count','episode_collision_count',
        'episode_collision_rate','route_progress_mean','route_completion_rate',
        'collision_beyond_execution_prefix_count','executed_prefix_collision_count']
    comparison = [{key: row.get(key) for key in columns} for row in agents]
    return agents, scenes, comparison, _summary(episodes, windows)


def _report(result, agents, scenes, comparison, overall):
    episodes = result.get('episodes', [])
    end_audit = []
    for scene in result.get('audit', {}).get('scene_audit', []):
        ended = [e for e in episodes if e.get('scene_id') == scene.get('scene_id') and e.get('termination_reason') == 'log_end']
        native_end = scene.get('scene_end_time')
        for end_time in sorted({e['end_time'] for e in ended if _finite(e.get('end_time'))}):
            end_audit.append(dict(scene_id=scene.get('scene_id'), episode_end_time=end_time, native_scene_end=native_end,
                unexecuted_native_tail_s=native_end-end_time if _finite(native_end) else None,
                episode_count=sum(e.get('end_time')==end_time for e in ended)))
    table_columns = [('agent_name','Agent'),('open_loop_window_count','Windows'),
        ('open_loop_collision_window_count','Horizon collision windows'),
        ('open_loop_collision_window_rate','Horizon collision rate'),('valid_episode_count','Valid episodes'),
        ('episode_collision_count','Episode collisions'),('episode_collision_rate','Episode collision rate'),
        ('route_progress_mean','Mean route progress'),('route_completion_rate','Route completion rate')]
    lines = ['# Episode-level receding-horizon evaluation', '',
        '## 1. Open-loop collision and episode collision', '',
        'An open-loop collision is detected anywhere in an independently evaluated planning horizon. '
        'An episode collision occurs inside the prefix executed before the next decision. A collision '
        'beyond that prefix does not itself terminate the episode. This distinction does not prove that replanning avoided a collision.', '',
        '## 2. Execution semantics and source audit', '',
        'Execution duration follows consecutive decision timestamps, not a fixed presumed replanning rate. '
        'The final decision uses the configured terminal policy: nominal interval or saved-plan horizon commit, capped by available prediction and scene time. '
        'The prefix endpoint is interpolated when needed. Source coordinates, timing, input hashes and '
        'route provenance are recorded in the audit below.', '', '```json',
        json.dumps(jsonable(result.get('audit', {})),ensure_ascii=False,indent=2,allow_nan=False), '```', '',
        '## 3. Feedback boundary', '',
        'Episode-level evaluation executes only the prefix of each logged-state plan until the next replanning '
        'instant. It does not model full environment feedback and therefore is not claimed as true closed-loop simulation.', '',
        'This is logged-background counterfactual execution: other actors follow synchronized logged trajectories '
        'and do not respond to the ego plan. Logged-state prefixes are not a continuous stateful physical rollout; '
        'successive observations and predictions retain their logged-state conditioning.', '',
        '## 4. Episode construction', '',
        'Each scene × agent pair starts at its first common valid decision and advances in timestamp order. '
        'Every source window remains in the execution CSV, including those not executed after termination.', '',
        'The common valid planning windows require a complete three-second prediction horizon, so the decision '
        'log can end before the native scene ends. Here log_end means available decision plans are exhausted; '
        'it does not imply the last native scene frame was executed. A remaining route at this cutoff is not '
        'automatically an agent failure. Native scene time beyond the final executed prefix is not credited as progress.', '',
        *_table(end_audit, [('scene_id','Scene'),('episode_count','Episodes'),
            ('episode_end_time','Episode end (s)'),('native_scene_end','Native scene end (s)'),
            ('unexecuted_native_tail_s','Unexecuted native tail (s)')]), '',
        *_table([dict(metric=k,value=v) for k,v in overall.items()], [('metric','Metric'),('value','Observed value')]), '',
        '## 5. Collision termination', '',
        'The first event on the executed prefix terminates the episode; simultaneous collision and goal '
        'events give collision priority. Later windows do not execute or contribute route progress or terminal '
        'responsibility. The collision geometry and temporal refinement policy are recorded by the evaluator.', '',
        *_table(episodes,[('episode_id','Episode'),('termination_reason','Termination'),('end_time','End time (s)'),
            ('time_to_collision','Time to collision (s)'),('executed_decision_count','Executed windows'),
            ('available_decision_count','Available windows')]), '',
        '## 6. Route progress', '',
        'States are projected onto the declared reference polyline. Net progress is s_current − s_start; '
        'maximum progress is max_t(s_t − s_start). The progress ratio divides maximum progress by '
        's_goal − s_start and clips to [0,1]. An unavailable route remains unavailable, without a Euclidean '
        'substitute. A logged-trajectory reference, if used, is a route proxy rather than a verified mission route. '
        'Progress freezes at termination. Original EP remains an open_loop_progress_proxy.', '',
        *_table(episodes,[('episode_id','Episode'),('route_available','Route available'),
            ('route_kind','Route source'),('route_progress_distance','Progress (m)'),('route_total_distance','Remaining route (m)'),
            ('route_progress_ratio','Progress ratio'),('route_completed','Completed')]), '',
        '## 7. Route completion', '',
        'Completion means the configured reference endpoint criterion was reached before termination. '
        'For the logged_ego_path_proxy, this is logged-path endpoint completion, not verified mission success. '
        'Partial progress and completion remain separate. Completion averages use valid episodes with an observed route; their count is exported explicitly.', '',
        '## 8. Open-loop versus episode results', '', *_table(comparison,table_columns), '',
        'The denominators differ: planning windows versus valid episodes. Rates are not interchangeable.', '',
        '## 9. Agent and scene results', '',
        *_table(agents,[('agent_name','Agent'),('episode_count','Episodes'),('invalid_episode_count','Invalid'),
            ('collision_free_episode_rate','Collision-free rate'),('route_progress_median','Median route progress'),
            ('mean_time_to_collision','Mean terminal TTC (s)'),('mean_terminal_ego_responsibility','Terminal ego ratio'),
            ('ego_dominant_terminal_collision_rate','Ego-dominant terminal rate'),
            ('other_dominant_terminal_collision_rate','Other-dominant terminal rate')]), '',
        *_table(scenes,[('scene_id','Scene'),('episode_count','Episodes'),('episode_collision_count','Collisions'),
            ('route_progress_mean','Mean progress'),('route_completion_rate','Completion rate')]), '',
        '## 10. Terminal collision responsibility', '',
        'Frozen Responsibility V3 applies only to the terminal collision. Full-horizon V1/V2/V3 outputs remain '
        'separate. Responsibility ratios are model-based normalized evidence proportions, not legal fault. '
        'Confidence describes reliability and does not suppress a finite attribution.', '']
    terminal = result.get('terminal_responsibility', [])
    if terminal:
        lines += _table(terminal,[(key,key) for key in ('episode_id','event_id','ego_responsibility_ratio',
            'other_responsibility_ratio','ego_responsibility_class','responsibility_dominance_strength',
            'responsibility_confidence','cars_status')])
    else:
        lines += ['No terminal collision responsibility rows were produced. This is not evidence of zero responsibility; '
                  'consult episode termination reasons and status. The empty CSV retains its schema.']
    lines += ['', '## 11. Traffic compliance until termination', '',
        'Only executed prefixes may contribute. Logged future states after termination and unexecuted horizon '
        'tails are excluded. Scores and coverage remain unavailable where the evaluator cannot support them. '
        'Kinematic derivatives across logged-state resets are not a valid continuous comfort measure.', '',
        *_table(episodes,[('episode_id','Episode'),('traffic_compliance_score','Executed compliance'),
            ('traffic_compliance_coverage','Coverage'),('offroad_before_termination','Offroad before termination')]), '',
        '## 12. Synthetic tests', '',
        _display(result.get('tests', 'Test results are documented in the final run validation report; this exporter does not infer test success.')), '',
        '## 13. Limitations', '',
        'This is descriptive validation on two scenes and seven agents, not a statistically representative '
        'ranking. Reused logged-state plans provide no fresh observation feedback, responsive background actors '
        'or stateful simulated dynamics. Short execution prefixes may differ greatly from full-horizon collision '
        'diagnostics without demonstrating policy superiority. Route source and endpoint assumptions constrain '
        'progress and completion interpretations.', '',
        ('Collision timelines mark decision instants, executed prefixes, predicted horizon collision times and actual '
         'terminal collisions; prefix segments after termination do not appear.' if any(e.get('termination_reason')=='collision' for e in episodes)
         else 'No collision-terminated episode was observed; no artificial collision timeline was generated.'), '',
        '## 14. Regression', '',
        _display(result.get('regression', 'Frozen-input and original-metric regression results are documented in the final run validation report; no pass is inferred here.')), '',
        'This export adds episode diagnostics. It does not replace original video, trajectory, consistency, '
        'traffic, agent or Responsibility V1/V2/V3 results.', '']
    return '\n'.join(lines)


def _plots(result, agents, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    episodes, windows = result.get('episodes', []), result.get('windows', [])
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('png','pdf'):
            fig.savefig(directory/f'{name}.{suffix}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5))
    for i,row in enumerate(agents):
        value=row['route_progress_mean']
        if _finite(value):
            ax.bar(i,value,color='#3978ab',alpha=.65)
            values=[e['route_progress_ratio'] for e in episodes if e.get('agent_name')==row['agent_name'] and _finite(e.get('route_progress_ratio'))]
            ax.scatter([i]*len(values),values,color='#222222',s=24,zorder=3)
        else:
            ax.text(i,.02,'unavailable',ha='center',rotation=90)
    ax.set(xticks=range(len(agents)),xticklabels=[r['agent_name'] for r in agents],ylim=(0,1.05),
           ylabel='Route progress ratio',title='Reference-route progress (bars: means; points: episodes)')
    ax.tick_params(axis='x',rotation=25)
    save(fig,'episode_route_progress_by_agent')
    fig,ax=plt.subplots(figsize=(10,4.5))
    reasons=sorted({e.get('termination_reason','unavailable') for e in episodes})
    bottom=[0]*len(agents)
    for reason in reasons:
        values=[sum(e.get('agent_name')==a['agent_name'] and e.get('termination_reason')==reason for e in episodes) for a in agents]
        ax.bar(range(len(agents)),values,bottom=bottom,label=reason)
        bottom=[a+b for a,b in zip(bottom,values)]
    ax.set(xticks=range(len(agents)),xticklabels=[r['agent_name'] for r in agents],ylabel='Episode count',title='Termination reasons')
    ax.tick_params(axis='x',rotation=25)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if reasons: ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    save(fig,'episode_termination_reasons')
    for number,episode in enumerate(episodes,1):
        if episode.get('termination_reason')!='collision':
            continue
        rows=[w for w in windows if w.get('episode_id')==episode['episode_id']]
        rows.sort(key=lambda w:w['decision_time'])
        fig,ax=plt.subplots(figsize=(10,max(3,len(rows)*.14+1.5)))
        for i,row in enumerate(rows):
            start=row['decision_time']; horizon=row.get('executed_prefix_duration',row.get('execution_horizon_s'))
            executed=row.get('window_execution_status')=='executed'
            if executed and _finite(horizon):
                ax.plot([start,start+horizon],[i,i],color='#3978ab',linewidth=3)
            ax.plot(start,i,'|',color='#333333' if executed else '#bbbbbb')
            predicted=row.get('open_loop_first_collision_time')
            if _finite(predicted): ax.plot(predicted,i,'x',color='#d99b49')
            terminal=row.get('executed_prefix_first_collision_time')
            if row.get('executed_prefix_collision') and _finite(terminal): ax.plot(terminal,i,'o',color='#c33434')
        ax.plot([],[],color='#3978ab',linewidth=3,label='Executed prefix')
        ax.plot([],[],'|',color='#333333',label='Executed decision')
        ax.plot([],[],'|',color='#bbbbbb',label='Not executed decision')
        ax.plot([],[],'x',color='#d99b49',label='Predicted horizon collision')
        ax.plot([],[],'o',color='#c33434',label='Terminal prefix collision')
        ax.set(xlabel='Scene timestamp (s)',ylabel='Decision index (chronological)',title=episode['episode_id'])
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.invert_yaxis();ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        save(fig,f'episode_collision_timeline_{number:02d}')


def export(result, directory):
    """Write into all_metrics_export and return Markdown for its metric catalog."""
    from driveaudit_eval.utils.io import write_csv
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    agents, scenes, comparison, overall=_summaries(result)
    tables=[('episode_rollout_window_execution.csv',result.get('windows',[]),['episode_id','scene_id','agent_name','decision_frame_id','decision_time','prediction_horizon_s','execution_horizon_s','open_loop_collision','open_loop_first_collision_time','executed_prefix_collision','executed_prefix_first_collision_time','window_execution_status','episode_done_after_window','route_s_before','route_s_after','route_progress_ratio_after']),
        ('episode_rollout_events.csv',result.get('episodes',[]),['episode_id','scene_id','agent_name','termination_reason']),
        ('episode_rollout_agent_summary.csv',agents,['agent_name','episode_count']),
        ('episode_rollout_scene_summary.csv',scenes,['scene_id','episode_count']),
        ('episode_terminal_responsibility.csv',result.get('terminal_responsibility',[]),['episode_id','event_id','scene_id','agent_name','ego_responsibility_ratio','other_responsibility_ratio','ego_responsibility_class','dominant_responsible_actor','responsibility_dominance_strength','responsibility_confidence','cars_status']),
        ('open_loop_vs_episode_comparison.csv',comparison,['agent_name'])]
    for filename,rows,headers in tables:
        write_csv(directory/filename,rows,headers)
    with (directory/'route_progress_timeseries.jsonl').open('w',encoding='utf-8') as handle:
        for row in result.get('route_timeseries',[]):
            handle.write(json.dumps(jsonable(row),ensure_ascii=False,allow_nan=False)+'\n')
    write_json(directory/'episode_rollout_overall.json',overall)
    write_json(directory/'episode_rollout_audit.json',result.get('audit',{}))
    (directory/'EPISODE_ROLLOUT_REPORT.md').write_text(_report(result,agents,scenes,comparison,overall),encoding='utf-8')
    _plots(result,agents,directory)
    return CATALOG


if __name__=='__main__':
    import csv
    import tempfile
    sample=dict(episodes=[dict(episode_id='scene/agent',scene_id='scene',agent_name='agent',termination_reason='log_end',
                route_available=False,route_progress_ratio=None,route_completed=False)],
                windows=[dict(episode_id='scene/agent',scene_id='scene',agent_name='agent',decision_time=0,
                window_execution_status='executed',open_loop_collision=True,executed_prefix_collision=False)])
    with tempfile.TemporaryDirectory() as temporary:
        export(sample,temporary)
        root=Path(temporary)
        header=next(csv.reader((root/'episode_terminal_responsibility.csv').open()))
        assert 'ego_responsibility_ratio' in header
        overall=json.loads((root/'episode_rollout_overall.json').read_text())
        assert overall['route_progress_mean'] is None and overall['episode_collision_count']==0
        assert overall['collision_beyond_execution_prefix_count']==1
        assert len(list(root.glob('*.png')))==2
        assert (root/'EPISODE_ROLLOUT_REPORT.md').read_text().count('## ')==14
        print('episode_reporting self-check passed')
