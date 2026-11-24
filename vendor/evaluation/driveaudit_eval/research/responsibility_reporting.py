"""Lossless V2 exports and simple descriptive diagnostic figures."""
from collections import Counter
import json
import math
from pathlib import Path

from driveaudit_eval.utils.io import jsonable, write_json


def _number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _display(value):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return 'unavailable'
    if isinstance(value, float):
        return f'{value:.4f}'
    if isinstance(value, dict) and 'raw' in value:
        raw = value.get('raw') or {}
        return '; '.join(f'{label}={_display(source.get(key))}' for source,key,label in [
            (value,'status','status'),(raw,'preventable','preventable'),(raw,'min_ttc','min TTC (s)'),
            (raw,'max_braking_deficit','max braking deficit (m)')])
    if isinstance(value, (dict, list)):
        return json.dumps(jsonable(value), ensure_ascii=False)
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _table(rows, columns):
    return ['| ' + ' | '.join(title for key, title in columns) + ' |',
            '| ' + ' | '.join('---' for _ in columns) + ' |',
            *['| ' + ' | '.join(_display(row.get(key)) for key, title in columns) + ' |' for row in rows]]


def _explain(event):
    """Keep the measured evidence in view without printing trajectory arrays."""
    def compact(value):
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()
                    if key not in ('states', 'actor_a_states', 'actor_b_states', 'frame_evidence', 'frames',
                                   'kinematic_series', 'relative_coordinates', 'native_dimensions', 'observed_frame_ids',
                                   'violation_frame_ids', 'frame_ids', 'timeseries')
                    and not (isinstance(item, dict) and item.get('observable') is False)}
        if isinstance(value, list):
            if value and all(item is None or isinstance(item, (int,float)) for item in value):
                finite = [item for item in value if _number(item)]
                return dict(sample_count=len(value), finite_count=len(finite),
                            minimum=min(finite) if finite else None, maximum=max(finite) if finite else None)
            return [compact(item) for item in value]
        return round(value, 5) if isinstance(value, float) and math.isfinite(value) else value
    ratio = event.get('ego_responsibility_ratio')
    if not event.get('ego_involved', False):
        interpretation = 'Non-ego pair: interpret actor A/B evidence; ego attribution is unavailable.'
    elif not _number(ratio):
        interpretation = 'The evidence does not support a finite ego ratio; no equal split is imputed.'
    elif event.get('responsibility_class') == 'uncertain':
        interpretation = 'A finite evidence ratio is available, but confidence does not support a primary/shared class.'
    else:
        interpretation = 'Model-based evidence suggests ' + ('stronger ego attribution.' if ratio > .5 else
                           'stronger other-actor attribution.' if ratio < .5 else 'equal normalized attribution.')
    return [interpretation, '', 'Measured evidence (null means unavailable; numeric arrays show finite sample count and range; full arrays and unavailable rule records are in JSONL):', '', '```json',
            json.dumps(jsonable(compact(event.get('evidence', {}))), ensure_ascii=False, indent=2, allow_nan=False), '```']


def _report_ratios(result):
    overall = result.get('overall', {})
    events = result.get('events', [])
    total = len(events)
    old = overall.get('v1_scorable_count')
    new = overall.get('responsibility_scorable_count')
    lines = ['# Collision responsibility V2', '', '## Method and interpretation', '',
        result.get('method', 'model-based responsibility attribution'), '',
        result.get('disclaimer', 'Responsibility ratio 表示基于运动学、交通规则、碰撞因果和可避免性证据得到的模型化责任证据比例，并非法律责任比例。'), '',
        'This run validates the evaluation methodology and produces descriptive results on two scenes; '
        'it is not a statistically representative benchmark.', '',
        'V1 required usable CARS vehicle replay. V2 first evaluates motion, rule, conflict-causation and '
        'avoidability evidence; CARS availability is a separate state.', '',
        'For each actor, S = sum(w_k * score_k) / sum(available w_k). '
        'R_A = S_A / (S_A + S_B), R_B = 1 - R_A. These are normalized evidence scores, not calibrated probabilities. '
        'Missing evidence is excluded from the weighted mean; absent discriminative scores leave the ratio null.', '',
        'Where CARS is valid, its preventable vote ratio augments basic avoidability using the configured alpha. '
        'Unavailable CARS contributes no vote score. Confidence includes coverage, score separation, '
        'history, type, geometry and agreement terms; it is not the maximum responsibility ratio.', '',
        'Only observed applicable traffic rules contribute. Missing red-light, speed-limit, solid-line or '
        'priority information is not evidence of compliance. Geometric crossing evidence does not establish right of way.', '',
        'Resolved configuration:', '', '```json',
        json.dumps(jsonable(result.get('config', {})), ensure_ascii=False, indent=2, allow_nan=False), '```', '',
        '## Coverage and classes', '',
        *_table([dict(method='CARS-only V1', scorable=old, unscorable=total-old if old is not None else None,
                      rate=old/total if old is not None and total else None),
                 dict(method='Responsibility V2', scorable=new, unscorable=overall.get('unscorable_count'),
                      rate=overall.get('responsibility_scorable_rate'))],
                [('method','Method'),('scorable','Scorable'),('unscorable','Unscorable'),('rate','Scorable rate')]), '']
    if total and old is not None and new is not None:
        lines += [f'Coverage change: {(new-old)/total*100:.2f} percentage points. Scorable includes uncertain evidence; '
                  'finite-ratio coverage is reported separately.', '']
    keys = ['ego_primary_count','other_primary_count','shared_count','uncertain_count','unscorable_count',
            'responsibility_ratio_count','cars_available_count','cars_changed_final_class_count',
            'mean_evidence_coverage','mean_confidence']
    lines += _table([dict(metric=k,value=overall.get(k)) for k in keys], [('metric','Metric'),('value','Value')])
    unscorable = [e for e in events if e.get('basic_status') == 'unscorable']
    lines += ['', 'Unscorable reasons: ' + _display(dict(Counter(str(e.get('reason')) for e in unscorable))), '',
        '## Agent comparison', '',
        'Rates use each row’s documented denominator. Ego means use finite ego event ratios only. '
        'No collision means no conditional responsibility estimate, not zero responsibility. '
        'Attributed burden is an experimental event-weighted diagnostic divided by all evaluated windows; '
        'unknown ratios produce bounds. It does not replace NC or PDMS.', '']
    lines += _table(result.get('agent_summary', []), [
        ('agent_name','Agent'),('collision_event_count','Events'),('raw_collision_rate','Raw window collision rate'),
        ('ego_responsibility_mean','Ego mean'),('ego_primary_rate','Ego primary rate'),
        ('other_primary_rate','Other primary rate'),('shared_rate','Shared rate'),
        ('ego_attributed_collision_burden','Attributed burden'),('mean_confidence','Confidence')])
    lines += ['', '## Evidence ablation and sensitivity', '']
    lines += _table(result.get('experiment_summary', []), [
        ('experiment_family','Family'),('experiment','Setting'),('responsibility_scorable_count','Scorable'),
        ('ego_responsibility_mean','Ego mean'),('mean_confidence','Confidence'),
        ('mean_evidence_coverage','Coverage'),('class_distribution','Class distribution')])
    lines += ['', 'Per-event stability (primary flip rate compares pairs of primary-class settings only; '
              'unavailable means no eligible comparison):', '']
    lines += _table(result.get('stability', []), [('event_id','Event'),('experiment_family','Family'),
        ('ratio_std','Ratio std'),('primary_actor_flip_rate','Primary flip rate'),
        ('class_consistency','Class consistency'),('class_flip_rate_vs_default','Class flip vs default')])
    unstable = [r for r in result.get('stability', []) if _number(r.get('primary_actor_flip_rate')) and r['primary_actor_flip_rate'] > 0]
    lines += ['', ('Responsibility result not robust for settings/events with primary-actor flips; inspect the table above.'
                  if unstable else 'No primary-actor flips were observed in eligible comparisons; this does not establish general robustness.'), '',
        '## CARS incremental evidence', '', 'Only CARS-valid events are compared below. Ratios refer to actor A consistently.', '']
    lines += _table([r for r in result.get('cars_comparison', []) if r.get('cars_status') in ('valid','cars_valid')],
        [('event_id','Event'),('ratio_before','Basic ratio'),('ratio_after','Basic + CARS ratio'),
         ('class_before','Basic class'),('class_after','Final class'),('confidence_before','Basic confidence'),
         ('confidence_after','Final confidence')])
    lines += ['', '## Collision types', '']
    lines += _table(result.get('collision_type_summary', []), [('collision_type','Type'),
        ('collision_event_count','Events'),('ego_responsibility_mean','Ego mean'),('other_responsibility_mean','Other mean'),('mean_confidence','Confidence')])
    lines += ['', '## Per-event explanations', '',
        'A/B identities and CARS outcomes remain explicit; no unavailable value is silently interpreted as zero. '
        'Full timestamped state arrays are retained in collision_responsibility_evidence_v2.jsonl.', '']
    for index, event in enumerate(events, 1):
        lines += [f"### {index}. {event.get('event_id', 'unavailable')}", '',
            f"Scene / agent / frame: {_display(event.get('scene_id'))} / {_display(event.get('agent_name'))} / {_display(event.get('decision_frame_id'))}", '',
            f"Pair A / B: {_display(event.get('actor_a_id'))} ({_display(event.get('actor_a_type'))}) / {_display(event.get('actor_b_id'))} ({_display(event.get('actor_b_type'))})", '',
            *_table([dict(field=k,value=event.get(k)) for k in ['collision_time','collision_type','basic_status','cars_status',
                'actor_a_responsibility_ratio','actor_b_responsibility_ratio','ego_responsibility_ratio','other_responsibility_ratio',
                'responsibility_class','ego_responsibility_class','responsibility_confidence','confidence_terms','evidence_coverage',
                'active_evidence','unavailable_evidence','reason']], [('field','Field'),('value','Observed result')]), '',
            *_table([dict(actor=actor, **{key:event.get(f'{actor}_{key}') for key in
                        ('rule_score','causal_score','avoidability_score','maneuver_score','raw_responsibility_score')},
                        **{key:event.get(f'cars_{actor}_{key}') for key in ('fsm','ccjp','rss','votes')})
                     for actor in ('actor_a','actor_b')], [('actor','Actor'),('rule_score','Rule'),('causal_score','Causal'),
                        ('avoidability_score','Avoidability'),('maneuver_score','Maneuver'),('raw_responsibility_score','Raw score'),
                        ('fsm','FSM'),('ccjp','CCJP'),('rss','RSS'),('votes','CARS votes')]), '', *_explain(event), '']
    lines += ['## Limitations', '', 'Evidence ratios are heuristic and uncalibrated against human attribution labels. '
              'Short history, missing map rules, unknown actor categories and center-trajectory geometric proxies '
              'limit interpretation. No causal intervention on the real scene is observed. '
              'All original evaluation metrics remain separate; regression and synthetic-test evidence are documented '
              'in the run-level RESPONSIBILITY_V2_REPORT.md.', '']
    return '\n'.join(lines)


def _figures(result, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    events = result.get('events', [])
    def save(fig, name):
        fig.tight_layout()
        for extension in ('png','pdf'):
            fig.savefig(directory / f'collision_responsibility_{name}.{extension}', dpi=160, bbox_inches='tight')
        plt.close(fig)
    labels = [f"{i+1:02d} {e.get('scene_id', '?')} / {e.get('agent_name') or 'GT'} / {e.get('decision_frame_id', '?')}" for i,e in enumerate(events)]
    fig, ax = plt.subplots(figsize=(11, max(3, len(events)*.32+1.5)))
    legend_done = False
    for i,e in enumerate(events):
        a,b=e.get('ego_responsibility_ratio'),e.get('other_responsibility_ratio')
        if e.get('ego_involved') and _number(a) and _number(b):
            ax.barh(i,a,color='#3978ab',label='Ego' if not legend_done else None)
            ax.barh(i,b,left=a,color='#d99b49',label='Other' if not legend_done else None)
            legend_done=True
        else:
            ax.text(.02,i,'unavailable (non-ego pair)' if not e.get('ego_involved') else 'unavailable ratio',va='center',color='#777777')
    ax.set(yticks=range(len(events)),yticklabels=labels,xlim=(0,1),xlabel='Normalized responsibility evidence ratio',title='Collision events (two-scene descriptive evaluation)')
    ax.invert_yaxis()
    if legend_done: ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    save(fig,'event_ratios_v2')
    agents=result.get('agent_summary', [])
    fig,ax=plt.subplots(figsize=(9,max(3,len(agents)*.45+1.5)))
    for i,row in enumerate(agents):
        value=row.get('ego_responsibility_mean')
        if _number(value): ax.barh(i,value,color='#3978ab')
        else: ax.text(.02,i,'unavailable: no collisions' if row.get('collision_event_count')==0 else 'unavailable: no finite ego ratios',va='center')
    ax.set(yticks=range(len(agents)),yticklabels=[r.get('agent_name','?') for r in agents],xlim=(0,1),xlabel='Mean over finite ego collision-event ratios',title='Agent conditional ego attribution')
    ax.invert_yaxis();save(fig,'agent_means_v2')
    counts=Counter(e.get('ego_responsibility_class') if e.get('ego_involved') else 'non-ego pair' for e in events)
    classes=['ego_primary','other_primary','shared','uncertain','unscorable','non-ego pair']
    fig,ax=plt.subplots(figsize=(9,4))
    bars=ax.bar(range(len(classes)),[counts[c] for c in classes],color='#3978ab')
    ax.bar_label(bars,padding=3)
    ax.set(xticks=range(len(classes)),xticklabels=[c.replace('_','\n') for c in classes],ylabel='Collision event count',title='Ego-oriented classes (non-ego pair shown separately)')
    ax.set_ylim(0,max(1,max(counts.values(),default=0))*1.2);save(fig,'class_distribution_v2')
    eligible=[r for r in result.get('cars_comparison',[]) if r.get('cars_status') in ('valid','cars_valid') and _number(r.get('ratio_before')) and _number(r.get('ratio_after'))]
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot([0,1],[0,1],'--',color='gray',linewidth=1)
    grouped = {}
    for i,row in enumerate(eligible,1):
        event_index=next((j+1 for j,e in enumerate(events) if e.get('event_id')==row.get('event_id')),i)
        grouped.setdefault((row['ratio_before'],row['ratio_after']),[]).append(f'E{event_index:02d}')
    for i,((before,after),ids) in enumerate(grouped.items()):
        ax.scatter(before,after,color='#3978ab')
        ax.annotate(', '.join(ids),(before,after),xytext=(5,5+i*10),textcoords='offset points',fontsize=8)
    if not eligible: ax.text(.5,.5,'No eligible finite CARS comparison',ha='center')
    ax.set(xlim=(-.04,1.04),ylim=(-.04,1.04),xlabel='Basic actor A evidence ratio',ylabel='Basic + CARS actor A evidence ratio',title=f'CARS increment ({len(eligible)} eligible events)')
    save(fig,'basic_vs_cars_v2')


def export_v2(result, directory):
    """Write flat files so the existing export ZIP includes all V2 diagnostics."""
    from driveaudit_eval.utils.io import write_csv
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    for key,suffix in [('events','events_v2'),('agent_summary','agent_summary_v2'),('scene_summary','scene_summary_v2'),
        ('ablation','ablation'),('weight_sensitivity','weight_sensitivity'),('window_sensitivity','window_sensitivity'),
        ('size_sensitivity','size_sensitivity_v2'),('cars_comparison','cars_comparison_v2'),('collision_type_summary','collision_type_summary_v2'),
        ('experiment_summary','experiment_summary_v2'),('stability','stability_v2')]:
        rows=result.get(key,[])
        if key=='events':
            rows=[{k:v for k,v in row.items() if k not in ('evidence','actor_a_states','actor_b_states')} for row in rows]
        write_csv(directory/f'collision_responsibility_{suffix}.csv',rows)
    with (directory/'collision_responsibility_evidence_v2.jsonl').open('w',encoding='utf-8') as handle:
        for row in result.get('events',[]):
            handle.write(json.dumps(jsonable(row),ensure_ascii=False,allow_nan=False)+'\n')
    write_json(directory/'collision_responsibility_overall_v2.json',result.get('overall',{}))
    (directory/'collision_responsibility_report.md').write_text(_report_ratios(result),encoding='utf-8')
    _figures(result,directory)
    return directory



"""V3 mandatory-attribution exports; no recomputation of evidence or ratios."""
import json
from pathlib import Path

from driveaudit_eval.utils.io import jsonable, write_json


def _rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [dict(metric=key, value=item) for key, item in value.items()]
    return []


def _observed_table(value):
    rows = _rows(value)
    if not rows:
        return ['No observations available.']
    keys = list(dict.fromkeys(key for row in rows for key in row))
    return _table(rows, [(key, key) for key in keys])


def _report_attribution(result):
    lines = ['# Responsibility V3 — Mandatory Attribution', '',
        '## Method', '',
        'Model-based responsibility attribution: responsibility ratios are normalized evidence scores, '
        'not calibrated probabilities or legal responsibility proportions.', '',
        'Responsibility ratio 表示基于运动学、交通规则、碰撞因果和可避免性证据得到的模型化责任证据比例，并非法律责任比例。', '',
        'V3 preserves the V2 rule, causal, avoidability and maneuver scores, weights, available-evidence '
        'normalization and CARS fusion. Only the mapping from evidence ratios to attribution changes. '
        'The two actor ratios and confidence values remain available for direct V2/V3 comparison.', '',
        'For finite ratios, the larger ratio determines actor_a_dominant or actor_b_dominant. '
        'Within the configured tie epsilon, the deterministic comparison order is causal, avoidability, '
        'rule, maneuver, then CARS votes. Exact equal ratios with no discriminating tie-break evidence '
        'produce shared_equal, a valid equal attribution. Ratios are never adjusted to obtain a class. '
        'Only an unavailable ratio prevents attribution.', '',
        'Confidence describes evidence reliability and never gates classification. Low confidence, low coverage '
        'and unavailable CARS do not remove an available direction. Dominance margin = abs(R_A - R_B); '
        'dominance strength measures separation, whereas confidence level describes reliability. '
        'A strong direction with low confidence is therefore a valid output.', '',
        'Resolved V3 configuration:', '', '```json',
        json.dumps(jsonable(result.get('config', {})), ensure_ascii=False, indent=2, allow_nan=False), '```', '',
        '## Overall coverage and directions', '', *_observed_table(result.get('overall')), '',
        '## Agent comparison', '',
        'Raw collision rate counts collision windows over all evaluated windows. Responsibility-weighted '
        'collision count sums ego event ratios; attributed collision burden divides this sum by all agent '
        'windows. These are additional diagnostics and do not replace NC, PDMS or other prior metrics. '
        'A no-collision agent has no conditional ego responsibility mean; unavailable is not zero.', '',
        *_table(result.get('agent_summary', []), [
            ('agent_name', 'Agent'), ('total_windows', 'Windows'), ('collision_event_count', 'Raw events'),
            ('raw_collision_rate', 'Raw collision rate'), ('ego_dominant_collision_count', 'Ego dominant'),
            ('other_dominant_collision_count', 'Other dominant'), ('ego_responsibility_mean', 'Ego ratio mean'),
            ('responsibility_weighted_collision_count', 'Weighted collision count'),
            ('ego_attributed_collision_burden', 'Attributed burden'), ('mean_dominance_margin', 'Mean margin'),
            ('mean_confidence', 'Mean confidence')]), '',
        '## Scene comparison', '', *_table(result.get('scene_summary', []), [('scene_id','Scene'), ('agent_name','Agent'),
            ('collision_event_count','Events'), ('finite_ratio_count','Finite'),
            ('ego_dominant_collision_count','Ego dominant'), ('other_dominant_collision_count','Other dominant'),
            ('class_distribution','Pairwise classes'), ('ego_responsibility_mean','Ego mean'), ('mean_confidence','Confidence')]), '',
        '## V2 versus V3', '',
        'The comparison retains the old V2 class as historical context. It is not a V3 output class. '
        'Ratio agreement is checked at tolerance 1e-12 in the run regression report.', '',
        *_observed_table(result.get('comparison')), '',
        '## Ego ratio distribution', '', *_observed_table(result.get('ratio_distribution')), '',
        '## Attribution and dominance strength', '', *_table(result.get('strength_summary', []), [('attribution_strength','Strength'),
            ('collision_event_count','Events'), ('ego_dominant_collision_count','Ego dominant'),
            ('other_dominant_collision_count','Other dominant'), ('class_distribution','Pairwise classes'),
            ('mean_dominance_margin','Mean margin'), ('mean_confidence','Confidence')]), '',
        '## Confidence versus direction', '',
        'Low, medium and high are reliability annotations only; each finite event retains its attribution.', '',
        *_table(result.get('confidence_summary', []), [('confidence_level','Reliability'),
            ('collision_event_count','Events'), ('finite_ratio_count','Finite'), ('dominant_attribution_count','Dominant'),
            ('ego_dominant_collision_count','Ego dominant'), ('other_dominant_collision_count','Other dominant'),
            ('class_distribution','Pairwise classes')]), '',
        '## CARS increment', '',
        'Basic and Basic + CARS are compared using V3 dominant directions. A direction flip means '
        'actor A versus actor B changes; a confidence-level change alone is not a direction flip. '
        'Unavailable CARS does not constitute an intervention.', '',
        *_observed_table(result.get('cars_comparison')), '',
        '## Direction stability', '',
        'Weight and time-window sensitivity compare dominant actors, not the former confidence-gated classes. '
        'Exact shared ties and unavailable ratios are explicit; consult eligible comparison counts before '
        'interpreting a flip rate. Stability is descriptive over the tested configurations.', '',
        *_observed_table(result.get('direction_stability')), '',
        'Per-setting event rows are preserved in responsibility_weight_sensitivity_v3.csv, '
        'responsibility_window_sensitivity_v3.csv, responsibility_ablation_v3.csv and '
        'responsibility_size_sensitivity_v3.csv.', '',
        '## Every event', '',
        'Ego fields are unavailable for background pairs. Actor A/B ratios remain the authoritative pairwise '
        'result. Full evidence and all exported values are retained in collision_responsibility_evidence_v3.jsonl.', '']
    fields = ['scene_id', 'agent_name', 'decision_frame_id', 'actor_a_id', 'actor_b_id',
        'actor_a_type', 'actor_b_type', 'collision_type', 'responsibility_status',
        'actor_a_responsibility_ratio', 'actor_b_responsibility_ratio',
        'ego_responsibility_ratio', 'other_responsibility_ratio',
        'dominant_responsible_actor', 'responsibility_class', 'ego_responsibility_class',
        'responsibility_dominance_margin', 'responsibility_dominance_strength',
        'attribution_strength', 'responsibility_confidence',
        'responsibility_confidence_level', 'evidence_coverage', 'cars_status',
        'rule_coverage', 'causal_coverage', 'avoidability_coverage', 'maneuver_coverage',
        'overall_evidence_coverage', 'responsibility_direction_stability', 'tie_break_decision', 'reason']
    for i, event in enumerate(result.get('events', []), 1):
        lines += [f"### {i}. {event.get('event_id', 'unavailable')}", '',
            *_table([dict(field=key, value=event.get(key)) for key in fields if key in event],
                    [('field', 'Field'), ('value', 'Result')]), '']
        direction = event.get('ego_responsibility_class') if event.get('ego_involved') else event.get('responsibility_class')
        if event.get('responsibility_status') == 'unscorable':
            explanation = 'The responsibility ratio is unavailable; no direction or equal split is fabricated.'
        else:
            explanation = (f"Current model-based evidence direction: {_display(direction)}. "
                           f"Evidence reliability: {_display(event.get('responsibility_confidence_level'))}. "
                           'Reliability does not change the formal attribution.')
        lines += [explanation, '']
    lines += ['## Limitations and regression boundary', '',
        'This run validates the evaluation methodology and produces descriptive results on two scenes; '
        'it is not a statistically representative benchmark. Repeated overlapping collision windows are not '
        'independent crashes. Mandatory attribution does not establish real-world fault or stronger evidence '
        'than V2. Missing rules, short histories, actor-type ambiguity and geometric approximations remain '
        'limitations even when the responsibility direction is strong.', '',
        'Synthetic tests, frozen-input verification, V2/V3 ratio equality and regression of previous modules '
        'are documented in RESPONSIBILITY_V3_REPORT.md. V2 exports remain separate and unchanged.', '']
    return '\n'.join(lines)


def export_v3(result, directory):
    """Export existing V3 values without modifying the input or earlier files."""
    from driveaudit_eval.utils.io import write_csv
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = {
        'events': 'collision_responsibility_events_v3.csv',
        'agent_summary': 'collision_responsibility_agent_summary_v3.csv',
        'scene_summary': 'collision_responsibility_scene_summary_v3.csv',
        'comparison': 'responsibility_v2_v3_comparison.csv',
        'direction_stability': 'responsibility_direction_stability.csv',
        'weight_sensitivity': 'responsibility_weight_sensitivity_v3.csv',
        'window_sensitivity': 'responsibility_window_sensitivity_v3.csv',
        'ablation': 'responsibility_ablation_v3.csv',
        'size_sensitivity': 'responsibility_size_sensitivity_v3.csv',
        'cars_comparison': 'responsibility_cars_comparison_v3.csv',
        'ratio_distribution': 'responsibility_ratio_distribution_v3.csv',
        'confidence_summary': 'responsibility_confidence_summary_v3.csv',
        'strength_summary': 'responsibility_strength_summary_v3.csv',
    }
    for key, filename in names.items():
        rows = _rows(result.get(key))
        if key == 'events':
            rows = [{k: v for k, v in row.items() if k not in ('evidence', 'actor_a_states', 'actor_b_states')}
                    for row in rows]
        write_csv(directory / filename, rows)
    write_json(directory / 'collision_responsibility_overall_v3.json', result.get('overall', {}))
    with (directory / 'collision_responsibility_evidence_v3.jsonl').open('w', encoding='utf-8') as handle:
        for event in result.get('events', []):
            handle.write(json.dumps(jsonable(event), ensure_ascii=False, allow_nan=False) + '\n')
    (directory / 'collision_responsibility_report_v3.md').write_text(_report_attribution(result), encoding='utf-8')
    return directory


