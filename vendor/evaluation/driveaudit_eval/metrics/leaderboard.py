"""Registry-directed ranks with equal group weight and explicit missing values."""
import csv
import json
import math
from statistics import mean

from driveaudit_eval.metrics.registry import METRICS, CORE_IDS, OFFICIAL_IDS, GROUP_COLUMNS

COLUMNS = ['model']
for key in OFFICIAL_IDS:
    COLUMNS.append(key)
    if key in ('tvc', 'trs'):
        COLUMNS.append(METRICS[key]['coverage'])
COLUMNS += ['qtes_definition_version', 'validity', 'invalid_reason', 'overall_rank']


from driveaudit_eval.utils.aggregation import average_ranks


def rank_rows(rows, registry=None):
    registry = registry or METRICS
    result = [dict(row) for row in rows]
    if len({r['model'] for r in result}) != len(result):
        raise ValueError('Leaderboard model identities must be unique')
    ranks = {}
    for key in OFFICIAL_IDS:
        values = []
        for row in result:
            value = row.get(key)
            lo, hi = registry[key]['range']
            if value is not None and (not math.isfinite(value) or
                                      (lo is not None and value < lo) or (hi is not None and value > hi)):
                raise ValueError(f'{key} outside registry range: {value}')
            values.append(value)
        ranks[key] = average_ranks(values, registry[key]['higher_is_better'])
    for i, row in enumerate(result):
        for group, column in GROUP_COLUMNS.items():
            values = [ranks[k][i] for k in CORE_IDS if registry[k]['group'] == group]
            row[column] = mean(values) if all(v is not None for v in values) else None
        values = [row[c] for c in GROUP_COLUMNS.values()]
        row['avg_rank'] = mean(values) if all(v is not None for v in values) else None
    for row, rank in zip(result, average_ranks([r.get('qtes') if r.get('validity') == 1 and r.get('qtes_definition_version') == 'gt_relative_v1' else None for r in result], True)):
        row['overall_rank'] = rank
    return result


def export_leaderboard(rows, output_dir, registry=None, markdown_name='LEADERBOARD.md'):
    registry = registry or METRICS
    rows = sorted(rank_rows(rows, registry), key=lambda r: (r['overall_rank'] is None, r['overall_rank'] or 0, r['model']))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/'leaderboard.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, COLUMNS, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    keys = [*OFFICIAL_IDS]
    headers = ['Rank', 'Model'] + [registry[k]['short_name'] + ('↑' if registry[k]['higher_is_better'] else '↓') for k in keys]
    lines = ['# Leaderboard', '', '| ' + ' | '.join(headers) + ' |', '|'+ '|'.join(['---']*len(headers))+'|']
    def cell(row, key):
        value = row.get(key)
        if value is None:
            return 'N/A'
        unique = sorted({r[key] for r in rows if r.get(key) is not None},
                        reverse=key != 'avg_rank' and registry[key]['higher_is_better'])
        formatted = f'{value:.6f}'
        return '**'+formatted+'**' if value == unique[0] else '<u>'+formatted+'</u>' if len(unique)>1 and value == unique[1] else formatted
    for row in rows:
        lines.append('| '+str(row['overall_rank'] if row['overall_rank'] is not None else 'N/A')+' | '+str(row['model']).replace('|','\\|')+' | '+' | '.join(cell(row,k) for k in keys)+' |')
    lines.extend(['', 'QTES definition: gt_relative_v1. Official ranks use GT-relative QTES descending; ties receive average ranks. Invalid, missing-QTES or old absolute-QTES runs have no official rank. A gated zero retains its invalid reasons; missing component values remain unavailable.',
                  'Model denotes the evaluated scene/video source. Driving agents are evaluators; their individual scores are exported separately. A single source has rank 1 by construction, not evidence of superiority.',
                  'Coverage and scientific status are recorded in metrics.json.', '',
                  '## Supplementary Avg.Rank', '',
                  'Equal metric weights within each of the original five groups, then equal group weights; TE/QTES do not enter Avg.Rank.', '',
                  '| Model | Avg.Rank↓ |', '| --- | --- |'])
    for row in rows:
        lines.append('| '+str(row['model']).replace('|','\\|')+' | '+cell(row,'avg_rank')+' |')
    from driveaudit_eval.utils.io import write_csv
    write_csv(output_dir/'supplementary_ranks.csv', [
        {key:row.get(key) for key in ('model', *GROUP_COLUMNS.values(), 'avg_rank')} for row in rows])
    (output_dir/markdown_name).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return rows


def build_cross_dataset_leaderboards(results, output_dir, expected_agents=None, score='PDMS_open_loop'):
    """Separate complete common datasets from each dataset's available agents.

    Missing scores or failed inference exclude a dataset from the common set.
    Every agent must cover the same evaluated scene set; no missing-as-zero.
    """
    from pathlib import Path
    from driveaudit_eval.agents.registry import FORMAL_AGENTS
    from driveaudit_eval.utils.aggregation import metric_number
    agents = tuple(FORMAL_AGENTS if expected_agents is None else expected_agents)
    source = results.get('agents', results)
    indexed = {}; statuses = {}; datasets = set()
    def identity(value):
        return str(value).replace('__', '/')
    def dataset(sid):
        return sid.split('/')[0].split('_challenger')[0].replace('nuscenes300', 'nuscenes')
    for agent in agents:
        value = source.get(agent, {})
        for status in value.get('scene_status', []):
            sid = identity(status.get('scene_id') or status.get('scene', ''))
            ds = status.get('dataset') or dataset(sid)
            datasets.add(ds)
            statuses[(agent, ds, sid.split('/')[-1])] = status
        for row in value.get('per_scene', []):
            sid = identity(row['scene_id']); ds = row.get('dataset') or dataset(sid)
            key = (agent, ds, sid.split('/')[-1])
            if key in indexed:
                raise ValueError('Duplicate agent/dataset/scene in leaderboard')
            indexed[key] = row; datasets.add(ds)
    tables = {}; shared = []
    for ds in sorted(datasets):
        scenes = {key[2] for key in set(indexed) | set(statuses) if key[1] == ds}
        rows = []
        for agent in agents:
            values = []; reasons = []
            for sid in sorted(scenes):
                key = (agent, ds, sid); item = indexed.get(key, {}); status = statuses.get(key, {})
                value = metric_number(item.get(score))
                if status and str(status.get('inference_status', '')).lower() not in ('success', 'completed', 'available', 'valid'):
                    value = None
                if value is None:
                    reasons.append(status.get('failure_reason') or item.get(score + '_reason') or 'required_inference_or_metric_not_available')
                values.append(value)
            complete = bool(values) and all(value is not None for value in values)
            rows.append(dict(agent=agent, dataset=ds, **{score: mean(values) if complete else None},
                             status='available' if complete else 'unavailable', scenes=len(scenes),
                             available_scenes=sum(value is not None for value in values),
                             reason='; '.join(sorted(set(str(reason) for reason in reasons))) or None))
        tables[ds] = rows
        if rows and all(row['status'] == 'available' for row in rows):
            shared.append(ds)
    full = [dict(agent=agent, **{score: mean(tables[ds][i][score] for ds in shared) if shared else None},
                 status='available' if shared else 'unavailable', datasets=len(shared),
                 reason=None if shared else 'no_common_fully_supported_dataset') for i, agent in enumerate(agents)]
    result = dict(score=score, expected_agents=list(agents),
                  full_agent=dict(datasets=shared, rows=full), dataset_specific=tables,
                  aggregation='scene equal within dataset; dataset equal within common subset; no missing-score reweighting')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / 'cross_dataset_leaderboards.json').write_text(json.dumps(result, indent=2) + '\n')
    for filename, rows in [('FULL_AGENT_LEADERBOARD.csv', full),
                           ('DATASET_SPECIFIC_LEADERBOARD.csv', [row for rows in tables.values() for row in rows])]:
        with (output / filename).open('w', newline='') as stream:
            columns = list(rows[0]) if rows else ['agent', score, 'status', 'reason']
            writer = csv.DictWriter(stream, columns); writer.writeheader()
            writer.writerows({key: 'N/A' if value is None else value for key, value in row.items()} for row in rows)
    lines = ['# Cross-dataset leaderboards', '', 'Full-Agent common datasets: ' + (', '.join(shared) or 'N/A'),
             '', '| Scope | Agent | ' + score + ' | Reason |', '| --- | --- | --- | --- |']
    for label, rows in [('Full-Agent', full), *tables.items()]:
        for row in rows:
            value = 'N/A' if row[score] is None else f'{row[score]:.6f}'
            lines.append(f"| {label} | {row['agent']} | {value} | {row['reason'] or ''} |")
    lines += ['', result['aggregation'] + '.', 'Dataset-specific N/A entries remain visible and are not assigned zero.']
    (output / 'CROSS_DATASET_LEADERBOARDS.md').write_text('\n'.join(lines) + '\n')
    return result
