"""Render measured camera-only progress, accepted scores and paired ablations."""
import csv,json
from pathlib import Path
OUT=Path.cwd()
DATASETS=('nuscenes','nuplan','av2','pandaset')
from driveaudit_eval.agents.registry import FORMAL_AGENTS as AGENTS
from driveaudit_eval.metrics.registry import CORE_IDS as CORE
def read(name,default=None):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else default
def csvrows(name):
    p=OUT/name
    return list(csv.DictReader(p.open())) if p.exists() else []
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def exact_pairs(rows):
    keys=[(r.get('dataset'),r.get('agent')) for r in rows]
    return len(keys)==24 and set(keys)=={(d,a) for d in DATASETS for a in AGENTS}


def verified_purity(purity):
    verified={}
    for dataset in DATASETS:
        rows=purity.get(dataset,[])
        if len(rows)!=6 or {r.get('agent') for r in rows}!=set(AGENTS):continue
        valid=[]
        for row in rows:
            agent=row['agent'];expected=190 if agent in ('uniad','vad') else 100
            manifest=read(f'formal/{dataset}/real_agents/{agent}/agent_input_manifest.json',{})
            decisions=manifest.get('decisions',[])
            keys=[(r.get('scene'),r.get('decision_frame')) for r in decisions]
            if (row.get('scored_decisions')==100 and row.get('actual_forward_count')==expected
                and row.get('real_lidar_used') is False and row.get('sensor_leakage_detected') is False
                and manifest.get('protocol')=='camera_only' and manifest.get('sensor_leakage_detected') is False
                and len(keys)==len(set(keys))==expected
                and sum(r.get('scored_decision') is True for r in decisions)==100
                and all(r.get('rgb_used') is True and all(r.get(k) is False for k in
                    ('lidar_used','real_lidar_accessed','gt_future_used','gt_boxes_used')) for r in decisions)):
                valid.append(row)
        if len(valid)==6:verified[dataset]=valid
    return verified


def inference_complete(rows):
    return exact_pairs(rows) and all(r.get('expected_scenes')==r.get('completed_scenes')==10
        and r.get('expected_decisions')==r.get('completed_decisions')==100
        and r.get('input_validation')=='VALID' for r in rows)


def rgb_ready(metadata,decode):
    counts={'av2':75,'nuplan':100,'nuscenes':250,'pandaset':75}
    return (metadata.get('status')=='METADATA_READY' and metadata.get('scenes')==metadata.get('valid_scenes')==500
        and metadata.get('invalid_scenes')==0 and metadata.get('rgb_files')==270000
        and metadata.get('dataset_counts')==counts and decode.get('status')=='PASS'
        and decode.get('checked_scenes')==decode.get('passed_scenes')==500
        and decode.get('checked_files')==decode.get('passed_files')==270000 and decode.get('bad_file_count')==0
        and not decode.get('bad_files') and all(decode.get('datasets',{}).get(d,{}).get('passed_scenes')==n
            and decode['datasets'][d].get('passed_files')==n*540 for d,n in counts.items()))


def main():
    gate=read('core_universality_gate.json',{});inference=read('stability_inference.json',[]);purity=read('sensor_purity.json',{})
    ablation=read('paired_ablation.json',{}).get('rows',[]);metadata=read('rgb_500_readiness.json',{});decode=read('rgb_500_decode_audit.json',{})
    values=csvrows('UNIVERSAL_CORE_METRICS.csv');cells=csvrows('core_metric_evidence.csv')
    purity=verified_purity(purity)
    cell_keys=[(r.get('dataset'),r.get('agent'),r.get('metric')) for r in cells]
    passed=(gate.get('status')=='PASS' and gate.get('valid_cells')==216 and exact_pairs(values)
        and len(cell_keys)==216 and set(cell_keys)=={(d,a,m) for d in DATASETS for a in AGENTS for m in CORE}
        and all(r.get('gate_status')=='CORE_VALID' for r in cells) and inference_complete(inference))
    pairs_ready=exact_pairs(ablation) and all(r['status'] in
        (('PASS','DEGRADED') if r['agent'] in ('goalflow','diffusiondrivev2') else ('PASS',)) for r in ablation)
    ready=passed and pairs_ready and set(purity)==set(DATASETS) and rgb_ready(metadata,decode)
    matrix=[]
    for agent in AGENTS:
        states=[r for r in ablation if r['agent']==agent]
        status=('NOT_READY' if len(states)!=4 or {r.get('dataset') for r in states}!=set(DATASETS) else 'FAILED' if any(r['status']=='FAILED' for r in states) else
                'REGRESSION_MISMATCH' if any(r['status']=='REGRESSION_MISMATCH' for r in states) else
                'NOT_READY' if any(r['status']=='NOT_READY' for r in states) else 'DEGRADED' if any(r['status']=='DEGRADED' for r in states) else 'PASS')
        valid_datasets=[d for d in DATASETS if any(r['agent']==agent for r in purity.get(d,[]))]
        if len(valid_datasets)!=4 and status in ('PASS','DEGRADED'):status='NOT_READY'
        counts=[];ego=[];command=[]
        for d in valid_datasets:
            manifest=read(f'formal/{d}/real_agents/{agent}/agent_input_manifest.json',{})
            for r in manifest.get('decisions',[]):counts.append(r['camera_count']);ego.append(r['ego_state_used']);command.append(r['command_used'])
        matrix.append(dict(agent=agent,RGB='YES' if len(valid_datasets)==4 else 'NOT_VERIFIED',camera_count=','.join(map(str,sorted(set(counts)))) or 'NOT_VERIFIED',
            EgoState='YES' if len(valid_datasets)==4 and ego and all(ego) else 'NOT_VERIFIED',Command='YES' if len(valid_datasets)==4 and command and all(command) else 'NOT_VERIFIED',
            RealLiDAR='NO' if len(valid_datasets)==4 else 'NOT_VERIFIED',verified_datasets=','.join(valid_datasets),
            lidar_mode='zero_lidar' if agent in ('diffusiondrivev2','goalflow') else 'native_camera_only',status=status))
    with (OUT/'CAMERA_ONLY_AGENT_MATRIX.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(matrix[0]));writer.writeheader();writer.writerows(matrix)
    scored=sum(r['completed_decisions'] for r in inference)
    sections = ['# Camera-only validation',
        table(list(matrix[0]), [list(row.values()) for row in matrix]),
        f'Core gate verified: {passed}; paired comparisons complete: {pairs_ready}; RGB input audit ready: {rgb_ready(metadata, decode)}.',
        'Readiness requires complete inference, metric, sensor and paired evidence. Missing evidence never passes.']
    (OUT/'CAMERA_ONLY_PROTOCOL_REPORT.md').write_text('\n\n'.join(sections)+'\n')
    terminal=['========== Camera-Only Driving Benchmark ==========','Formal protocol: Environment perception: RGB only','Agents:']
    terminal += [f"{r['agent']}: real_lidar_used={r['RealLiDAR']}, mode={r['lidar_mode']}, status={r['status']}" for r in matrix]
    terminal += [f'40-scene stability decisions: {scored}/2400',f'Core metrics valid: {gate.get("valid_cells",0)}/216',f'Paired reference complete: {len(ablation)}/24','Real LiDAR cache required: NO',f'Ready for 500-scene RGB benchmark: {"YES" if ready else "NO (formal gate or paired validation incomplete)"}','500-scene model run launched: NO','=================================================']
    (OUT/'terminal_summary.txt').write_text('\n'.join(terminal)+'\n');print('\n'.join(terminal))


