"""Install a process-wide Python file-access guard before importing model code."""
import os
import sys
from pathlib import Path


def forbidden_sensor_path(path):
    if not isinstance(path,(str,bytes,os.PathLike)):
        return False
    original=Path(os.fsdecode(path))
    for p in (original,original.resolve()):
        name=str(p).lower(); suffix=p.suffix.lower()
        # Source modules and released fusion weights are not point measurements.
        if suffix in ('.py','.pyc','.so','.pth','.pt','.safetensors'):
            continue
        if (suffix=='.pcd' or any(s in name for s in ('lidar','point_cloud','pointcloud'))
                or p.name in ('agents.csv','future_trajectories.csv','semantic_map.npz')):
            return True
    return False


def install_guard(audit_path):
    import json
    sink=os.open(str(audit_path),os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    def record(row):
        os.write(sink,(json.dumps(row)+'\n').encode())
    def audit(event,args):
        if event=='open' and args and forbidden_sensor_path(args[0]):
            record(dict(event='blocked',path=os.fsdecode(args[0])))
            raise RuntimeError('Real LiDAR is forbidden in camera-only leaderboard mode: '+os.fsdecode(args[0]))
    sys.addaudithook(audit)
    record(dict(event='guard_installed',protocol='camera_only',resolved_paths=True,pid=os.getpid()))


def validate_camera_only_run(directory, agents=None):
    """Accept only measured, unique decision manifests backed by guarded services."""
    import json
    from driveaudit_eval.agents.registry import FORMAL_AGENTS
    from driveaudit_eval.agents.adapters.camera_only_temporal import stable_request_seed
    directory=Path(directory)
    source=json.loads((directory/'input_manifest.json').read_text())
    if source.get('sensor_protocol')!='camera_only':
        raise ValueError('Camera-only input manifest is required')
    expected={(r['scene_id'],r['frame_id']) for r in source['common_windows']}
    def require(condition):
        if not condition:
            raise ValueError('Camera-only sensor evidence is missing, inconsistent or contaminated')
    result=[]
    selected = list(FORMAL_AGENTS if agents is None else agents)
    require(bool(selected) and len(selected)==len(set(selected)) and set(selected)<=set(FORMAL_AGENTS))
    for agent in selected:
        path=directory/agent
        manifest=json.loads((path/'agent_input_manifest.json').read_text())
        audit=[json.loads(s) for s in (path/'sensor_access_audit.jsonl').read_text().splitlines()]
        decisions=manifest['decisions'];keys=[(r['scene'],r['decision_frame']) for r in decisions]
        require(manifest['protocol']=='camera_only' and manifest['sensor_leakage_detected'] is False)
        require(len(keys)==len(set(keys)) and {k for k,r in zip(keys,decisions) if r['scored_decision']}==expected)
        require(any(r['event']=='guard_installed' and r.get('resolved_paths') is True for r in audit) and not any(r['event']=='blocked' for r in audit))
        require(all(r['rgb_used'] and not any(r[k] for k in ('lidar_used','real_lidar_accessed','gt_future_used','gt_boxes_used')) for r in decisions))
        raw=[json.loads(s) for s in (path/'raw_outputs.jsonl').read_text().splitlines()]
        require({(r['scene_id'],r['frame_id']) for r in raw}==set(keys) and len(raw)==len(keys))
        for row in raw:
            value=row['output']
            if agent in ('goalflow','diffusiondrivev2'):
                require(value['sensor_mode']=='zero_lidar' and value['uses_lidar'] is False)
                require(value['lidar_feature_nonzero']==0 and value['lidar_feature_shape']==[1,1,256,256])
            if agent in ('uniad','vad'):
                require(value['request_seed']==stable_request_seed(row['scene_id'],row['frame_id']))
        result.append(dict(agent=agent,scored_decisions=len(expected),actual_forward_count=len(decisions),real_lidar_used=False,sensor_leakage_detected=False))
    return result

if __name__=='__main__':
    import argparse
    import runpy
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit',required=True)
    parser.add_argument('--service',nargs=argparse.REMAINDER,required=True)
    args=parser.parse_args()
    install_guard(args.audit)
    os.environ['DRIVEAUDIT_SENSOR_PROTOCOL']='camera_only'
    sys.argv=args.service
    runpy.run_path(args.service[0],run_name='__main__')
