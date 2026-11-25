"""Audit persisted state feedback and actual rendered inputs after a completed run."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from driveaudit.closed_loop.core import atomic_json,execute_prefix
from driveaudit.closed_loop.goal_navigation import apply_goal_terminal

def verify(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());cfg=json.loads((root/'config.json').read_text())
    if summary['status']!='completed':raise ValueError('run not completed')
    records=[json.loads(p.read_text()) for p in sorted((root/'steps').glob('*.json'))]
    if len(records)!=summary['steps']:raise ValueError('step count mismatch')
    previous=None;distance=0.;hashes=[];actor_counts=[];times={k:[] for k in ['render','policy','traffic']};max_error=0.
    for i,r in enumerate(records):
        assert r['step']==i and r['time_s']==.5*i
        before=r['before'];after=r['after'];pose=np.array(before['pose']);committed=np.array(after['committed_ego_path'])
        if previous is not None:np.testing.assert_allclose(pose,previous,atol=1e-6,rtol=0)
        np.testing.assert_allclose(r['render']['render_pose'],pose,atol=1e-6,rtol=0)
        np.testing.assert_allclose(after['previous_pose'],pose,atol=1e-6,rtol=0)
        native=r['policy']['native_trajectory'][0];x,y,yaw=native;c,s=math.cos(pose[3]),math.sin(pose[3])
        target=np.array([pose[0]+c*x-s*y,pose[1]+s*x+c*y,pose[2],pose[3]+math.atan2(math.sin(yaw),math.cos(yaw))])
        if cfg.get('goal_navigation',{}).get('enabled'):
            prefix=execute_prefix(pose,r['policy']['native_trajectory'],r['policy']['native_time_offsets'])
            prefix,terminal=apply_goal_terminal(prefix,pose,cfg['goal_navigation'],i>0 and records[i-1]['goal_terminal']['reached'])
            np.testing.assert_allclose(committed,prefix,atol=2e-5,rtol=0)
            assert terminal==r['goal_terminal']
            target=prefix[-1]
        error=float(np.max(np.abs(np.array(after['pose'])-target)));max_error=max(max_error,error)
        if error>2e-5:raise ValueError('executed pose differs from policy prefix')
        np.testing.assert_allclose(committed[-1],after['pose'],atol=2e-5,rtol=0)
        assert after['tick']==i+1
        directory=root/'ipc/policy';request=json.loads((directory/f'{i:04d}.request.json').read_text())
        assert request['model_input']['camera_images']==r['render']['camera_images']
        if cfg.get('policy_image_protocol')=='generated_only_per_step_sha256':
            from driveaudit.closed_loop.image_contract import attest
            local_paths={cam:str(root/'images'/f'{i:04d}'/(cam+'.png')) for cam in r['render']['camera_images']}
            proof=attest(local_paths,root,i)
            assert r['policy']['image_source_audit']==proof==request['generated_image_proof']
        for path in r['render']['camera_images'].values():
            # Resolve archived files by their run-relative image path.
            name=Path(path);local=root/'images'/name.parent.name/name.name
            assert local.is_file();hashes.append(hashlib.sha256(local.read_bytes()).hexdigest())
        distance+=float(np.linalg.norm(np.diff(np.vstack((pose[:2],committed[:,:2])),axis=0),axis=1).sum())
        actor_counts.append(len(before['actors']));previous=after['pose']
        for role in times:times[role].append(r['after' if role=='traffic' else role]['wall_s'])
    return dict(status='PASS',steps=len(records),duration_s=len(records)*.5,images=len(hashes),unique_images=len(set(hashes)),travel_distance_m=distance,max_feedback_error=max_error,actor_count_range=[min(actor_counts),max(actor_counts)],mean_inference_s={k:float(np.mean(v)) for k,v in times.items()},catk_source_integrity=records[-1]['after']['model_load']['source_integrity'],scope='trace_consistency_and_single_scene_execution; not safety_or_generalization_validation')


def main():
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();result=verify(a.run);atomic_json(Path(a.run)/'verification.json',result);print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
