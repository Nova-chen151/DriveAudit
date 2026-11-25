"""One-scene, synchronous state-feedback simulation with three real models."""
import argparse,json,os,subprocess,sys,time,shutil,hashlib
from pathlib import Path
import numpy as np
from driveaudit.generation import MODEL_DIRS
from driveaudit.resources import resource_root
from driveaudit.closed_loop.core import atomic_json,execute_prefix,pose_matrix
from driveaudit.closed_loop.goal_navigation import command_for_goal,apply_goal_terminal

class Service:
    def __init__(self,role,cfg,config,out,gpu):
        self.ipc=out/'ipc'/role;self.ipc.mkdir(parents=True);self.n=0
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        env.update(OPENBLAS_CORETYPE='Haswell',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        self.log=(out/(role+'.log')).open('w')
        env.update(cfg.get('role_env',{}).get(role,{}))
        if role=='renderer' and 'CUDA_HOME' in env:
            cuda=Path(env['CUDA_HOME'])
            env['LD_LIBRARY_PATH']=str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH','')
            env['PATH']=str(cuda/'bin')+os.pathsep+env['PATH']
        command=[cfg['python'][role]]
        if role=='renderer' and cfg.get('renderer_processes',1)>1:
            command+=['-m','torch.distributed.run','--standalone','--nproc_per_node',str(cfg['renderer_processes'])]
        command += [str(out/'code_snapshot'/'worker.py'),role,str(config),str(self.ipc)]
        self.proc=subprocess.Popen(command,env=env,cwd=cfg.get('role_cwd',{}).get(role),stdout=self.log,stderr=subprocess.STDOUT)
    def wait(self,path,timeout=1800):
        deadline=time.monotonic()+timeout
        while not path.exists():
            if (self.ipc/'error.json').exists():raise RuntimeError((self.ipc/'error.json').read_text())
            if self.proc.poll() is not None:raise RuntimeError('worker exited; see '+self.log.name)
            if time.monotonic()>deadline:raise TimeoutError(str(path))
            time.sleep(.2)
        return json.loads(path.read_text())
    def call(self,value):
        i=self.n;self.n+=1;atomic_json(self.ipc/f'{i:04d}.request.json',value)
        return self.wait(self.ipc/f'{i:04d}.response.json')
    def close(self):
        self.proc.terminate()
        try:self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
        self.log.close()

def timing_report(records,ready,total_s,rollout_s):
    result=dict(total_wall_s=total_s,rollout_wall_s=rollout_s,steps=len(records),simulated_s=len(records)*.5,roles={},load=ready)
    for role,key in [('renderer','render'),('policy','policy'),('traffic','after')]:
        values=np.array([r[key]['wall_s'] for r in records])
        result['roles'][role]=dict(total_s=float(values.sum()),mean_s=float(values.mean()),median_s=float(np.median(values)),p95_s=float(np.percentile(values,95)),per_step_s=values.tolist())
    return result

def snapshot_sources(out, model):
    """Freeze the shared loop and the selected model workers before spawning them."""
    snapshot = out / 'code_snapshot'
    snapshot.mkdir()
    sources = {p.name: p for p in Path(__file__).parent.glob('*.py')}
    models = resource_root() / 'models'
    sources.update({
        'catk_traffic.py': models / 'generation/trajectory/CAT-K/closed_loop.py',
        'drivor_policy.py': models / 'agents/DrivoR/closed_loop.py',
        'render_' + model + '.py': MODEL_DIRS[model] / 'closed_loop.py',
    })
    hashes = {}
    for name, source in sources.items():
        target = snapshot / name
        shutil.copy2(source, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_json(out / 'code_manifest.json', hashes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=20);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU execution requires Slurm')
    if not 2<=a.steps<=79:raise ValueError('steps must be 2..79')
    cfg=json.loads(Path(a.config).read_text());out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    snapshot_sources(out, cfg.get('model', 'basedreamer'))
    cfg.update(run_output=str(out),runtime_dir=str(out/'runtime'))
    if not cfg.get('goal_navigation',{}).get('enabled'):raise ValueError('Explicit goal configuration required')
    goal_cfg=cfg['goal_navigation'];reached=False;terminal_time=None
    atomic_json(out/'config.json',cfg);gpus=os.environ['CUDA_VISIBLE_DEVICES'].split(',');allocated_gpus=gpus.copy()
    if len(gpus)==1:gpus=gpus*3
    elif len(gpus)!=3 and not cfg.get('gpu_layout'):raise ValueError('nonstandard GPU count requires gpu_layout')
    if cfg.get('gpu_layout'):gpus=[gpus[0]]*3

    key='/'.join([Path(cfg['scene']).parents[1].name,Path(cfg['scene']).name])

    services={};records=[];started=time.monotonic();ready={}
    try:
        for role,gpu in zip(['traffic','renderer','policy'],gpus):
            layout=cfg.get('gpu_layout',{}).get(role)
            assigned=','.join(allocated_gpus[int(i)] for i in str(layout).split(',')) if layout is not None else gpu
            services[role]=Service(role,cfg,out/'config.json',out,str(assigned))
        for role,v in services.items():ready[role]=v.wait(v.ipc/'ready.json',timeout=cfg.get('load_timeout_s',3600))
        rollout_start=time.monotonic()
        state=services['traffic'].call(dict(op='initial'));velocity=np.array(state['velocity']);previous_velocity=velocity.copy()
        for step in range(a.steps):
            rendered=services['renderer'].call(dict(state=state,output=str(out/'images'/f'{step:04d}')))
            R=pose_matrix(state['pose'])[:2,:2]
            local_v=velocity@R;local_a=(velocity-previous_velocity)/.5@R
            command,navigation=command_for_goal(state['pose'],goal_cfg)
            model_input=dict(camera_images=rendered['camera_images'],camera_calibration=rendered['camera_calibration'],ego_state=dict(timestamp=step*.5,velocity=[*local_v,0.],acceleration=[*local_a,0.]),command=command)
            from driveaudit.closed_loop.image_contract import attest
            proof=attest(rendered['camera_images'],out,step)
            atomic_json(out/'images'/f'{step:04d}'/'generated_manifest.json',proof)
            planned=services['policy'].call(dict(model_input=model_input,generated_image_proof=proof))
            path=execute_prefix(state['pose'],planned['native_trajectory'],planned['native_time_offsets'])
            path,terminal=apply_goal_terminal(path,state['pose'],goal_cfg,reached)
            if terminal['reached'] and not reached:terminal_time=step*.5+terminal['first_reached_offset_s']
            reached=terminal['reached']
            next_state=services['traffic'].call(dict(op='step',ego_path=path.tolist()))
            np.testing.assert_allclose(next_state['pose'],path[-1],atol=1e-5)
            previous_velocity=velocity;velocity=np.array(next_state['velocity'])
            record=dict(step=step,time_s=step*.5,before=state,render=rendered,policy=planned,after=next_state,navigation=navigation,goal_terminal=terminal,first_goal_time_s=terminal_time)
            atomic_json(out/'steps'/f'{step:04d}.json',record);records.append(record);state=next_state
            print('STEP',step+1,'pose',state['pose'],flush=True)
        atomic_json(out/'timing.json',timing_report(records,ready,time.monotonic()-started,time.monotonic()-rollout_start))
        atomic_json(out/'summary.json',dict(model=cfg.get('model','basedreamer'),status='completed',scene=key,steps=len(records),duration_s=len(records)*.5,execution='ideal_0.5s_prefix',state_feedback=True,future_file_reads_blocked=not cfg.get('allow_future_conditioning',False),policy_images_generated_only=True,final_pose=state['pose'],navigation='goal_bearing_to_native_discrete_command',goal_navigation=goal_cfg,goal_reached=reached,first_goal_time_s=terminal_time,terminal_convention='absorbing_ego_state_after_arrival; not_learned_braking',scope='single_scene_DrivoR_plus_goal_navigation_adapter'))
    except Exception as exc:
        atomic_json(out/'summary.json',dict(status='failed',completed_steps=len(records),error=str(exc)));raise
    finally:
        for v in services.values():v.close()
if __name__=='__main__':main()
