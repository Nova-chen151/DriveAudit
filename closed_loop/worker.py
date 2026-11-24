"""File RPC workers; optional future conditioning, generated-only policy images."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from core import atomic_json,pose_matrix

def deny_future(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=os.fsdecode(args[0]).replace('\\','/')
        if '/future/' in name:
            # importlib_resources has a Python subpackage literally named future.
            library=os.path.join(sys.prefix,'lib','python%d.%d'%sys.version_info[:2],
                                 'site-packages','importlib_resources','future')
            is_library_code=(os.path.realpath(name).startswith(os.path.realpath(library)+os.sep)
                             and name.endswith(('.py','.pyc')))
            if not is_library_code:raise RuntimeError('Forbidden future-data read: '+name)

class Traffic:
    def __init__(self,cfg):
        import yaml
        from driveaudit_traffic.trajectory.catk.adapter import prepare_scene,build_graph
        from portable_catk import CATKPredictor
        self.scene=prepare_scene(cfg['participant_scene'])
        self.ego=self.scene.ids.index('__ego__');self.tick=0
        adapter=Path(cfg['catk_adapter']);official=Path(cfg['official'])
        c=yaml.safe_load((adapter/'original_config.yaml').read_text())['catk']
        c.update(root=str(official),selected_checkpoint=str(official/'checkpoints/clsft_E9.ckpt'),
                 checkpoint_dir=str(official/'checkpoints'),migration_manifest=str(adapter/'official_source.sha256'))
        self.predictor=CATKPredictor(c);self.seed=cfg['seed']
        # Predictor registers the official CAT-K source before build_graph imports src.smart.
        self.graph=build_graph(self.scene)
        self.metadata=self.scene.actor_metadata
    def state(self):
        a=self.graph['agent'];i=self.ego
        pose=[*a['position'][i,10].tolist(),float(a['heading'][i,10])]
        actors=[]
        for j,m in enumerate(self.metadata):
            if j==i or not bool(a['valid_mask'][j,10]):continue
            length,width,height=a['shape'][j].tolist()
            actors.append(dict(id=m['agent_id'],category=m['category'],xyz=a['position'][j,10].tolist(),yaw=float(a['heading'][j,10]),wlh=[width,length,height]))
        for r in self.scene.static_current:
            actors.append(dict(id=r['agent_id'],category=r['category'],xyz=[float(r[k]) for k in ['x','y','z']],yaw=float(r['yaw']),wlh=[float(r[k]) for k in ['width','length','height']]))
        return dict(pose=pose,actors=actors,velocity=a['velocity'][i,10].tolist(),tick=self.tick)
    def call(self,q):
        if q['op']=='initial':return self.state()
        import torch
        path=np.asarray(q['ego_path'],float)
        if path.shape!=(5,4) or not np.isfinite(path).all():raise ValueError('bad committed prefix')
        before=self.state();p=self.predictor.infer_graphs([self.graph],self.seed+self.tick)
        pos=p['position'][:,:5].copy();head=p['heading'][:,:5].copy();valid=p['valid'][:,:5].copy()
        pos[self.ego]=path[:,:3];head[self.ego]=path[:,3];valid[self.ego]=True
        a=self.graph['agent'];old=a['position'][:,10:11,:2].numpy()
        vel=np.diff(np.concatenate((old,pos[:,:,:2]),axis=1),axis=1)/.1
        for key,value in [('position',pos),('heading',head),('velocity',vel),('valid_mask',valid)]:
            a[key][:,:11]=torch.cat((a[key][:,5:11].clone(),torch.from_numpy(value)),dim=1)
            a[key][:,11:]=0
        self.tick+=1
        actor_steps=[]
        for k in range(5):
            rows=[]
            for j,m in enumerate(self.metadata):
                if j==self.ego or not bool(valid[j,k]):continue
                length,width,height=a['shape'][j].tolist()
                rows.append(dict(id=m['agent_id'],category=m['category'],xyz=pos[j,k].tolist(),yaw=float(head[j,k]),wlh=[width,length,height]))
            rows.extend(dict(id=r['agent_id'],category=r['category'],xyz=[float(r[x]) for x in ['x','y','z']],yaw=float(r['yaw']),wlh=[float(r[x]) for x in ['width','length','height']]) for r in self.scene.static_current)
            actor_steps.append(dict(offset_s=(k+1)*.1,actors=rows))
        return dict(**self.state(),previous_pose=before['pose'],committed_ego_path=path.tolist(),committed_actor_states=actor_steps,sampling=self.predictor.last_sampling_stats,model_load=self.predictor.load_info)

class Renderer:
    def __init__(self,cfg):
        from types import SimpleNamespace
        from genbench_gt2.schema import SceneRef,SceneInputs
        from genbench_gt2.dataset import _calibration,_map,_history_rgb
        from genbench_gt2.constants import CAMERAS
        from PIL import Image
        import infer_genbench_gt2 as impl
        self.impl=impl;self.cfg=cfg;self.n=0
        scene=Path(cfg['scene']);root=scene.parents[2]
        ref=SceneRef(0,str(scene.relative_to(root)),'nuscenes',scene.name,root)
        order,calib=_calibration(scene)
        context=json.loads((scene/'scene_context.json').read_text())
        self.inputs=SceneInputs(ref,order,_history_rgb(ref),calib,np.repeat(np.eye(4)[None],90,axis=0),np.zeros((80,3)),{},_map(scene),context['generation_prompt'],np.arange(90)*500000)
        self.pipe,self.model_cfg,self.dtype=impl.load_pipeline(SimpleNamespace(checkpoint=Path(cfg['render_checkpoint']),sd15_root=Path(cfg['sd15'])))
        self.reference=impl.image_tensor([Image.open(scene/'history/rgb'/cam/'000009.jpg').convert('RGB') for cam in CAMERAS])
    def call(self,q):
        import torch
        from genbench_gt2.schema import AgentBox
        from gt2_native import ego_to_native
        from PIL import Image
        f=10+self.n;pose=pose_matrix(q['state']['pose'])
        if self.n==0:self.inputs.ego_to_anchor[9]=pose
        self.inputs.ego_to_anchor[f]=pose
        self.inputs.agents_by_frame[f]=tuple(AgentBox(f,a['id'],a['category'],np.array(a['xyz']),np.array(a['wlh']),a['yaw']) for a in q['state']['actors'])
        kwargs,dropped=self.impl.build_conditions(self.inputs,f,f-1)
        for k in ['bev_hdmap','camera_param','rel_pose','layout_canvas']:kwargs[k]=kwargs[k].to(device='cuda',dtype=self.dtype)
        kwargs['ref_images']=self.reference.to(device='cuda',dtype=self.dtype)
        with torch.inference_mode(),torch.autocast('cuda'):
            rows=self.pipe(prompt=[self.inputs.prompt],height=224,width=400,generator=torch.Generator(device='cuda').manual_seed(self.cfg['seed']+self.n),**kwargs,**dict(self.model_cfg.runner.pipeline_param)).images
        images=rows[0]
        if len(images)!=6 or any(im.size!=(400,224) for im in images):raise ValueError('invalid render views')
        self.reference=self.impl.image_tensor(images)
        folder=Path(q['output']);folder.mkdir(parents=True,exist_ok=True)
        paths={};calibs={};rig=ego_to_native(self.inputs)
        for name,im in zip(self.inputs.camera_order,images):
            p=folder/(name+'.png');im.save(p);paths[name]=str(p)
            c=self.inputs.calibrations[name];extrinsic=rig@c.camera_to_ego
            calibs[name]=dict(sensor2lidar_rotation=extrinsic[:3,:3].tolist(),sensor2lidar_translation=extrinsic[:3,3].tolist(),camera_intrinsics=c.intrinsics.tolist())
        self.n+=1
        return dict(camera_images=paths,camera_calibration=calibs,render_pose=q['state']['pose'],dropped=dropped)

class Policy:
    def __init__(self,cfg):
        from driveaudit.agents.native.drivor import DrivoRAdapter
        self.adapter=DrivoRAdapter(Path(cfg['policy_checkpoint']))
        self.run_output=cfg['run_output']
    def call(self,q):
        from image_contract import validate
        proof=validate(q,self.run_output)
        result=self.adapter.predict(q['model_input'])
        result['image_source_audit']=proof
        return result

def synchronize_cuda():
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()

def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['traffic','renderer','policy']);p.add_argument('config');p.add_argument('ipc');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());ipc=Path(a.ipc);ipc.mkdir(parents=True,exist_ok=True)
    future_reads=[]
    if a.role=='policy' or not cfg.get('allow_future_conditioning',False):
        sys.addaudithook(deny_future)
    else:
        def audit_future(event,args):
            if event=='open' and isinstance(args[0],(str,bytes)):
                name=os.fsdecode(args[0]).replace('\\','/')
                if '/future/' in name and 'site-packages/importlib_resources/future/' not in name:
                    if name not in future_reads:future_reads.append(name)
        sys.addaudithook(audit_future)
    try:
        load_start=time.monotonic()
        cls={'traffic':Traffic,'renderer':Renderer,'policy':Policy}[a.role]
        if a.role=='renderer' and cfg.get('renderer_module'):
            import importlib
            cls=importlib.import_module(cfg['renderer_module']).Renderer
        worker=cls(cfg)
        synchronize_cuda()
        if int(os.environ.get('RANK','0'))==0:
            atomic_json(ipc/'ready.json',dict(ready=True,load_wall_s=time.monotonic()-load_start,model=cfg.get('model','basedreamer'),future_data_paths_opened=list(future_reads)))
        index=0
        while True:
            request=ipc/f'{index:04d}.request.json'
            if not request.exists():time.sleep(.1);continue
            q=json.loads(request.read_text())
            synchronize_cuda()
            start=time.monotonic()
            result=worker.call(q)
            synchronize_cuda()
            if int(os.environ.get('RANK','0'))==0:
                result['wall_s']=time.monotonic()-start
                result['future_data_paths_opened']=list(future_reads)
                atomic_json(ipc/f'{index:04d}.response.json',result)
            index+=1
    except Exception:
        atomic_json(ipc/'error.json',dict(error=traceback.format_exc()));raise
if __name__=='__main__':main()
