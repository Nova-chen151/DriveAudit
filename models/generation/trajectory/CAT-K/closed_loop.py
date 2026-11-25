"""CAT-K traffic state and committed policy feedback for one scene."""
from pathlib import Path

import numpy as np


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
