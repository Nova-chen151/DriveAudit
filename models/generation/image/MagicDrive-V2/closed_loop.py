"""MagicDrive-V2 (Challenger MagicDriveDiT stage3-40k-ft) feedback adapter.

Uses regeneration_v2's full-sequence rflow sampler and checkpoint, replanning a
33-frame proposal at every call. Only slot 9 is committed; all later outputs
are discarded. Native 89-frame open-loop results are never read.
"""
import copy
import json
import os
from collections import deque
from pathlib import Path
import numpy as np
from driveaudit.closed_loop.rendering import initial_inputs, publish


class Renderer:
    def __init__(self,cfg):
        import torch
        import torch.distributed as dist
        import magicdrivedit.utils.module_contrib
        from mmengine.config import Config
        from colossalai.cluster import ProcessGroupMesh
        from magicdrivedit.acceleration.parallel_states import set_sequence_parallel_group,set_data_parallel_group
        from magicdrivedit.registry import MODELS,SCHEDULERS,build_module
        from accelerate.big_modeling import cpu_offload_with_hook
        from challenge_generation import infer as geom
        from challenge_generation.genbench_gt2 import load_map
        from challenge_generation.model_conditioning import training_calibration
        self.cfg=cfg;self.geom=geom;self.n=0
        self.inputs,images=initial_inputs(cfg)
        self.references=deque([images]*9,maxlen=9)
        self.states=deque(maxlen=9)
        self.device=torch.device('cuda',int(os.environ.get('LOCAL_RANK','0')))
        torch.cuda.set_device(self.device)
        if not dist.is_initialized():dist.init_process_group('nccl')
        self.sp_size=dist.get_world_size()
        self.mesh=ProcessGroupMesh(1,self.sp_size)
        self.sp_group=self.mesh.get_group_along_axis(1)
        set_sequence_parallel_group(self.sp_group)
        set_data_parallel_group(self.mesh.get_group_along_axis(0))
        self.dtype=torch.bfloat16
        mc=Config.fromfile(cfg['render_config'])
        mc.model.from_pretrained=cfg['render_checkpoint']
        mc.model.enable_layernorm_kernel=False
        mc.model.bbox_embedder_param.enable_layernorm_kernel=False
        mc.model.frame_emb_param.enable_layernorm_kernel=False
        mc.model.force_pad_h_for_sp_size=4
        mc.vae.from_pretrained=cfg['vae_checkpoint']
        mc.text_encoder.from_pretrained=cfg['text_checkpoint']
        mc.scheduler.type='rflow'
        self.mc=mc
        self.text=build_module(mc.text_encoder,MODELS,device='cpu')
        self.vae=build_module(mc.vae,MODELS).to('cpu',self.dtype).eval()
        self.model=build_module(mc.model,MODELS,input_size=(None,None,None),in_channels=self.vae.out_channels,
            caption_channels=self.text.output_dim,model_max_length=self.text.model_max_length,
            enable_sequence_parallelism=self.sp_size>1).to('cpu',self.dtype).eval()
        self.text.y_embedder=self.model.y_embedder
        self.scheduler=build_module(mc.scheduler,SCHEDULERS)
        self.text.t5.model.to('cpu');self.text.t5.device=torch.device('cpu')
        self.model,hook=cpu_offload_with_hook(self.model,self.device)
        self.vae,self.last_hook=cpu_offload_with_hook(self.vae,self.device,hook)
        calibration=training_calibration(json.loads((Path(cfg['scene'])/'calibration/cameras.json').read_text()))
        self.origin=geom._renderer_origin(calibration)
        self.rotation=geom._renderer_to_ego_rotation(calibration)
        self.cameras,self.to_cameras=geom._camera_matrices(calibration,self.origin)
        self.rig=np.eye(4);self.rig[:3,:3]=self.rotation;self.rig[:3,3]=self.origin
        self.map_mask,self.map_meta=load_map(Path(cfg['scene']))

    def _conditions(self,state):
        import torch
        from driveaudit.closed_loop.core import pose_matrix
        from magicdrivedit.utils.misc import collate_bboxes_to_maxlen,move_to
        from einops import rearrange,repeat
        # Current state is the only forecast: frozen continuation. No oracle
        # poses/agents are loaded, including for padded speculative slots.
        if not self.states:self.states.extend([copy.deepcopy(state)]*9)
        states=list(self.states)+[state]*24
        ids=sorted({str(a['id']) for s in states for a in s['actors']})
        n=max(1,len(ids));T=len(states)
        boxes=np.zeros((T,1,n,8,3),np.float32)
        classes=np.full((T,1,n),-1,np.int32)
        masks=np.zeros((T,6,n),np.float32)
        poses=[];positions=[];yaws=[]
        for t,s in enumerate(states):
            ego=pose_matrix(s['pose']);pose=ego@self.rig;poses.append(pose)
            positions.append(ego[:2,3]);yaws.append(np.arctan2(ego[1,0],ego[0,0]))
            inv=np.linalg.inv(pose)
            for a in s['actors']:
                if a['category'] not in self.geom.CATEGORY_TO_CLASS:continue
                j=ids.index(str(a['id']))
                corners=self.geom._box_corners(np.asarray(a['xyz']),*a['wlh'],a['yaw'])
                corners=(np.column_stack([corners,np.ones(8)])@inv.T)[:,:3]
                boxes[t,0,j]=corners;classes[t,0,j]=self.geom.CATEGORY_TO_CLASS[a['category']]
                for c,ext in enumerate(self.to_cameras):masks[t,c,j]=np.any((np.column_stack([corners,np.ones(8)])@ext.T)[:,2]>.05)
        bbox=collate_bboxes_to_maxlen([{k:torch.from_numpy(v) for k,v in dict(bboxes=boxes,classes=classes,masks=masks).items()}],self.device,self.dtype,6,T)
        maps=self.geom._warp_maps(self.map_mask,np.array(positions),np.array(yaws),self.origin,self.map_meta,self.rotation)
        cams=torch.from_numpy(np.repeat(self.cameras[None,None],T,axis=1).astype(np.float32))
        rel=torch.from_numpy(np.stack([np.linalg.inv(p)@poses[0] for p in poses]).astype(np.float32))[None]
        args=dict(maps=torch.from_numpy(maps[None]),bbox=bbox,
            cams=rearrange(cams,'B T NC ... -> (B NC) T 1 ...'),
            rel_pos=repeat(rel,'B T ... -> (B NC) T 1 ...',NC=6),fps=torch.tensor([2.]),
            drop_cond_mask=torch.ones(1),drop_frame_mask=torch.ones(1,T),
            height=torch.tensor([224]),width=torch.tensor([400]),num_frames=torch.tensor([T]))
        args=move_to(args,device=self.device,dtype=self.dtype)
        args.update(mv_order_map=self.mc.mv_order_map,t_order_map=self.mc.t_order_map)
        return args

    def call(self,q):
        import torch
        import torch.distributed as dist
        from PIL import Image
        from einops import rearrange
        from functools import partial
        from magicdrivedit.utils.misc import add_box_latent,move_to
        from magicdrivedit.utils.inference_utils import apply_mask_strategy,add_null_condition
        from magicdrivedit.utils.train_utils import sp_vae
        from scripts.adv_inference_magicdrive import make_box_latent_sampler
        args=self._conditions(q['state'])
        # Model camera order equals geom.CAMERAS, independently of publication order.
        ref=np.stack([np.stack([np.asarray(row[self.inputs.camera_order.index(c)].resize((400,224)),dtype=np.float32) for row in self.references]) for c in self.geom.CAMERAS])
        reference=torch.from_numpy(ref).permute(0,4,1,2,3).to(self.device,self.dtype)/127.5-1
        with torch.inference_mode():
            latent=self.vae.encode(reference)
            latent=rearrange(latent,'NC C T H W -> 1 (C NC) T H W')
            torch.manual_seed(self.cfg['seed']+self.n)
            z=torch.randn(1,self.vae.out_channels*6,*self.vae.get_latent_size((33,224,400)),device=self.device,dtype=self.dtype)
            bbox=add_box_latent(args['bbox'],1,6,33,make_box_latent_sampler(self.model,'random',1024))
            args['bbox']=move_to({k:rearrange(v,'B T NC ... -> (B NC) T ...') for k,v in bbox.items()},device=self.device,dtype=self.dtype)
            args=add_null_condition(args,self.model.camera_embedder.uncond_cam.to(self.device),self.model.frame_embedder.uncond_cam.to(self.device),prepend=False)
            mask=apply_mask_strategy(z,[[latent[0]]],[f'0,0,0,0,{latent.shape[2]},0'],0,align=None)
            samples=self.scheduler.sample(self.model,self.text,z=z,prompts=[self.inputs.prompt],neg_prompts=None,device=self.device,additional_args=args,progress=False,mask=mask)
            samples=rearrange(samples,'B (C NC) T H W -> (B NC) C T H W',NC=6)
            samples=sp_vae(samples.to(self.dtype),partial(self.vae.decode,num_frames=args['num_frames']),self.sp_group) if self.sp_size>1 else self.vae.decode(samples.to(self.dtype),num_frames=args['num_frames'])
            if samples.shape[:3]!=(6,3,33) or not torch.isfinite(samples).all():raise ValueError('Invalid MagicDrive-V2 clip')
            selected=samples[:,:,9].float().cpu().permute(0,2,3,1).numpy()
        self.last_hook.offload()
        by_cam={c:Image.fromarray(((a.clip(-1,1)+1)*127.5).round().astype('uint8')) for c,a in zip(self.geom.CAMERAS,selected)}
        images=[by_cam[c] for c in self.inputs.camera_order]
        result=publish(self.inputs,images,q['state']['pose'],q['output']) if dist.get_rank()==0 else {}
        self.references.append(images);self.states.append(copy.deepcopy(q['state']));self.n+=1
        result['temporal_commit']={'native_frames':33,'committed_slot':9,'discarded_generated_frames':23,'reference':'history_000009_repeated_then_committed_prefix','source_version':'regeneration_v2_checkpoint_full_sequence_rflow'}
        return result
