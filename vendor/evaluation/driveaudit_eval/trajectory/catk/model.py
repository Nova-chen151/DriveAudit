"""Load only official tokenizer/decoder weights; no Lightning/WOSAC execution."""
import hashlib
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from .adapter import build_graph, sha256


def validate_sampling(sampling):
    if sampling['criterium']=='greedy_map':
        if int(sampling['num_k'])!=64 or float(sampling['temperature'])!=1:
            raise ValueError('Constrained greedy requires a 64-token initial pool and temperature=1')
        return
    if sampling['criterium']!='topk_prob': raise ValueError('Only no-GT topk_prob sampling is permitted')
    if int(sampling['num_k'])!=48 or float(sampling['temperature'])!=1.0:
        raise ValueError('Formal sampling requires K=48, temperature=1')


def scene_seed(dataset,scene_id,global_seed):
    return int.from_bytes(hashlib.sha256(f'{dataset}\0{scene_id}\0{global_seed}'.encode()).digest()[:4],'big')


class CATKPredictor:
    def __init__(self,config):
        validate_sampling(config['sampling'])
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch
        from omegaconf import OmegaConf
        self.config=config; self.device=config.get('device','cuda:0')
        self.root=Path(config['root']).resolve()
        if not (self.root/'src/smart/tokens/token_processor.py').is_file(): raise FileNotFoundError(self.root)
        sys.path.insert(0,str(self.root))
        from src.smart.tokens.token_processor import TokenProcessor
        from src.smart.modules.smart_decoder import SMARTDecoder
        self.repo_sha=subprocess.check_output(['git','-C',str(self.root),'rev-parse','HEAD'],text=True).strip()
        if subprocess.check_output(['git','-C',str(self.root),'status','--porcelain'],text=True).strip():
            raise ValueError('Official CAT-K checkout must be clean')
        if self.repo_sha!=config['repo_sha']: raise ValueError('CAT-K repo SHA mismatch')
        checkpoint=Path(config['selected_checkpoint'])
        self.checkpoint_sha=sha256(checkpoint)
        if self.checkpoint_sha!=config['checkpoint_sha256']: raise ValueError('Checkpoint SHA mismatch')
        loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg=OmegaConf.create(loaded['hyper_parameters']['model_config'])
        if not cfg.finetune: raise ValueError('Selected checkpoint is not fine-tuned')
        if cfg.decoder.num_historical_steps!=11 or cfg.decoder.num_future_steps!=80:
            raise ValueError('Checkpoint time protocol mismatch')
        self.vocabulary_hashes={}
        for name in ('map_token_file','agent_token_file'):
            path=self.root/'src/smart/tokens'/cfg.token_processor[name]
            self.vocabulary_hashes[str(path)]=sha256(path)
            if config['token_vocabulary_sha256'][path.name]!=self.vocabulary_hashes[str(path)]:
                raise ValueError('Vocabulary SHA mismatch')
        model=torch.nn.Module()
        model.token_processor=TokenProcessor(**cfg.token_processor)
        model.encoder=SMARTDecoder(**cfg.decoder,n_token_agent=model.token_processor.n_token_agent)
        loaded_keys=model.load_state_dict(loaded['state_dict'],strict=True)
        self.load_info=dict(missing_keys=loaded_keys.missing_keys,unexpected_keys=loaded_keys.unexpected_keys,
                            parameter_count=sum(p.numel() for p in model.parameters()),state_keys=len(loaded['state_dict']))
        self.model=model.eval().requires_grad_(False).to(self.device)
        self.sampling=OmegaConf.create(dict(criterium='topk_prob',num_k=48,temp=1.0))
        torch.set_num_threads(config.get('cpu_threads',4))
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        self.model_config=OmegaConf.to_container(cfg,resolve=True)
        self.last_sampling_stats={}

    def infer_graphs(self,graphs,seed):
        import torch
        from torch_geometric.data import Batch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        data=Batch.from_data_list([g.clone() for g in graphs]).to(self.device)
        # A second guard protects direct callers and continuation from future-state injection.
        for key in ('position','heading','velocity','valid_mask'): data['agent'][key][:,11:]=0
        with torch.inference_mode():
            current_position=data['agent']['position'][:,10,:2].clone()
            current_heading=data['agent']['heading'][:,10].clone()
            token_map,token_agent=self.model.token_processor(data)
            if self.config['sampling']['criterium']=='greedy_map':
                if len(graphs)!=1: raise ValueError('Map-constrained decoding requires batch_size=1')
                from unittest.mock import patch
                from driveaudit_eval.utils.map_provider.canonical_map import load_map_context
                from .constrained import ConstrainedGreedy
                sampler=ConstrainedGreedy(load_map_context(graphs[0].map_scene_path),data['agent'],64,
                                         collision_checks=self.config.get('decoding_policy',{}).get('collision_checks',False))
                sampler.stats['max_history_quantization_offset_m']=float(torch.linalg.vector_norm(token_agent['gt_pos'][:,1]-current_position,dim=-1).max())
                # Quantization supplies history token IDs; rollout starts at the actual observation.
                # Continuations use the previous prediction as their observed prefix.
                token_agent['gt_pos'][:,1]=current_position
                token_agent['gt_heading'][:,1]=current_heading
                # Scoped hook: official weights, decoder and token coordinates remain unchanged.
                with patch('src.smart.modules.agent_decoder.sample_next_token_traj',sampler):
                    pred=self.model.encoder.inference(token_map,token_agent,self.sampling)
                self.last_sampling_stats=dict(sampler.stats)
            else:
                pred=self.model.encoder.inference(token_map,token_agent,self.sampling)
                self.last_sampling_stats={}
        position=torch.cat((pred['pred_traj_10hz'],pred['pred_z_10hz'][...,None]),dim=-1)
        result=dict(position=position.cpu().numpy(),heading=pred['pred_head_10hz'].cpu().numpy(),
                    valid=pred['pred_valid'][:,2:].repeat_interleave(5,dim=1).cpu().numpy())
        if result['position'].shape!=(data['agent'].num_nodes,80,3): raise ValueError('Expected 80 future states')
        if not all(np.isfinite(result[k]).all() for k in ('position','heading')): raise ValueError('Nonfinite CAT-K prediction')
        return result

    def infer(self,scene,seed):
        return self.infer_graphs([build_graph(scene)],seed)

    def rollout(self,scene,seed):
        """Each segment is 80 steps; later histories contain only previous predictions."""
        import torch
        graph=build_graph(scene)
        count=math.ceil((scene.timestamps[-1]-scene.timestamps[9]-1e-7)/8)
        results=[];diagnostics=[]
        for segment in range(count):
            result=self.infer_graphs([graph],seed if segment==0 else scene_seed(scene.dataset,str(seed)+f'/segment/{segment}',self.config['global_seed']))
            results.append(result)
            diagnostics.append(dict(self.last_sampling_stats))
            if segment+1<count:
                position=result['position'][:,-12:]
                graph['agent']['position'][:,:11]=torch.from_numpy(position[:,1:].copy())
                graph['agent']['heading'][:,:11]=torch.from_numpy(result['heading'][:,-11:].copy())
                graph['agent']['velocity'][:,:11]=torch.from_numpy(np.diff(position[:,:,:2],axis=1)/.1)
                graph['agent']['valid_mask'][:,:11]=torch.from_numpy(result['valid'][:,-11:].copy())
        self.last_rollout_diagnostics=diagnostics
        return {key:np.concatenate([r[key] for r in results],axis=1) for key in results[0]}
