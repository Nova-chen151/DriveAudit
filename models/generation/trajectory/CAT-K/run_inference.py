#!/usr/bin/env python3
"""Actual CAT-K inference from a participant archive. No original dataset argument."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'participant_tools'))
from participant_io import extract_inputs,write,read,sha,write_actor_csv,PROTOCOL

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scene',nargs='+',default=['case_000001']);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=20260914);a=p.parse_args()
 archive=a.archive.resolve();output=a.output.resolve()
 if output.exists():raise FileExistsError('Use a new run output directory: '+str(output))
 if output==archive or output in archive.parents:raise ValueError('Output overlaps input archive')
 output.mkdir(parents=True);started=time.perf_counter();opened=set()
 # Deny original/private dataset access. Inputs are extracted from the exact supplied archive.
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  path=Path(os.fsdecode(args[0])).absolute()
  if '/genbench-gt2/' in str(path) and path!=archive:raise PermissionError('External dataset access denied: '+str(path))
  if output in path.parents:opened.add(str(path.relative_to(output)))
 sys.addaudithook(audit)
 try:
  inputs=output/'inputs/participant';input_report=extract_inputs(archive,inputs,a.scene);write(output/'input_verification.json',input_report)
  import yaml,torch,numpy as np
  from portable_catk import CATKPredictor,scene_seed
  from driveaudit_traffic.trajectory.catk.adapter import prepare_scene
  from native_output import native_prediction,qc
  cfg=yaml.safe_load((ROOT/'original_config.yaml').read_text())['catk'];official=Path(os.environ['DRIVEAUDIT_CATK_OFFICIAL']).resolve()
  cfg.update(root=str(official),selected_checkpoint=str(official/'checkpoints/clsft_E9.ckpt'),checkpoint_dir=str(official/'checkpoints'),migration_manifest=str(ROOT/'official_source.sha256'),device=a.device,global_seed=a.seed,cpu_threads=4)
  for key in ('input_root','output_root','work_root'):cfg.pop(key,None)
  load_start=time.perf_counter();model=CATKPredictor(cfg);load_seconds=time.perf_counter()-load_start
  submission=output/'submission';write(submission/'submission.json',dict(protocol=PROTOCOL,team='organizer-baseline',method='CAT-K clsft_E9 + existing constrained token sampler',seed=a.seed,checkpoint=cfg['checkpoint_sha256'],training_data='Existing pretrained CAT-K checkpoint; see upstream README',external_data='No external scenario data at inference; public package history and map only'))
  records=[]
  for name in a.scene:
   source=inputs/'scenes'/name;prep=time.perf_counter();scene=prepare_scene(source);prep_seconds=time.perf_counter()-prep
   assert not scene.agent['valid_mask'][:,11:].any()
   if a.device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   begin=time.perf_counter();pred=model.rollout(scene,scene_seed(scene.dataset,name,a.seed))
   if a.device.startswith('cuda'):torch.cuda.synchronize()
   infer_seconds=time.perf_counter()-begin;native=native_prediction(scene,pred)
   if not all(np.isfinite(native[k]).all() for k in ('position','heading','velocity')):raise ValueError('Nonfinite output')
   csv_report=write_actor_csv(scene,native,source,submission/'scenes'/name/'future/future_trajectories.csv')
   diag=output/'diagnostics'/name;diag.mkdir(parents=True);np.savez_compressed(diag/'prediction_10hz.npz',**pred,agent_ids=np.asarray(scene.ids))
   record=dict(scene_id=name,prepare_seconds=prep_seconds,inference_seconds=infer_seconds,gpu_peak_memory_mb=torch.cuda.max_memory_allocated()/2**20 if a.device.startswith('cuda') else None,history=scene.history,actor_metadata=scene.actor_metadata,model_input_files=scene.input_hashes,output=csv_report,qc=qc(scene,native),sampling=model.last_rollout_diagnostics)
   write(diag/'report.json',record);records.append(record);print('INFERENCE_DONE',name,infer_seconds,'seconds',flush=True)
  from validate_submission import check
  for name in a.scene:check(inputs,submission,name,trajectories_only=True)
  # The public checker requires all dataset IDs for a non-single scene call; validate selected scenes separately.
  from prepare_render_inputs import build
  sys.path.insert(0,str(ROOT.parent/'gen-benchmark'))
  from genbench_gt2.dataset import load_generation_set,load_scene_inputs
  rendering=[]
  for name in a.scene:
   bridge=output/'render_inputs'/name;build(inputs,submission,bridge,name)
   os.environ['GENBENCH_USE_CATK_EGO_PREDICTION']='0'
   refs=load_generation_set(bridge,'participant_joint');loaded=load_scene_inputs(refs[0])
   assert len(loaded.history_rgb)==6
   assert sha(inputs/'scenes'/name/'conditioning/ego_trajectory.csv')==sha(bridge/'nuscenes300_challenger_v1/scenes'/name/'future/ego_trajectory.csv')
   rendering.append(dict(scene_id=name,status='PASS',path=str(bridge),generated_actor_csv_sha256=sha(bridge/'nuscenes300_challenger_v1/scenes'/name/'future/future_trajectories.csv'),public_ego_unchanged=True))
  write(output/'io_audit.json',dict(external_original_dataset_reads='blocked',opened_output_paths=sorted(opened)))
  report=dict(status='PASS',input=input_report,checkpoint_sha256=model.checkpoint_sha,official_source_verification=model.source_integrity,config=cfg,model_load_seconds=load_seconds,total_seconds=time.perf_counter()-started,device=torch.cuda.get_device_name() if a.device.startswith('cuda') else 'cpu',torch=torch.__version__,numpy=np.__version__,records=records,renderer_loader_checks=rendering,future_actor_gt_used=False,future_rgb_used=False,ego_policy='Decoder predicts ego jointly from history; its ego output is diagnostic only. Renderer uses publicly supplied ego motion. No future-ego-conditioned decoding claim.',rendered_rgb=False)
  write(output/'COMPLETED.json',report);print('PARTICIPANT_CATK_PASS',output,flush=True)
 except BaseException:
  write(output/'FAILED.json',dict(error=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started));raise
if __name__=='__main__':main()
