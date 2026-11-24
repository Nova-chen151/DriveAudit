"""Participant archive boundary and actor-only output; no hidden-data dependencies."""
import csv,hashlib,json,re,tarfile
from pathlib import Path,PurePosixPath
import numpy as np
FIELDS=['frame_index','agent_id','category','is_dynamic','x','y','z','width','length','height','yaw','vx','vy','valid']
PROTOCOL='nuscenes_ego_conditioned_v1'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(Path(path).read_text())
def extract_inputs(archive,destination,scenes):
 if any(not re.fullmatch(r'case_[0-9]{6}',s) for s in scenes):raise ValueError('Invalid scene ID')
 if destination.exists():raise FileExistsError(destination)
 destination.mkdir(parents=True);seen=set();hashes={};manifest=None
 with tarfile.open(archive,'r|gz') as tar:
  for member in tar:
   parts=PurePosixPath(member.name).parts
   if not parts or parts[0]!='participant' or '..' in parts or PurePosixPath(member.name).is_absolute():raise ValueError('Unsafe archive path')
   if not (member.isfile() or member.isdir()):raise ValueError('Archive links/devices forbidden')
   if member.isdir():continue
   if member.name in seen:raise ValueError('Duplicate archive entry')
   seen.add(member.name)
   relative=PurePosixPath(*parts[1:]);rel=str(relative)
   wanted=len(parts)==2 or (len(parts)>3 and parts[1]=='scenes' and parts[2] in scenes)
   if not wanted:continue
   if 'future' in relative.parts or 'organizer_only' in relative.parts:raise ValueError('Unexpected hidden targets')
   target=destination/Path(*parts[1:]);target.parent.mkdir(parents=True,exist_ok=True)
   h=hashlib.sha256()
   with tar.extractfile(member) as src,target.open('xb') as dst:
    for b in iter(lambda:src.read(1024*1024),b''):h.update(b);dst.write(b)
   if rel=='manifest.sha256.json':manifest=read(target)
   else:hashes[rel]=h.hexdigest()
 if manifest is None or any(manifest.get(k)!=v for k,v in hashes.items()):raise ValueError('Input manifest mismatch')
 expected={k for k in manifest if len(PurePosixPath(k).parts)==1 or (len(PurePosixPath(k).parts)>2 and PurePosixPath(k).parts[:2][0]=='scenes' and PurePosixPath(k).parts[1] in scenes)}
 if set(hashes)!=expected:raise ValueError('Selected archive files are incomplete')
 meta=read(destination/'dataset.json')
 if meta['protocol']!=PROTOCOL or not set(scenes)<=set(meta['scenes']):raise ValueError('Unexpected dataset or scene ID')
 for scene in scenes:
  if read(destination/'scenes'/scene/'task.json')['source_dataset']!='nuscenes':raise ValueError('Only nuScenes supported')
 return dict(archive_sha256=sha(archive),verified_selected_files=len(hashes),scene_count=len(scenes),scene_ids=scenes)
def write_actor_csv(scene,native,source,destination):
 """All historical IDs get rows; absent-at-cutoff/unsupported actors explicitly invalid."""
 rows=list(csv.DictReader((source/'history/agents.csv').read_text().splitlines()));last={}
 for row in rows:
  if row['agent_id'] not in last or int(row['frame_index'])>int(last[row['agent_id']]['frame_index']):last[row['agent_id']]=row
 predicted={aid:i for i,aid in enumerate(scene.ids) if aid!='__ego__'}
 statics={r['agent_id'] for r in scene.static_current}
 modes={aid:'catk' if aid in predicted else 'static_hold' if aid in statics else 'inactive_not_predicted' for aid in last}
 if not predicted:raise ValueError('No traffic agents predicted; cannot claim successful traffic inference')
 destination.parent.mkdir(parents=True,exist_ok=True)
 with destination.open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
  for j,frame in enumerate(range(10,90)):
   for aid,row in sorted(last.items()):
    out={k:row[k] for k in FIELDS};out['frame_index']=frame
    if aid in predicted:
     i=predicted[aid];p=native['position'][i,j];v=native['velocity'][i,j];shape=scene.agent['shape'][i]
     out.update(x=float(p[0]),y=float(p[1]),z=float(p[2]),yaw=float(native['heading'][i,j]),vx=float(v[0]),vy=float(v[1]),length=float(shape[0]),width=float(shape[1]),height=float(shape[2]),valid=bool(native['valid'][i,j]))
    else:
     out.update(vx=0.,vy=0.,valid=aid in statics)
    writer.writerow(out)
 return dict(history_actor_count=len(last),model_actor_count=len(predicted),static_held_count=len(statics),inactive_count=sum(v=='inactive_not_predicted' for v in modes.values()),actor_modes=modes,rows=len(last)*80,csv_sha256=sha(destination))
