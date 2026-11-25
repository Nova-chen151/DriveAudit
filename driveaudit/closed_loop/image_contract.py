"""DrivoR receives only files attested by this run's renderer RPC."""
import hashlib,json
from pathlib import Path
CAMERAS={'CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT'}
def attest(paths,root,step):
    root=Path(root).resolve();folder=root/'images'/f'{step:04d}'
    if set(paths)!=CAMERAS:raise ValueError('Exactly six generated cameras required')
    hashes={}
    for cam,name in paths.items():
        path=Path(name).resolve()
        if path!=folder/(cam+'.png') or not path.is_file():raise ValueError('Policy image is outside current generated frame: '+name)
        hashes[cam]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(source='renderer_output',step=step,camera_sha256=hashes)
def validate(request,root):
    proof=request['generated_image_proof'];step=proof['step']
    actual=attest(request['model_input']['camera_images'],root,step)
    manifest=Path(root)/'images'/f'{step:04d}'/'generated_manifest.json'
    if actual!=proof or json.loads(manifest.read_text())!=actual:raise ValueError('Generated image attestation mismatch')
    return actual
