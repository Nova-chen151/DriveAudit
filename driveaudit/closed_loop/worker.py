"""File RPC workers; optional future conditioning, generated-only policy images."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from importlib import import_module

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

def synchronize_cuda():
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()

def main():
    from driveaudit.closed_loop.core import atomic_json
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
        renderer = cfg.get('renderer_module') or 'render_' + cfg.get('model', 'basedreamer')
        module, name = {'traffic': ('catk_traffic', 'Traffic'), 'policy': ('drivor_policy', 'Policy'),
                        'renderer': (renderer, 'Renderer')}[a.role]
        worker = getattr(import_module(module), name)(cfg)
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
if __name__ == '__main__':
    # The launcher runs this file from code_snapshot in each native environment.
    # Resolve shared imports in model workers against those same frozen sources.
    import driveaudit.closed_loop
    driveaudit.closed_loop.__path__.insert(0, str(Path(__file__).resolve().parent))
    main()
