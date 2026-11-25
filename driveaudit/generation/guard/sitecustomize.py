"""Block accidental reads of original benchmark inputs in copied demo renderers."""
import os,sys
from pathlib import Path
if os.environ.get('DEMO_ROOT'):
 def guard(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   path=Path(os.fsdecode(args[0])).absolute()
   value=str(path)
   if '/data/genbench-gt2/' in value or '/data/nuscenes/' in value or '/data/nuscenes_mmdet3d' in value:
    raise PermissionError('Demo must use extracted participant inputs, not original dataset: '+value)
 sys.addaudithook(guard)
