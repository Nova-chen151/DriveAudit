"""Verify a git-less migration snapshot against its original checksum manifest."""
import hashlib
from pathlib import Path

def verify_source(root,manifest):
    root=Path(root);manifest=Path(manifest);prefix='dependencies/catk-official/'
    expected={}
    for line in manifest.read_text().splitlines():
        sha,sep,name=line.partition('  ')
        if sep and name.startswith(prefix+'src/') and name.endswith('.py'):expected[name[len(prefix):]]=sha
    actual={str(p.relative_to(root)) for p in (root/'src').rglob('*.py')}
    if not expected or actual!=set(expected):raise ValueError('CAT-K source inventory differs from migration manifest')
    for name,sha in expected.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:raise ValueError('CAT-K source checksum mismatch: '+name)
    return dict(method='migration_sha256',source_files=len(expected),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
