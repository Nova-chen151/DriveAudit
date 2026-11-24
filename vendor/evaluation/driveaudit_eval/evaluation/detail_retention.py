"""Lossless scene-local storage for reviewed, unconsumed metric detail arrays.

Aggregate metrics and scientific eligibility remain inline.  A reference replaces
only a named diagnostic payload after every metric kernel has finished using it.
Archives use paths relative to the evaluation output, so retention may move them.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

from driveaudit_eval.utils.io import file_sha256, jsonable

POLICY = 'lossless_compact_v1'
SCHEMA = 'driveaudit_eval_metric_details_v1'


def enabled(config):
    policy = config.get('metric_detail_retention')
    if policy not in (None, POLICY):
        raise ValueError(f'Unknown metric_detail_retention policy: {policy}')
    return policy == POLICY and config.get('retain_intermediates', True) is False


def _pointer(parts):
    return '/' + '/'.join(str(p).replace('~', '~0').replace('/', '~1') for p in parts)


def _lookup(value, parts):
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _rewrite(value, replacements, parts=()):
    if parts in replacements:
        return replacements[parts]
    # Copy only the ancestors of replaced payloads.  Unchanged metrics stay shared
    # until the ordinary JSON writer normalizes them; no full scene copy is needed.
    descendants = {path: replacement for path, replacement in replacements.items()
                   if path[:len(parts)] == parts}
    if not descendants:
        return value
    return {key: _rewrite(child, descendants, parts + (key,))
            for key, child in value.items()}


def _compact(result, fields, archive, artifact_root):
    if '_retained_details' in result:
        raise ValueError('Metric result already has retained details')
    payloads = {parts: _lookup(result, parts) for parts in fields}
    payloads = {parts: value for parts, value in payloads.items()
                if isinstance(value, (dict, list, tuple)) and len(value)}
    if not payloads:
        return result
    root = Path(artifact_root).resolve()
    archive = Path(archive).resolve()
    relative = archive.relative_to(root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + archive.name + '.', dir=archive.parent)
    raw_hash = hashlib.sha256()
    raw_bytes = 0
    try:
        with os.fdopen(descriptor, 'wb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed:
                pending, pending_chars = [], 0
                def write(text, flush=False):
                    nonlocal raw_bytes, pending_chars
                    pending.append(text)
                    pending_chars += len(text)
                    if pending_chars < 65536 and not flush:
                        return
                    block = ''.join(pending).encode('utf-8')
                    pending.clear()
                    pending_chars = 0
                    compressed.write(block)
                    raw_hash.update(block)
                    raw_bytes += len(block)
                write('{"schema":' + json.dumps(SCHEMA) + ',"payloads":{')
                for index, (parts, value) in enumerate(payloads.items()):
                    write((',' if index else '') + json.dumps(_pointer(parts)) + ':')
                    # Normalize one field at a time, never the entire scene.
                    for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False,
                            separators=(',', ':')).iterencode(jsonable(value)):
                        write(chunk)
                write('}}', flush=True)
            raw.flush()
            os.fsync(raw.fileno())
        # Verify compressed storage before the caller can release native payloads.
        verified = hashlib.sha256()
        with gzip.open(temporary, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                verified.update(block)
        if verified.hexdigest() != raw_hash.hexdigest():
            raise ValueError('Retained detail archive content hash mismatch')
        os.replace(temporary, archive)
    finally:
        Path(temporary).unlink(missing_ok=True)
    replacements = {parts: {'$retained_detail': _pointer(parts), 'item_count': len(value)}
                    for parts, value in payloads.items()}
    compact = _rewrite(result, replacements)
    compact['_retained_details'] = dict(schema=SCHEMA, policy=POLICY, path=str(relative),
        compression='gzip', sha256=file_sha256(archive), content_sha256=raw_hash.hexdigest(),
        bytes=archive.stat().st_size, uncompressed_bytes=raw_bytes, field_count=len(payloads),
        field_paths=[_pointer(parts) for parts in payloads])
    return compact


def compact_consistency_result(result, scene_output, artifact_root):
    fields = []
    for backend, prefix, groups in (
            ('mtpc', ('primary_metrics',), ('presence', 'projection_geometry', 'image_motion', 'multiview')),
            ('tcvv', ('primary_metrics',), ('presence', 'local_motion', 'projection_geometry', 'multiview')),
            ('tcvv', ('history_calibration', 'metrics'), ('presence', 'local_motion', 'projection_geometry', 'multiview'))):
        for group in groups:
            leaf = 'per_window' if backend == 'tcvv' and group in ('local_motion', 'projection_geometry') else 'per_frame'
            fields.append(('backends', backend, 'result') + prefix + (group, leaf))
    archive = Path(scene_output) / 'retained_details' / 'metric_payloads.json.gz'
    return _compact(result, fields, archive, artifact_root)


def compact_trajectory_result(result, artifact_root):
    fields = [('metrics', 'interaction', 'values', 'by_interaction_type')]
    values = result.get('metrics', {}).get('kinematics', {}).get('values', {})
    if isinstance(values, dict):
        fields.extend(('metrics', 'kinematics', 'values', actor, 'series') for actor in values)
    name = str(result['scene_id']).replace('/', '__') + '.details.json.gz'
    archive = Path(artifact_root) / 'trajectory_scenes' / name
    return _compact(result, fields, archive, artifact_root)


def restore_metric_details(result, artifact_root):
    """Recover the original JSON-equivalent scene, failing closed on corruption."""
    metadata = result.get('_retained_details')
    if metadata is None:
        return result
    _validate_references(result, metadata)
    archive = _verify_descriptor(metadata, artifact_root)
    with gzip.open(archive, 'rt', encoding='utf-8') as stream:
        document = json.load(stream)
    if document.get('schema') != SCHEMA or set(document['payloads']) != set(metadata['field_paths']):
        raise ValueError('Retained detail archive fields mismatch')
    replacements = {}
    for pointer, payload in document['payloads'].items():
        parts = tuple(p.replace('~1', '/').replace('~0', '~') for p in pointer.split('/')[1:])
        reference = _lookup(result, parts)
        if (not isinstance(reference, dict) or reference.get('$retained_detail') != pointer
                or reference.get('item_count') != len(payload)):
            raise ValueError('Retained detail reference mismatch')
        replacements[parts] = payload
    restored = _rewrite(result, replacements)
    del restored['_retained_details']
    return restored


def _verify_descriptor(metadata, artifact_root):
    if (metadata.get('schema') != SCHEMA or metadata.get('policy') != POLICY
            or metadata.get('compression') != 'gzip'):
        raise ValueError('Unsupported retained detail schema')
    root = Path(artifact_root).resolve()
    archive = (root / metadata['path']).resolve()
    archive.relative_to(root)
    if file_sha256(archive) != metadata['sha256']:
        raise ValueError('Retained detail archive hash mismatch')
    verified = hashlib.sha256()
    size = 0
    with gzip.open(archive, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            verified.update(block)
            size += len(block)
    if verified.hexdigest() != metadata['content_sha256'] or size != metadata['uncompressed_bytes']:
        raise ValueError('Retained detail content hash mismatch')
    return archive


def _validate_references(result, metadata):
    expected = metadata.get('field_paths')
    if (not isinstance(expected, list) or len(set(expected)) != len(expected)
            or metadata.get('field_count') != len(expected)):
        raise ValueError('Retained detail reference manifest mismatch')
    actual = set()
    def visit(value, parts=()):
        if isinstance(value, dict):
            if '$retained_detail' in value:
                pointer = _pointer(parts)
                if (value['$retained_detail'] != pointer or type(value.get('item_count')) is not int
                        or value['item_count'] <= 0):
                    raise ValueError('Retained detail reference location mismatch')
                actual.add(pointer)
                return
            for key, child in value.items():
                if key != '_retained_details':
                    visit(child, parts + (key,))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, parts + (index,))
    visit(result)
    if set(expected) != actual:
        raise ValueError('Retained detail reference fields mismatch')


def verify_metric_detail_archives(summary, artifact_root, *, include_quality_children=True, include_native=True):
    """Stream-check all archive hashes, including separately retained GT quality."""
    root = Path(artifact_root).resolve()
    records, native_expected = {}, {}
    def visit(value, base):
        if isinstance(value, dict):
            native = value.get('_native_metric_retention')
            if native is not None:
                if native.get('schema') != 'lossless_terminal_native_metrics_v1':
                    raise ValueError('Unsupported native metric retention schema')
                manifest = (base / native['manifest']).resolve()
                archive = (base / native['archive']).resolve()
                native_expected[str(manifest.relative_to(root))] = dict(
                    archive=str(archive.relative_to(root)), sha256=native['archive_sha256'])
            metadata = value.get('_retained_details')
            if metadata is not None:
                _validate_references(value, metadata)
                archive = _verify_descriptor(metadata, base)
                relative = str(archive.relative_to(root))
                records[relative] = dict(path=relative, sha256=metadata['sha256'],
                    content_sha256=metadata['content_sha256'], bytes=archive.stat().st_size,
                    uncompressed_bytes=metadata['uncompressed_bytes'])
                return
            if '$retained_detail' in value:
                raise ValueError('Retained detail orphan reference has no archive descriptor')
            for child in value.values():
                visit(child, base)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, base)
    visit(summary, root)
    if include_quality_children:
        for path in sorted((root / 'gt_quality').glob('*/summary.json')):
            with path.open() as stream:
                visit(json.load(stream), path.parent)
    detail_count, native_count = len(records), 0
    if include_native:
        from tools.cleanup_output_gt_scratch import verify_native_metric_archives
        native = verify_native_metric_archives(root, expected_manifest_paths=native_expected,
                                              include_quality_children=include_quality_children)
        native_count = native['archive_count']
        for row in native['archives']:
            expected = native_expected[row['manifest']]
            if expected['archive'] != row['archive'] or expected['sha256'] != row['archive_sha256']:
                raise ValueError('Native metric archive descriptor mismatch')
            records[row['archive']] = dict(row, path=row['archive'], kind='native', sha256=row['archive_sha256'])
    return dict(schema=SCHEMA, status='verified', archive_count=len(records),
        detail_archive_count=detail_count, native_archive_count=native_count,
        archive_bytes=sum(row['bytes'] for row in records.values()), archives=list(records.values()))
