#!/usr/bin/env python3
"""Reconstruct DriveAudit vector outlines from the original visual-reference.png.

This traces the supplied artwork; it does not recover an original font/SVG and
never reads a GIF. All processing is local. Requires Pillow, NumPy, SciPy,
contourpy and PyMuPDF. Existing output files and the source tree are protected.

Example:
    python -I -B tools/render_wordmark.py --source /path/to/visual-reference.png \
        --output-dir media/wordmark
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import contourpy
import numpy as np
import pymupdf
from PIL import Image
from scipy import ndimage

SOURCE_SIZE = (2172, 724)
SEARCH_CROP = (460, 40, 1720, 235)
PADDING = 10
THRESHOLD = 0.5
SIMPLIFY_ERROR = 0.30
CORNER_ANGLE = math.radians(40)
ROUND_RADIUS = 1.5


def area(points: np.ndarray) -> float:
    return abs(float(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                            - points[:, 1] * np.roll(points[:, 0], -1)))) / 2


def simplify_open(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker with a source-pixel error bound."""
    if len(points) <= 2:
        return points
    delta = points[-1] - points[0]
    length = float(delta @ delta)
    if length == 0:
        distances = np.linalg.norm(points - points[0], axis=1)
    else:
        t = np.clip((points - points[0]) @ delta / length, 0, 1)
        distances = np.linalg.norm(points - (points[0] + t[:, None] * delta), axis=1)
    index = int(np.argmax(distances))
    if distances[index] <= tolerance:
        return points[[0, -1]]
    return np.vstack((simplify_open(points[:index + 1], tolerance)[:-1],
                      simplify_open(points[index:], tolerance)))


def simplify_closed(points: np.ndarray) -> np.ndarray:
    points = points[:-1] if np.allclose(points[0], points[-1]) else points
    first = int(np.lexsort((points[:, 1], points[:, 0]))[0])
    points = np.roll(points, -first, axis=0)
    opposite = int(np.argmax(np.linalg.norm(points - points[0], axis=1)))
    if opposite == 0:
        return points
    a = simplify_open(points[:opposite + 1], SIMPLIFY_ERROR)
    b = simplify_open(np.vstack((points[opposite:], points[0])), SIMPLIFY_ERROR)
    return np.vstack((a[:-1], b[:-1]))


def point_text(point: np.ndarray) -> str:
    return f'{point[0]:.3f},{point[1]:.3f}'


def outline_path(points: np.ndarray) -> str:
    """Round only gentle contour joins; preserve hard italic cuts and corners.

    A rounding radius <=1.5 source pixels moves the shallow joins by less than
    0.3 pixels. Strong corners retain their actual traced intersection.
    """
    vertices = simplify_closed(points)
    entries, exits, curved = [], [], []
    for index, vertex in enumerate(vertices):
        incoming = vertex - vertices[index - 1]
        outgoing = vertices[(index + 1) % len(vertices)] - vertex
        a, b = float(np.linalg.norm(incoming)), float(np.linalg.norm(outgoing))
        turn = math.acos(float(np.clip(incoming @ outgoing / max(a * b, 1e-12), -1, 1)))
        rounded = a > 0 and b > 0 and turn < CORNER_ANGLE
        radius = min(ROUND_RADIUS, a * 0.25, b * 0.25) if rounded else 0
        entries.append(vertex - incoming / max(a, 1e-12) * radius)
        exits.append(vertex + outgoing / max(b, 1e-12) * radius)
        curved.append(rounded)
    commands = [f'M{point_text(exits[-1])}']
    for index, vertex in enumerate(vertices):
        commands.append(f'L{point_text(entries[index])}')
        if curved[index]:
            commands.append(f'Q{point_text(vertex)} {point_text(exits[index])}')
    commands.append('Z')
    return ' '.join(commands)


def contours(field: np.ndarray) -> list[np.ndarray]:
    padded = np.pad(field, 1)
    generator = contourpy.contour_generator(
        x=np.arange(-1, field.shape[1] + 1, dtype=float) + 0.5,
        y=np.arange(-1, field.shape[0] + 1, dtype=float) + 0.5,
        z=padded, line_type='Separate', corner_mask=False,
    )
    return [loop for loop in generator.lines(THRESHOLD)
            if len(loop) >= 4 and np.allclose(loop[0], loop[-1]) and area(loop) > 0.15]


def trace(field: np.ndarray) -> str:
    return ' '.join(outline_path(loop) for loop in contours(field))


def masked_field(coverage: np.ndarray, selected: np.ndarray,
                 competing: np.ndarray | None = None) -> np.ndarray:
    support = ndimage.binary_dilation(selected, iterations=1)
    if competing is not None:
        support &= ~competing
    return np.where(support, coverage, 0)


def hex_color(rgb: np.ndarray) -> str:
    values = np.clip(np.rint(rgb), 0, 255).astype(int)
    return '#' + ''.join(f'{value:02x}' for value in values)


def interior_mask(selected: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(selected, iterations=2)
    return eroded if eroded.any() else selected


def gradient(name: str, rgb: np.ndarray, selected: np.ndarray,
             coverage: np.ndarray, horizontal: bool = False) -> tuple[str, dict]:
    """Estimate a smooth gradient from opaque original pixels, not the matte."""
    mask = interior_mask(selected)
    y, x = np.nonzero(mask)
    colors = rgb[mask]
    coordinates = np.column_stack((x + 0.5, y + 0.5))
    center = coordinates.mean(axis=0)
    if horizontal:
        direction = np.array([1.0, 0.0])
    else:
        design = np.column_stack((coordinates - center, np.ones(len(x))))
        coefficients = np.linalg.lstsq(design, colors, rcond=None)[0][:2]
        direction = np.linalg.svd(coefficients, full_matrices=False)[0][:, 0]
        if direction[np.argmax(np.abs(direction))] < 0:
            direction = -direction
    projection = (coordinates - center) @ direction
    low, high = np.quantile(projection, [0.01, 0.99])
    start, end = center + direction * low, center + direction * high
    stops = []
    for t in [0, 0.25, 0.5, 0.75, 1]:
        target = low + (high - low) * t
        band = np.abs(projection - target) <= max(2, (high - low) * 0.08)
        if not band.any():
            band[np.argmin(np.abs(projection - target))] = True
        stops.append((t, hex_color(np.median(colors[band], axis=0))))
    definition = (f'<linearGradient id="{name}" gradientUnits="userSpaceOnUse" '
                  f'x1="{start[0]:.3f}" y1="{start[1]:.3f}" '
                  f'x2="{end[0]:.3f}" y2="{end[1]:.3f}">'
                  + ''.join(f'<stop offset="{t:g}" stop-color="{color}"/>' for t, color in stops)
                  + '</linearGradient>')
    return definition, {'start': start.round(3).tolist(), 'end': end.round(3).tolist(),
                        'stops': [{'offset': t, 'color': color} for t, color in stops]}


def render_svg(svg: str, width: int, height: int, scale: int) -> bytes:
    """Render vector coverage and SVG user-space gradients without a white matte.

    MuPDF's lightweight SVG importer drops gradient fills. Render each actual
    vector path as coverage, then composite its standard sRGB gradient in straight
    alpha. The PNG is still rasterized from these SVG paths, never the source PNG.
    This deliberately supports only the path/gradient subset generated above.
    """
    root = ET.fromstring(svg)
    namespace = {'s': 'http://www.w3.org/2000/svg'}
    gradients = {node.attrib['id']: node for node in root.findall('s:defs/s:linearGradient', namespace)}
    pixel_width, pixel_height = width * scale, height * scale
    yy = (np.arange(pixel_height, dtype=np.float32)[:, None] + 0.5) / scale
    xx = (np.arange(pixel_width, dtype=np.float32)[None, :] + 0.5) / scale
    alpha = np.zeros((pixel_height, pixel_width), dtype=np.float32)
    premultiplied = np.zeros((pixel_height, pixel_width, 3), dtype=np.float32)

    def rgb(value: str) -> np.ndarray:
        return np.array([int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)], dtype=np.float32)

    for node in root.findall('s:path', namespace):
        d = escape(node.attrib['d'])
        mask_svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                    f'viewBox="0 0 {width} {height}"><path fill="#ffffff" fill-rule="evenodd" d="{d}"/></svg>')
        with pymupdf.open(stream=mask_svg.encode(), filetype='svg') as document:
            pixmap = document[0].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=True)
            samples = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixel_height, pixel_width, pixmap.n)
            coverage = samples[..., -1].astype(np.float32) / 255
        fill = node.attrib['fill']
        if fill.startswith('url(#'):
            gradient_node = gradients[fill[5:-1]]
            x1, y1, x2, y2 = (float(gradient_node.attrib[key]) for key in ('x1', 'y1', 'x2', 'y2'))
            dx, dy = x2 - x1, y2 - y1
            t = np.clip(((xx - x1) * dx + (yy - y1) * dy) / (dx * dx + dy * dy), 0, 1)
            stops = gradient_node.findall('s:stop', namespace)
            offsets = [float(stop.attrib['offset']) for stop in stops]
            colors = np.array([rgb(stop.attrib['stop-color']) for stop in stops])
            color = np.stack([np.interp(t, offsets, colors[:, channel]) for channel in range(3)], axis=-1).astype(np.float32)
        else:
            color = rgb(fill)
        premultiplied *= (1 - coverage[..., None])
        premultiplied += color * coverage[..., None]
        alpha = coverage + alpha * (1 - coverage)
    straight = np.divide(premultiplied, alpha[..., None], out=np.zeros_like(premultiplied), where=alpha[..., None] > 0)
    rgba = np.dstack((straight, alpha))
    image = Image.fromarray(np.clip(np.rint(rgba * 255), 0, 255).astype(np.uint8))
    output = BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


def rebuild(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    # The original repository is input only, including its tools/ parents.
    source_repository = next((parent for parent in source.parents if (parent / '.git').exists()), source.parent)
    if destination == source_repository or source_repository in destination.parents:
        raise ValueError('Output must not be inside the source repository')
    if destination.exists():
        raise FileExistsError(f'Use a fresh output directory: {destination}')
    source_bytes = source.read_bytes()
    with Image.open(source) as opened:
        if opened.format != 'PNG' or opened.size != SOURCE_SIZE:
            raise ValueError(f'Expected original {SOURCE_SIZE} reference PNG, not a GIF or scaled copy')
        raw = np.asarray(opened.convert('RGB').crop(SEARCH_CROP), dtype=float)
    # For this cyan/purple/navy ink on white, the darkest channel gives continuous
    # foreground coverage. Iso-contours are traced, never exported as binary alpha.
    coverage = 1 - raw.min(axis=2) / 255
    labels, _ = ndimage.label(coverage >= THRESHOLD)
    components = []
    for identifier, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        selected = labels == identifier
        pixels = int(selected.sum())
        if pixels < 4:
            continue  # Isolated one-pixel color dust, not a road dash or cutout.
        y, x = region
        median = np.median(raw[interior_mask(selected)], axis=0)
        components.append({'id': identifier, 'pixels': pixels,
                           'bounds': [x.start, y.start, x.stop, y.stop],
                           'median': median.tolist()})
    if len(components) != 17:
        raise ValueError(f'Unexpected source topology ({len(components)} components); review artwork before tracing')
    kept_ids = [item['id'] for item in components]
    foreground = np.isin(labels, kept_ids)
    ys, xs = np.nonzero(foreground)
    bounds = (max(0, int(xs.min()) - PADDING), max(0, int(ys.min()) - PADDING),
              min(raw.shape[1], int(xs.max()) + PADDING + 1),
              min(raw.shape[0], int(ys.max()) + PADDING + 1))
    left, top, right, bottom = bounds
    raw = raw[top:bottom, left:right]
    coverage = coverage[top:bottom, left:right]
    labels = labels[top:bottom, left:right]
    height, width = coverage.shape

    groups: dict[str, list[int]] = {'cyan': [], 'purple': [], 'letters': [], 'underline': [], 'accents': []}
    for item in components:
        x0, y0, x1, y1 = item['bounds']
        red, green, blue = item['median']
        if y0 >= 150:
            group = 'underline' if item['pixels'] > 1000 else 'accents'
        elif green > red + 50 and green > 80:
            group = 'cyan'
        elif red > green + 35 and blue > green + 70:
            group = 'purple'
        else:
            group = 'letters'
        groups[group].append(item['id'])
    if [len(groups[key]) for key in ['cyan', 'purple', 'letters', 'underline', 'accents']] != [2, 2, 8, 1, 4]:
        raise ValueError('Unexpected wordmark color groups; review source instead of guessing')

    definitions, gradients, paths, details = [], {}, {}, {}
    for name in ['cyan', 'purple', 'letters', 'underline']:
        selected = np.isin(labels, groups[name])
        field = masked_field(coverage, selected, (labels > 0) & ~selected)
        paths[name] = trace(field)
        if name != 'letters':
            definition, data = gradient(name, raw, selected, coverage, horizontal=name == 'underline')
            definitions.append(definition)
            gradients[name] = data
        if name in ('cyan', 'purple'):
            # These enclosed white areas are the original lane/bridge markings;
            # open counters remain transparent, as do the ordinary R/D counters.
            holes = ndimage.binary_fill_holes(selected) & ~selected
            detail = masked_field(1 - coverage, holes, selected)
            details[name] = trace(detail)

    # Recover the white lane cutout WITHIN the lower bridge arch. The gap above
    # that arch is the A's open counter and must remain transparent on dark pages.
    upper_id = min(groups['purple'], key=lambda identifier: np.nonzero(labels == identifier)[0].min())
    lower_id = next(identifier for identifier in groups['purple'] if identifier != upper_id)
    bridge = np.zeros_like(coverage, dtype=bool)
    for x in range(width):
        rows = np.nonzero(labels[:, x] == lower_id)[0]
        runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
        for before, after in zip(runs, runs[1:]):
            a, b = int(before[-1]) + 1, int(after[0])
            if 0 < b - a <= 14:
                bridge[a:b, x] = True
    bridge_path = trace(masked_field(1 - coverage, bridge, labels > 0))
    accents = []
    for identifier in groups['accents']:
        selected = labels == identifier
        accents.append((trace(masked_field(coverage, selected, (labels > 0) & ~selected)),
                        hex_color(np.median(raw[interior_mask(selected)], axis=0))))
    ordinary_color = hex_color(np.median(raw[interior_mask(np.isin(labels, groups['letters']))], axis=0))

    outputs: dict[str, bytes] = {}
    for theme, letter_color in [('on-dark', '#ffffff'), ('on-light', ordinary_color)]:
        pieces = [f'<?xml version="1.0" encoding="UTF-8"?>\n'
                  f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                  f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="logo-title logo-description">',
                  '<title id="logo-title">DriveAudit</title>',
                  '<desc id="logo-description">Source-reconstructed DriveAudit wordmark with cyan D, purple bridge A, and speed underline.</desc>',
                  '<defs>' + ''.join(definitions) + '</defs>']
        for name in ['cyan', 'letters', 'purple', 'underline']:
            fill = letter_color if name == 'letters' else f'url(#{name})'
            pieces.append(f'<path fill="{fill}" fill-rule="evenodd" d="{escape(paths[name])}"/>')
        for path in details.values():
            pieces.append(f'<path fill="#ffffff" fill-rule="evenodd" d="{escape(path)}"/>')
        if bridge_path:
            pieces.append(f'<path fill="#ffffff" fill-rule="evenodd" d="{escape(bridge_path)}"/>')
        pieces.extend(f'<path fill="{color}" fill-rule="evenodd" d="{escape(path)}"/>' for path, color in accents)
        pieces.append('</svg>\n')
        svg = '\n'.join(pieces)
        stem = f'driveaudit-{theme}'
        outputs[f'{stem}.svg'] = svg.encode()
        for scale in [2, 3]:
            outputs[f'{stem}@{scale}x.png'] = render_svg(svg, width, height, scale)

    manifest = {
        'description': 'Source-derived vector reconstruction, not recovered original vector artwork.',
        'renderer': 'Vector path coverage via MuPDF; user-space SVG gradients composited in straight-alpha sRGB.',
        'dependencies': {name: version(name) for name in ['Pillow', 'numpy', 'scipy', 'contourpy', 'PyMuPDF']},
        'source': source.name, 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'source_size': list(SOURCE_SIZE), 'search_crop': list(SEARCH_CROP),
        'source_bounds': [SEARCH_CROP[0] + left, SEARCH_CROP[1] + top,
                          SEARCH_CROP[0] + right, SEARCH_CROP[1] + bottom],
        'viewbox': [0, 0, width, height], 'threshold': THRESHOLD,
        'simplify_error_source_px': SIMPLIFY_ERROR, 'maximum_gentle_join_radius_source_px': ROUND_RADIUS,
        'components': components, 'groups': groups, 'gradients': gradients,
        'outputs': {name: hashlib.sha256(content).hexdigest() for name, content in sorted(outputs.items())},
    }
    outputs['source.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    # Create/write only after all parsing, tracing and rendering succeeded.
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in outputs.items():
        (destination / name).write_bytes(content)
    if source.read_bytes() != source_bytes:
        raise RuntimeError('The read-only source changed during generation')
    print(json.dumps({'directory': str(destination), 'size': [width, height],
                      'files': {name: len(content) for name, content in outputs.items()}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    rebuild(args.source, args.output_dir)


if __name__ == '__main__':
    main()
