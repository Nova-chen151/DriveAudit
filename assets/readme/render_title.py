"""Draw the original DriveAudit geometric wordmark; no font files required.

python assets/readme/render_title.py [--output PATH]
The D contains a closed driving circuit. All glyphs are original vector outlines.
"""

import argparse
from bisect import bisect_right
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


WIDTH, HEIGHT = 960, 440
SCALE = 2
FRAMES, DURATION = 72, 60
GLYPH_SCALE = 1.32
SLANT = 0.16
NAVY = (20, 32, 58)
CYAN = (7, 161, 189)
VIOLET = (105, 77, 195)

# Width, outer contour, counters. Coordinates are in a 120-unit cap-height grid.
GLYPHS = {
    'D': (80, [(0, 0), (58, 0), (80, 22), (80, 98), (58, 120), (0, 120)],
          [[(25, 24), (48, 24), (55, 31), (55, 89), (48, 96), (25, 96)]]),
    'R': (82, [(0, 0), (63, 0), (82, 19), (82, 56), (63, 75), (83, 120),
              (55, 120), (35, 78), (25, 78), (25, 120), (0, 120)],
          [[(25, 24), (52, 24), (57, 29), (57, 49), (52, 54), (25, 54)]]),
    'I': (32, [(3, 0), (29, 0), (29, 120), (3, 120)], []),
    'V': (88, [(0, 0), (27, 0), (44, 83), (61, 0), (88, 0), (59, 120), (29, 120)], []),
    'E': (78, [(0, 0), (78, 0), (78, 24), (26, 24), (26, 48), (68, 48),
              (68, 71), (26, 71), (26, 96), (78, 96), (78, 120), (0, 120)], []),
    'A': (94, [(0, 120), (32, 0), (62, 0), (94, 120), (67, 120), (60, 94),
              (34, 94), (27, 120)], [[(40, 70), (54, 70), (47, 36)]]),
    'U': (82, [(0, 0), (26, 0), (26, 87), (35, 96), (47, 96), (56, 87),
              (56, 0), (82, 0), (82, 101), (63, 120), (19, 120), (0, 101)], []),
    'T': (86, [(0, 0), (86, 0), (86, 25), (56, 25), (56, 120), (30, 120),
              (30, 25), (0, 25)], []),
}


def mix(a, b, t):
    return tuple(round(x * (1 - t) + y * t) for x, y in zip(a, b))


def gradient(colors, t):
    p = max(0, min(1, t)) * (len(colors) - 1)
    i = min(len(colors) - 2, int(p))
    return mix(colors[i], colors[i + 1], p - i)


def point(p, origin=(0, 0)):
    x, y = p
    return ((origin[0] + GLYPH_SCALE * (x + SLANT * (120 - y))) * SCALE,
            (origin[1] + GLYPH_SCALE * y) * SCALE)


def glyph(draw, character, origin):
    _, outer, holes = GLYPHS[character]
    draw.polygon([point(p, origin) for p in outer], fill=255)
    for hole in holes:
        draw.polygon([point(p, origin) for p in hole], fill=0)


def lettering(size):
    full = Image.new('L', size)
    accents = [Image.new('L', size), Image.new('L', size)]
    origins = [(98, 44), (262, 236)]
    for row, (word, origin) in enumerate(zip(('DRIVE', 'AUDIT'), origins)):
        x, y = origin
        for i, character in enumerate(word):
            glyph(ImageDraw.Draw(full), character, (x, y))
            if i == 0:
                glyph(ImageDraw.Draw(accents[row]), character, (x, y))
            x += (GLYPHS[character][0] + 14) * GLYPH_SCALE
    return full, accents, origins


def vertical_material(size, stops):
    image = Image.new('RGB', size)
    draw = ImageDraw.Draw(image)
    # One material cycle per line; the fine bevel is added from the glyph mask.
    for y in range(size[1]):
        local_y = (y / SCALE - 44) % 192
        color = gradient(stops, local_y / 158)
        draw.line((0, y, size[0], y), fill=color)
    return image


def closed_path(vertices, origin, count=720):
    points = [point(p, origin) for p in vertices + vertices[:1]]
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    result = []
    for i in range(count):
        d = lengths[-1] * i / count
        j = min(len(points) - 2, bisect_right(lengths, d) - 1)
        t = (d - lengths[j]) / (lengths[j + 1] - lengths[j])
        result.append(tuple(a * (1 - t) + b * t for a, b in zip(points[j], points[j + 1])))
    return result


def surface(mask, accents):
    size = mask.size
    image = Image.new('RGB', size, 'white')
    shadow = ImageChops.offset(mask, 0, 4 * SCALE).filter(ImageFilter.GaussianBlur(4 * SCALE))
    image.paste((93, 105, 133), mask=shadow.point(lambda a: a // 8))
    image.paste(vertical_material(size, [(36, 55, 88), (17, 28, 53), (48, 58, 88)]), mask=mask)
    image.paste(vertical_material(size, [(3, 116, 155), (9, 169, 191), (13, 98, 145)]), mask=accents[0])
    image.paste(vertical_material(size, [(92, 64, 167), (129, 101, 207), (62, 48, 126)]), mask=accents[1])
    # Subpixel highlights are derived from the actual contours, not a text outline.
    upper = ImageChops.subtract(mask, ImageChops.offset(mask, 0, SCALE))
    image.paste((174, 202, 226), mask=upper.point(lambda a: round(a * 0.48)))
    return image


def palette_for():
    # Reserve equal tonal precision for both small accent glyphs and the navy
    # letters; a histogram palette otherwise bands the smaller violet surface.
    colors = [(255, 255, 255)]
    for stops in [
        [(36, 55, 88), (17, 28, 53), (48, 58, 88)],
        [(3, 116, 155), (9, 169, 191), (13, 98, 145)],
        [(92, 64, 167), (129, 101, 207), (62, 48, 126)],
    ]:
        colors.extend(gradient(stops, i / 63) for i in range(64))
    colors.extend(mix(NAVY, (134, 191, 227), i / 45) for i in range(16))
    for color in [NAVY, CYAN, VIOLET]:
        colors.extend(mix(color, (255, 255, 255), i / 9) for i in range(10))
    colors.extend(mix((190, 239, 245), (238, 253, 255), i / 7) for i in range(8))
    colors.extend(mix((220, 225, 235), (255, 255, 255), i / 7) for i in range(8))
    flat = [channel for color in colors for channel in color]
    image = Image.new('P', (1, 1))
    image.putpalette(flat + [255] * (768 - len(flat)))
    return image


def render(output):
    size = (WIDTH * SCALE, HEIGHT * SCALE)
    mask, accents, origins = lettering(size)
    base = surface(mask, accents)
    # A road is inscribed in the first D itself, rather than enclosing the name.
    road = closed_path([(12, 12), (52, 12), (67, 28), (67, 92),
                        (52, 108), (12, 108)], origins[0])
    road_layer = Image.new('RGBA', size)
    draw = ImageDraw.Draw(road_layer)
    for i in range(0, len(road), 45):
        draw.line(road[i:i + 18], fill=(190, 239, 245, 105), width=SCALE)
    base = Image.alpha_composite(base.convert('RGBA'), road_layer)
    palette = palette_for()
    frames = []
    for frame in range(FRAMES):
        phase = frame / FRAMES
        image = base.copy()
        # Directional reflection, narrow enough that the wordmark remains dark.
        scan = -200 + phase * (WIDTH + 400)
        reflection = Image.new('L', size)
        draw = ImageDraw.Draw(reflection)
        draw.polygon([(scan * SCALE, 0), ((scan + 46) * SCALE, 0),
                      ((scan - 104) * SCALE, size[1]), ((scan - 150) * SCALE, size[1])], fill=85)
        reflection = reflection.filter(ImageFilter.GaussianBlur(9 * SCALE))
        image.paste((134, 191, 227), mask=ImageChops.multiply(mask, reflection))
        # The driving loop has an actual moving vehicle-shaped lozenge.
        packet = Image.new('RGBA', size)
        draw = ImageDraw.Draw(packet)
        head = int(phase * len(road)) % len(road)
        for age in range(70, 0, -1):
            i = (head - age) % len(road)
            draw.line([road[i], road[(i + 1) % len(road)]],
                      fill=(190, 244, 248, round(155 * (1 - age / 70))), width=3 * SCALE)
        hx, hy = road[head]
        ax, ay = road[(head + 3) % len(road)]
        angle = math.atan2(ay - hy, ax - hx)
        body = []
        for u, v in [(-5, -2.3), (3, -2.3), (5, -1), (5, 1), (3, 2.3), (-5, 2.3)]:
            body.append((hx + SCALE * (u * math.cos(angle) - v * math.sin(angle)),
                         hy + SCALE * (u * math.sin(angle) + v * math.cos(angle))))
        draw.polygon(body, fill=(238, 253, 255, 255))
        glow = packet.filter(ImageFilter.GaussianBlur(2 * SCALE))
        image = Image.alpha_composite(image, glow)
        image = Image.alpha_composite(image, packet)
        image = image.convert('RGB').resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        frames.append(image.quantize(palette=palette, dither=Image.Dither.NONE))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=DURATION,
                   loop=0, disposal=1, optimize=False)
    print(f'{output}: {WIDTH}x{HEIGHT}, {FRAMES} frames, {output.stat().st_size:,} bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('title.gif'))
    args = parser.parse_args()
    render(args.output)
