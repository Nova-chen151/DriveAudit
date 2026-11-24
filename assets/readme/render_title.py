"""Render the README wordmark with Pillow; no font files are redistributed.

Usage (macOS): python assets/readme/render_title.py
For another platform, supply an installed bold italic face with --font PATH.
"""

import argparse
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


def bezier(points, steps=160):
    for i in range(steps):
        t = i / (steps - 1)
        yield tuple(
            (1 - t) ** 3 * points[0][axis]
            + 3 * (1 - t) ** 2 * t * points[1][axis]
            + 3 * (1 - t) * t * t * points[2][axis]
            + t ** 3 * points[3][axis]
            for axis in (0, 1)
        )


def render(font_path, output):
    width, height = 1600, 360
    blue = (29, 128, 190)
    ink = (30, 40, 50)
    font = ImageFont.truetype(str(font_path), 218, index=9 if font_path.suffix == '.ttc' else 0)
    word = 'DriveAudit'
    bbox = font.getbbox(word)
    x = (width - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = 38 - bbox[1]
    text_mask = Image.new('L', (width, height))
    ImageDraw.Draw(text_mask).text((x, y), word, font=font, fill=255)
    split = x + font.getlength('Drive')
    base = Image.new('RGB', (width, height), 'white')
    base.paste(ink, mask=text_mask)
    drive_mask = text_mask.copy()
    ImageDraw.Draw(drive_mask).rectangle((split, 0, width, height), fill=0)
    base.paste(blue, mask=drive_mask)

    # One continuous driving path: an approach, a smooth lane change,
    # and a returning arc. Its moving segment represents feedback.
    points = []
    for controls in [
        [(170, 291), (450, 291), (520, 291), (760, 291)],
        [(760, 291), (925, 291), (923, 255), (1080, 255)],
        [(1080, 255), (1135, 255), (1240, 255), (1320, 255)],
        [(1320, 255), (1400, 255), (1400, 321), (1320, 321)],
        [(1320, 321), (1060, 321), (700, 321), (600, 321)],
    ]:
        points.extend(bezier(controls))
    # Resample by arc length so the trajectory head moves at steady speed.
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    sampled = []
    cursor = 0
    for index in range(1000):
        distance = lengths[-1] * index / 999
        while cursor < len(lengths) - 2 and lengths[cursor + 1] < distance:
            cursor += 1
        fraction = (distance - lengths[cursor]) / max(0.001, lengths[cursor + 1] - lengths[cursor])
        sampled.append(tuple(points[cursor][axis] * (1 - fraction) + points[cursor + 1][axis] * fraction for axis in (0, 1)))

    frames = []
    palette_image = Image.new('RGB', (1, 1))
    palette = []
    # A fixed palette prevents flicker between frames.
    for color in [blue, ink, (189, 221, 240), (228, 240, 248)]:
        for i in range(32):
            palette.extend(round(channel + (255 - channel) * i / 31) for channel in color)
    palette_image = palette_image.quantize(colors=128)
    palette_image.putpalette(palette + [255] * (768 - len(palette)))
    for frame in range(72):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        draw.line(sampled, fill=(228, 240, 248), width=3, joint='curve')
        # The seam is held on a fully readable wordmark, not a blank reveal.
        progress = min(1.0, frame / 57)
        head = int(progress * (len(sampled) - 1))
        tail = max(0, head - 195)
        if head > tail:
            for start in range(tail, head, 8):
                strength = (start - tail) / max(1, head - tail)
                color = tuple(round(210 + (c - 210) * strength) for c in blue)
                draw.line(sampled[start:min(start + 9, head + 1)], fill=color, width=6, joint='curve')
            hx, hy = sampled[head]
            draw.ellipse((hx - 5, hy - 5, hx + 5, hy + 5), fill=blue)
        if 5 <= frame <= 47:
            # A narrow diagonal highlight travels once across the solid wordmark.
            scan_x = -180 + (frame - 5) / 42 * (width + 360)
            sheen = Image.new('L', (width, height))
            ImageDraw.Draw(sheen).polygon([(scan_x, 20), (scan_x + 28, 20), (scan_x - 62, 247), (scan_x - 90, 247)], fill=90)
            image.paste('white', mask=ImageChops.multiply(sheen, text_mask))
        frames.append(image.quantize(palette=palette_image, dither=Image.Dither.NONE))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True, disposal=1)
    print(f'{output}: {width}x{height}, {len(frames)} frames, {output.stat().st_size:,} bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, default=Path('/System/Library/Fonts/Avenir Next.ttc'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('title.gif'))
    args = parser.parse_args()
    render(args.font, args.output)
