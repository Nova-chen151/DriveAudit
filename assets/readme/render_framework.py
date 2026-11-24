#!/usr/bin/env python3
"""Render four original, seamless conceptual animations (Pillow + stdlib).

Run from any directory: python -I /path/to/render_framework.py [--output-dir DIR]
All geometry is drawn here; no fonts, source images, or sibling imports are used.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

SIZE = 384
SCALE = 3
FRAMES = 60
DURATION = 50
TAU = math.tau
NAVY = (23, 37, 65)
CYAN = (16, 171, 196)
PURPLE = (124, 73, 210)
WHITE = (255, 255, 255)


def mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def rgba(color, alpha=255):
    return (*color[:3], round(alpha))


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def cubic(a, b, c, d, t):
    u = 1 - t
    return (
        u**3 * a[0] + 3 * u * u * t * b[0] + 3 * u * t * t * c[0] + t**3 * d[0],
        u**3 * a[1] + 3 * u * u * t * b[1] + 3 * u * t * t * c[1] + t**3 * d[1],
    )


def bezier(segments, steps=48):
    points = []
    for segment in segments:
        points.extend(cubic(*segment, i / steps) for i in range(steps))
    points.append(segments[-1][-1])
    return points


class Surface:
    """Small floating-point drawing layer with supersampled geometry."""

    def __init__(self, width=SIZE, height=SIZE, transparent=False):
        self.image = Image.new(
            "RGBA", (round(width * SCALE), round(height * SCALE)),
            (0, 0, 0, 0) if transparent else WHITE + (255,),
        )
        self.draw = ImageDraw.Draw(self.image)

    def points(self, points):
        return [(round(x * SCALE), round(y * SCALE)) for x, y in points]

    def box(self, box):
        return tuple(round(v * SCALE) for v in box)

    def line(self, points, color=NAVY, width=2, closed=False):
        points = list(points)
        if closed:
            points.append(points[0])
        self.draw.line(self.points(points), fill=color, width=max(1, round(width * SCALE)), joint="curve")

    def stroke(self, points, color=NAVY, width=2):
        points = list(points)
        if not points:
            return
        self.line(points, color, width)
        r = width / 2
        for x, y in (points[0], points[-1]):
            self.ellipse((x - r, y - r, x + r, y + r), color)

    def polygon(self, points, fill, outline=None, width=1):
        self.draw.polygon(self.points(points), fill=fill)
        if outline:
            self.line(points, outline, width, closed=True)

    def ellipse(self, box, fill, outline=None, width=1):
        self.draw.ellipse(self.box(box), fill, outline, max(1, round(width * SCALE)))

    def rounded(self, box, radius, fill, outline=None, width=1):
        self.draw.rounded_rectangle(
            self.box(box), round(radius * SCALE), fill, outline, max(1, round(width * SCALE)),
        )

    def composite(self, layer, xy=(0, 0)):
        image = layer.image if isinstance(layer, Surface) else layer
        self.image.alpha_composite(image, tuple(round(v * SCALE) for v in xy))
        self.draw = ImageDraw.Draw(self.image)

    def shadow(self, box, opacity=20, blur=8):
        layer = Surface(self.image.width / SCALE, self.image.height / SCALE, True)
        layer.ellipse(box, rgba(NAVY, opacity))
        self.composite(layer.image.filter(ImageFilter.GaussianBlur(blur * SCALE)))

    def glow(self, xy, color=CYAN, radius=18, opacity=50):
        layer = Surface(self.image.width / SCALE, self.image.height / SCALE, True)
        x, y = xy
        layer.ellipse((x - radius, y - radius, x + radius, y + radius), rgba(color, opacity))
        self.composite(layer.image.filter(ImageFilter.GaussianBlur(radius * SCALE * 0.6)))

    def gradient(self, box, top, bottom, radius=0, polygon=None):
        x0, y0, x1, y1 = self.box(box)
        w, h = max(1, x1 - x0), max(1, y1 - y0)
        strip = Image.new("RGBA", (1, h))
        strip.putdata([rgba(mix(top, bottom, y / max(1, h - 1))) for y in range(h)])
        fill = strip.resize((w, h))
        mask = Image.new("L", (w, h))
        d = ImageDraw.Draw(mask)
        if polygon:
            d.polygon([(round(x * SCALE) - x0, round(y * SCALE) - y0) for x, y in polygon], fill=255)
        else:
            d.rounded_rectangle((0, 0, w - 1, h - 1), radius=round(radius * SCALE), fill=255)
        fill.putalpha(mask)
        self.image.alpha_composite(fill, (x0, y0))
        self.draw = ImageDraw.Draw(self.image)

    def finish(self):
        return self.image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def side_car(surface, x, y, scale=1.0, tracked=True):
    def p(points):
        return [(x + a * scale, y + b * scale) for a, b in points]

    def box(values):
        a, b, c, d = values
        return (x + a * scale, y + b * scale, x + c * scale, y + d * scale)

    surface.ellipse(box((-21, 8, 23, 15)), mix(NAVY, WHITE, 0.89))
    surface.polygon(p([(-21, 0), (-12, -3), (-5, -13), (11, -13), (20, -3), (25, 0), (24, 9), (-22, 9)]), NAVY)
    surface.polygon(p([(-8, -3), (-2, -10), (9, -10), (15, -3)]), mix(CYAN, WHITE, 0.55))
    surface.line(p([(3, -10), (3, -3)]), NAVY, 1.7 * scale)
    surface.rounded(box((-20, 0, 24, 7)), 2.5 * scale, CYAN)
    surface.line(p([(-18, 1), (20, 1)]), mix(CYAN, WHITE, 0.6), 1 * scale)
    for wx in (-12, 15):
        surface.ellipse(box((wx - 4.5, 5, wx + 4.5, 14)), NAVY)
        surface.ellipse(box((wx - 2, 7.5, wx + 2, 11.5)), WHITE)
    surface.rounded(box((20, 0, 24, 3)), 1 * scale, WHITE)
    if tracked:
        c = mix(CYAN, WHITE, 0.1)
        for sx in (-1, 1):
            for sy in (-1, 1):
                bx, by = sx * 32, sy * 24
                surface.stroke(p([(bx - sx * 8, by), (bx, by), (bx, by - sy * 7)]), c, 2.3 * scale)


def video_card(t, history, width=240, height=181):
    card = Surface(width, height, True)
    card.gradient((0, 0, width, height), WHITE, mix(CYAN, WHITE, 0.92), 15)
    scene = Surface(width, height, True)
    scene.gradient((5, 5, width - 5, height - 5), mix(CYAN, WHITE, 0.94), mix(CYAN, WHITE, 0.84), 11)
    parallax = 4 * math.sin(TAU * t)
    # A compact, coherent roadside scene, seen at three nearby times.
    scene.polygon([(5, 87), (40 + parallax, 66), (78 + parallax, 86), (132, 61), (198, 84), (235, 70), (235, 127), (5, 127)], mix(CYAN, WHITE, 0.79))
    scene.polygon([(5, 104), (72, 92), (137, 105), (194, 90), (235, 100), (235, 134), (5, 134)], mix(CYAN, WHITE, 0.88))
    for bx, by, bw, bh, tint in [(39, 62, 25, 44, CYAN), (78, 43, 30, 63, NAVY), (177, 66, 30, 41, PURPLE)]:
        bx += parallax * 0.45
        scene.polygon([(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)], mix(tint, WHITE, 0.78))
        scene.polygon([(bx + bw, by), (bx + bw + 7, by + 5), (bx + bw + 7, by + bh), (bx + bw, by + bh)], mix(tint, WHITE, 0.62))
        scene.line([(bx + 6, by + 10), (bx + 6, by + bh - 9)], mix(WHITE, tint, 0.13), 3)
        if bw > 26:
            scene.line([(bx + 15, by + 10), (bx + 15, by + bh - 9)], mix(WHITE, tint, 0.13), 3)
    scene.polygon([(5, 113), (235, 113), (235, 169), (5, 169)], mix(NAVY, WHITE, 0.92))
    scene.line([(5, 114), (235, 114)], mix(NAVY, WHITE, 0.76), 1.4)
    scene.line([(5, 163), (235, 163)], WHITE, 2.4)
    for x in range(-40, 270, 48):
        scene.stroke([(x, 145), (x + 22, 145)], WHITE, 2)
    # Wrapped motion is clipped by the glass, not reset in the visible scene.
    position = ((t - history) * 296 + 106) % 296 - 28
    for dx in (-296, 0, 296):
        x = position + dx
        if -60 < x < width + 60:
            side_car(scene, x, 128, 0.91)
    # Small second moving landmark gives the scene depth, not just a scan line.
    pedestrian = 155 + 13 * math.sin(TAU * (t - history) + 0.3)
    scene.ellipse((pedestrian - 3, 96, pedestrian + 3, 102), PURPLE)
    scene.stroke([(pedestrian, 104), (pedestrian, 112)], mix(PURPLE, WHITE, 0.2), 3)
    scene.line([(pedestrian, 110), (pedestrian - 4, 115)], PURPLE, 1.8)
    scene.line([(pedestrian, 110), (pedestrian + 4, 115)], PURPLE, 1.8)
    mask = Image.new("L", scene.image.size)
    ImageDraw.Draw(mask).rounded_rectangle(card.box((6, 6, width - 6, height - 6)), radius=10 * SCALE, fill=255)
    scene.image.putalpha(ImageChops.multiply(scene.image.getchannel("A"), mask))
    card.composite(scene)
    # Restrained glass highlights and a solid optical frame.
    sheen = Surface(width, height, True)
    sheen.polygon([(8, 8), (100, 8), (30, 175), (8, 175)], rgba(WHITE, 48))
    sheen.image.putalpha(ImageChops.multiply(sheen.image.getchannel("A"), mask))
    card.composite(sheen)
    card.rounded((1.8, 1.8, width - 1.8, height - 1.8), 14, None, NAVY, 3.6)
    card.rounded((6, 6, width - 6, height - 6), 10, None, rgba(WHITE, 240), 1.3)
    card.rounded((16, 15, 36, 21), 3, CYAN)
    card.ellipse((43, 15, 49, 21), mix(PURPLE, WHITE, 0.35))
    return card


def observation(t):
    out = Surface()
    out.shadow((62, 290, 331, 322), 25, 10)
    for i, (x, y, delay) in enumerate([(44, 66, 0.16), (66, 88, 0.08), (88, 110, 0)]):
        bob = 2.5 * math.sin(TAU * t + i * 0.32)
        shadow = Surface(254, 198, True)
        shadow.rounded((5, 7, 245, 188), 15, rgba(NAVY, 23))
        out.composite(shadow.image.filter(ImageFilter.GaussianBlur(4 * SCALE)), (x - 3, y + bob))
        out.composite(video_card(t, delay), (x, y + bob))
    return out.finish()


# Orthographic projection keeps all views registered to one physical object.
def project(point):
    x, y, z = point
    return (192 + 0.87 * (x - y), 244 + 0.44 * (x + y) - z)


def depth(point):
    return point[0] + point[1] + 1.0 * point[2]


def prism_faces(corners, height, colors):
    bottom = [(x, y, z) for x, y, z in corners]
    top = [(x, y, z + height) for x, y, z in corners]
    result = [(top, colors[0])]
    for i in range(4):
        j = (i + 1) % 4
        result.append(([bottom[i], bottom[j], top[j], top[i]], colors[1 + i % (len(colors) - 1)]))
    return result


def draw_faces(out, faces, outline=NAVY, width=1.4):
    for points, color in sorted(faces, key=lambda face: sum(depth(p) for p in face[0]) / len(face[0])):
        out.polygon([project(p) for p in points], color, outline, width)


def scene_cube(out, t):
    corners = [(-31, -31, 8), (31, -31, 8), (31, 31, 8), (-31, 31, 8)]
    draw_faces(out, prism_faces(corners, 77, [mix(CYAN, WHITE, 0.79), mix(PURPLE, WHITE, 0.78), mix(CYAN, WHITE, 0.91), mix(CYAN, WHITE, 0.75), mix(PURPLE, WHITE, 0.83)]), width=2.5)
    # Large material insets remain legible when the GIF is displayed at 128 px.
    out.polygon([project(p) for p in [(-17, -15, 85.1), (15, -15, 85.1), (15, 16, 85.1), (-17, 16, 85.1)]], mix(CYAN, WHITE, 0.36), mix(CYAN, WHITE, 0.05), 1.6)
    out.polygon([project(p) for p in [(31.1, -18, 25), (31.1, 17, 25), (31.1, 17, 67), (31.1, -18, 67)]], mix(CYAN, WHITE, 0.5), mix(CYAN, WHITE, 0.1), 1.4)
    out.line([project(p) for p in [(31.3, -15, 59), (31.3, 14, 59)]], mix(CYAN, WHITE, 0.86), 2)
    out.line([project(p) for p in [(-17, 31.1, 28), (15, 31.1, 28), (15, 31.1, 65), (-17, 31.1, 65), (-17, 31.1, 28)]], mix(PURPLE, WHITE, 0.23), 2.3)
    # One shared feature, not unrelated view-dependent targets.
    target = project((0, 0, 85))
    out.ellipse((target[0] - 4, target[1] - 3, target[0] + 4, target[1] + 3), WHITE, CYAN, 1.4)


def camera_geometry(angle, t, index):
    radius = 112
    position = (radius * math.cos(angle), radius * math.sin(angle), 55 + 4 * math.sin(TAU * t + index * 2.1))
    f = (-math.cos(angle), -math.sin(angle))
    s = (-f[1], f[0])

    def local(u, v, z):
        return (position[0] + u * s[0] + v * f[0], position[1] + u * s[1] + v * f[1], position[2] + z)

    corners = [local(-14, -11, -10), local(14, -11, -10), local(14, 8, -10), local(-14, 8, -10)]
    faces = prism_faces(corners, 22, [mix(NAVY, WHITE, 0.39), NAVY, mix(NAVY, WHITE, 0.14), mix(NAVY, WHITE, 0.23), NAVY])
    # A real lens barrel, built in the same 3D coordinate system as the scene.
    n = 20
    rear, front = [], []
    for i in range(n):
        a = TAU * i / n
        rear.append(local(8.3 * math.cos(a), 8, 8.3 * math.sin(a)))
        front.append(local(8.3 * math.cos(a), 20, 8.3 * math.sin(a)))
    for i in range(n):
        j = (i + 1) % n
        faces.append(([rear[i], rear[j], front[j], front[i]], mix(CYAN, NAVY, 0.28 + 0.17 * math.sin(TAU * i / n))))
    faces.append((front, CYAN))
    glass = [local(5.5 * math.cos(TAU * i / n), 20.2, 5.5 * math.sin(TAU * i / n)) for i in range(n)]
    faces.append((glass, mix(CYAN, WHITE, 0.76)))
    return position, local(0, 22, 0), faces


def scenario(t):
    out = Surface()
    out.shadow((66, 281, 322, 321), 26, 12)
    floor = [(-78, -78, -9), (78, -78, -9), (78, 78, -9), (-78, 78, -9)]
    draw_faces(out, prism_faces(floor, 12, [mix(NAVY, WHITE, 0.97), mix(PURPLE, WHITE, 0.85), mix(CYAN, WHITE, 0.85), mix(NAVY, WHITE, 0.88)]), mix(NAVY, WHITE, 0.64), 1.3)
    ring = [(88 * math.cos(i * TAU / 100), 88 * math.sin(i * TAU / 100), 4) for i in range(101)]
    for i in range(0, 100, 5):
        out.line([project(p) for p in ring[i:i + 3]], mix(NAVY, WHITE, 0.86), 1.5)
    cameras = [camera_geometry(TAU * t + TAU * i / 3 + 0.35, t, i) for i in range(3)]
    target = (0, 0, 85)

    def rays(camera, index):
        position, lens, _ = camera
        col = CYAN if index != 1 else PURPLE
        a = project(lens)
        b = project(target)
        layer = Surface(transparent=True)
        # Faint optical cone and a crisp common-target ray.
        tip1 = project((-15, -14, 85))
        tip2 = project((15, 14, 85))
        layer.polygon([a, tip1, tip2], rgba(col, 15))
        layer.line([a, b], rgba(col, 95), 1.6)
        for k in range(2):
            q = (2 * t + k / 2 + index * 0.1) % 1
            x, y = (a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q)
            layer.ellipse((x - 2.2, y - 2.2, x + 2.2, y + 2.2), rgba(col, 200))
        out.composite(layer)

    ordered = sorted(enumerate(cameras), key=lambda item: item[1][0][0] + item[1][0][1])
    for i, camera in ordered:
        if camera[0][0] + camera[0][1] < 0:
            rays(camera, i)
            draw_faces(out, camera[2], width=1)
    scene_cube(out, t)
    for i, camera in ordered:
        if camera[0][0] + camera[0][1] >= 0:
            rays(camera, i)
            draw_faces(out, camera[2], width=1)
    return out.finish()


ROAD_SEGMENTS = [
    ((185, 76), (248, 54), (307, 101), (299, 157)),
    ((299, 157), (291, 213), (273, 282), (213, 292)),
    ((213, 292), (154, 304), (72, 263), (72, 202)),
    ((72, 202), (72, 146), (113, 98), (185, 76)),
]


class ArcPath:
    """Length parameterization gives the car a constant physical speed."""

    def __init__(self, points, closed=False):
        self.points = points
        self.closed = closed
        self.lengths = [0.0]
        for a, b in zip(points, points[1:]):
            self.lengths.append(self.lengths[-1] + math.dist(a, b))
        self.total = self.lengths[-1]

    def point(self, fraction):
        import bisect

        fraction = fraction % 1 if self.closed else max(0, min(1, fraction))
        length = fraction * self.total
        i = min(len(self.points) - 2, max(0, bisect.bisect_right(self.lengths, length) - 1))
        a, b = self.points[i:i + 2]
        q = (length - self.lengths[i]) / max(1e-9, self.lengths[i + 1] - self.lengths[i])
        return (a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q)

    def part(self, start, end, steps=60):
        return [self.point(start + (end - start) * i / steps) for i in range(steps + 1)]


ROAD = ArcPath(bezier(ROAD_SEGMENTS, 100), True)


def top_car(out, position, angle):
    x, y = position
    # Local +y points rearward, local -y points in the direction of travel.
    forward = (math.cos(angle), math.sin(angle))
    right = (-forward[1], forward[0])

    def p(points, offset=(0, 0)):
        return [(x + 1.1 * (u * right[0] - v * forward[0]) + offset[0], y + 1.1 * (u * right[1] - v * forward[1]) + offset[1]) for u, v in points]

    outline = [(-8, -25), (8, -25), (13, -17), (13, 18), (9, 25), (-9, 25), (-13, 18), (-13, -17)]
    shadow = Surface(transparent=True)
    shadow.polygon(p(outline, (2, 5)), rgba(NAVY, 65))
    out.composite(shadow.image.filter(ImageFilter.GaussianBlur(3.8 * SCALE)))
    for side in (-1, 1):
        for v in (-14, 13):
            out.polygon(p([(side * 11, v - 5), (side * 15, v - 5), (side * 15, v + 5), (side * 11, v + 5)]), NAVY)
    out.polygon(p(outline), NAVY)
    out.polygon(p([(-7, -22), (7, -22), (10, -16), (10, 17), (7, 22), (-7, 22), (-10, 17), (-10, -16)]), mix(CYAN, WHITE, 0.2))
    out.polygon(p([(-7, -8), (7, -8), (8, 10), (-8, 10)]), mix(CYAN, WHITE, 0.87))
    out.polygon(p([(-8, -16), (8, -16), (7, -8), (-7, -8)]), NAVY)
    out.polygon(p([(-8, 11), (8, 11), (7, 17), (-7, 17)]), mix(NAVY, WHITE, 0.25))
    out.line(p([(-7, -20), (7, -20)]), WHITE, 2.1)
    out.line(p([(-6, 20), (-2, 20)]), PURPLE, 2)
    out.line(p([(2, 20), (6, 20)]), PURPLE, 2)
    out.line(p([(-9, -6), (-9, 8)]), mix(CYAN, WHITE, 0.8), 1.2)


def policy(t):
    out = Surface()
    road_points = ROAD.points
    shadow = Surface(transparent=True)
    shadow.line([(x, y + 5) for x, y in road_points], rgba(NAVY, 19), 43)
    out.composite(shadow.image.filter(ImageFilter.GaussianBlur(5 * SCALE)))
    out.line(road_points, mix(NAVY, WHITE, 0.76), 45)
    out.line(road_points, mix(NAVY, WHITE, 0.965), 40)
    # The inner island and beveled roadside objects make this a place, not a plot.
    island = bezier([
        ((173, 134), (205, 111), (248, 129), (248, 171)),
        ((248, 171), (247, 220), (224, 247), (190, 246)),
        ((190, 246), (145, 242), (119, 219), (124, 186)),
        ((124, 186), (127, 162), (145, 145), (173, 134)),
    ])
    out.polygon(island, mix(CYAN, WHITE, 0.97), mix(CYAN, WHITE, 0.87), 1.5)
    # A low, abstract block inside the bend, intentionally secondary to the car.
    out.shadow((157, 199, 226, 218), 14, 5)
    out.polygon([(164, 178), (200, 167), (219, 181), (182, 194)], mix(CYAN, WHITE, 0.76), mix(CYAN, WHITE, 0.42), 1.3)
    out.polygon([(164, 178), (182, 194), (182, 207), (164, 191)], mix(CYAN, WHITE, 0.53))
    out.polygon([(182, 194), (219, 181), (219, 194), (182, 207)], mix(CYAN, WHITE, 0.86))
    for i in range(36):
        start = i / 36
        out.line(ROAD.part(start, start + 0.012, 5), mix(NAVY, WHITE, 0.76), 1.5)
    q = (t + 0.075) % 1
    # Continuous observed history, segmented prediction, both tied to the vehicle.
    for i in range(24):
        a = q - 0.29 + i * 0.29 / 24
        b = a + 0.29 / 24 + 0.001
        out.stroke(ROAD.part(a, b, 5), mix(CYAN, WHITE, 0.82 - 0.79 * (i / 23)), 7)
    for i in range(8):
        a = q + 0.055 + i * 0.027
        out.stroke(ROAD.part(a, a + 0.013, 5), mix(PURPLE, WHITE, 0.19 + 0.56 * i / 8), 4.6)
    a, b = ROAD.point(q - 0.003), ROAD.point(q + 0.003)
    top_car(out, ROAD.point(q), math.atan2(b[1] - a[1], b[0] - a[0]))
    return out.finish()


BEAM_PATHS = [
    ArcPath(bezier([((48, 170), (95, 166), (124, 156), (180, 164)), ((180, 164), (217, 169), (241, 176), (273, 171))], 100)),
    ArcPath(bezier([((48, 247), (105, 249), (128, 213), (180, 205)), ((180, 205), (217, 200), (241, 178), (273, 171))], 100)),
]


def gate_arc(out, cx, cy, rx, ry, start, end, color, width):
    points = [(cx + rx * math.cos(start + (end - start) * i / 80), cy + ry * math.sin(start + (end - start) * i / 80)) for i in range(81)]
    out.stroke(points, color, width)


def shield(out, activation):
    points = [(235, 122), (277, 103), (319, 122), (316, 179), (302, 211), (277, 233), (252, 211), (238, 179)]
    out.polygon([(x + 8, y + 6) for x, y in points], mix(PURPLE, WHITE, 0.76), mix(PURPLE, WHITE, 0.35), 2)
    out.gradient((235, 103, 320, 234), mix(CYAN, WHITE, 0.92), mix(CYAN, WHITE, 0.64), polygon=points)
    out.line(points, NAVY, 4.2, closed=True)
    inner = [(245, 128), (277, 113), (309, 128), (306, 178), (295, 203), (277, 220), (259, 203), (248, 178)]
    out.line(inner, mix(CYAN, WHITE, 0.42), 1.5, closed=True)
    out.polygon([(277, 114), (308, 128), (305, 177), (277, 218)], mix(CYAN, WHITE, 0.81))
    out.polygon([(246, 129), (277, 114), (277, 218), (259, 202), (249, 177)], mix(CYAN, WHITE, 0.9))
    out.glow((276, 169), CYAN, 19 + 5 * activation, 25 + 55 * activation)
    # A generously sized physical check motif, not a measurement or progress bar.
    check = [(258, 167), (271, 180), (295, 151)]
    out.stroke([(x + 1, y + 2) for x, y in check], mix(CYAN, NAVY, 0.35), 8)
    out.stroke(check, mix(CYAN, WHITE, 0.08 + activation * 0.22), 6.5)
    out.line([(242, 127), (273, 113)], WHITE, 2)


def testing(t):
    out = Surface()
    q0 = (t + 0.08) % 1
    # Two paired packets per cycle; gate opening and shield receipt are coupled.
    crossing = sum(math.exp(-(((q0 + k * 0.5 - 0.56 + 0.5) % 1 - 0.5) / 0.075) ** 2) for k in range(2))
    received = sum(math.exp(-(((q0 + k * 0.5 - 0.91 + 0.5) % 1 - 0.5) / 0.09) ** 2) for k in range(2))
    breathe = 1 + 0.045 * math.sin(2 * TAU * t) + 0.055 * crossing
    cx, cy = 179, 184
    rx, ry = 25 * breathe, 73 * breathe
    out.shadow((128, 263, 220, 290), 22, 7)
    out.shadow((223, 255, 338, 287), 22, 8)
    out.polygon([(225, 260), (278, 243), (333, 261), (279, 280)], mix(NAVY, WHITE, 0.95), mix(NAVY, WHITE, 0.83), 1.5)
    # Back half of the thick oval portal is behind the incoming packets.
    gate_arc(out, cx + 5, cy + 2, rx, ry, -math.pi / 2, math.pi / 2, mix(PURPLE, WHITE, 0.5), 11)
    gate_arc(out, cx, cy, rx, ry, -math.pi / 2, math.pi / 2, NAVY, 9)
    gate_arc(out, cx - 1, cy, rx - 1, ry - 1, -math.pi / 2, math.pi / 2, mix(CYAN, WHITE, 0.34), 4)
    for i, path in enumerate(BEAM_PATHS):
        col = CYAN if i == 0 else PURPLE
        out.stroke(path.points, mix(col, WHITE, 0.89), 11)
        out.stroke(path.points, mix(col, WHITE, 0.36), 2.8)
        x, y = path.point(0)
        out.ellipse((x - 7, y - 7, x + 7, y + 7), WHITE, mix(col, WHITE, 0.22), 2.7)
        out.ellipse((x - 3, y - 3, x + 3, y + 3), col)
        for k in range(2):
            q = (q0 + k / 2) % 1
            fade = smoothstep(q / 0.07) * smoothstep((1 - q) / 0.06)
            if fade < 0.01:
                continue
            for j in range(12):
                a = max(0, q - 0.105 + j * 0.105 / 12)
                b = max(0, q - 0.105 + (j + 1) * 0.105 / 12)
                out.stroke(path.part(a, b, 3), mix(col, WHITE, 1 - fade * (0.12 + 0.88 * j / 11)), 7.2)
            x, y = path.point(q)
            out.ellipse((x - 4.2, y - 4.2, x + 4.2, y + 4.2), mix(col, WHITE, 0.15))
            out.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), WHITE)
    # Front half occludes the beams for an unambiguous through-the-gate action.
    gate_arc(out, cx + 5, cy + 2, rx, ry, math.pi / 2, 3 * math.pi / 2, mix(PURPLE, WHITE, 0.5), 11)
    gate_arc(out, cx, cy, rx, ry, math.pi / 2, 3 * math.pi / 2, NAVY, 9)
    gate_arc(out, cx - 1, cy, rx - 1, ry - 1, math.pi / 2, 3 * math.pi / 2, mix(CYAN, WHITE, 0.24 + 0.3 * crossing), 4)
    shield(out, received)
    return out.finish()


def shared_palette():
    # One deterministic palette for every frame and every icon. No error-diffusion
    # dithering: this avoids crawling noise in the pale glass and soft shadows.
    colors = [WHITE]
    for a, b, count in [
        (NAVY, WHITE, 63), (CYAN, WHITE, 64), (PURPLE, WHITE, 48),
        (NAVY, CYAN, 40), (NAVY, PURPLE, 40),
    ]:
        colors.extend(mix(a, b, i / (count - 1)) for i in range(count))
    palette = Image.new("P", (1, 1))
    palette.putpalette([channel for color in colors for channel in color])
    return palette


RENDERERS = [
    ("observation-quality", observation),
    ("scenario-fidelity", scenario),
    ("policy-response", policy),
    ("testing-effectiveness", testing),
]


def render(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = shared_palette()
    for name, renderer in RENDERERS:
        frames = [renderer(i / FRAMES).quantize(palette=palette, dither=Image.Dither.NONE) for i in range(FRAMES)]
        path = output_dir / f"{name}.gif"
        frames[0].save(
            path, save_all=True, append_images=frames[1:], duration=DURATION,
            # Delta-frame transparency reduces bytes; the supplied global
            # palette still fixes the RGB value of every color in every frame.
            loop=0, disposal=1, optimize=True, palette=palette,
        )
        print(f"{path}: {SIZE}x{SIZE}, {FRAMES} frames, {FRAMES * DURATION} ms, {path.stat().st_size:,} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    render(args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
