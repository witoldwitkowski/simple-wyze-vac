"""One look for robot vacuum maps: an SVG drawn from a floor grid, rooms and a path.

The same file is used by the shark2mqtt fork and the simple-wyze-vac fork, so
their maps match; change both copies together. Standard library only.

Every picture has the same layout and the same width, so text and line sizes
match between vacuums: a title saying what you are looking at, the floor plan,
and a key. Map coordinates are grid cells, already turned the way they are shown
(x to the right, y down).

The swept area (a soft band as wide as the vacuum cleans) shows where a run
cleaned; the thin line on top is the route it took.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable
from xml.sax.saxutils import escape

Point = tuple[float, float]

# cell classes in Scene.cells
NOTHING, FLOOR, WALL, OBSTACLE = 0, 1, 2, 3

WIDTH = 1000
MARGIN = 40
HEADER = 142
MAP_WIDTH = 936
MAP_MIN_HEIGHT = 420
MAP_MAX_HEIGHT = 880
SWEPT_METRES = 0.30
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

PAGE = "#ffffff"
INK = "#0f172a"
MUTED = "#5b6b82"
EDGE = "#dbe2ea"
FLOOR_COLOUR = "#e9eef4"
WALL_COLOUR = "#52627a"
OBSTACLE_COLOUR = "#b6c2d1"
ACCENT = "#1d5fd6"
SWEPT = "#4b8cf5"
NO_GO = "#d6264c"
STATE_COLOURS = {"docked": "#94a3b8", "waiting": ACCENT, "cleaning": ACCENT,
                 "returning": "#c26a05", "paused": "#c26a05", "error": NO_GO}


@dataclass
class Room:
    name: str
    label: Point                 # where the name is written
    outline: str = ""            # SVG path data of the room's area
    hidden: bool = False         # rooms that only group others are not labelled


@dataclass
class Scene:
    width: int
    height: int
    cells: bytes                 # NOTHING / FLOOR / WALL / OBSTACLE, row by row
    cell_metres: float
    rooms: list[Room] = field(default_factory=list)
    zones: list[list[Point]] = field(default_factory=list)      # no-go areas
    barriers: list[list[Point]] = field(default_factory=list)   # no-go lines
    dock: Point | None = None
    path: list[list[Point]] = field(default_factory=list)       # recorded strokes
    path_picture: str = ""       # or the recorded path as a base64 PNG ...
    path_box: tuple[float, float, float, float] = (0, 0, 0, 0)  # ... covering this area
    swept: str = ""              # SVG path data of the swept area of that picture
    trail: list[Point] = field(default_factory=list)            # estimated route
    samples: list[Point] = field(default_factory=list)          # positions reported
    robot: Point | None = None


@dataclass
class Caption:
    state: str                   # a key of STATE_COLOURS
    title: str
    detail: str = ""


# --- words ---------------------------------------------------------------------

def room_list(names: Iterable[str]) -> str:
    names = list(names)
    if len(names) > 3:
        return f"{names[0]}, {names[1]} + {len(names) - 2} more"
    return ", ".join(names)


def when(moment: datetime) -> str:
    """A date and time as written under the title, e.g. "Wed, Nov 25 · 7:22 AM"."""
    hour = moment.hour % 12 or 12
    return f"{moment:%a}, {moment:%b} {moment.day} · {hour}:{moment:%M} {'PM' if moment.hour >= 12 else 'AM'}"


def main_rooms(counts: dict[str, int], total: int, share: float = 0.2) -> list[str]:
    """The rooms a path was mostly in, busiest first.

    ``counts`` holds path points per room out of ``total``. Rooms only passed
    through on the way stay below ``share``.
    """
    if total <= 0 or not counts:
        return []
    ranked = sorted(counts, key=lambda name: -counts[name])
    chosen = [name for name in ranked if counts[name] / total >= share]
    return chosen or ranked[:1]


def text_width(text: str, size: float, bold: bool = True) -> float:
    """A good estimate of a line's width in a sans-serif font."""
    units = 0.0
    for char in text:
        if char in "iIjl.,:;'|!":
            units += 0.28
        elif char in "ftr()[]- ·":
            units += 0.35
        elif char in "mMW@":
            units += 0.86
        elif char in "wOQGDHNU&%²":
            units += 0.73
        elif char.isupper():
            units += 0.65
        else:
            units += 0.56
    return units * size * (1.04 if bold else 1.0)


# --- grid ----------------------------------------------------------------------

def run_path(cells: bytes, width: int, value: int) -> str:
    """SVG path covering every cell holding ``value``, as horizontal runs."""
    parts = []
    for row in range(len(cells) // width):
        line = cells[row * width:(row + 1) * width]
        col = line.find(value)
        while col >= 0:
            end = col + 1
            while end < width and line[end] == value:
                end += 1
            parts.append(f"M{col} {row}h{end - col}v1h-{end - col}z")
            col = line.find(value, end)
    return "".join(parts)


def _extent(scene: Scene) -> tuple[float, float, float, float]:
    rows = [r for r in range(scene.height)
            if any(scene.cells[r * scene.width:(r + 1) * scene.width])]
    if not rows:
        return 0, 0, scene.width, scene.height
    left, right = scene.width, 0
    for r in rows:
        line = scene.cells[r * scene.width:(r + 1) * scene.width]
        left = min(left, len(line) - len(line.lstrip(b"\x00")))
        right = max(right, len(line.rstrip(b"\x00")))
    x0, y0, x1, y1 = left, rows[0], right, rows[-1] + 1
    for points in [*scene.zones, *scene.barriers]:
        for x, y in points:
            x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
    return x0 - 1, y0 - 1, x1 + 1, y1 + 1


def _points(points: Iterable[Point]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _line(points: list[Point]) -> str:
    return "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in points)


# --- picture -------------------------------------------------------------------

def _dock(x: float, y: float, r: float) -> str:
    bolt = [(0.12, -0.62), (-0.38, 0.1), (-0.02, 0.1), (-0.14, 0.62), (0.38, -0.12), (0.02, -0.12)]
    return (f'<g class="dock" transform="translate({x:.1f} {y:.1f})"><circle r="{r}" fill="{INK}" '
            f'stroke="{PAGE}" stroke-width="{r * 0.22:.1f}"/><polygon fill="{PAGE}" '
            f'points="{_points((px * r, py * r) for px, py in bolt)}"/></g>')


def _robot(x: float, y: float, r: float) -> str:
    return (f'<g class="robot" transform="translate({x:.1f} {y:.1f})"><circle class="pulse" r="{r * 2:.1f}" '
            f'fill="{ACCENT}" fill-opacity="0.2"/><circle r="{r}" fill="{ACCENT}" stroke="{PAGE}" '
            f'stroke-width="{r * 0.28:.1f}"/><circle r="{r * 0.32:.1f}" fill="{PAGE}"/></g>')


def render(
    scene: Scene,
    caption: Caption,
    *,
    cleaned: Iterable[str] = (),
    cleaning: Iterable[str] = (),
) -> str:
    """The map as an SVG.

    ``cleaned`` names the rooms of a finished run (their labels are filled in);
    ``cleaning`` names rooms being cleaned now (their area is also marked).
    """
    cleaned = {name.lower() for name in cleaned}
    cleaning = {name.lower() for name in cleaning}
    x0, y0, x1, y1 = _extent(scene)
    scale = min(MAP_WIDTH / (x1 - x0), MAP_MAX_HEIGHT / (y1 - y0))
    map_height = (y1 - y0) * scale
    area_height = max(map_height, MAP_MIN_HEIGHT)
    ox = (WIDTH - (x1 - x0) * scale) / 2 - x0 * scale
    oy = HEADER + (area_height - map_height) / 2 - y0 * scale
    key_y = HEADER + area_height + 54
    height = key_y + 44

    def at(point: Point) -> Point:
        return ox + point[0] * scale, oy + point[1] * scale

    def unit(canvas: float) -> float:
        """A length given in picture units, expressed in cells."""
        return canvas / scale

    band = SWEPT_METRES / scene.cell_metres
    floor = run_path(scene.cells, scene.width, FLOOR)
    walls = run_path(scene.cells, scene.width, WALL)
    obstacles = run_path(scene.cells, scene.width, OBSTACLE)
    marked = [r for r in scene.rooms if r.name.lower() in cleaning]
    strokes = [s for s in scene.path if len(s) > 1]
    has_path = bool(strokes or scene.path_picture)
    has_swept = bool(strokes or scene.swept)
    colour = STATE_COLOURS.get(caption.state, STATE_COLOURS["docked"])

    region = (f'filterUnits="userSpaceOnUse" x="{x0 - 4:.0f}" y="{y0 - 4:.0f}" '
              f'width="{x1 - x0 + 8:.0f}" height="{y1 - y0 + 8:.0f}" color-interpolation-filters="sRGB"')
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:.0f}" width="{WIDTH}" '
        f'height="{height:.0f}" font-family="{FONT}">',
        '<style>.pulse{transform-box:fill-box;transform-origin:center;animation:pulse 2.4s ease-in-out infinite}'
        '@keyframes pulse{0%,100%{transform:scale(.72);opacity:.45}50%{transform:scale(1.12);opacity:1}}'
        '@media (prefers-reduced-motion:reduce){.pulse{animation:none}}</style>',
        f'<title>{escape(caption.title)}. {escape(caption.detail)}</title>',
        '<defs>',
        # rounds off the square grid cells without losing one-cell walls
        f'<filter id="soft" {region}><feGaussianBlur stdDeviation="0.5"/><feComponentTransfer>'
        '<feFuncA type="linear" slope="5" intercept="-1.7"/></feComponentTransfer></filter>',
        f'<pattern id="nogo" width="{unit(15):.2f}" height="{unit(15):.2f}" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)"><rect width="100%" height="100%" fill="{NO_GO}" fill-opacity="0.08"/>'
        f'<rect width="{unit(4):.2f}" height="100%" fill="{NO_GO}" fill-opacity="0.4"/></pattern>',
        f'<clipPath id="floor"><path d="{floor}{obstacles}"/></clipPath>',
        '</defs>',
        f'<rect width="100%" height="100%" fill="{PAGE}"/>',
        # header
        f'<circle class="{"pulse" if caption.state in ("cleaning", "waiting") else "state"}" cx="{MARGIN + 11}" '
        f'cy="52" r="11" fill="{colour}"/>',
    ]
    available = WIDTH - 2 * MARGIN - 38
    title_size = min(46.0, 46.0 * available / max(text_width(caption.title, 46), 1))
    out.append(f'<text class="title" x="{MARGIN + 38}" y="68" font-size="{title_size:.0f}" font-weight="700" '
               f'fill="{INK}">{escape(caption.title)}</text>')
    if caption.detail:
        size = min(29.0, 29.0 * available / max(text_width(caption.detail, 29, False), 1))
        out.append(f'<text class="detail" x="{MARGIN + 38}" y="112" font-size="{size:.0f}" fill="{MUTED}">'
                   f'{escape(caption.detail)}</text>')

    # floor plan
    out.append(f'<g class="plan" transform="translate({ox:.2f} {oy:.2f}) scale({scale:.4f})">')
    out.append(f'<path d="{floor}{obstacles}" fill="{FLOOR_COLOUR}" filter="url(#soft)"/>')
    for room in marked:
        # clipped to the floor: a room's outline is a rough box that can reach past its walls
        out.append(f'<g clip-path="url(#floor)"><path class="cleaning" d="{room.outline}" fill="{SWEPT}" '
                   f'fill-opacity="0.24" stroke="{ACCENT}" stroke-width="{unit(5):.2f}" '
                   f'stroke-dasharray="{unit(9):.2f} {unit(7):.2f}" stroke-linejoin="round"/></g>')
    if has_swept:
        # one group opacity, so crossing strokes do not darken
        out.append('<g class="swept" opacity="0.3">')
        if strokes:
            out.append(f'<path d="{"".join(_line(s) for s in strokes)}" fill="none" stroke="{SWEPT}" '
                       f'stroke-width="{band:.2f}" stroke-linecap="round" stroke-linejoin="round"/>')
        if scene.swept:
            out.append(f'<path d="{scene.swept}" fill="{SWEPT}" filter="url(#soft)"/>')
        out.append("</g>")
    out.append(f'<path d="{obstacles}" fill="{OBSTACLE_COLOUR}" filter="url(#soft)"/>')
    out.append(f'<path d="{walls}" fill="{WALL_COLOUR}" filter="url(#soft)"/>')
    for zone in scene.zones:
        out.append(f'<polygon class="nogo" points="{_points(zone)}" fill="url(#nogo)" stroke="{NO_GO}" '
                   f'stroke-width="{unit(2.5):.2f}" stroke-linejoin="round"/>')
    for barrier in scene.barriers:
        out.append(f'<polyline class="nogo" points="{_points(barrier)}" fill="none" stroke="{NO_GO}" '
                   f'stroke-width="{unit(5):.2f}" stroke-linecap="round"/>')
    if strokes:
        out.append(f'<path class="path" d="{"".join(_line(s) for s in strokes)}" fill="none" stroke="{ACCENT}" '
                   f'stroke-width="{unit(2.4):.2f}" stroke-linecap="round" stroke-linejoin="round"/>')
    if scene.path_picture:
        x, y, w, h = scene.path_box
        out.append(f'<image class="path" x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                   f'preserveAspectRatio="none" href="data:image/png;base64,{scene.path_picture}"/>')
    if len(scene.trail) > 1:
        out.append(f'<path class="trail" d="{_line(scene.trail)}" fill="none" stroke="{ACCENT}" '
                   f'stroke-width="{unit(5):.2f}" stroke-dasharray="{unit(12):.2f} {unit(10):.2f}" '
                   'stroke-linecap="round" stroke-linejoin="round"/>')
    for x, y in scene.samples:
        out.append(f'<circle class="sample" cx="{x:.1f}" cy="{y:.1f}" r="{unit(4.5):.2f}" fill="{ACCENT}" '
                   f'stroke="{PAGE}" stroke-width="{unit(1.6):.2f}"/>')
    out.append("</g>")

    placed: list[tuple[float, float, float, float]] = []   # centre x, y, half width, half height
    if scene.dock:
        x, y = at(scene.dock)
        out.append(_dock(x, y, 19))
        placed.append((x, y, 19, 19))
    if scene.robot:
        x, y = at(scene.robot)
        out.append(_robot(x, y, 18))
        placed.append((x, y, 20, 20))

    # room names as chips, nudged down when they would cover one another
    for room in scene.rooms:
        if room.hidden:
            continue
        size, half_h = 27.0, 23.0
        half_w = text_width(room.name, size) / 2 + 17
        x, y = at(room.label)
        x = min(max(x, half_w + 8), WIDTH - half_w - 8)
        for _ in range(12):
            if not any(abs(x - px) < half_w + pw + 4 and abs(y - py) < half_h + ph + 3 for px, py, pw, ph in placed):
                break
            y += half_h
        placed.append((x, y, half_w, half_h))
        on = room.name.lower() in cleaned or room.name.lower() in cleaning
        fill, stroke, text = (ACCENT, PAGE, PAGE) if on else (PAGE, EDGE, INK)
        out.append(f'<g class="{"room on" if on else "room"}"><rect x="{x - half_w:.1f}" y="{y - half_h:.1f}" '
                   f'width="{2 * half_w:.1f}" height="{2 * half_h:.1f}" rx="{half_h:.0f}" fill="{fill}" '
                   f'fill-opacity="0.95" stroke="{stroke}" stroke-width="2"/>'
                   f'<text x="{x:.1f}" y="{y + size * 0.35:.1f}" text-anchor="middle" font-size="{size:.0f}" '
                   f'font-weight="600" fill="{text}">{escape(room.name)}</text></g>')

    # key: only what is in the picture
    entries: list[tuple[str, str]] = []
    if has_swept:
        entries.append(("Cleaned", f'<rect y="-12" width="34" height="24" rx="7" fill="{SWEPT}" fill-opacity="0.3"/>'))
    if has_path:
        entries.append(("Path", f'<path d="M2 5Q12 -13 18 0T32 -5" fill="none" stroke="{ACCENT}" '
                                'stroke-width="3.5" stroke-linecap="round"/>'))
    if marked:
        entries.append(("Cleaning", f'<rect y="-12" width="34" height="24" rx="5" fill="{SWEPT}" fill-opacity="0.24" '
                                    f'stroke="{ACCENT}" stroke-width="2.5" stroke-dasharray="7 5"/>'))
    if scene.robot:
        entries.append(("Vacuum", _robot(17, 0, 11).replace('class="pulse" ', "")))
    if len(scene.trail) > 1:
        entries.append(("Likely route", f'<path d="M3 0H31" stroke="{ACCENT}" stroke-width="5" '
                                        'stroke-dasharray="8 8" stroke-linecap="round"/>'))
    if scene.zones or scene.barriers:
        entries.append(("No-go", f'<rect y="-12" width="34" height="24" rx="5" fill="{NO_GO}" fill-opacity="0.16" '
                                 f'stroke="{NO_GO}" stroke-width="2.5"/>'
                                 f'<path d="M6 8L18 -8M17 8L29 -8" stroke="{NO_GO}" stroke-width="3"/>'))
    if scene.dock:
        entries.append(("Dock", _dock(17, 0, 14).replace('class="dock" ', "")))
    widths = [44 + text_width(label, 25, False) for label, _ in entries]
    room_for_gaps = WIDTH - 2 * MARGIN - sum(widths)
    gap = min(36.0, room_for_gaps / max(len(entries) - 1, 1))
    x = float(MARGIN)
    out.append(f'<path d="M{MARGIN} {key_y - 38:.0f}H{WIDTH - MARGIN}" stroke="{EDGE}" stroke-width="2"/>')
    for (label, swatch), width in zip(entries, widths):
        out.append(f'<g class="key" transform="translate({x:.1f} {key_y:.0f})">{swatch}'
                   f'<text x="44" y="9" font-size="25" fill="{MUTED}">{label}</text></g>')
        x += width + gap
    out.append("</svg>")
    return "".join(out)


def distance(a: Point, b: Point) -> float:
    return math.dist(a, b)
