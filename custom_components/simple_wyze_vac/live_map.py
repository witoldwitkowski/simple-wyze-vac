"""Live map: draw the current map with the vacuum's position and the path seen so far.

Wyze only reports the newest position, so the path is built from the samples collected while a run is active.
Requests: while cleaning or returning, a status and a position request every ten seconds; while paused, the
same once a minute; while idle, one status request every five minutes. A command from Home Assistant makes it
look at once. A rate limit from Wyze stops all of it for ten minutes.
"""
import asyncio
import base64
import heapq
import io
import logging
import time
import zlib
from datetime import timedelta

import blackboxprotobuf
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from homeassistant.components.vacuum.const import VacuumActivity
from homeassistant.helpers.event import async_track_time_interval

from wyze_sdk.models.devices import VacuumMode
from wyze_sdk.models.devices.vacuums import VacuumMap

from .wyze_errors import is_rate_limited, is_token_expired

_LOGGER = logging.getLogger(__name__)

TICK = timedelta(seconds=10)
IDLE_CHECK_SECONDS = 300
PAUSED_CHECK_SECONDS = 60
ERROR_PAUSE_SECONDS = 60
EAGER_SECONDS = 120  # keep checking every tick for this long after a command from Home Assistant
RATE_LIMIT_PAUSE_SECONDS = 600
RENEW_RETRY_SECONDS = 900
MAX_PATH_POINTS = 3000
MOVING = (VacuumActivity.CLEANING, VacuumActivity.RETURNING)
ACTIVE = MOVING + (VacuumActivity.PAUSED,)  # a run is under way

SCALE = 5
SMOOTH = 2  # shapes are drawn this much larger, then scaled down for soft edges
MARGIN = 10  # map cells around the floor
LEGEND_HEIGHT = 90
# pale floor tints; blue is kept for the vacuum and red for no-go areas
ROOM_COLORS = [(209, 239, 228), (252, 236, 196), (225, 221, 246), (249, 224, 234),
               (219, 237, 205), (252, 224, 211), (214, 236, 240), (233, 230, 220)]
OUTLINE = (112, 128, 144)
INK = (36, 52, 71)
VACUUM = (28, 92, 171)
NO_GO = (201, 52, 52)
DOCK = (43, 58, 74)


def activity_from_mode(mode):
    """Translate a Wyze vacuum mode into a Home Assistant activity."""
    if mode in [VacuumMode.SWEEPING, VacuumMode.CLEANING, VacuumMode.QUICK_MAPPING_MAPPING]:
        return VacuumActivity.CLEANING
    if mode in [VacuumMode.IDLE, VacuumMode.BREAK_POINT, VacuumMode.DOCKED_NOT_COMPLETE,
                VacuumMode.QUICK_MAPPING_DOCKED_NOT_COMPLETE]:
        return VacuumActivity.DOCKED
    if mode in [VacuumMode.ON_WAY_CHARGE, VacuumMode.FULL_FINISH_SWEEPING_ON_WAY_CHARGE,
                VacuumMode.FINISHED_RETURNING_TO_CHARGE, VacuumMode.RETURNING_TO_CHARGE,
                VacuumMode.QUICK_MAPPING_COMPLETED_RETURNING_TO_CHARGE]:
        return VacuumActivity.RETURNING
    if mode in [VacuumMode.PAUSED, VacuumMode.PAUSE, VacuumMode.QUICK_MAPPING_PAUSED]:
        return VacuumActivity.PAUSED
    return VacuumActivity.ERROR


def _items(value):
    """A repeated protobuf field with one entry is decoded as that entry, not as a list."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _text(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _smooth(mask, size, threshold=127):
    """Scale a one-pixel-per-cell mask up to the picture size with rounded, antialiased edges."""
    large = mask.resize((size[0] * SMOOTH, size[1] * SMOOTH), Image.Resampling.BILINEAR)
    return large.point(lambda v: 255 if v > threshold else 0).resize(size, Image.Resampling.LANCZOS)


def _label_cell(mask):
    """The cell deepest inside a room, so the name sits clear of walls and markers."""
    while True:
        smaller = mask.filter(ImageFilter.MinFilter(3))
        if not smaller.getbbox():
            break
        mask = smaller
    box = mask.getbbox()
    if not box:
        return None
    middle = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    if mask.getpixel(middle):
        return middle
    index = next(i for i, v in enumerate(mask.tobytes()) if v)
    return (index % mask.width, index // mask.width)


def _draw_dock(draw, x, y, size):
    draw.ellipse((x - size, y - size, x + size, y + size), fill=DOCK, outline=(255, 255, 255), width=max(size // 6, 2))
    bolt = [(0.12, -0.62), (-0.38, 0.1), (-0.02, 0.1), (-0.14, 0.62), (0.38, -0.12), (0.02, -0.12)]
    draw.polygon([(x + dx * size, y + dy * size) for dx, dy in bolt], fill=(255, 255, 255))


def _draw_vacuum(draw, x, y, size):
    draw.ellipse((x - size * 1.9, y - size * 1.9, x + size * 1.9, y + size * 1.9), fill=VACUUM + (45,))
    draw.ellipse((x - size, y - size, x + size, y + size), fill=VACUUM, outline=(255, 255, 255), width=max(size // 4, 2))
    draw.ellipse((x - size * 0.32, y - size * 0.32, x + size * 0.32, y + size * 0.32), fill=(255, 255, 255))


def _draw_trail(draw, points, width, samples):
    """Dashes, because the line between two samples is only an estimate of the route."""
    dash, gap, offset = width * 2.4, width * 1.8, 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        position = -offset
        while position < length:
            start, end = max(position, 0), min(position + dash, length)
            if end > start:
                draw.line([(x1 + (x2 - x1) * t / length, y1 + (y2 - y1) * t / length) for t in (start, end)],
                          fill=VACUUM + (215,), width=width)
            position += dash + gap
        offset = (offset + length) % (dash + gap)
    for x, y in samples:
        draw.ellipse((x - width, y - width, x + width, y + width), fill=VACUUM, outline=(255, 255, 255), width=width // 3)


def _draw_no_go(image, points):
    """A hatched, outlined area."""
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    layer = Image.new("RGBA", image.size, NO_GO + (0,))
    draw = ImageDraw.Draw(layer)
    draw.rectangle((0, 0) + image.size, fill=NO_GO + (34,))
    for start in range(-image.height, image.width, 18):
        draw.line((start, image.height, start + image.height, 0), fill=NO_GO + (120,), width=4)
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), mask))
    image.alpha_composite(layer)
    ImageDraw.Draw(image).line(points + points[:1], fill=NO_GO + (255,), width=5, joint="curve")


class Router:
    """Estimates the route between two position samples: around walls, not through them."""

    def __init__(self, grid):
        self._width, self._height = grid.size
        blocked = grid.point(lambda v: 255 if v in (0, 1, 255) else 0)
        near = blocked.filter(ImageFilter.MaxFilter(3))
        self._blocked = blocked.tobytes()
        # walls stay passable at a high price, so a sample on an obstacle still gets a route
        self._cost = [40 if wall else 4 if close else 1 for wall, close in zip(self._blocked, near.tobytes())]
        self._cache = {}

    def _index(self, x, y):
        column = min(max(int(x), 0), self._width - 1)
        row = min(max(int(y), 0), self._height - 1)
        return row * self._width + column

    def clear(self, a, b):
        steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        return not any(self._blocked[self._index(a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps)]
                       for i in range(steps + 1))

    def between(self, a, b):
        """Waypoints (in cells) from a to b."""
        if (a, b) not in self._cache:
            self._cache[(a, b)] = [a, b] if self.clear(a, b) else self._search(a, b)
        return self._cache[(a, b)]

    def _search(self, a, b):
        width = self._width
        start, goal = self._index(*a), self._index(*b)
        goal_y, goal_x = divmod(goal, width)
        best, came, queue = {start: 0.0}, {}, [(0.0, start)]
        while queue:
            _, here = heapq.heappop(queue)
            if here == goal:
                break
            y, x = divmod(here, width)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < self._height):
                    continue
                there = ny * width + nx
                total = best[here] + self._cost[there] * (1.414 if dx and dy else 1.0)
                if total < best.get(there, float("inf")):
                    best[there], came[there] = total, here
                    heapq.heappush(queue, (total + ((nx - goal_x) ** 2 + (ny - goal_y) ** 2) ** 0.5, there))
        cells, here = [], goal
        while here != start and here in came:
            here = came[here]
            y, x = divmod(here, width)
            cells.append((x + 0.5, y + 0.5))
        cells = [a] + cells[::-1][1:] + [b]
        # keep only the corners that are needed
        route, i = [a], 0
        while i < len(cells) - 1:
            j = next((j for j in range(len(cells) - 1, i, -1) if self.clear(cells[i], cells[j])), i + 1)
            route.append(cells[j])
            i = j
        return route

    def curve(self, points):
        """Round the corners (Catmull-Rom), staying straight wherever a curve would cut a wall."""
        smooth = points[:1]
        for i in range(len(points) - 1):
            p0, p1, p2, p3 = points[max(i - 1, 0)], points[i], points[i + 1], points[min(i + 2, len(points) - 1)]
            piece = [tuple(0.5 * (2 * p1[k] + (p2[k] - p0[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t * t
                                  + (3 * p1[k] - p0[k] - 3 * p2[k] + p3[k]) * t ** 3) for k in (0, 1))
                     for t in (n / 8 for n in range(1, 9))]
            hits = any(not self.clear(u, v) for u, v in zip([p1] + piece, piece))
            smooth.extend([p2] if hits else piece)
        return smooth


class BaseMap:
    """The floor plan of one saved map, drawn once and reused for every frame."""

    def __init__(self, blob):
        raw = zlib.decompress(base64.b64decode(blob))
        data, _ = blackboxprotobuf.decode_message(raw, VacuumMap._robot_map_proto())
        head = data["mapHeadInfo_"]
        width, height = head["sizeX_"], head["sizeY_"]
        self._min_x, self._min_y = head["minX_"], head["minY_"]
        self._resolution = head["resolution_"]

        cells = data["mapData_"]["mapData_"]
        if isinstance(cells, str):
            cells = cells.encode("latin-1")
        rooms = sorted(_items(data.get("12")), key=lambda room: room["roomId_"])

        # every cell holds 0 (outside), 1 (obstacle), 255 (wall) or a room id
        grid = Image.frombytes("L", (width, height), cells)
        floor = grid.point(lambda v: 0 if v in (0, 1) else 255)
        box = floor.getbbox() or (0, 0, width, height)
        self._left = max(box[0] - MARGIN, 0)
        self._top = max(box[1] - MARGIN, 0)
        crop = (self._left, self._top, min(box[2] + MARGIN, width), min(box[3] + MARGIN, height))
        grid, floor = grid.crop(crop), floor.crop(crop)
        self._rows = grid.height
        size = (grid.width * SCALE, grid.height * SCALE)
        # close pinholes and interior wall lines so each room reads as one clean surface
        floor = floor.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))

        def flip(picture):  # map rows count upwards, picture rows count downwards
            return picture.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        plan = Image.new("RGBA", size, (0, 0, 0, 0))
        plan.paste(OUTLINE + (255,), mask=_smooth(flip(floor.filter(ImageFilter.MaxFilter(3))), size))
        shape = _smooth(flip(floor), size)
        plan.paste((238, 240, 242, 255), mask=shape)
        for index, room in enumerate(rooms):
            mask = grid.point(lambda v, wanted=room["roomId_"]: 255 if v == wanted else 0)
            grown = ImageChops.multiply(mask.filter(ImageFilter.MaxFilter(5)), floor)
            plan.paste(ROOM_COLORS[index % len(ROOM_COLORS)] + (255,),
                       mask=ImageChops.multiply(_smooth(flip(grown), size), shape))

        walls = grid.point(lambda v: 255 if v == 255 else 0)
        plan.paste(OUTLINE + (255,), mask=_smooth(flip(walls), size, 95))
        self._router = Router(grid)

        areas = []
        for area in _items(data.get("virtualWalls_")):
            points = [self.pixel(q["x_"], q["y_"]) for q in _items(area.get("points_"))]
            if len(points) > 2:
                _draw_no_go(plan, points)
                areas.append(points)
            elif len(points) == 2:
                ImageDraw.Draw(plan).line(points, fill=NO_GO + (255,), width=7)

        draw = ImageDraw.Draw(plan, "RGBA")
        dock = data.get("chargeStation_")
        dock_at = self.pixel(dock.get("x_", 0), dock.get("y_", 0)) if dock else None
        if dock_at:
            _draw_dock(draw, *dock_at, 22)

        font = ImageFont.load_default(34)
        for room in rooms:
            free = flip(grid.point(lambda v, wanted=room["roomId_"]: 255 if v == wanted else 0))
            blocked = ImageDraw.Draw(free)
            for points in areas:
                blocked.polygon([(x / SCALE, y / SCALE) for x, y in points], fill=0)
            if dock_at:
                x, y = dock_at[0] / SCALE, dock_at[1] / SCALE
                blocked.ellipse((x - 14, y - 14, x + 14, y + 14), fill=0)
            cell = _label_cell(free)
            if not cell:
                continue
            name = " & ".join(part.strip() for part in _text(room.get("roomName_", "")).split("&"))
            half = draw.textlength(name, font=font) / 2 + 20
            center = (min(max(cell[0] * SCALE, half + 4), size[0] - half - 4), cell[1] * SCALE)
            draw.rounded_rectangle((center[0] - half, center[1] - 28, center[0] + half, center[1] + 28), radius=28,
                                   fill=(255, 255, 255, 235))
            draw.text(center, name, font=font, fill=INK, anchor="mm")

        self.image = Image.new("RGBA", (size[0], size[1] + LEGEND_HEIGHT), (0, 0, 0, 0))
        self.image.alpha_composite(plan)
        self._legend(size)

    def _legend(self, size):
        draw = ImageDraw.Draw(self.image, "RGBA")
        font = ImageFont.load_default(28)
        entries = ["Vacuum", "Path", "Dock", "No-go area"]
        widths = [46 + draw.textlength(name, font=font) for name in entries]
        total = sum(widths) + 40 * (len(entries) - 1)
        x, y = (size[0] - total) / 2, size[1] + LEGEND_HEIGHT / 2
        draw.rounded_rectangle((x - 26, y - 32, x + total + 26, y + 32), radius=32, fill=(255, 255, 255, 245),
                               outline=(215, 221, 228, 255), width=2)
        for name, width in zip(entries, widths):
            if name == "Vacuum":
                _draw_vacuum(draw, x + 16, y, 12)
            elif name == "Path":
                _draw_trail(draw, [(x, y), (x + 34, y)], 6, [(x, y)])
            elif name == "Dock":
                _draw_dock(draw, x + 16, y, 16)
            else:
                draw.rectangle((x, y - 15, x + 32, y + 15), fill=NO_GO + (60,), outline=NO_GO + (255,), width=3)
                draw.line((x + 4, y + 11, x + 28, y - 11), fill=NO_GO + (200,), width=3)
            draw.text((x + 46, y), name, font=font, fill=INK, anchor="lm")
            x += width + 40

    def _cell(self, x, y):
        """Convert map metres to cells of the cropped grid."""
        return ((x - self._min_x) / self._resolution - self._left, (y - self._min_y) / self._resolution - self._top)

    def _from_cell(self, cell):
        return (cell[0] * SCALE, (self._rows - cell[1]) * SCALE)

    def pixel(self, x, y):
        """Convert map metres to picture pixels."""
        return self._from_cell(self._cell(x, y))

    def route(self, path):
        """The estimated route through all samples, in picture pixels."""
        cells = [self._cell(x, y) for x, y in path]
        points = cells[:1]
        for a, b in zip(cells, cells[1:]):
            points.extend(self._router.between(a, b)[1:])
        return [self._from_cell(c) for c in self._router.curve(points)]

    def frame(self, path):
        """Return a PNG of the floor plan with this run's path and the newest position on top."""
        image = self.image.copy()
        points = [self.pixel(x, y) for x, y in path]
        if points:
            large = Image.new("RGBA", (image.width * SMOOTH, image.height * SMOOTH), VACUUM + (0,))
            draw = ImageDraw.Draw(large, "RGBA")
            scaled = [(x * SMOOTH, y * SMOOTH) for x, y in points]
            line = [(x * SMOOTH, y * SMOOTH) for x, y in self.route(path)]
            _draw_trail(draw, line, 5 * SMOOTH, scaled[:-1])
            _draw_vacuum(draw, *scaled[-1], 18 * SMOOTH)
            image.alpha_composite(large.resize(image.size, Image.Resampling.LANCZOS))
        output = io.BytesIO()
        image.save(output, "PNG")
        return output.getvalue()


class LiveMap:
    """Follows one vacuum and keeps the newest live map picture."""

    def __init__(self, mac):
        self._mac = mac
        self._vacuum = None
        self._base = None
        self._base_key = None
        self._task_id = None
        self._pose_id = None
        self._path = []
        self._image = None
        self._lock = asyncio.Lock()
        self._next_check = 0.0
        self._eager_until = 0.0
        self._paused_until = 0.0
        self._next_login = 0.0
        self._was_active = False
        self._moving = False

    def start(self, hass, config_entry, vacuum):
        """Begin following the vacuum entity."""
        self._vacuum = vacuum
        config_entry.async_on_unload(async_track_time_interval(hass, self._async_tick, TICK))

    def poke(self):
        """A command was just sent: look again soon and keep looking for a while."""
        self._eager_until = time.monotonic() + EAGER_SECONDS

    async def async_image(self, hass):
        """Newest picture; draws the plain floor plan if no run has been seen yet."""
        if self._image is None and self._vacuum is not None and time.monotonic() >= self._paused_until:
            async with self._lock:
                if self._image is None:
                    try:
                        await hass.async_add_executor_job(self._redraw)
                    except Exception as err:  # the SDK raises several unrelated error types
                        self._failed(err)
        return self._image

    async def _async_tick(self, _now):
        vacuum = self._vacuum
        if vacuum is None or vacuum.hass is None or self._lock.locked():
            return
        now = time.monotonic()
        if now < self._paused_until:
            return
        if not (self._moving or now < self._eager_until or now >= self._next_check):
            return
        async with self._lock:
            try:
                activity, battery = await vacuum.hass.async_add_executor_job(self._read_status)
                active = activity in ACTIVE
                if active:
                    await vacuum.hass.async_add_executor_job(self._follow, not self._was_active)
            except Exception as err:  # the SDK raises several unrelated error types
                relogin = self._failed(err)
                if relogin:
                    await vacuum.get_new_client(err)
                return

        finished = self._was_active and not active
        self._was_active = active
        self._moving = activity in MOVING
        self._next_check = now + (PAUSED_CHECK_SECONDS if activity == VacuumActivity.PAUSED else IDLE_CHECK_SECONDS)
        if finished:
            # back to the clean floor plan; the next run starts a new path
            self._path = []
            self._pose_id = None
            if self._base is not None:
                self._image = await vacuum.hass.async_add_executor_job(self._base.frame, [])
        vacuum.async_live_status(activity, battery, finished)

    def _venus(self):
        return self._vacuum._client.vacuums._venus_client()

    def _read_status(self):
        props = self._venus().get_iot_prop(did=self._mac, keys=["mode", "battary"])["data"]["props"]
        mode = VacuumMode.parse(code=int(props["mode"]))
        battery = props.get("battary")
        return activity_from_mode(mode), None if battery is None else int(battery)

    def _follow(self, run_started):
        """Add the newest position to the path and redraw."""
        positions = self._venus().get_current_position(did=self._mac)["data"]
        if positions:
            newest = _items(positions)[-1]
            if newest.get("task_id") != self._task_id:
                self._task_id = newest.get("task_id")
                self._path = []
                self._pose_id = None
                run_started = True
            if newest.get("pose_id") != self._pose_id:
                self._pose_id = newest.get("pose_id")
                self._path.append((float(newest["x"]), float(newest["y"])))
                del self._path[:-MAX_PATH_POINTS]
        self._redraw(refresh_map=run_started)

    def _redraw(self, refresh_map=True):
        if refresh_map or self._base is None:
            current = self._venus().get_current_map(did=self._mac)["data"]
            key = (current.get("mapId"), current.get("updateTime"))
            if key != self._base_key:
                self._base = BaseMap(current["map"])
                self._base_key = key
        self._image = self._base.frame(self._path)

    def _failed(self, err):
        """Back off after an error. Returns True when renewing the session is worth one try."""
        now = time.monotonic()
        if is_rate_limited(err):
            self._paused_until = now + RATE_LIMIT_PAUSE_SECONDS
            _LOGGER.warning("Wyze rate limit reached; live map paused for %s seconds", RATE_LIMIT_PAUSE_SECONDS)
            return False
        self._paused_until = now + ERROR_PAUSE_SECONDS
        _LOGGER.warning("Live map update failed: %s", type(err).__name__)
        if is_token_expired(err) and now >= self._next_login:
            self._next_login = now + RENEW_RETRY_SECONDS
            return True
        return False
