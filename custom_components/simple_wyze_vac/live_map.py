"""Draw the current map with the vacuum's route while it runs and the last run's path once it is done.

Wyze only reports the newest position, so the route shown during a run is estimated from the samples collected
while it is active. After the run Wyze saves a record with a picture holding the route actually driven; that route
is copied onto our own map (drawn by ``floor_style``).

Requests: while cleaning or returning, a status and a position request every ten seconds; while paused, the
same once a minute; while idle, one status request every five minutes and the run records once an hour. After a
run the records are asked for every 30 seconds until the new one appears (at most six minutes). A command from
Home Assistant makes it look at once. A rate limit from Wyze stops all of it for ten minutes.
"""
import asyncio
import base64
import heapq
import io
import logging
import time
import urllib.request
import zlib
from dataclasses import replace
from datetime import datetime, timedelta

import blackboxprotobuf
from PIL import Image, ImageChops, ImageFilter

from homeassistant.components.vacuum.const import VacuumActivity
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util

from wyze_sdk.models.devices import VacuumMode
from wyze_sdk.models.devices.vacuums import VacuumMap

from . import floor_style
from .floor_style import Caption, Room, Scene, main_rooms, room_list, run_path, when
from .wyze_errors import is_rate_limited, is_token_expired

_LOGGER = logging.getLogger(__name__)

TICK = timedelta(seconds=10)
IDLE_CHECK_SECONDS = 300
PAUSED_CHECK_SECONDS = 60
ERROR_PAUSE_SECONDS = 60
EAGER_SECONDS = 120  # keep checking every tick for this long after a command from Home Assistant
RATE_LIMIT_PAUSE_SECONDS = 600
RENEW_RETRY_SECONDS = 900
RECORD_CHECK_SECONDS = 30       # after a run, until its record appears ...
RECORD_WAIT_SECONDS = 360       # ... but no longer than this
RECORD_REFRESH_SECONDS = 3600   # while idle, in case a run went unseen
MAX_PICTURE_BYTES = 10_000_000
MAX_PATH_POINTS = 3000
SQUARE_FEET = 10.7639
MOVING = (VacuumActivity.CLEANING, VacuumActivity.RETURNING)
ACTIVE = MOVING + (VacuumActivity.PAUSED,)  # a run is under way

MARGIN = 10  # map cells around the floor
PICTURE_WALL = (0, 208, 185, 255)    # colours in the picture Wyze saves with a run record
PICTURE_PATH = (240, 240, 240, 255)


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


def _colour_mask(picture, colour):
    """255 where the picture has exactly this colour."""
    mask = Image.new("L", picture.size, 255)
    for band, wanted in zip(picture.split(), colour):
        mask = ImageChops.multiply(mask, band.point(lambda v, w=wanted: 255 if v == w else 0))
    return mask


def _span(points, axis, limit):
    values = [p[axis] for p in points]
    return max(int(min(values)), 0), min(int(max(values)) + 1, limit)


def _clock(moment):
    return when(moment).rsplit(" · ", 1)[1]


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
    """The floor plan of one saved map, prepared once and reused for every picture."""

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
        self._router = Router(grid)

        def flip(picture):  # map rows count upwards, picture rows count downwards
            return picture.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # close pinholes so each room reads as one clean surface
        self._floor = flip(floor.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3)))
        self._grid = flip(grid)
        self._walls = self._grid.point(lambda v: 255 if v == 255 else 0)
        classes = Image.new("L", grid.size, floor_style.NOTHING)
        classes.paste(floor_style.FLOOR, mask=self._floor)
        classes.paste(floor_style.WALL, mask=self._walls)

        zones, barriers = [], []
        for area in _items(data.get("virtualWalls_")):
            points = [self.cell(q["x_"], q["y_"]) for q in _items(area.get("points_"))]
            if len(points) > 2:
                zones.append(points)
            elif len(points) == 2:
                barriers.append(points)

        dock = data.get("chargeStation_")
        dock_at = self.cell(dock.get("x_", 0), dock.get("y_", 0)) if dock else None

        self._names = {}
        labelled = []
        for room in rooms:
            name = " & ".join(part.strip() for part in _text(room.get("roomName_", "")).split("&"))
            self._names[room["roomId_"]] = name
            mask = self._grid.point(lambda v, wanted=room["roomId_"]: 255 if v == wanted else 0)
            free = mask.copy()
            for points in zones:
                for x, y in ((x, y) for y in range(*_span(points, 1, free.height))
                             for x in range(*_span(points, 0, free.width))):
                    free.putpixel((x, y), 0)
            cell = _label_cell(free) or _label_cell(mask)
            if cell:
                labelled.append(Room(name, (cell[0] + 0.5, cell[1] + 0.5)))

        self.scene = Scene(grid.width, grid.height, classes.tobytes(), self._resolution, rooms=labelled,
                           zones=zones, barriers=barriers, dock=dock_at)

    def cell(self, x, y):
        """Convert map metres to picture cells."""
        return ((x - self._min_x) / self._resolution - self._left,
                self._rows - ((y - self._min_y) / self._resolution - self._top))

    def route(self, path):
        """The estimated route through all samples, in picture cells."""
        cells = [(x, self._rows - y) for x, y in (self.cell(*p) for p in path)]
        points = cells[:1]
        for a, b in zip(cells, cells[1:]):
            points.extend(self._router.between(a, b)[1:])
        return [(x, self._rows - y) for x, y in self._router.curve(points)]

    def recorded_run(self, picture_bytes):
        """The route in Wyze's picture of a run, placed on this map.

        The picture shows the same map at a few pixels per cell, so its walls give the scale and position. Returns
        None if there is no route or the walls do not line up (for example after a new map).
        """
        picture = Image.open(io.BytesIO(picture_bytes)).convert("RGBA")
        walls, path = _colour_mask(picture, PICTURE_WALL), _colour_mask(picture, PICTURE_PATH)
        seen, known, route = walls.getbbox(), self._walls.getbbox(), path.getbbox()
        if not (seen and known and route):
            return None
        scale_x = (seen[2] - seen[0]) / (known[2] - known[0])
        scale_y = (seen[3] - seen[1]) / (known[3] - known[1])
        if abs(scale_x - scale_y) > 0.05 * scale_x or scale_x < 1:
            return None
        expected = self._walls.crop(known).resize((seen[2] - seen[0], seen[3] - seen[1]), Image.Resampling.NEAREST)
        found = walls.crop(seen)
        total = found.histogram()[255]
        if ImageChops.multiply(found, expected).histogram()[255] < 0.5 * total:
            return None

        # the picture area holding the route, in whole map cells
        left = int((route[0] - seen[0]) // scale_x) + known[0]
        top = int((route[1] - seen[1]) // scale_y) + known[1]
        right = int(-(-(route[2] - seen[0]) // scale_x)) + known[0]
        bottom = int(-(-(route[3] - seen[1]) // scale_y)) + known[1]
        pixels = (round(seen[0] + (left - known[0]) * scale_x), round(seen[1] + (top - known[1]) * scale_y),
                  round(seen[0] + (right - known[0]) * scale_x), round(seen[1] + (bottom - known[1]) * scale_y))
        line = path.crop(pixels)

        size = (right - left, bottom - top)
        covered = Image.new("L", self._grid.size, 0)
        covered.paste(line.resize(size, Image.Resampling.BOX).point(lambda v: 255 if v > 10 else 0), (left, top))
        reach = max(round(floor_style.SWEPT_METRES / self._resolution) | 1, 3)
        swept = ImageChops.multiply(covered.filter(ImageFilter.MaxFilter(reach)), self._floor)

        counts, on_floor = {}, 0
        for room_id, inside in zip(self._grid.tobytes(), covered.tobytes()):
            if inside and room_id in self._names:
                name = self._names[room_id]
                counts[name] = counts.get(name, 0) + 1
                on_floor += 1

        thin = line.filter(ImageFilter.MinFilter(3)) if scale_x >= 4 else line
        coloured = Image.new("RGBA", line.size, floor_style.ACCENT)
        coloured.putalpha(thin)
        output = io.BytesIO()
        coloured.save(output, "PNG", optimize=True)
        return {"picture": base64.b64encode(output.getvalue()).decode(), "box": (left, top) + size,
                "swept": run_path(swept.tobytes(), swept.width, 255), "rooms": main_rooms(counts, on_floor)}

    def draw(self, caption, path=(), run=None, cleaned=()):
        """The map as an SVG, with this run's estimated route or a recorded run on top."""
        scene = replace(self.scene)
        if run:
            scene.path_picture, scene.path_box, scene.swept = run["picture"], run["box"], run["swept"]
        if path:
            points = [self.cell(*p) for p in path]
            scene.trail = self.route(path) if len(path) > 1 else []
            scene.samples, scene.robot = points[:-1], points[-1]
        return floor_style.render(scene, caption, cleaned=cleaned).encode()


class LiveMap:
    """Follows one vacuum and keeps the newest map picture."""

    def __init__(self, mac):
        self._mac = mac
        self._vacuum = None
        self._base = None
        self._base_key = None
        self._task_id = None
        self._pose_id = None
        self._path = []
        self._activity = VacuumActivity.DOCKED
        self._started = None        # when the current run was first seen, if seen starting
        self._seen_idle = False
        self._record = None         # the newest run record and its route on the map
        self._records_due = 0.0
        self._waiting_for = None    # task id of a finished run whose record is expected
        self._wait_until = 0.0
        self._image = None
        self._listeners = []
        self._lock = asyncio.Lock()
        self._next_check = 0.0
        self._eager_until = 0.0
        self._paused_until = 0.0
        self._next_login = 0.0
        self._was_active = False
        self._moving = False

    @property
    def image(self):
        """The newest picture (SVG), or None before the first one."""
        return self._image

    def listen(self, changed):
        """Call ``changed`` whenever there is a new picture. Returns a function that stops it."""
        self._listeners.append(changed)
        return lambda: self._listeners.remove(changed)

    def start(self, hass, config_entry, vacuum):
        """Begin following the vacuum entity."""
        self._vacuum = vacuum
        config_entry.async_on_unload(async_track_time_interval(hass, self._async_tick, TICK))

    def poke(self):
        """A command was just sent: look again soon and keep looking for a while."""
        self._eager_until = time.monotonic() + EAGER_SECONDS

    async def _async_tick(self, _now):
        vacuum = self._vacuum
        if vacuum is None or vacuum.hass is None or self._lock.locked():
            return
        now = time.monotonic()
        if now < self._paused_until:
            return
        if not (self._moving or now < self._eager_until or now >= self._next_check):
            return
        before = self._image
        async with self._lock:
            try:
                activity, battery = await vacuum.hass.async_add_executor_job(self._read_status)
                active = activity in ACTIVE
                finished = self._was_active and not active
                self._activity = activity
                if active:
                    if not self._was_active:
                        self._started = dt_util.now() if self._seen_idle else None
                        self._waiting_for = None
                    await vacuum.hass.async_add_executor_job(self._follow, not self._was_active)
                else:
                    self._seen_idle = True
                    if finished:
                        self._waiting_for = self._task_id or 0
                        self._wait_until = now + RECORD_WAIT_SECONDS
                        self._path = []
                        self._pose_id = None
                    await vacuum.hass.async_add_executor_job(self._idle, finished)
            except Exception as err:  # the SDK raises several unrelated error types
                relogin = self._failed(err)
                if relogin:
                    await vacuum.get_new_client(err)
                return

        self._was_active = active
        self._moving = activity in MOVING
        if activity == VacuumActivity.PAUSED:
            wait = PAUSED_CHECK_SECONDS
        elif self._waiting_for is not None:
            wait = RECORD_CHECK_SECONDS
        else:
            wait = IDLE_CHECK_SECONDS
        self._next_check = now + wait
        if self._image != before:
            for changed in list(self._listeners):
                changed()
        vacuum.async_live_status(activity, battery, finished)

    def _venus(self):
        return self._vacuum._client.vacuums._venus_client()

    def _read_status(self):
        props = self._venus().get_iot_prop(did=self._mac, keys=["mode", "battary"])["data"]["props"]
        mode = VacuumMode.parse(code=int(props["mode"]))
        battery = props.get("battary")
        return activity_from_mode(mode), None if battery is None else int(battery)

    def _follow(self, run_started):
        """Add the newest position to the route and redraw."""
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
        if run_started or self._base is None:
            self._load_map()
        self._image = self._base.draw(self._running_caption(), path=self._path)

    def _running_caption(self):
        state, title = {
            VacuumActivity.RETURNING: ("returning", "Returning to the dock"),
            VacuumActivity.PAUSED: ("paused", "Paused"),
        }.get(self._activity, ("cleaning", "Cleaning"))
        detail = [f"Started {_clock(self._started)}"] if self._started else []
        detail.append("Route is estimated until the run ends")
        return Caption(state, title, " · ".join(detail))

    def _idle(self, just_finished):
        """Show the last run, fetching its record when one is due."""
        now = time.monotonic()
        expected = self._waiting_for
        if just_finished or self._base is None:
            self._load_map()
        if expected is not None or now >= self._records_due:
            self._records_due = now + RECORD_REFRESH_SECONDS
            self._read_record()
            if expected is not None:
                arrived = self._record is not None and (not expected or self._record["id"] == expected)
                if arrived or now >= self._wait_until:
                    self._waiting_for = None
        if self._activity == VacuumActivity.ERROR:
            self._image = self._base.draw(Caption("error", "Needs attention", "Check the vacuum or the Wyze app"))
        elif self._waiting_for is not None:
            self._image = self._base.draw(Caption("waiting", "Docked", "Waiting for the cleaning path"))
        else:
            self._image = self._draw_last_run()

    def _read_record(self):
        records = self._venus().get_sweep_records(did=self._mac, keys=[], limit=1, since=datetime.now())
        records = records["data"]["data"]
        if not records:
            return
        newest = records[0]
        number = int(newest["timeBegin"])
        if self._record and self._record["id"] == number:
            return
        run = None
        try:
            with urllib.request.urlopen(newest["map_img_big_url"], timeout=30) as response:
                picture = response.read(MAX_PICTURE_BYTES)
            run = self._base.recorded_run(picture)
        except Exception as err:  # the caption is still worth showing without the route
            _LOGGER.warning("Could not read the picture of the last run: %s", type(err).__name__)
        minutes = int(newest.get("cleanTime") or 0)
        self._record = {"id": number, "run": run, "map": self._base_key, "minutes": minutes,
                        "square_metres": int(newest.get("cleanSize") or 0) / 100,
                        "ended": dt_util.as_local(dt_util.utc_from_timestamp(number + minutes * 60))}

    def _draw_last_run(self):
        record = self._record
        if record is None:
            return self._base.draw(Caption("docked", "Docked", "No cleaning recorded yet"))
        # a route is only drawn on the map it was recorded on
        run = record["run"] if record["map"] == self._base_key else None
        rooms = run["rooms"] if run else []
        detail = [when(record["ended"])]
        if record["minutes"]:
            detail.append(f"{record['minutes']} min")
        if record["square_metres"]:
            detail.append(f"{record['square_metres'] * SQUARE_FEET:.0f} sq ft")
        title = f"Cleaned {room_list(rooms)}" if rooms else "Last clean"
        return self._base.draw(Caption("docked", title, " · ".join(detail)), run=run, cleaned=rooms)

    def _load_map(self):
        current = self._venus().get_current_map(did=self._mac)["data"]
        key = (current.get("mapId"), current.get("updateTime"))
        if key != self._base_key:
            self._base = BaseMap(current["map"])
            self._base_key = key

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
