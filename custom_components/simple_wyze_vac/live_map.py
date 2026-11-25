"""Live map: draw the current map with the vacuum's position and the path seen so far.

Wyze only reports the newest position, so the path is built from the samples collected while a run is active.
While the vacuum is idle this makes one small status request per minute; while it is active, a status and a
position request every ten seconds.
"""
import asyncio
import base64
import io
import logging
import time
import zlib
from datetime import timedelta

import blackboxprotobuf
from PIL import Image, ImageDraw, ImageFont

from homeassistant.components.vacuum.const import VacuumActivity
from homeassistant.helpers.event import async_track_time_interval

from wyze_sdk.models.devices import VacuumMode
from wyze_sdk.models.devices.vacuums import VacuumMap

_LOGGER = logging.getLogger(__name__)

TICK = timedelta(seconds=10)
IDLE_CHECK_SECONDS = 60
EAGER_SECONDS = 90  # keep checking every tick for this long after a command from Home Assistant
RATE_LIMIT_PAUSE_SECONDS = 600
LOGIN_RETRY_SECONDS = 900
MAX_PATH_POINTS = 3000
ACTIVE = (VacuumActivity.CLEANING, VacuumActivity.RETURNING, VacuumActivity.PAUSED)

SCALE = 5
MARGIN = 12  # map cells around the floor
ROOM_COLORS = [(168, 208, 255), (255, 205, 150), (180, 230, 180), (225, 190, 240),
               (255, 235, 140), (170, 230, 230), (255, 190, 200), (210, 210, 170)]


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
        rooms = _items(data.get("12"))

        # every cell holds 0 (outside), 1 (obstacle), 255 (wall) or a room id
        grid = Image.frombytes("L", (width, height), cells)
        box = grid.getbbox() or (0, 0, width, height)
        self._left = max(box[0] - MARGIN, 0)
        self._top = max(box[1] - MARGIN, 0)
        grid = grid.crop((self._left, self._top, min(box[2] + MARGIN, width), min(box[3] + MARGIN, height)))
        self._rows = grid.height

        palette = [255, 255, 255] * 256
        palette[1 * 3:2 * 3] = (60, 60, 60)
        palette[255 * 3:256 * 3] = (150, 150, 150)
        for index, room in enumerate(rooms):
            room_id = room["roomId_"]
            palette[room_id * 3:room_id * 3 + 3] = ROOM_COLORS[index % len(ROOM_COLORS)]
        picture = Image.frombytes("P", grid.size, grid.tobytes())
        picture.putpalette(palette)
        picture = picture.convert("RGB").resize((grid.width * SCALE, grid.height * SCALE), Image.Resampling.NEAREST)
        # map rows count upwards, image rows count downwards
        self.image = picture.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        draw = ImageDraw.Draw(self.image, "RGBA")
        for area in _items(data.get("virtualWalls_")):
            points = [self.pixel(p["x_"], p["y_"]) for p in _items(area.get("points_"))]
            if len(points) > 2:
                draw.polygon(points, fill=(220, 0, 0, 60), outline=(220, 0, 0, 255), width=3)
            elif len(points) == 2:
                draw.line(points, fill=(220, 0, 0, 255), width=5)

        dock = data.get("chargeStation_")
        if dock:
            x, y = self.pixel(dock.get("x_", 0), dock.get("y_", 0))
            draw.rectangle((x - 12, y - 12, x + 12, y + 12), fill=(0, 150, 60), outline=(0, 0, 0), width=3)

        font = ImageFont.load_default(32)
        for room in rooms:
            position = room.get("roomNamePost_")
            if not position:
                continue
            center = self.pixel(position["x_"], position["y_"])
            name = _text(room.get("roomName_", ""))
            box = draw.textbbox(center, name, font=font, anchor="mm")
            draw.rounded_rectangle((box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6), radius=8,
                                   fill=(255, 255, 255, 215), outline=(0, 0, 0, 255), width=2)
            draw.text(center, name, font=font, fill=(0, 0, 0), anchor="mm")

    def pixel(self, x, y):
        """Convert map metres to image pixels."""
        column = (x - self._min_x) / self._resolution - self._left
        row = (y - self._min_y) / self._resolution - self._top
        return (column * SCALE, (self._rows - row) * SCALE)

    def frame(self, path):
        """Return a JPEG of the floor plan with the path and the newest position on top."""
        image = self.image.copy()
        draw = ImageDraw.Draw(image, "RGBA")
        points = [self.pixel(x, y) for x, y in path]
        if len(points) > 1:
            draw.line(points, fill=(20, 60, 200, 190), width=5, joint="curve")
        if points:
            x, y = points[-1]
            draw.ellipse((x - 26, y - 26, x + 26, y + 26), fill=(230, 40, 40, 70))
            draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=(230, 40, 40), outline=(255, 255, 255), width=4)
        output = io.BytesIO()
        image.save(output, "JPEG", quality=90)
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
        if not (self._was_active or now < self._eager_until or now >= self._next_check):
            return
        self._next_check = now + IDLE_CHECK_SECONDS

        async with self._lock:
            try:
                activity, battery = await vacuum.hass.async_add_executor_job(self._read_status)
                active = activity in ACTIVE
                if active:
                    await vacuum.hass.async_add_executor_job(self._follow, not self._was_active)
            except Exception as err:  # the SDK raises several unrelated error types
                relogin = self._failed(err)
                if relogin:
                    await vacuum.get_new_client()
                return

        finished = self._was_active and not active
        self._was_active = active
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
        """Back off after an error. Returns True when a new login is worth one try."""
        status = getattr(getattr(err, "response", None), "status_code", None)
        now = time.monotonic()
        if status == 429 or "429" in str(err):
            self._paused_until = now + RATE_LIMIT_PAUSE_SECONDS
            _LOGGER.warning("Wyze rate limit reached; live map paused for %s seconds", RATE_LIMIT_PAUSE_SECONDS)
            return False
        self._paused_until = now + IDLE_CHECK_SECONDS
        _LOGGER.warning("Live map update failed: %s", type(err).__name__)
        expired = status == 401 or "token" in str(err).lower()
        if expired and now >= self._next_login:
            self._next_login = now + LOGIN_RETRY_SECONDS
            return True
        return False
