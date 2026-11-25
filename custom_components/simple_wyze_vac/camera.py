import logging

from .const import WYZE_VACUUMS, \
                   DOMAIN

from homeassistant.components.camera import Camera
from homeassistant.components.local_file.camera import LocalFile

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, config_entry, async_add_entities):

    cameras = []
    for pl in hass.data[WYZE_VACUUMS]:
        cameras.append(SWVCamera(pl,f"www/{DOMAIN}/{pl['name']}_last_map.jpg"))
        cameras.append(SWVLiveCamera(pl))

    async_add_entities(cameras)

class SWVCamera(LocalFile):
    def __init__(self, pl, file_path):
        unique_id = "swv_" + pl["name"] + "_camera"
        super().__init__(pl["name"], file_path, unique_id)
        self._model = pl["model"]

    @property
    def name(self):
        return self._attr_name + " Camera"

    @property
    def unique_id(self):
        return "swv_" + self._attr_name + "_camera"

    @property
    def device_info(self):
        """Return device registry information for this entity."""
        return {
            "identifiers": {(DOMAIN, self._attr_name)},
            "name": self._attr_name,
            "manufacturer": "Wyze Inc.",
            "model": self._model,
        }


class SWVLiveCamera(Camera):
    """The current map with the vacuum's position, kept up to date while it runs."""

    _attr_should_poll = False

    def __init__(self, pl):
        super().__init__()
        self._live = pl["live"]
        self._device_name = pl["name"]
        self._model = pl["model"]
        self.content_type = "image/png"
        self._attr_name = pl["name"] + " Live Map"
        self._attr_unique_id = "swv_" + pl["name"] + "_live_map"

    @property
    def device_info(self):
        """Return device registry information for this entity."""
        return {
            "identifiers": {(DOMAIN, self._device_name)},
            "name": self._device_name,
            "manufacturer": "Wyze Inc.",
            "model": self._model,
        }

    async def async_camera_image(self, width=None, height=None):
        return await self._live.async_image(self.hass)
