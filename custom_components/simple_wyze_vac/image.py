"""The floor plan with the vacuum's route, as an image entity."""
from homeassistant.components.image import ImageEntity
from homeassistant.core import callback
from homeassistant.util import dt as dt_util

from .const import WYZE_VACUUMS, DOMAIN


async def async_setup_entry(hass, config_entry, async_add_entities):
    async_add_entities([SWVFloorPlan(hass, pl) for pl in hass.data[WYZE_VACUUMS]])


class SWVFloorPlan(ImageEntity):
    """Updates by itself: the live map pushes each new picture."""

    _attr_should_poll = False
    _attr_content_type = "image/svg+xml"

    def __init__(self, hass, pl):
        super().__init__(hass)
        self._live = pl["live"]
        self._device_name = pl["name"]
        self._model = pl["model"]
        self._attr_name = pl["name"] + " Floor Plan"
        self._attr_unique_id = "swv_" + pl["name"] + "_floor_plan"

    @property
    def device_info(self):
        """Return device registry information for this entity."""
        return {
            "identifiers": {(DOMAIN, self._device_name)},
            "name": self._device_name,
            "manufacturer": "Wyze Inc.",
            "model": self._model,
        }

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        self.async_on_remove(self._live.listen(self._changed))
        if self._live.image:
            self._changed()

    @callback
    def _changed(self):
        self._attr_image_last_updated = dt_util.utcnow()
        self.async_write_ha_state()

    async def async_image(self):
        return self._live.image
