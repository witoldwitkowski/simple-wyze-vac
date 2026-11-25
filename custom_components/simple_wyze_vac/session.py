"""Keep one Wyze session instead of logging in with the password again and again.

A password login gives an access token and a refresh token. Both are saved in Home Assistant's private
storage and reused after a restart or reload. A rejected access token is renewed with the refresh token.
The password is used only when nothing is saved or Wyze rejects the refresh token, and at most once in
ten minutes, because Wyze limits how often an account may log in.
"""
import asyncio
import logging
import time

from wyze_sdk import Client

from homeassistant.helpers.storage import Store

from .const import DOMAIN
from .wyze_errors import is_rate_limited, is_refresh_refused, is_token_expired

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
LOGIN_PAUSE_SECONDS = 600
RATE_LIMIT_PAUSE_SECONDS = 600
RENEW_PAUSE_SECONDS = 300


class LoginPaused(Exception):
    """A password login was tried too recently to try again."""


class RateLimited(Exception):
    """Wyze asked for fewer requests, so none are being sent for now."""


def storage_for(hass, entry_id):
    """Where the tokens of one config entry are kept (readable by Home Assistant only)."""
    return Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry_id}", private=True)


class WyzeSession:
    """Owns the single Wyze client of a config entry."""

    def __init__(self, hass, entry_id, email, password, key_id, api_key, totp_key=None):
        self._hass = hass
        self._store = storage_for(hass, entry_id)
        self._credentials = {"email": email, "password": password, "key_id": key_id,
                             "api_key": api_key, "totp_key": totp_key}
        self._data = {}
        self._lock = asyncio.Lock()
        self._renewed_at = None
        self.client = None

    async def async_connect(self):
        """Return a working client, reusing the saved tokens when Wyze still accepts them."""
        saved = await self._store.async_load() or {}
        self._data = saved if saved.get("account") == self._credentials["email"] else {}
        self._check_quiet()

        if not (self._data.get("access_token") and self._data.get("refresh_token")):
            async with self._lock:
                await self._async_login()
            return self.client

        # a client that is given a token does not log in
        self.client = await self._hass.async_add_executor_job(
            lambda: Client(token=self._data["access_token"], refresh_token=self._data["refresh_token"],
                           **self._credentials))
        try:
            await self._hass.async_add_executor_job(self.client.vacuums.list)
        except Exception as err:  # the SDK raises several unrelated error types
            await self._async_note_rate_limit(err)
            if not is_token_expired(err):
                raise
            await self.async_renew()
        else:
            _LOGGER.info("Reusing the saved Wyze session")
        return self.client

    async def async_renew(self):
        """Replace the access token. The client object stays the same, so every user of it sees the new one."""
        async with self._lock:
            if self._renewed_at is not None and time.monotonic() - self._renewed_at < RENEW_PAUSE_SECONDS:
                return
            self._check_quiet()
            try:
                await self._hass.async_add_executor_job(self.client.refresh_token)
            except Exception as err:  # the SDK raises several unrelated error types
                await self._async_note_rate_limit(err)
                if not is_refresh_refused(err):
                    raise
                _LOGGER.warning("Wyze rejected the saved session; logging in with the password")
                await self._async_login()
                return
            self._renewed_at = time.monotonic()
            await self._async_save()
            _LOGGER.info("Renewed the Wyze access token")

    def _check_quiet(self):
        wait = self._data.get("quiet_until", 0) - time.time()
        if wait > 0:
            raise RateLimited(f"waiting {int(wait)} seconds after a Wyze rate limit")

    async def _async_note_rate_limit(self, err):
        """Remember a rate limit, also across restarts, and stay quiet for a while."""
        if is_rate_limited(err):
            self._data = {**self._data, "account": self._credentials["email"],
                          "quiet_until": time.time() + RATE_LIMIT_PAUSE_SECONDS}
            await self._store.async_save(self._data)

    async def _async_login(self):
        self._check_quiet()
        wait = self._data.get("next_login", 0) - time.time()
        if wait > 0:
            raise LoginPaused(f"next password login allowed in {int(wait)} seconds")
        # note the attempt before making it, so neither a failure nor a restart can repeat it right away
        self._data = {**self._data, "account": self._credentials["email"],
                      "next_login": time.time() + LOGIN_PAUSE_SECONDS}
        await self._store.async_save(self._data)

        try:
            fresh = await self._hass.async_add_executor_job(lambda: Client(**self._credentials))
        except Exception as err:
            await self._async_note_rate_limit(err)
            raise
        if self.client is None:
            self.client = fresh
        else:
            self.client._update_session(access_token=fresh._token, refresh_token=fresh._refresh_token)
        self._renewed_at = time.monotonic()
        await self._async_save()

    async def _async_save(self):
        self._data = {**self._data, "account": self._credentials["email"],
                      "access_token": self.client._token, "refresh_token": self.client._refresh_token}
        await self._store.async_save(self._data)
