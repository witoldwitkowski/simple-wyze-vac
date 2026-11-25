"""Tell apart the ways Wyze turns a request down."""

RATE_LIMIT_CODES = {"3044"}  # "requests too frequent"
EXPIRED_TOKEN_CODES = {"2001"}  # access token expired or unknown


def _reply(err):
    """HTTP status, Wyze error code and message carried by an SDK or requests error."""
    response = getattr(err, "response", None)
    status = getattr(response, "status_code", None)
    body = response if isinstance(response, dict) else None
    if body is None and response is not None:
        try:
            body = response.json()
        except Exception:  # not every error response holds JSON
            body = getattr(response, "data", None)
    if not isinstance(body, dict):
        body = {}
    code = str(body.get("code", body.get("errorCode", "")))
    message = str(body.get("msg", body.get("description", "")))
    return status, code, message


def is_rate_limited(err):
    """Wyze wants fewer requests: wait, and do not refresh or log in."""
    status, code, message = _reply(err)
    squeezed = message.lower().replace(" ", "")
    return status == 429 or code in RATE_LIMIT_CODES or "toofrequent" in squeezed or "toomany" in squeezed


def is_token_expired(err):
    """The access token no longer works and should be refreshed."""
    if is_rate_limited(err):
        return False
    status, code, message = _reply(err)
    return status == 401 or code in EXPIRED_TOKEN_CODES or message == "AccessTokenError"


def is_refresh_refused(err):
    """Wyze will not accept the refresh token, so only a password login can help.

    Wyze answers an unknown refresh token with the general code 1001, so any reply from Wyze to the refresh
    call counts, except a rate limit. Network trouble carries no reply and does not count.
    """
    if is_rate_limited(err):
        return False
    status, code, _ = _reply(err)
    if status is not None and status >= 500:  # trouble on Wyze's side says nothing about the token
        return False
    return status in (400, 401) or code != ""
