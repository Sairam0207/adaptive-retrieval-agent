"""Single shared Gemini client with retry-with-backoff on transient server
errors and per-minute rate limits (both common on the free tier), plus
fallback to a secondary model when the primary's *daily* quota is exhausted
(retrying that in place would be pointless — it won't reset for hours).
Used by every node, the naive baseline, and the eval judge, instead of each
creating its own client and repeating this logic."""
import logging
import re
import time

import httpx
from google import genai
from google.genai import errors, types

from src.config import settings

logger = logging.getLogger(__name__)

client = genai.Client(
    api_key=settings.google_api_key,
    http_options=types.HttpOptions(timeout=30_000),
)

_MAX_ATTEMPTS = 4
_BASE_DELAY_SECONDS = 3.0
_MAX_DELAY_SECONDS = 20.0
_RATE_LIMIT_RETRY_PATTERN = re.compile(r"Please retry in ([\d.]+)s")

# Free-tier per-minute quotas are shared across every call site that uses a
# given model; proactively spacing calls avoids paying a ~60s reactive 429
# penalty for bursts that would otherwise stay under the limit if paced.
_MIN_INTERVAL_SECONDS = 4.5
_last_call_time: dict[str, float] = {}


def _throttle(model: str) -> None:
    now = time.monotonic()
    wait = _MIN_INTERVAL_SECONDS - (now - _last_call_time.get(model, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_call_time[model] = time.monotonic()


def _is_daily_quota_error(exc: errors.ClientError) -> bool:
    return "PerDay" in str(exc)


def _rate_limit_delay(exc: errors.ClientError) -> float:
    match = _RATE_LIMIT_RETRY_PATTERN.search(str(exc))
    return float(match.group(1)) + 1.0 if match else _MAX_DELAY_SECONDS


def _generate_with_retry(model: str, contents: str, config: types.GenerateContentConfig):
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            _throttle(model)
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.ServerError as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(min(_BASE_DELAY_SECONDS * (2 ** attempt), _MAX_DELAY_SECONDS))
        except errors.ClientError as exc:
            if exc.code == 429 and not _is_daily_quota_error(exc) and attempt < _MAX_ATTEMPTS - 1:
                last_exc = exc
                time.sleep(_rate_limit_delay(exc))
            else:
                raise
        except httpx.TransportError as exc:
            # Raw network faults (timeouts, dropped connections) never reach
            # google-genai's own error types, so they need their own retry arm.
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(min(_BASE_DELAY_SECONDS * (2 ** attempt), _MAX_DELAY_SECONDS))
    raise last_exc


def generate(
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    fallback_model: str | None = None,
) -> types.GenerateContentResponse:
    try:
        return _generate_with_retry(model, contents, config)
    except (errors.ServerError, httpx.TransportError) as exc:
        if fallback_model is None or fallback_model == model:
            raise
        reason = "server overload" if isinstance(exc, errors.ServerError) else "network error"
        logger.warning("%s exhausted retries (%s), falling back to %s", model, reason, fallback_model)
        return _generate_with_retry(fallback_model, contents, config)
    except errors.ClientError as exc:
        if exc.code == 429 and fallback_model and fallback_model != model:
            logger.warning("%s quota exhausted, falling back to %s", model, fallback_model)
            return _generate_with_retry(fallback_model, contents, config)
        raise
