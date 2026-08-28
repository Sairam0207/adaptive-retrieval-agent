"""Thin observability wrapper. Uses Langfuse's @observe decorator when
credentials are configured (auto-nests spans within one trace per query via
contextvars); otherwise degrades to console timing logs so the pipeline
always runs without requiring an external account."""
import functools
import logging
import time

from src.config import settings

logger = logging.getLogger(__name__)

_langfuse_enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

if _langfuse_enabled:
    import os

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
    from langfuse import observe as _observe
else:
    def _observe(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if not args else args[0]


def traced(name: str):
    """Decorator for agent node functions: records latency always, and sends
    a full nested trace to Langfuse when configured."""
    def outer(fn):
        langfuse_wrapped = _observe(name=name)(fn) if _langfuse_enabled else fn

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            start = time.perf_counter()
            result = langfuse_wrapped(*args, **kwargs)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.info("%s (%sms)", name, elapsed_ms)
            return result

        return inner

    return outer
