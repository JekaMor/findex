"""``@timed``: log how long each call took (logger ``findex.timing``, DEBUG)."""

import functools
import logging
import time
from collections.abc import Callable

log = logging.getLogger("findex.timing")


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    # wraps copies __name__, __qualname__, __doc__ and __wrapped__ onto the
    # wrapper; without it every decorated function would be called "wrapper".
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            log.debug("%s took %.1f ms", fn.__qualname__, elapsed)

    return wrapper
