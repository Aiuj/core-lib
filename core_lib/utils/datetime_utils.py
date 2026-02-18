"""Datetime utilities for consistent timezone handling.

Provides timezone-aware replacements for the deprecated ``datetime.utcnow()``
so that every service in the ecosystem uses the same helper instead of
scattering ``datetime.now(timezone.utc)`` calls everywhere.

Usage::

    from core_lib.utils import utc_now, parse_iso_datetime, to_iso_string
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current UTC datetime **with** timezone info.

    This is the recommended replacement for the deprecated
    ``datetime.utcnow()`` which returns a *naive* datetime and is
    scheduled for removal in a future Python version.
    """
    return datetime.now(timezone.utc)


def parse_iso_datetime(iso_string: str) -> datetime:
    """Parse an ISO-8601 string into a timezone-aware datetime.

    If the string lacks timezone info the result is assumed to be UTC.
    """
    dt = datetime.fromisoformat(iso_string)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def to_iso_string(dt: datetime) -> str:
    """Convert a datetime to an ISO-8601 string."""
    return dt.isoformat()
