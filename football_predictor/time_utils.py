from __future__ import annotations

import re
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo


TIMEZONE_NAME = "Africa/Lagos"
_COMPACT_TIME = re.compile(r"^(\d{1,2})(\d{2})(am|pm)$", re.IGNORECASE)


def parse_kickoff_time(value: Any, *, default: time = time.min) -> time:
    if value is None:
        return default
    if isinstance(value, datetime):
        return value.time().replace(second=0, microsecond=0)
    if isinstance(value, time):
        return value.replace(second=0, microsecond=0)
    text = str(value).strip().replace(".", "").replace(" ", "")
    if not text or text.casefold() in {"nan", "nat", "none"}:
        return default
    compact = _COMPACT_TIME.fullmatch(text)
    if compact:
        text = f"{compact.group(1)}:{compact.group(2)}{compact.group(3)}"
    for pattern in ("%H:%M", "%H%M", "%I:%M%p", "%I%p"):
        try:
            return datetime.strptime(text.upper(), pattern).time()
        except ValueError:
            continue
    raise ValueError(f"invalid kickoff time {value!r}; use HH:MM, such as 10:15")


def kickoff_datetime(value: date | datetime, kickoff_time: time | str | None = None) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None, second=0, microsecond=0)
    return datetime.combine(value, parse_kickoff_time(kickoff_time))


def local_now() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE_NAME)).replace(tzinfo=None)


def kickoff_parts(value: date | datetime) -> tuple[date, time]:
    stamp = kickoff_datetime(value)
    return stamp.date(), stamp.time().replace(second=0, microsecond=0)


def display_kickoff(value: date | datetime) -> str:
    return kickoff_datetime(value).strftime("%Y-%m-%d %H:%M")
