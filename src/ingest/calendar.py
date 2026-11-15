"""Источник E: календарь праздников RU и US (библиотека holidays, без сети).
"""
import holidays
import pandas as pd

from ..db import logged_run
from .utils import _bulk, _count


def load_calendar(con, cfg: dict) -> None:
    src = cfg["sources"]["calendar"]
    years = src["years"]
    cal = {c: holidays.country_holidays(c, years=years) for c in src["countries"]}
    days = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31").date
    rows = pd.DataFrame({
        "date": days,
        "is_holiday_ru": [d in cal.get("RU", {}) for d in days],
        "is_holiday_us": [d in cal.get("US", {}) for d in days],
        "holiday_name": ["; ".join(f"{c}: {cal[c][d]}" for c in cal if d in cal[c]) or None for d in days],
    })
    with logged_run(con, "calendar", None, f"{min(years)}..{max(years)}") as stats:
        before = _count(con, "staging.calendar")
        _bulk(con, "insert into staging.calendar select * from df on conflict do nothing", rows)
        stats["fetched"], stats["inserted"] = len(rows), _count(con, "staging.calendar") - before
