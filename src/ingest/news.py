"""Источники C и D: GDELT DOC 2.0 (CSV, объём мировых новостей; не чаще 1 запроса в 5 секунд)
и Habr RSS (XML, русскоязычные статьи; только свежие, автор не сохраняется).
"""
import csv
import io
import json
import logging
import re
import time
from datetime import date, datetime

import feedparser
import pandas as pd

from ..db import logged_run, wm_date, wm_set
from ..http_client import SourceUnavailable, get, http_opts
from .utils import _bulk, _count

log = logging.getLogger("ingest.news")


# ---------------------------------------------------------------- C. GDELT
_gdelt_last_call = 0.0


def _gdelt_throttle(interval: float) -> None:
    global _gdelt_last_call
    wait = _gdelt_last_call + interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _gdelt_last_call = time.monotonic()


def gdelt_fetch(cfg: dict, topic: str, query: str, timespan: str) -> list[tuple]:
    src = cfg["sources"]["gdelt"]
    params = {"query": query, "mode": "timelinevolraw", "format": "csv", "timespan": timespan}
    attempts, text = max(cfg["http"]["retries"], 2), ""
    for attempt in range(1, attempts + 1):
        # интервал растёт с каждой попыткой: GDELT часто отвечает 429 даже при редких запросах
        _gdelt_throttle(src["min_interval_sec"] * attempt)
        try:
            text = get(src["url"], params=params, **{**http_opts(cfg), "retries": 1}).content.decode("utf-8-sig")
        except SourceUnavailable as exc:
            log.warning("gdelt/%s: попытка %d/%d: %s", topic, attempt, attempts, exc)
            continue
        if text.startswith("Date,Series,Value"):
            break
        log.warning("gdelt/%s: ответ не CSV (попытка %d/%d): %s", topic, attempt, attempts, text[:80])
    else:
        raise SourceUnavailable(f"GDELT недоступен или превышен лимит: {text[:120]!r}")
    rows = []
    for rec in csv.DictReader(io.StringIO(text)):
        # шаг 15 минут для коротких периодов, сутки — для длинных (timespan 3m)
        fmt = "%Y-%m-%d %H:%M:%S" if len(rec["Date"]) > 10 else "%Y-%m-%d"
        rows.append((topic, datetime.strptime(rec["Date"], fmt), rec["Series"], float(rec["Value"])))
    return rows


def load_gdelt(con, cfg: dict, topic: str, tcfg: dict) -> None:
    src = cfg["sources"]["gdelt"]
    mark = wm_date(con, "gdelt", topic)
    # без водяного знака — вся доступная история (~3 мес.), иначе — дни с момента последней загрузки
    span = src["history_timespan"] if not mark else f"{min(max((date.today() - mark).days + 1, 2), 90)}d"
    with logged_run(con, "gdelt", topic, span) as stats:
        rows = gdelt_fetch(cfg, topic, tcfg["gdelt_query"], span)
        stats["fetched"] = len(rows)
        before = _count(con, "raw.gdelt_timeline")
        if rows:
            df = pd.DataFrame(rows, columns=["topic", "ts", "series", "value"]).assign(run_id=stats["run_id"])
            _bulk(con, "insert into raw.gdelt_timeline select topic, ts, series, value, now()::timestamp, run_id "
                       "from df on conflict do nothing", df)
            wm_set(con, "gdelt", topic, max(r[1] for r in rows).date())
        stats["inserted"] = _count(con, "raw.gdelt_timeline") - before


# ---------------------------------------------------------------- D. Habr RSS
def _strip_authors(xml: str) -> str:
    return re.sub(r"<dc:creator>.*?</dc:creator>", "", xml, flags=re.S)


def load_habr(con, cfg: dict, topic: str, tcfg: dict) -> None:
    hub = tcfg["habr_hub"]
    with logged_run(con, "habr", topic, "rss") as stats:
        resp = get(cfg["sources"]["habr"]["url"].format(hub=hub), **http_opts(cfg))
        feed = feedparser.parse(_strip_authors(resp.content.decode("utf-8", "replace")))
        rows = []
        for e in feed.entries:
            published = datetime(*e.published_parsed[:6]) if e.get("published_parsed") else None
            payload = {"title": e.get("title"), "link": e.get("link"), "summary": e.get("summary"),
                       "tags": [t.get("term") for t in e.get("tags", [])],
                       "published": published.isoformat() if published else None}
            rows.append((e.get("id") or e.get("link"), topic, hub, published,
                         json.dumps(payload, ensure_ascii=False), stats["run_id"]))
        stats["fetched"] = len(rows)
        before = _count(con, "raw.habr_rss")
        if rows:
            _bulk(con, "insert into raw.habr_rss select guid, topic, hub, published, payload::json, "
                       "now()::timestamp, run_id from df on conflict do nothing",
                  pd.DataFrame(rows, columns=["guid", "topic", "hub", "published", "payload", "run_id"]))
        stats["inserted"] = _count(con, "raw.habr_rss") - before
