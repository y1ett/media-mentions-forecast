"""Источник B: Wikimedia Pageviews API. JSON, суточные просмотры статей (нужен User-Agent с контактом).
"""
import json
from datetime import date, datetime, timedelta
from urllib.parse import quote

import pandas as pd

from ..db import logged_run, wm_date, wm_set
from ..http_client import get, http_opts
from .utils import _bulk, last_complete_day


def load_wiki(con, cfg: dict, topic: str, tcfg: dict) -> None:
    src = cfg["sources"]["wiki"]
    start_cfg, end = date.fromisoformat(str(cfg["sources"]["start_date"])), last_complete_day()
    with logged_run(con, "wiki", topic, f"..{end}") as stats:
        for article in tcfg["wiki_articles"]:
            mark = wm_date(con, "wiki", article)
            start = max(start_cfg, mark - timedelta(days=1)) if mark else start_cfg
            if start > end:
                continue
            url = (f"{src['url']}/{src['project']}/{src['access']}/{src['agent']}/"
                   f"{quote(article, safe='')}/daily/{start:%Y%m%d}/{end:%Y%m%d}")
            items = get(url, **http_opts(cfg)).json().get("items", [])
            rows = [(article, datetime.strptime(it["timestamp"][:8], "%Y%m%d").date(), topic, it["views"],
                     json.dumps(it), stats["run_id"]) for it in items]
            if rows:
                _bulk(con, "insert or replace into raw.wiki_pageviews select article, date, topic, views, "
                           "payload::json, now()::timestamp, run_id from df",
                      pd.DataFrame(rows, columns=["article", "date", "topic", "views", "payload", "run_id"]))
                wm_set(con, "wiki", article, max(r[1] for r in rows))
            stats["fetched"] += len(rows)
            stats["inserted"] += len(rows)
