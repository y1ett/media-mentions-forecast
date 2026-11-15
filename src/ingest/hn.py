"""Источник A: Hacker News (Algolia API). JSON, слабоструктурированные тексты постов и комментариев.

Число документов по запросу темы за сутки (nbHits) — целевая переменная. Имя автора заменяется хэшем.
"""
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from ..db import logged_run, wm_date, wm_set
from ..http_client import SourceUnavailable, get, http_opts
from .utils import RateLimiter, _bulk, _count, last_complete_day

log = logging.getLogger("ingest.hn")


def _sanitize(hit: dict) -> dict:
    """Убирает служебные подсветки; имя автора заменяется хэшем."""
    hit = {k: v for k, v in hit.items() if k != "_highlightResult"}
    author = hit.pop("author", None)
    hit["author_hash"] = hashlib.sha256(author.encode()).hexdigest()[:16] if author else None
    hit["_tags"] = [t for t in hit.get("_tags", []) if not t.startswith("author_")]
    return hit


def _hn_day(cfg: dict, limiter: RateLimiter, query: str, day: date) -> dict:
    src = cfg["sources"]["hn"]
    start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
    params = {
        "query": query, "tags": src["tags"], "typoTolerance": "false",   # без опечаток: «llm» не находит «lim»
        "hitsPerPage": src["hits_per_page"], "attributesToHighlight": "",
        "attributesToRetrieve": "objectID,title,story_title,comment_text,story_text,points,"
                                "num_comments,created_at_i,_tags,url,author",
        "numericFilters": f"created_at_i>={start},created_at_i<{start + 86400}",
    }
    limiter.wait()
    data = get(src["url"], params=params, **http_opts(cfg)).json()
    if "nbHits" not in data:
        raise SourceUnavailable(f"HN: неожиданный ответ {str(data)[:100]}")
    hits = []
    for h in data.get("hits", []):
        h = _sanitize(h)
        for k in ("comment_text", "story_text"):
            if h.get(k):
                h[k] = h[k][: src["max_text_chars"]]
        hits.append(h)
    return {"day": day, "nb_hits": data["nbHits"],
            "exhaustive": bool((data.get("exhaustive") or {}).get("nbHits", True)), "hits": hits}


def load_hn(con, cfg: dict, topic: str, tcfg: dict) -> None:
    src = cfg["sources"]["hn"]
    start_cfg, end = date.fromisoformat(str(cfg["sources"]["start_date"])), last_complete_day()
    limiter = RateLimiter(src["requests_per_sec"])
    for query in tcfg["hn_queries"]:
        key = f"{topic}|{query}"
        mark = wm_date(con, "hn", key)
        start = max(start_cfg, mark - timedelta(days=src["refetch_days"] - 1)) if mark else start_cfg
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        if not days:
            continue
        with logged_run(con, "hn", topic, f"{query}: {start}..{end}") as stats:
            # блоками по 30 дней: после каждого блока сдвигаем водяной знак (можно прервать и продолжить)
            for i in range(0, len(days), 30):
                chunk = days[i:i + 30]
                with ThreadPoolExecutor(src["workers"]) as pool:
                    results = list(pool.map(lambda d: _hn_day(cfg, limiter, query, d), chunk))
                daily = pd.DataFrame([(topic, query, r["day"], r["nb_hits"], r["exhaustive"], stats["run_id"])
                                      for r in results],
                                     columns=["topic", "query", "date", "nb_hits", "exhaustive", "run_id"])
                _bulk(con, "insert or replace into raw.hn_daily select topic, query, date, nb_hits, exhaustive, "
                           "now()::timestamp, run_id from df", daily)
                items = pd.DataFrame([(h["objectID"], topic,
                                       datetime.fromtimestamp(h["created_at_i"], timezone.utc).replace(tzinfo=None),
                                       json.dumps({**h, "query": query}, ensure_ascii=False), stats["run_id"])
                                      for r in results for h in r["hits"]],
                                     columns=["object_id", "topic", "created_at", "payload", "run_id"])
                before = _count(con, "raw.hn_items")
                if len(items):
                    _bulk(con, "insert into raw.hn_items select distinct on (object_id, topic) object_id, topic, "
                               "created_at, payload::json, now()::timestamp, run_id from df on conflict do nothing", items)
                stats["inserted"] += _count(con, "raw.hn_items") - before
                stats["fetched"] += len(results)
                wm_set(con, "hn", key, chunk[-1])
                log.info("hn %s | %s: загружено по %s (%d/%d дней)", topic, query, chunk[-1], i + len(chunk), len(days))
            stats["message"] = f"дней {stats['fetched']}, новых текстов {stats['inserted']}"
