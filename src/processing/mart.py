"""Слой staging и витрина mart.topic_daily (тема x день). Пересборка из raw детерминирована и идемпотентна.

Решения по обработке:
- целевая переменная mentions = сумма nbHits по запросам темы за сутки UTC (документ, подходящий под два запроса
  темы, считается дважды — это «упоминания», а не уникальные документы);
- пропуск дня по одному из запросов -> mentions пустой -> линейная интерполяция (не более max_ffill_days подряд),
  флаг mentions_imputed; такие дни не участвуют в оценке качества;
- просмотры Википедии: всплески > wiki_spike_factor медиан за прошлые 28 дней — накрутка/боты, заменяются медианой
  (флаг wiki_imputed); короткие пропуски — предыдущим значением;
- всплески целевой переменной НЕ удаляются: это реальные события, которые и нужно прогнозировать.
"""
import json
import logging
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .text import clean_text, daily_text_features

log = logging.getLogger("processing")


def build_documents(con) -> pd.DataFrame:
    hn = con.execute("select object_id, topic, created_at, payload from raw.hn_items").df()
    rows = []
    for r in hn.itertuples(index=False):
        p = json.loads(r.payload)
        tags = p.get("_tags") or []
        kind = "story" if "story" in tags and "comment" not in tags else "comment"
        title = p.get("title") or p.get("story_title")
        body = p.get("comment_text") or p.get("story_text") or ""
        text = clean_text((p.get("title") or "") + " " + body) if kind == "story" else clean_text(body)
        url = p.get("url")
        rows.append((f"hn:{r.object_id}", "hn", r.topic, r.created_at, kind, title, text, url,
                     urlparse(url).netloc if url else None, p.get("points"), p.get("num_comments"), "en",
                     p.get("author_hash")))
    habr = con.execute("select guid, topic, published, payload from raw.habr_rss").df()
    for r in habr.itertuples(index=False):
        p = json.loads(r.payload)
        if pd.isna(r.published):
            continue
        text = clean_text((p.get("title") or "") + " " + (p.get("summary") or ""))
        rows.append((f"habr:{r.guid}", "habr", r.topic, r.published, "article", p.get("title"), text,
                     p.get("link"), "habr.com", None, None, "ru", None))
    cols = ["doc_id", "source", "topic", "ts_utc", "kind", "title", "text_clean", "url", "domain", "score",
            "n_comments", "lang", "author_hash"]
    docs = pd.DataFrame(rows, columns=cols)
    docs["ts_utc"] = pd.to_datetime(docs["ts_utc"])
    docs["date"] = docs["ts_utc"].dt.date
    docs = docs.drop_duplicates(["doc_id", "topic"])
    con.execute("delete from staging.documents")
    con.register("docs_df", docs)
    con.execute("insert into staging.documents by name select * from docs_df")
    con.unregister("docs_df")
    return docs


def build_series(con, cfg: dict) -> None:
    con.execute("delete from staging.pageviews")
    con.execute("insert into staging.pageviews select topic, article, date, views from raw.wiki_pageviews")
    con.execute("delete from staging.news_volume")
    con.execute("""
        insert into staging.news_volume
        select topic, ts::date, sum(value) filter (where series = 'Article Count'),
               sum(value) filter (where series = 'Total Monitored Articles')
        from raw.gdelt_timeline group by 1, 2""")


def _fill_short_gaps(s: pd.Series, limit: int) -> tuple[pd.Series, pd.Series]:
    filled = s.interpolate(limit=limit, limit_area="inside")
    return filled, s.isna() & filled.notna()


def build_mart(con, cfg: dict, docs: pd.DataFrame) -> pd.DataFrame:
    p = cfg["processing"]
    topics = list(cfg["topics"])
    n_queries = {t: len(v["hn_queries"]) for t, v in cfg["topics"].items()}
    hn = con.execute("select topic, date, sum(nb_hits) mentions, count(*) n_q from raw.hn_daily group by 1, 2").df()
    if hn.empty:
        raise RuntimeError("raw.hn_daily пуст: сначала выполните загрузку (make ingest)")
    hn.loc[hn.apply(lambda r: r["n_q"] < n_queries.get(r["topic"], 0), axis=1), "mentions"] = np.nan
    start, end = pd.to_datetime(hn["date"]).min(), pd.to_datetime(hn["date"]).max()
    grid = pd.MultiIndex.from_product([topics, pd.date_range(start, end)], names=["topic", "date"]).to_frame(index=False)

    hn["date"] = pd.to_datetime(hn["date"])
    df = grid.merge(hn[["topic", "date", "mentions"]], how="left", on=["topic", "date"])

    wiki = con.execute("select topic, date, sum(views) wiki_views from staging.pageviews group by 1, 2").df()
    wiki["date"] = pd.to_datetime(wiki["date"])
    df = df.merge(wiki, how="left", on=["topic", "date"])

    news = con.execute("select topic, date, articles gdelt_articles, articles / nullif(total_monitored, 0) gdelt_share "
                       "from staging.news_volume").df()
    news["date"] = pd.to_datetime(news["date"])
    df = df.merge(news, how="left", on=["topic", "date"])

    habr = docs[docs["source"] == "habr"].groupby(["topic", "date"]).size().rename("habr_articles").reset_index()
    habr["date"] = pd.to_datetime(habr["date"])
    df = df.merge(habr, how="left", on=["topic", "date"])

    hn_docs = docs[docs["source"] == "hn"].rename(columns={"text_clean": "text", "score": "points",
                                                           "n_comments": "num_comments"})
    txt = daily_text_features(hn_docs[["topic", "date", "kind", "text", "points", "num_comments"]],
                              p["event_words"], p["hashing_features"])
    txt["date"] = pd.to_datetime(txt["date"])
    df = df.merge(txt, how="left", on=["topic", "date"])

    cal = con.execute("select date, is_holiday_us, is_holiday_ru from staging.calendar").df()
    cal["date"] = pd.to_datetime(cal["date"])
    df = df.merge(cal, how="left", on="date")

    parts = []
    for _, g in df.sort_values("date").groupby("topic"):
        g = g.copy()
        g["mentions"], g["mentions_imputed"] = _fill_short_gaps(g["mentions"], p["max_ffill_days"])
        w = g["wiki_views"]
        med = w.shift(1).rolling(28, min_periods=7).median()
        spike = w > p["wiki_spike_factor"] * med
        w = w.mask(spike, med)
        w_filled = w.ffill(limit=p["max_ffill_days"])
        g["wiki_imputed"] = spike | (w.isna() & w_filled.notna())
        g["wiki_views"] = w_filled
        g["habr_articles"] = g["habr_articles"].fillna(0)
        g["txt_docs"] = g["txt_docs"].fillna(0)
        parts.append(g)
    df = pd.concat(parts)
    df["date"] = df["date"].dt.date
    cols = [c[0] for c in con.execute("describe mart.topic_daily").fetchall()]
    df = df[cols]
    con.execute("delete from mart.topic_daily")
    con.register("mart_df", df)
    con.execute("insert into mart.topic_daily by name select * from mart_df")
    con.unregister("mart_df")
    return df


def build_all(con, cfg: dict) -> pd.DataFrame:
    """staging + витрина (полная детерминированная пересборка из raw)."""
    docs = build_documents(con)
    build_series(con, cfg)
    mart = build_mart(con, cfg, docs)
    log.info("staging.documents: %d, mart.topic_daily: %d строк (%s .. %s)", len(docs), len(mart),
             mart["date"].min(), mart["date"].max())
    return mart
