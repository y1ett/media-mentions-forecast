"""Хранилище DuckDB: слои raw -> staging -> mart + служебный meta, журнал загрузок, водяные знаки.

Первичные ключи заданы заранее: запись через INSERT OR REPLACE / ON CONFLICT DO NOTHING идемпотентна.
"""
import logging
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import duckdb

log = logging.getLogger("db")


SCHEMA = """
create schema if not exists raw;
create schema if not exists staging;
create schema if not exists mart;
create schema if not exists meta;

-- ============ meta: служебные таблицы ============
create table if not exists meta.topics (
    topic         varchar primary key,
    title         varchar,
    hn_queries    varchar,
    wiki_articles varchar,
    gdelt_query   varchar,
    habr_hub      varchar
);

create table if not exists meta.load_log (
    run_id        varchar primary key,
    source        varchar not null,
    topic         varchar,
    period        varchar,
    started_at    timestamp not null,
    finished_at   timestamp,
    status        varchar not null,      -- running / success / failed
    rows_fetched  bigint,
    rows_inserted bigint,
    message       varchar
);

create table if not exists meta.watermarks (   -- для инкрементальной загрузки
    source     varchar not null,
    topic      varchar not null,
    value      varchar not null,
    updated_at timestamp not null,
    primary key (source, topic)
);

create table if not exists meta.dq_results (
    checked_at timestamp,
    layer      varchar,
    check_name varchar,
    value      double,
    threshold  double,
    passed     boolean,
    severity   varchar                    -- critical (останавливает пайплайн) / warning
);

create table if not exists meta.experiments (   -- журнал экспериментов
    run_id        varchar primary key,
    created_at    timestamp not null,
    kind          varchar not null,       -- train / tune / reproduce
    model         varchar not null,
    params        json,
    features      json,
    data_version  varchar,                -- хэш витрины + последняя дата
    data_end      date,
    seed          integer,
    cv_metrics    json,
    holdout_metrics json,
    note          varchar
);

create table if not exists meta.monitoring (    -- результаты мониторинга модели
    checked_at    timestamp not null,
    model_version varchar not null,
    metric        varchar not null,       -- psi:<признак> / recent_WAPE / days_since_training / decision
    value         double,
    threshold     double,
    alert         boolean,
    details       varchar
);

-- ============ raw: сырые данные источников как есть ============
create table if not exists raw.hn_items (       -- A: Hacker News (JSON, тексты)
    object_id  varchar not null,
    topic      varchar not null,                -- один документ может относиться к нескольким темам
    created_at timestamp not null,
    payload    json not null,
    loaded_at  timestamp not null,
    run_id     varchar not null,
    primary key (object_id, topic)
);

create table if not exists raw.hn_daily (       -- A: число документов по запросу за сутки (nbHits)
    topic      varchar not null,
    query      varchar not null,
    date       date not null,
    nb_hits    bigint not null,
    exhaustive boolean,
    loaded_at  timestamp not null,
    run_id     varchar not null,
    primary key (topic, query, date)
);

create table if not exists raw.wiki_pageviews ( -- B: Wikimedia (JSON, временной ряд)
    article   varchar not null,
    date      date not null,
    topic     varchar not null,
    views     bigint,
    payload   json,
    loaded_at timestamp not null,
    run_id    varchar not null,
    primary key (article, date)
);

create table if not exists raw.gdelt_timeline ( -- C: GDELT (CSV, объём новостей)
    topic     varchar not null,
    ts        timestamp not null,
    series    varchar not null,
    value     double,
    loaded_at timestamp not null,
    run_id    varchar not null,
    primary key (topic, ts, series)
);

create table if not exists raw.habr_rss (       -- D: Habr (XML/RSS, русские тексты)
    guid      varchar primary key,
    topic     varchar not null,
    hub       varchar not null,
    published timestamp,
    payload   json not null,
    loaded_at timestamp not null,
    run_id    varchar not null
);

-- ============ staging: очищенные таблицы единой схемы ============
create table if not exists staging.documents (
    doc_id        varchar not null,          -- <source>:<id>
    source        varchar not null,          -- hn / habr
    topic         varchar not null,
    ts_utc        timestamp not null,
    date          date not null,
    kind          varchar not null,          -- story / comment / article
    title         varchar,
    text_clean    varchar,
    url           varchar,
    domain        varchar,
    score         integer,
    n_comments    integer,
    lang          varchar,
    author_hash   varchar,                   -- sha256, имя автора не хранится
    primary key (doc_id, topic)
);

create table if not exists staging.pageviews (
    topic   varchar not null,
    article varchar not null,
    date    date not null,
    views   bigint,
    primary key (article, date)
);

create table if not exists staging.news_volume (
    topic            varchar not null,
    date             date not null,
    articles         double,
    total_monitored  double,
    primary key (topic, date)
);

create table if not exists staging.calendar (
    date           date primary key,
    is_holiday_ru  boolean,
    is_holiday_us  boolean,
    holiday_name   varchar
);

-- ============ mart: витрина topic x date (пересобирается целиком в src/processing/mart.py) ============
create table if not exists mart.topic_daily (
    topic             varchar not null,
    date              date not null,
    mentions          double,                -- ЦЕЛЕВАЯ переменная: документов HN по теме за сутки
    mentions_imputed  boolean,
    wiki_views        double,
    wiki_imputed      boolean,
    gdelt_articles    double,
    gdelt_share       double,
    habr_articles     double,
    txt_docs          double,
    txt_share_story   double,
    txt_mean_points   double,
    txt_mean_comments double,
    txt_mean_len      double,
    txt_sentiment     double,
    txt_neg_share     double,
    txt_event_share   double,
    txt_novelty       double,
    is_holiday_us     boolean,
    is_holiday_ru     boolean,
    primary key (topic, date)
);

create table if not exists mart.forecasts (     -- прогнозы: бэктест на отложенной выборке и будущие дни
    model_version varchar not null,
    kind          varchar not null,      -- holdout / future
    topic         varchar not null,
    feature_date  date not null,         -- дата, по которую известны данные
    target_date   date not null,
    horizon       integer not null,
    y_pred        double,
    y_true        double,
    created_at    timestamp,
    primary key (model_version, kind, topic, feature_date, horizon)
);
"""


def connect(cfg: dict) -> duckdb.DuckDBPyConnection:
    """Открывает БД и создаёт схему (если её ещё нет). Повторный вызов ничего не меняет."""
    Path(cfg["paths"]["db"]).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(cfg["paths"]["db"])
    con.execute(SCHEMA)
    sync_topics(con, cfg)
    return con


def sync_topics(con, cfg: dict) -> None:
    """Справочник тем из конфига (upsert, чтобы повторный запуск был идемпотентным)."""
    for name, t in cfg["topics"].items():
        con.execute(
            "insert or replace into meta.topics values (?, ?, ?, ?, ?, ?)",
            [name, t["title"], "; ".join(t["hn_queries"]), "; ".join(t["wiki_articles"]),
             t["gdelt_query"], t["habr_hub"]],
        )


# ---------------------------------------------------------------- журнал загрузок (meta.load_log)
@contextmanager
def logged_run(con, source: str, topic: str | None, period: str):
    """Пишет в журнал running -> success/failed.

    В блоке нужно заполнить stats['fetched'] и stats['inserted']; run_id доступен как stats['run_id'].
    """
    run_id = uuid.uuid4().hex[:12]
    con.execute(
        "insert into meta.load_log values (?, ?, ?, ?, ?, null, 'running', null, null, null)",
        [run_id, source, topic, period, datetime.now()],
    )
    stats = {"run_id": run_id, "fetched": 0, "inserted": 0, "message": None}
    try:
        yield stats
    except Exception as exc:
        con.execute(
            "update meta.load_log set finished_at = ?, status = 'failed', message = ? "
            "where run_id = ?",
            [datetime.now(), f"{type(exc).__name__}: {exc}"[:1000], run_id],
        )
        log.error("[%s %s] ошибка: %s", source, topic, exc)
        raise
    else:
        con.execute(
            "update meta.load_log set finished_at = ?, status = 'success', rows_fetched = ?, "
            "rows_inserted = ?, message = ? where run_id = ?",
            [datetime.now(), stats["fetched"], stats["inserted"], stats["message"], run_id],
        )
        log.info("[%s %s] получено %s, новых %s", source, topic, stats["fetched"], stats["inserted"])


# ---------------------------------------------------------------- водяные знаки инкрементальной загрузки
def wm_get(con, source: str, key: str) -> str | None:
    row = con.execute("select value from meta.watermarks where source = ? and topic = ?",
                      [source, key]).fetchall()
    return row[0][0] if row else None


def wm_date(con, source: str, key: str) -> date | None:
    value = wm_get(con, source, key)
    return date.fromisoformat(value) if value else None


def wm_set(con, source: str, key: str, value) -> None:
    con.execute("insert or replace into meta.watermarks values (?, ?, ?, ?)",
                [source, key, str(value), datetime.now()])
