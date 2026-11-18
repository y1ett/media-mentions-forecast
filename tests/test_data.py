"""Хранилище, конфигурация, загрузка: идемпотентность, устойчивость к сбоям, отсутствие персональных данных."""
from src import ingest
from src.config import load_config
from src.db import connect, logged_run
from src.http_client import SourceUnavailable
from src.ingest import hn, news, wiki

EXPECTED = {
    "raw": {"hn_items", "hn_daily", "wiki_pageviews", "gdelt_timeline", "habr_rss"},
    "staging": {"documents", "pageviews", "news_volume", "calendar"},
    "mart": {"topic_daily", "forecasts"},
    "meta": {"topics", "load_log", "watermarks", "dq_results", "experiments", "monitoring"},
}


def _tables(con):
    out = {}
    for schema, table in con.execute("select table_schema, table_name from information_schema.tables "
                                     "where table_schema in ('raw','staging','mart','meta')").fetchall():
        out.setdefault(schema, set()).add(table)
    return out


def test_schema_created_and_connect_is_idempotent(cfg):
    con = connect(cfg)
    assert _tables(con) == EXPECTED
    con.close()
    con = connect(cfg)                     # повторный запуск ничего не ломает и не дублирует
    assert _tables(con) == EXPECTED
    assert con.execute("select count(*) from meta.topics").fetchall()[0][0] == len(cfg["topics"])


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("HTTP_RETRIES", "9")
    monkeypatch.setenv("CONTACT_EMAIL", "me@test.org")
    cfg = load_config()
    assert cfg["http"]["retries"] == 9 and "me@test.org" in cfg["project"]["user_agent"]
    for name, t in cfg["topics"].items():
        for key in ("title", "hn_queries", "wiki_articles", "gdelt_query", "habr_hub"):
            assert t.get(key), f"{name}: нет {key}"


class FakeResp:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


def _fake_wiki(views):
    def fake_get(url, **kw):
        return FakeResp({"items": [{"timestamp": f"202601{d:02d}00", "views": v} for d, v in views]})
    return fake_get


def test_incremental_load_is_idempotent(cfg, con, monkeypatch):
    monkeypatch.setattr(wiki, "get", _fake_wiki([(1, 10), (2, 20), (3, 30)]))
    tcfg = {"wiki_articles": ["Bitcoin"]}
    wiki.load_wiki(con, cfg, "crypto", tcfg)
    wiki.load_wiki(con, cfg, "crypto", tcfg)            # повторный запуск — те же данные, без дублей
    assert con.execute("select count(*), sum(views) from raw.wiki_pageviews").fetchall()[0] == (3, 60)
    assert con.execute("select value from meta.watermarks where source = 'wiki'").fetchall()[0][0] == "2026-01-03"


def test_failed_source_is_logged_and_others_continue(cfg, con, monkeypatch):
    def broken(*a, **k):
        raise SourceUnavailable("источник недоступен")
    monkeypatch.setattr(ingest, "LOADERS", {"hn": broken, "wiki": wiki.load_wiki})
    monkeypatch.setattr(wiki, "get", _fake_wiki([(1, 10)]))
    failed = ingest.run(con, cfg, ["hn", "wiki"], {"crypto": {"wiki_articles": ["Bitcoin"]}})
    assert failed == 1
    assert con.execute("select count(*) from raw.wiki_pageviews").fetchall()[0][0] == 1   # wiki загружен


def test_logged_run_records_failure(con):
    try:
        with logged_run(con, "hn", "crypto", "test"):
            raise SourceUnavailable("HTTP 503")
    except SourceUnavailable:
        pass
    status, msg = con.execute("select status, message from meta.load_log").fetchall()[0]
    assert status == "failed" and "503" in msg


def test_author_names_are_not_stored():
    hit = hn._sanitize({"author": "bryan0", "_tags": ["comment", "author_bryan0", "story_1"], "objectID": "1"})
    assert "author" not in hit and "bryan0" not in str(hit) and hit["author_hash"]
    assert "someone" not in news._strip_authors("<item><dc:creator>someone</dc:creator></item>")
