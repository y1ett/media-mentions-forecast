"""Источники: A — Hacker News (Algolia API, JSON, тексты; число документов по запросу = целевая переменная),
B — Wikimedia Pageviews (JSON), C — GDELT DOC 2.0 (CSV), D — Habr RSS (XML), E — праздники (библиотека holidays).

Принципы:
- инкрементальность: по каждому (источник, ключ) хранится водяной знак в meta.watermarks, запрашиваются только
  даты после него (плюс refetch_days последних дней — индекс HN догоняет с задержкой);
- идемпотентность: запись через INSERT OR REPLACE / ON CONFLICT DO NOTHING по первичным ключам;
- устойчивость: повторы и 429 в common.get, ограничение частоты; сбой одного источника/темы не останавливает остальные;
- журнал: каждая операция пишет строку в meta.load_log;
- персональные данные не хранятся: имя автора HN -> хэш, автор Habr отбрасывается.
"""
from .calendar import load_calendar
from .hn import load_hn
from .news import load_gdelt, load_habr
from .wiki import load_wiki

__all__ = ["LOADERS", "run", "load_calendar", "load_hn", "load_wiki", "load_gdelt", "load_habr"]


LOADERS = {"hn": load_hn, "wiki": load_wiki, "gdelt": load_gdelt, "habr": load_habr}


def run(con, cfg: dict, sources: list[str], topics: dict) -> int:
    """Запускает загрузчики; ошибка одного (источник, тема) пишется в журнал и не мешает остальным."""
    failed = 0
    for name in sources:
        for topic, tcfg in topics.items():
            try:
                LOADERS[name](con, cfg, topic, tcfg)
            except Exception:
                failed += 1
    return failed
