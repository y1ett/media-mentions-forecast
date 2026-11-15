"""Вспомогательные функции загрузки: пакетная запись, ограничение частоты запросов.
"""
import threading
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd


def last_complete_day() -> date:
    return datetime.now(timezone.utc).date() - timedelta(days=1)


def _bulk(con, sql: str, df: pd.DataFrame) -> None:
    """Пакетная запись DataFrame (в разы быстрее построчного executemany)."""
    con.register("df", df)
    try:
        con.execute(sql)
    finally:
        con.unregister("df")


def _count(con, table: str) -> int:
    return con.execute(f"select count(*) from {table}").fetchall()[0][0]


class RateLimiter:
    """Не более rps запросов в секунду суммарно по всем потокам."""

    def __init__(self, rps: float):
        self.interval, self.lock, self.next_at = 1.0 / rps, threading.Lock(), 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next_at - now
            self.next_at = max(now, self.next_at) + self.interval
        if delay > 0:
            time.sleep(delay)
