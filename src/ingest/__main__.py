"""Инкрементальная загрузка: python -m src.ingest [all|hn|wiki|gdelt|habr|calendar] [--topics a,b]
"""
import argparse
import logging

from ..config import load_config
from ..db import connect
from . import LOADERS, load_calendar, run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="all", choices=["all", "calendar", *LOADERS])
    ap.add_argument("--topics", default="", help="через запятую; по умолчанию все темы из конфига")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    con = connect(cfg)
    wanted = [t for t in args.topics.split(",") if t]
    topics = {k: v for k, v in cfg["topics"].items() if not wanted or k in wanted}
    failed = 0
    if args.source in ("all", "calendar"):
        try:
            load_calendar(con, cfg)
        except Exception:
            failed += 1
    failed += run(con, cfg, list(LOADERS) if args.source == "all" else
                  [s for s in LOADERS if s == args.source], topics)
    print(f"Загрузка завершена, ошибок: {failed} (подробности — meta.load_log)")
    return 0     # частичные сбои не останавливают пайплайн: недогруженное доберётся следующим запуском


if __name__ == "__main__":
    raise SystemExit(main())
