"""python -m src.processing build   # пересобрать staging и витрину
python -m src.processing dq      # проверки качества (код 1 при критичных ошибках)
python -m src.processing status  # сводка по таблицам и журналу загрузок
"""
import argparse
import logging

from ..config import load_config
from ..db import connect
from .mart import build_all
from .quality import run_dq


def status(con) -> None:
    print("== Таблицы по слоям ==")
    for schema, table in con.execute(
            "select table_schema, table_name from information_schema.tables "
            "where table_schema in ('raw','staging','mart','meta') order by 1, 2").fetchall():
        print(f"  {schema}.{table:<18}{con.execute(f'select count(*) from {schema}.{table}').fetchall()[0][0]:>10}")
    print("\n== Последние загрузки (meta.load_log) ==")
    print(con.execute(
        "select source, coalesce(topic,'-') topic, status, rows_fetched fetched, rows_inserted inserted, "
        "strftime(started_at, '%d.%m %H:%M') started, message "
        "from meta.load_log order by started_at desc limit 20").df().to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "dq", "status"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    con = connect(cfg)
    if args.cmd == "build":
        build_all(con, cfg)
    elif args.cmd == "dq":
        return run_dq(con, cfg)
    else:
        status(con)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
