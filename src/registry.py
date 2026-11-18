"""Реестр версий моделей и журнал экспериментов.

Правило вывода в эксплуатацию: новая версия становится активной, если её WAPE на отложенной выборке не хуже
активной более чем на promote_tolerance (по умолчанию: строго лучше) и лучше базовой модели.
Откат: python -m src.train rollback — активной становится предыдущая версия.
"""
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


def _root(cfg) -> Path:
    p = Path(cfg["paths"]["models"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read(cfg) -> dict:
    f = _root(cfg) / "registry.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"active": None, "history": [], "versions": {}}


def _write(cfg, reg: dict) -> None:
    (_root(cfg) / "registry.json").write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")


def register(cfg, model, meta: dict) -> tuple[str, bool]:
    """Сохраняет версию; возвращает (версия, стала ли активной)."""
    version = datetime.now().strftime("v%Y%m%d-%H%M%S")
    d = _root(cfg) / version
    d.mkdir()
    joblib.dump(model, d / "model.pkl")
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    reg = _read(cfg)
    reg["versions"][version] = {"model": meta["model"], "holdout_WAPE": meta["holdout"]["WAPE"],
                                "baseline_WAPE": meta["baseline_holdout_WAPE"], "created": version[1:]}
    current = reg["versions"].get(reg["active"]) if reg["active"] else None
    better_than_base = meta["holdout"]["WAPE"] < meta["baseline_holdout_WAPE"]
    promote = better_than_base and (current is None or meta["holdout"]["WAPE"] <= current["holdout_WAPE"]
                                    or meta.get("force_promote", False))
    if promote:
        if reg["active"]:
            reg["history"].append(reg["active"])
        reg["active"] = version
    _write(cfg, reg)
    return version, promote


def load_active(cfg):
    reg = _read(cfg)
    if not reg["active"]:
        raise SystemExit("нет активной модели: выполните make train")
    d = _root(cfg) / reg["active"]
    return reg["active"], joblib.load(d / "model.pkl"), json.loads((d / "meta.json").read_text(encoding="utf-8"))


def rollback(cfg) -> str:
    reg = _read(cfg)
    if not reg["history"]:
        raise SystemExit("нет предыдущей версии для отката")
    reg["active"] = reg["history"].pop()
    _write(cfg, reg)
    return reg["active"]


# ---------------------------------------------------------------- журнал экспериментов (meta.experiments)
def data_version(mart: pd.DataFrame) -> str:
    """Версия данных: хэш содержимого витрины (устойчив к порядку строк) + последняя дата."""
    m = mart.sort_values(["topic", "date"]).reset_index(drop=True)
    m = m.assign(date=m["date"].astype(str)).round(8)
    h = hashlib.sha256(pd.util.hash_pandas_object(m, index=False).values.tobytes()).hexdigest()[:12]
    return f"{h}@{pd.to_datetime(m['date']).max():%Y-%m-%d}"


def log_experiment(con, *, kind: str, model: str, params: dict, features: list[str], data_ver: str,
                   data_end, seed: int, cv: dict | None, holdout: dict | None, note: str = "") -> str:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    con.execute("insert into meta.experiments values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [run_id, datetime.now(), kind, model, json.dumps(params, default=str), json.dumps(features),
                 data_ver, data_end, seed, json.dumps(cv), json.dumps(holdout), note])
    return run_id


def load_mart(con, data_end=None) -> pd.DataFrame:
    mart = con.execute("select * from mart.topic_daily order by topic, date").df()
    if data_end is not None:
        mart = mart[pd.to_datetime(mart["date"]) <= pd.to_datetime(data_end)]
    return mart.reset_index(drop=True)
