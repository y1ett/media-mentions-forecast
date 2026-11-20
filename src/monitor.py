"""Мониторинг качества модели и правило переобучения.

python -m src.monitor                                  # проверка -> meta.monitoring
python -m src.monitor --auto-retrain                   # переобучить, если сработало правило
python -m src.monitor --simulate-degradation ai_llm    # демонстрация: сдвиг данных темы (в памяти)

Что проверяется:
1. дрейф входных данных — PSI ключевых признаков (топ по важности, без календаря и темы):
   эталон = 180 дней до обучения модели, текущее окно = до 28 последних дней ПОСЛЕ обучения (не менее min_new_days);
2. качество на новых фактах — WAPE активной модели на датах цели после даты обучения (данные, которых модель
   не видела); заодно в mart.forecasts проставляются фактические значения для прошлых прогнозов;
3. возраст модели.

ПРАВИЛО ПЕРЕОБУЧЕНИЯ (явное): переобучить, если выполнено хотя бы одно:
  a) WAPE на новых фактах > retrain_wape_factor x WAPE модели на отложенной выборке (при >= min_rows строк);
  b) PSI > psi_threshold хотя бы у psi_min_features ключевых признаков;
  c) с момента обучения прошло > max_model_age_days дней.
Вывод новой версии в эксплуатацию и откат — правило в src/models.py.
"""
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from .config import load_config
from .db import connect
from .explain import gain_importance
from .features import build_features, to_prediction
from .models import metrics
from .registry import load_active, load_mart

log = logging.getLogger("monitor")


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index по квантильным корзинам эталона."""
    ref, cur = ref.dropna().to_numpy(), cur.dropna().to_numpy()
    if len(ref) < 50 or len(cur) < 20:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def check(cfg: dict, mart: pd.DataFrame | None = None, note: str = "",
          new_from: pd.Timestamp | None = None) -> tuple[pd.DataFrame, bool]:
    mon = cfg["monitoring"]
    con = connect(cfg)
    version, model, meta = load_active(cfg)
    mart = load_mart(con) if mart is None else mart
    df = build_features(mart, list(cfg["topics"]), cfg["model"]["horizons"], cfg["model"]["min_history_days"])
    trained_end = pd.Timestamp(meta["data_end"]) if new_from is None else new_from
    rows = []

    # 1. дрейф ключевых признаков: данные, которых модель не видела, против данных обучения.
    #    Календарь и тема не проверяются — их «сдвиг» ожидаем (наступил другой месяц).
    key = [f for f in gain_importance(model, meta["features"]).index
           if not f.startswith(("topic_", "target_")) and f != "horizon"][: mon["psi_top_features"]]
    h1 = df[df["horizon"] == 1]
    ref = h1[(h1["feature_date"] <= trained_end) & (h1["feature_date"] > trained_end - pd.Timedelta(days=180))]
    cur = h1[h1["feature_date"] > trained_end]
    cur = cur[cur["feature_date"] > cur["feature_date"].max() - pd.Timedelta(days=mon["current_window_days"])] if len(cur) else cur
    new_days = cur["feature_date"].nunique()
    drifted = 0
    if new_days >= mon["min_new_days"]:
        for f in key:
            v = psi(ref[f], cur[f])
            alert = bool(v > mon["psi_threshold"]) if not np.isnan(v) else False
            drifted += alert
            rows.append(("psi:" + f, v, mon["psi_threshold"], alert, f"новых дней {new_days}"))
    else:
        rows.append(("psi", np.nan, mon["psi_threshold"], False,
                     f"мало новых данных после обучения ({new_days} < {mon['min_new_days']} дней)"))
    drift_alert = drifted >= mon["psi_min_features"]

    # 2. качество на новых фактах (даты цели после обучения модели)
    new = df[(df["target_date"] > trained_end) & df["y_true"].notna() & ~df["target_imputed"]]
    if len(new) >= mon["min_rows"]:
        pred = to_prediction(new["base"], model.predict(new[meta["features"]]))
        m = metrics(new["y_true"], pred)
        limit = mon["retrain_wape_factor"] * meta["holdout"]["WAPE"]
        quality_alert = m["WAPE"] > limit
        rows.append(("recent_WAPE", m["WAPE"], limit, quality_alert, f"строк {m['n']}, MAE {m['MAE']:.1f}"))
    else:
        quality_alert = False
        rows.append(("recent_WAPE", np.nan, np.nan, False, f"мало новых фактов ({len(new)} < {mon['min_rows']})"))

    # 3. возраст модели
    age = (pd.Timestamp(datetime.now().date()) - pd.Timestamp(meta["data_end"])).days
    age_alert = age > mon["max_model_age_days"]
    rows.append(("days_since_training", age, mon["max_model_age_days"], age_alert, ""))

    retrain = bool(quality_alert or drift_alert or age_alert)
    reasons = [n for n, a in (("качество", quality_alert), ("дрейф", drift_alert), ("возраст", age_alert)) if a]
    rows.append(("decision_retrain", float(retrain), 1, retrain,
                 (("переобучить: " + ", ".join(reasons)) if retrain else "переобучение не требуется")
                 + (f" [{note}]" if note else "")))

    res = pd.DataFrame(rows, columns=["metric", "value", "threshold", "alert", "details"])
    res.insert(0, "model_version", version)
    res.insert(0, "checked_at", datetime.now())
    con.register("mon", res)
    con.execute("insert into meta.monitoring by name select * from mon")
    con.unregister("mon")
    # факты для прошлых прогнозов
    con.execute("""update mart.forecasts f set y_true = m.mentions from mart.topic_daily m
                   where f.kind = 'future' and f.topic = m.topic and f.target_date = m.date and f.y_true is null""")
    return res, retrain


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto-retrain", action="store_true")
    ap.add_argument("--simulate-degradation", metavar="TOPIC",
                    help="умножить упоминания темы за последние 28 дней на 3 (только в памяти, БД не меняется)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    mart, note = None, ""
    if args.simulate_degradation:
        con = connect(cfg)
        mart = load_mart(con)
        con.close()
        d = pd.to_datetime(mart["date"])
        mask = (mart["topic"] == args.simulate_degradation) & (d > d.max() - pd.Timedelta(days=28))
        mart.loc[mask, "mentions"] *= 3
        mart.loc[mask, "wiki_views"] *= 3
        note = f"симуляция: {args.simulate_degradation} x3 за 28 дней"
    new_from = pd.to_datetime(mart["date"]).max() - pd.Timedelta(days=28) if mart is not None else None
    res, retrain = check(cfg, mart, note, new_from)
    pd.set_option("display.width", 200)
    print(res[["metric", "value", "threshold", "alert", "details"]].round(4).to_string(index=False))
    if retrain and args.auto_retrain and not args.simulate_degradation:
        print("\nСработало правило переобучения -> обучение новой версии")
        from .train import fit
        fit(connect(cfg), cfg)
    elif retrain:
        print("\nСработало правило переобучения (make retrain или python -m src.monitor --auto-retrain)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
