"""Обучение и эксплуатация моделей.

python -m src.train fit                 # отбор признаков, сравнение моделей, реестр версий
python -m src.train predict             # прогноз на 7 дней активной моделью
python -m src.train experiments         # журнал экспериментов
python -m src.train reproduce <run_id>  # воспроизвести эксперимент
python -m src.train versions            # реестр версий моделей
python -m src.train rollback            # откат к предыдущей версии

Этапы fit:
python -m src.train
1. признаки из витрины (src/features/build.py);
2. отбор групп признаков: абляция на кросс-валидации (LightGBM), группа удаляется, если без неё качество лучше;
3. сравнение всех моделей по единой схеме валидации (CV по времени) и на отложенной выборке;
4. выбор лучшей модели по CV (не по отложенной выборке!), разбивка ошибок по горизонтам и темам;
5. важность признаков перестановками на отложенной выборке;
6. журнал экспериментов, реестр версий, прогнозы на отложенной выборке -> mart.forecasts.
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .db import connect
from .features import build_features, feature_columns, feature_groups, to_prediction
from .models import BASELINES, MODEL_NAMES, breakdown, evaluate, fit_holdout, holdout_start, make_model
from .registry import _read, data_version, load_active, load_mart, log_experiment, register, rollback

log = logging.getLogger("train")
REPORTS = ROOT / "reports"


def best_params(cfg) -> dict:
    f = Path(cfg["paths"]["models"]) / "best_params.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def select_groups(df, cfg, topics, params) -> tuple[list[str], pd.DataFrame]:
    """Абляция групп признаков на CV. Возвращает отобранные признаки и таблицу результатов."""
    full = feature_columns(topics)
    base_cv, _ = evaluate(lambda: make_model("lightgbm", cfg, params), df, full, cfg)
    rows = [{"группа": "все признаки", "WAPE": base_cv["WAPE"], "MAE": base_cv["MAE"], "изменение WAPE": 0.0}]
    drop = []
    for g in feature_groups(topics):
        feats = feature_columns(topics, drop_groups=(g,))
        cv, _ = evaluate(lambda: make_model("lightgbm", cfg, params), df, feats, cfg)
        delta = cv["WAPE"] - base_cv["WAPE"]
        rows.append({"группа": f"без «{g}»", "WAPE": cv["WAPE"], "MAE": cv["MAE"], "изменение WAPE": delta})
        if delta < -0.002:          # без группы заметно лучше -> группа вредит
            drop.append(g)
        log.info("абляция: без %-9s WAPE %.4f (%+.4f)", g, cv["WAPE"], delta)
    table = pd.DataFrame(rows)
    table["решение"] = ["—"] + ["удалить" if g in drop else "оставить" for g in feature_groups(topics)]
    return feature_columns(topics, drop_groups=tuple(drop)), table


def permutation_importance(model, test: pd.DataFrame, features: list[str], seed: int, repeats: int = 3) -> pd.DataFrame:
    """Рост MAE (по y_rel) при перемешивании признака на отложенной выборке."""
    rng = np.random.default_rng(seed)
    X, y = test[features].copy(), test["y_rel"].to_numpy()
    base = np.abs(model.predict(X) - y).mean()
    out = []
    for f in features:
        scores = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            scores.append(np.abs(model.predict(Xp) - y).mean() - base)
        out.append({"feature": f, "importance": float(np.mean(scores)), "std": float(np.std(scores))})
    return pd.DataFrame(out).sort_values("importance", ascending=False)


def fit(con, cfg: dict) -> None:
    """Отбор признаков, сравнение моделей, регистрация версии, прогнозы на отложенной выборке."""
    REPORTS.mkdir(exist_ok=True)
    topics, seed = list(cfg["topics"]), cfg["model"]["seed"]
    mart = load_mart(con)
    ver, data_end = data_version(mart), pd.to_datetime(mart["date"]).max().date()
    df = build_features(mart, topics, cfg["model"]["horizons"], cfg["model"]["min_history_days"])
    log.info("строк признаков: %d, данные %s, отложенная выборка с %s", len(df), ver, holdout_start(df, cfg).date())

    lgb_params = best_params(cfg)
    features, ablation = select_groups(df, cfg, topics, lgb_params)
    ablation.to_csv(REPORTS / "ablation.csv", index=False)
    log.info("отобрано признаков: %d", len(features))

    results, preds_holdout, models = [], {}, {}
    for name in MODEL_NAMES:
        params = lgb_params if name == "lightgbm" else {}
        factory = lambda: make_model(name, cfg, params)
        # базовым моделям отбор признаков не нужен: им нужен только «тот же день неделю назад»
        feats = ["same_dow_rel"] if name in BASELINES else features
        cv, _ = evaluate(factory, df, feats, cfg)
        model, ho, p = fit_holdout(factory, df, feats, cfg)
        run_id = log_experiment(con, kind="train", model=name, params=params, features=feats, data_ver=ver,
                                data_end=data_end, seed=seed, cv=cv, holdout=ho)
        results.append({"model": name, "run_id": run_id, **{f"cv_{k}": v for k, v in cv.items()},
                        **{f"holdout_{k}": v for k, v in ho.items()}})
        preds_holdout[name], models[name] = p, model
        log.info("%-15s CV WAPE %.4f  MAE %.2f | holdout WAPE %.4f  MAE %.2f", name, cv["WAPE"], cv["MAE"],
                 ho["WAPE"], ho["MAE"])

    table = pd.DataFrame(results)
    naive_cv = table.set_index("model").loc["seasonal_naive", "cv_WAPE"]
    naive_ho = table.set_index("model").loc["seasonal_naive", "holdout_WAPE"]
    table["cv_gain_vs_naive_pct"] = 100 * (1 - table["cv_WAPE"] / naive_cv)
    table["holdout_gain_vs_naive_pct"] = 100 * (1 - table["holdout_WAPE"] / naive_ho)
    table.to_csv(REPORTS / "model_comparison.csv", index=False)

    # итоговая модель — из интерпретируемых (точные вклады признаков: TreeSHAP у LightGBM, коэффициенты у Ridge);
    # случайный лес остаётся в сравнении как ещё один класс моделей
    candidates = table[table["model"].isin(["lightgbm", "ridge"])]
    best = candidates.sort_values("cv_WAPE").iloc[0]
    best_name = best["model"]
    p = preds_holdout[best_name]
    breakdown(p, "horizon").to_csv(REPORTS / "holdout_by_horizon.csv")
    breakdown(p, "topic").to_csv(REPORTS / "holdout_by_topic.csv")

    start = holdout_start(df, cfg)
    test = df[(df["target_date"] >= start) & df["y_rel"].notna()]
    perm = permutation_importance(models[best_name], test, features, seed)
    perm.to_csv(REPORTS / "permutation_importance.csv", index=False)

    # финальная модель для эксплуатации: те же параметры, обучение на всех размеченных данных
    labeled = df[df["y_rel"].notna()]
    final = make_model(best_name, cfg, lgb_params if best_name == "lightgbm" else {}).fit(labeled[features],
                                                                                        labeled["y_rel"])
    meta = {"model": best_name, "run_id": best["run_id"], "features": features,
            "params": lgb_params if best_name == "lightgbm" else cfg["model"].get(best_name, {}),
            "data_version": ver, "data_end": str(data_end), "seed": seed,
            "cv": {k[3:]: best[k] for k in best.index if k.startswith("cv_") and "gain" not in k},
            "holdout": {k[8:]: best[k] for k in best.index if k.startswith("holdout_") and "gain" not in k},
            "baseline_holdout_WAPE": float(naive_ho), "holdout_start": str(start.date()),
            "trained_at": datetime.now().isoformat(timespec="seconds")}
    version, active = register(cfg, final, meta)

    # прогнозы всех моделей на отложенной выборке — для дашборда (факт против прогноза)
    con.execute("delete from mart.forecasts where kind = 'holdout'")
    for name, ph in preds_holdout.items():
        rows = ph.assign(model_version=f"{version}:{name}", kind="holdout", created_at=datetime.now())
        con.register("fc", rows)
        con.execute("insert or replace into mart.forecasts by name select model_version, kind, topic, "
                    "feature_date::date feature_date, target_date::date target_date, horizon, y_pred, y_true, "
                    "created_at from fc")
        con.unregister("fc")

    pd.set_option("display.width", 200)
    show = table[["model", "cv_MAE", "cv_WAPE", "holdout_MAE", "holdout_WAPE", "holdout_within_15pct",
                  "holdout_gain_vs_naive_pct"]]
    print("\n" + show.round(3).to_string(index=False))
    print(f"\nЛучшая по CV: {best_name}. На отложенной выборке ошибка в среднем ±{best['holdout_MAE']:.0f} "
          f"упоминаний в день (WAPE {best['holdout_WAPE']:.1%}), точнее наивного прогноза на "
          f"{100 * (1 - best['holdout_WAPE'] / naive_ho):.1f}%.")
    print(f"Версия {version} {'стала активной' if active else 'сохранена, но не активирована (хуже текущей)'}.")


def predict(con, cfg: dict) -> None:
    """Прогноз на 1..7 дней вперёд активной моделью -> mart.forecasts (kind = future)."""
    version, model, meta = load_active(cfg)
    mart = load_mart(con)
    df = build_features(mart, list(cfg["topics"]), cfg["model"]["horizons"], cfg["model"]["min_history_days"])
    last = df["feature_date"].max()
    rows = df[df["feature_date"] == last].copy()
    rows["y_pred"] = to_prediction(rows["base"], model.predict(rows[meta["features"]]))
    rows = rows.assign(model_version=version, kind="future", created_at=datetime.now())
    con.register("fc", rows)
    con.execute("insert or replace into mart.forecasts by name select model_version, kind, topic, "
                "feature_date::date feature_date, target_date::date target_date, horizon, y_pred, y_true, "
                "created_at from fc")
    con.unregister("fc")
    piv = rows.pivot_table(index="topic", columns=rows["target_date"].dt.strftime("%d.%m"), values="y_pred")
    print(f"Прогноз модели {version} ({meta['model']}) по данным на {last:%d.%m.%Y}:")
    print(piv.round(0).to_string())


def reproduce(con, cfg: dict, run_id: str) -> dict:
    """Повторяет эксперимент из журнала: те же модель, параметры, признаки, seed и срез данных."""
    row = con.execute("select model, params, features, data_version, data_end, seed, cv_metrics "
                      "from meta.experiments where run_id = ?", [run_id]).fetchall()
    if not row:
        raise SystemExit(f"эксперимент {run_id} не найден")
    model, params, features, ver, data_end, seed, cv = row[0]
    params, features, cv = json.loads(params), json.loads(features), json.loads(cv)
    cfg["model"]["seed"] = seed
    mart = load_mart(con, data_end)
    same_data = data_version(mart) == ver
    df = build_features(mart, list(cfg["topics"]), cfg["model"]["horizons"], cfg["model"]["min_history_days"])
    new_cv, _ = evaluate(lambda: make_model(model, cfg, params), df, features, cfg)
    print(f"Эксперимент {run_id}: модель {model}, данные {ver} ({'совпадают' if same_data else 'ИЗМЕНИЛИСЬ'})")
    print(f"{'метрика':<14}{'в журнале':>12}{'сейчас':>12}")
    for k in ("MAE", "RMSE", "WAPE", "sMAPE", "within_15pct"):
        print(f"{k:<14}{cv[k]:>12.4f}{new_cv[k]:>12.4f}")
    match = all(abs(cv[k] - new_cv[k]) < 1e-6 for k in ("MAE", "WAPE"))
    print("Результат воспроизведён" if match else "Метрики отличаются")
    log_experiment(con, kind="reproduce", model=model, params=params, features=features,
                   data_ver=data_version(mart), data_end=data_end, seed=seed, cv=new_cv, holdout=None,
                   note=f"воспроизведение {run_id}: {'совпало' if match else 'не совпало'}")
    return new_cv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "predict", "experiments", "reproduce", "versions", "rollback"])
    ap.add_argument("run_id", nargs="?")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    if a.cmd == "versions":
        reg = _read(cfg)
        for v, info in sorted(reg["versions"].items()):
            mark = "*" if v == reg["active"] else " "
            print(f"{mark} {v}  {info['model']:<14} holdout WAPE {info['holdout_WAPE']:.4f}  база {info['baseline_WAPE']:.4f}")
        return 0
    if a.cmd == "rollback":
        print("Активная версия после отката:", rollback(cfg))
        return 0
    con = connect(cfg)
    if a.cmd == "experiments":
        pd.set_option("display.width", 200)
        print(con.execute("select run_id, kind, model, data_version, cv_metrics->>'WAPE' cv_wape, "
                          "holdout_metrics->>'WAPE' holdout_wape, note from meta.experiments order by created_at")
              .df().to_string(index=False))
    elif a.cmd == "reproduce":
        if not a.run_id:
            raise SystemExit("укажите run_id: python -m src.train reproduce <run_id>")
        reproduce(con, cfg, a.run_id)
    else:
        {"fit": fit, "predict": predict}[a.cmd](con, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
