"""Автоматизированный подбор гиперпараметров LightGBM: python -m src.tune

Optuna (TPE, фиксированный seed); каждая попытка оценивается на той же CV по времени и пишется в журнал
экспериментов (kind = tune). Отложенная выборка в подборе не участвует — лучшая конфигурация проверяется на ней
один раз. Лучшие параметры сохраняются в models/best_params.json и используются python -m src.train fit.
"""
import json
import logging
from pathlib import Path

import optuna
import pandas as pd

from .config import load_config
from .db import connect
from .features import build_features, feature_columns
from .models import evaluate, fit_holdout, make_model
from .registry import data_version, load_mart, log_experiment

log = logging.getLogger("tune")


def tune(con, cfg: dict) -> None:
    """Optuna (TPE, seed) на той же CV; попытки -> журнал; лучшие параметры -> models/best_params.json."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed, tcfg = cfg["model"]["seed"], cfg["model"]["tuning"]
    mart = load_mart(con)
    ver, data_end = data_version(mart), pd.to_datetime(mart["date"]).max().date()
    topics = list(cfg["topics"])
    df = build_features(mart, topics, cfg["model"]["horizons"], cfg["model"]["min_history_days"])
    reg = Path(cfg["paths"]["models"])
    feats_file = reg / "registry.json"
    features = feature_columns(topics)
    if feats_file.exists():      # если уже есть обученная модель — подбираем на её отобранных признаках
        active = json.loads(feats_file.read_text(encoding="utf-8")).get("active")
        if active:
            features = json.loads((reg / active / "meta.json").read_text(encoding="utf-8"))["features"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 900, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "objective": trial.suggest_categorical("objective", ["regression", "huber", "regression_l1"]),
        }
        cv, _ = evaluate(lambda: make_model("lightgbm", cfg, params), df, features, cfg)
        run_id = log_experiment(con, kind="tune", model="lightgbm", params=params, features=features,
                                data_ver=ver, data_end=data_end, seed=seed, cv=cv, holdout=None,
                                note=f"optuna trial {trial.number}")
        trial.set_user_attr("run_id", run_id)
        log.info("попытка %d: CV WAPE %.4f (%s)", trial.number, cv["WAPE"], run_id)
        return cv["WAPE"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=tcfg["n_trials"], timeout=tcfg["timeout_sec"])
    best = study.best_trial
    _, ho, _ = fit_holdout(lambda: make_model("lightgbm", cfg, best.params), df, features, cfg)
    log_experiment(con, kind="tune", model="lightgbm", params=best.params, features=features, data_ver=ver,
                   data_end=data_end, seed=seed, cv={"WAPE": best.value}, holdout=ho,
                   note=f"лучшая попытка {best.number} ({best.user_attrs['run_id']}), проверка на отложенной выборке")
    reg.mkdir(exist_ok=True)
    (reg / "best_params.json").write_text(json.dumps(best.params, indent=2), encoding="utf-8")
    print(f"Лучшая попытка {best.number}: CV WAPE {best.value:.4f}, отложенная выборка WAPE {ho['WAPE']:.4f}")
    print("Параметры сохранены в models/best_params.json (используются make train)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    tune(connect(cfg), cfg)
