"""Модели и схема валидации.

- seasonal_naive — БАЗОВАЯ: значение того же дня недели неделю назад (y[t+h-7]);
- moving_average — вторая базовая: среднее за последние 7 дней (y_rel = 0);
- ridge — линейная модель с L2-регуляризацией (класс: линейные модели);
- random_forest — бэггинг деревьев (класс: ансамбли на бэггинге);
- lightgbm — градиентный бустинг деревьев (класс: бустинг).

Схема валидации с учётом времени и метрики.

Разбиение — по дате цели (target_date):
- отложенная выборка: последние holdout_days дней; не участвует ни в обучении, ни в выборе модели/признаков/параметров;
- до неё cv_folds окон по cv_fold_days дней, обучение на всех строках, у которых дата цели раньше начала окна
  (расширяющееся окно). Строки обучения с датой цели внутри окна валидации исключаются — так горизонты 1..7
  не «подглядывают» в период проверки.
Случайность есть только внутри моделей и зафиксирована seed из конфига.

Реестр версий моделей: models/<версия>/ (model.pkl + meta.json) и models/registry.json (активная версия).
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import to_prediction


BASELINES = ("seasonal_naive", "moving_average")


class SeasonalNaive:
    """Прогноз = значение того же дня недели неделю назад. Ничего не обучает."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        v = X["same_dow_rel"].to_numpy(dtype=float)
        return np.where(np.isnan(v), 0.0, v)


class MovingAverage:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def make_model(name: str, cfg: dict, params: dict | None = None):
    m, seed = cfg["model"], cfg["model"]["seed"]
    if name == "seasonal_naive":
        return SeasonalNaive()
    if name == "moving_average":
        return MovingAverage()
    if name == "ridge":
        p = {**m["ridge"], **(params or {})}
        # импутация и масштабирование обучаются внутри Pipeline только на обучающей части
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(**p))
    if name == "random_forest":
        p = {**m["random_forest"], **(params or {})}
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(random_state=seed, **p))
    if name == "lightgbm":
        p = {**m["lightgbm"], **(params or {})}
        return LGBMRegressor(random_state=seed, **p)
    raise ValueError(f"неизвестная модель {name}")


MODEL_NAMES = ["seasonal_naive", "moving_average", "ridge", "random_forest", "lightgbm"]


# ---------------------------------------------------------------- валидация по времени и метрики
def holdout_start(df: pd.DataFrame, cfg: dict) -> pd.Timestamp:
    last = df.loc[df["y_true"].notna(), "target_date"].max()
    return last - pd.Timedelta(days=cfg["model"]["holdout_days"] - 1)


def cv_folds(df: pd.DataFrame, cfg: dict) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    m = cfg["model"]
    end = holdout_start(df, cfg)
    folds = []
    for i in range(m["cv_folds"], 0, -1):
        start = end - pd.Timedelta(days=m["cv_fold_days"] * i)
        folds.append((start, start + pd.Timedelta(days=m["cv_fold_days"])))
    return folds


def split(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    labeled = df[df["y_rel"].notna()]
    train = labeled[labeled["target_date"] < start]
    valid = labeled[(labeled["target_date"] >= start) & (labeled["target_date"] < end)]
    return train, valid


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    denom = np.abs(y_true) + np.abs(y_pred)
    return {
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "WAPE": float(abs_err.sum() / max(np.abs(y_true).sum(), 1e-9)),
        "sMAPE": float(np.mean(np.where(denom > 0, 2 * abs_err / np.where(denom > 0, denom, 1), 0))),
        "within_15pct": float(np.mean(abs_err <= 0.15 * np.abs(y_true))),
        "n": int(len(y_true)),
    }


def evaluate(model_factory, df: pd.DataFrame, features: list[str], cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Кросс-валидация по времени. Возвращает средние метрики и прогнозы на всех окнах."""
    preds = []
    for k, (start, end) in enumerate(cv_folds(df, cfg)):
        train, valid = split(df, start, end)
        model = model_factory().fit(train[features], train["y_rel"])
        p = valid[["topic", "feature_date", "target_date", "horizon", "base", "y_true", "target_imputed"]].copy()
        p["y_pred"] = to_prediction(valid["base"], model.predict(valid[features]))
        p["fold"] = k
        preds.append(p)
    preds = pd.concat(preds, ignore_index=True)
    ok = preds[~preds["target_imputed"]]
    return metrics(ok["y_true"], ok["y_pred"]), preds


def fit_holdout(model_factory, df: pd.DataFrame, features: list[str], cfg: dict):
    """Обучение на всём до отложенной выборки, оценка на ней."""
    start = holdout_start(df, cfg)
    train, test = split(df, start, start + pd.Timedelta(days=10_000))
    model = model_factory().fit(train[features], train["y_rel"])
    p = test[["topic", "feature_date", "target_date", "horizon", "base", "y_true", "target_imputed"]].copy()
    p["y_pred"] = to_prediction(test["base"], model.predict(test[features]))
    ok = p[~p["target_imputed"]]
    return model, metrics(ok["y_true"], ok["y_pred"]), p


def breakdown(preds: pd.DataFrame, by: str) -> pd.DataFrame:
    ok = preds[~preds["target_imputed"]]
    return pd.DataFrame({k: metrics(g["y_true"], g["y_pred"]) for k, g in ok.groupby(by)}).T
