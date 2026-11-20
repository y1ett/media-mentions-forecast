"""Интерпретация модели: python -m src.explain --topic ai_llm --date 2026-10-01 --horizon 1

Интерпретация модели.

- важность признаков тремя способами: встроенная (gain) LightGBM, перестановками (reports/permutation_importance.csv),
  среднее |SHAP|;
- объяснение отдельного прогноза: вклад каждого признака (SHAP; для LightGBM — точные TreeSHAP через pred_contrib,
  для Ridge — коэффициент x стандартизованное значение);
- зависимость прогноза от значения признака (partial dependence);
- сравнение структуры важности между периодами (|SHAP| по кварталам + ранговая корреляция Спирмена).
"""
import argparse

import numpy as np
import pandas as pd

from .config import load_config
from .db import connect
from .features import build_features, to_prediction
from .registry import load_active, load_mart


def _final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    """Вклады признаков в прогноз y_rel (лог-отношение к среднему за 7 дней) + столбец bias."""
    est = _final_estimator(model)
    if est.__class__.__name__ == "LGBMRegressor":
        contrib = est.predict(X, pred_contrib=True)
        return pd.DataFrame(contrib, columns=list(X.columns) + ["bias"], index=X.index)
    if hasattr(model, "steps") and hasattr(est, "coef_"):
        Xt = X
        for _, step in model.steps[:-1]:
            Xt = step.transform(Xt)
        contrib = Xt * est.coef_
        out = pd.DataFrame(contrib, columns=X.columns, index=X.index)
        out["bias"] = est.intercept_
        return out
    raise ValueError("вклады признаков доступны для LightGBM и Ridge")


def load_context(cfg=None):
    cfg = cfg or load_config()
    con = connect(cfg)
    version, model, meta = load_active(cfg)
    df = build_features(load_mart(con), list(cfg["topics"]), cfg["model"]["horizons"], cfg["model"]["min_history_days"])
    return cfg, version, model, meta, df


def explain_row(model, meta, df, topic: str, feature_date, horizon: int) -> tuple[pd.DataFrame, dict]:
    row = df[(df["topic"] == topic) & (df["feature_date"] == pd.Timestamp(feature_date)) & (df["horizon"] == horizon)]
    if row.empty:
        raise ValueError("нет строки для такой темы, даты и горизонта")
    X = row[meta["features"]]
    sv = shap_values(model, X).iloc[0]
    contrib = sv.drop("bias").rename("shap").to_frame()
    contrib["value"] = X.iloc[0]
    contrib["effect_pct"] = 100 * (np.exp(contrib["shap"]) - 1)   # мультипликативный эффект на прогноз
    contrib = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index)
    pred = float(to_prediction(row["base"], model.predict(X))[0])
    info = {"topic": topic, "feature_date": pd.Timestamp(feature_date).date(), "horizon": horizon,
            "target_date": (pd.Timestamp(feature_date) + pd.Timedelta(days=horizon)).date(),
            "base_ma7": float(row["base"].iloc[0]), "bias": float(sv["bias"]), "prediction": pred,
            "actual": None if pd.isna(row["y_true"].iloc[0]) else float(row["y_true"].iloc[0])}
    return contrib, info


def gain_importance(model, features) -> pd.Series:
    est = _final_estimator(model)
    if hasattr(est, "booster_"):
        return pd.Series(est.booster_.feature_importance("gain"), index=features).sort_values(ascending=False)
    if hasattr(est, "feature_importances_"):
        return pd.Series(est.feature_importances_, index=features).sort_values(ascending=False)
    if hasattr(est, "coef_"):
        return pd.Series(np.abs(est.coef_), index=features).sort_values(ascending=False)
    return pd.Series(dtype=float)


def shap_importance_by_period(model, meta, df, freq: str = "Q") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Среднее |SHAP| по периодам и ранговая корреляция важностей между периодами."""
    lab = df[df["y_rel"].notna()].copy()
    lab["period"] = lab["feature_date"].dt.to_period(freq).astype(str)
    sample = lab.sample(frac=1.0, random_state=0).groupby("period").head(1500)
    sv = shap_values(model, sample[meta["features"]]).drop(columns="bias").abs()
    sv["period"] = sample["period"].values
    imp = sv.groupby("period").mean().T
    corr = imp.rank().corr(method="pearson")      # корреляция рангов = Спирмен
    return imp, corr


def partial_dependence(model, meta, df, feature: str, n_grid: int = 20, sample: int = 3000) -> pd.DataFrame:
    """Средний прогноз y_rel при подстановке значения признака всем строкам выборки."""
    lab = df[df["y_rel"].notna()].sample(min(sample, int(df["y_rel"].notna().sum())), random_state=0)
    X = lab[meta["features"]].copy()
    grid = np.unique(np.nanquantile(X[feature], np.linspace(0.02, 0.98, n_grid)))
    out = []
    for v in grid:
        Xv = X.copy()
        Xv[feature] = v
        p = model.predict(Xv)
        out.append({"value": v, "mean_y_rel": float(np.mean(p)), "effect_pct": float(100 * (np.exp(np.mean(p)) - 1))})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--date", required=True, help="дата t, по которую известны данные (YYYY-MM-DD)")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    cfg, version, model, meta, df = load_context()
    contrib, info = explain_row(model, meta, df, a.topic, a.date, a.horizon)
    print(f"Модель {version} ({meta['model']}). Тема {info['topic']}, данные по {info['feature_date']}, "
          f"прогноз на {info['target_date']} (горизонт {info['horizon']})")
    print(f"Среднее за 7 дней: {info['base_ma7']:.0f}; прогноз: {info['prediction']:.0f}; "
          f"факт: {info['actual'] if info['actual'] is not None else 'ещё неизвестен'}")
    print("\nВклад признаков (эффект на прогноз, %):")
    print(contrib.head(a.top)[["value", "effect_pct"]].round(3).to_string())


if __name__ == "__main__":
    main()
