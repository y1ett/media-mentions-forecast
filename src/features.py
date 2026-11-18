"""Признаки для моделей. Один и тот же код используется при обучении, оценке и прогнозе в эксплуатации.

Постановка (прямой многогоризонтный прогноз): строка = (тема, дата t, горизонт h).
Все признаки строки считаются только по данным, известным на конец дня t (сам день t завершён).
Цель строки — mentions в день t+h. Одна модель на все горизонты 1..7, горизонт — признак.

Цель моделей — относительная: y_rel = log1p(mentions[t+h]) - log1p(base_t), где base_t — среднее за 7 дней
t-6..t. Это выравнивает масштаб тем (от десятков до сотен упоминаний в день); прогноз = expm1(log1p(base) + y_rel).

Против утечки целевой переменной:
- признаки из ряда берутся по датам <= t, цель — по дате t+h > t (проверяется тестами);
- календарь даты t+h известен заранее;
- текстовые признаки дня t считаются без обучения (см. src/processing/text.py);
- масштабирование и кодирование для линейной модели обучаются внутри sklearn Pipeline только на train.
"""
import numpy as np
import pandas as pd

LAGS = [0, 1, 2, 3, 6, 13, 27]
TEXT_COLS = ["txt_share_story", "txt_mean_points", "txt_mean_comments", "txt_mean_len", "txt_sentiment",
             "txt_neg_share", "txt_event_share", "txt_novelty", "txt_docs"]
KEYS = ["topic", "feature_date", "horizon", "target_date"]


def feature_groups(topics: list[str]) -> dict[str, list[str]]:
    return {
        "lags": [f"lag{k}_rel" for k in LAGS] + ["same_dow_rel", "same_dow2_rel"],
        "rolling": ["log_base", "ma28_rel", "std7_rel", "max7_rel", "min7_rel", "ewm3_rel"],
        "calendar": ["horizon", "target_dow", "target_dow_sin", "target_dow_cos", "target_weekend",
                     "target_holiday_us", "target_holiday_ru", "target_month"],
        "wiki": ["wiki_rel", "wiki_trend", "wiki_log", "wiki_lag1_rel"],
        "text": [f"{c}" for c in TEXT_COLS] + ["txt_novelty_ma3", "txt_event_ma3", "txt_sentiment_ma7"],
        "topic": [f"topic_{t}" for t in topics],
    }


def feature_columns(topics: list[str], drop_groups: tuple[str, ...] = ()) -> list[str]:
    return [c for g, cols in feature_groups(topics).items() if g not in drop_groups for c in cols]


def _day_features(g: pd.DataFrame) -> pd.DataFrame:
    """Признаки на уровне (тема, день t): только прошлое и сам день t."""
    y = g["mentions"].astype(float)
    ly = np.log1p(y)
    base = y.rolling(7, min_periods=7).mean()
    lbase = np.log1p(base)
    out = pd.DataFrame(index=g.index)
    out["base"] = base
    out["log_base"] = lbase
    for k in LAGS:
        out[f"lag{k}_rel"] = ly.shift(k) - lbase
    out["ma28_rel"] = np.log1p(y.rolling(28, min_periods=28).mean()) - lbase
    out["std7_rel"] = y.rolling(7, min_periods=7).std() / (base + 1)
    out["max7_rel"] = np.log1p(y.rolling(7).max()) - lbase
    out["min7_rel"] = np.log1p(y.rolling(7).min()) - lbase
    out["ewm3_rel"] = np.log1p(y.ewm(span=3, adjust=False).mean()) - lbase

    w = g["wiki_views"].astype(float)
    lw7 = np.log1p(w.rolling(7, min_periods=5).mean())
    out["wiki_rel"] = np.log1p(w) - lw7
    out["wiki_trend"] = lw7 - np.log1p(w.rolling(28, min_periods=20).mean())
    out["wiki_log"] = lw7
    out["wiki_lag1_rel"] = np.log1p(w.shift(1)) - lw7

    for c in TEXT_COLS:
        out[c] = g[c].astype(float)
    out["txt_mean_len"] = np.log1p(out["txt_mean_len"])
    out["txt_novelty_ma3"] = g["txt_novelty"].astype(float).rolling(3, min_periods=1).mean()
    out["txt_event_ma3"] = g["txt_event_share"].astype(float).rolling(3, min_periods=1).mean()
    out["txt_sentiment_ma7"] = g["txt_sentiment"].astype(float).rolling(7, min_periods=3).mean()
    out["ly"] = ly
    out["imputed"] = g["mentions_imputed"].fillna(False).astype(bool)
    return out


def build_features(mart: pd.DataFrame, topics: list[str], horizons: list[int],
                   min_history_days: int = 28) -> pd.DataFrame:
    """mart: витрина mart.topic_daily. Возвращает строки (тема, t, h) с признаками, целью y_rel и y_true.

    Строки, у которых дата t+h ещё не наступила, тоже возвращаются (цель пустая) — это прогноз в эксплуатации.
    """
    mart = mart.copy()
    mart["date"] = pd.to_datetime(mart["date"])
    mart = mart.sort_values(["topic", "date"]).reset_index(drop=True)
    cal = mart.drop_duplicates("date").set_index("date")[["is_holiday_us", "is_holiday_ru"]].astype(float)

    frames = []
    for topic, g in mart.groupby("topic", sort=False):
        g = g.reset_index(drop=True)
        day = _day_features(g)
        day["topic"], day["feature_date"] = topic, g["date"]
        day = day.iloc[min_history_days - 1:]            # без полной истории окон строки не используем
        by_date = pd.Series(day["ly"].values, index=day["feature_date"])
        imputed = pd.Series(day["imputed"].values, index=day["feature_date"])
        ly_full = pd.Series(np.log1p(g["mentions"].astype(float)).values, index=g["date"])
        for h in horizons:
            f = day.copy()
            f["horizon"] = h
            f["target_date"] = f["feature_date"] + pd.Timedelta(days=h)
            f["y_log"] = f["target_date"].map(by_date)
            f["target_imputed"] = f["target_date"].map(imputed).fillna(False).astype(bool)
            # тот же день недели неделю и две назад относительно даты цели: t+h-7 <= t, известно на момент t
            f["same_dow_rel"] = (f["target_date"] - pd.Timedelta(days=7)).map(ly_full).values - f["log_base"]
            f["same_dow2_rel"] = (f["target_date"] - pd.Timedelta(days=14)).map(ly_full).values - f["log_base"]
            frames.append(f)
    df = pd.concat(frames, ignore_index=True)

    td = df["target_date"]
    df["target_dow"] = td.dt.day_of_week
    df["target_dow_sin"] = np.sin(2 * np.pi * df["target_dow"] / 7)
    df["target_dow_cos"] = np.cos(2 * np.pi * df["target_dow"] / 7)
    df["target_weekend"] = (df["target_dow"] >= 5).astype(int)
    hol = cal.reindex(td.values)
    df["target_holiday_us"] = hol["is_holiday_us"].fillna(0).values
    df["target_holiday_ru"] = hol["is_holiday_ru"].fillna(0).values
    df["target_month"] = td.dt.month
    # тема: one-hot — категорий мало (5), порядка между ними нет; target encoding не нужен и рискует утечкой
    for t in topics:
        df[f"topic_{t}"] = (df["topic"] == t).astype(int)

    df["y_rel"] = df["y_log"] - df["log_base"]
    df["y_true"] = np.expm1(df["y_log"])
    cols = KEYS + ["base", "y_true", "y_rel", "target_imputed"] + [c for c in feature_columns(topics) if c not in KEYS]
    return df[cols].sort_values(["feature_date", "topic", "horizon"]).reset_index(drop=True)


def to_prediction(base: pd.Series, y_rel_pred) -> np.ndarray:
    """Перевод относительного прогноза в число упоминаний."""
    return np.clip(np.expm1(np.log1p(base.values) + np.asarray(y_rel_pred)), 0, None)
