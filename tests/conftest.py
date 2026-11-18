"""Общие фикстуры и синтетическая витрина для тестов (без сети)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.db import connect  # noqa: E402

TOPICS = ["a", "b"]


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["paths"]["db"] = str(tmp_path / "test.duckdb")
    c["paths"]["data_dir"] = str(tmp_path / "data")
    c["paths"]["models"] = str(tmp_path / "models")
    return c


@pytest.fixture
def con(cfg):
    con = connect(cfg)
    yield con
    con.close()


def make_mart(days: int = 240, seed: int = 0) -> pd.DataFrame:
    """Синтетическая mart.topic_daily: недельная сезонность + медленная волна + шум."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days)
    rows = []
    for k, t in enumerate(TOPICS):
        weekly = 1 + 0.3 * np.sin(2 * np.pi * dates.day_of_week / 7)
        y = rng.poisson(100 * (k + 1) * weekly * (1 + 0.2 * np.sin(np.arange(days) / 30)))
        wiki = rng.poisson(5000 * weekly)
        for i, d in enumerate(dates):
            rows.append({"topic": t, "date": d.date(), "mentions": float(y[i]), "mentions_imputed": False,
                         "wiki_views": float(wiki[i]), "wiki_imputed": False, "gdelt_articles": np.nan,
                         "gdelt_share": np.nan, "habr_articles": 0.0, "txt_docs": 40.0,
                         "txt_share_story": rng.random(), "txt_mean_points": rng.random() * 3,
                         "txt_mean_comments": rng.random() * 3, "txt_mean_len": 200 + rng.random() * 50,
                         "txt_sentiment": rng.normal(0, 0.2), "txt_neg_share": rng.random(),
                         "txt_event_share": rng.random() * 0.2, "txt_novelty": rng.random(),
                         "is_holiday_us": d.month == 1 and d.day == 1, "is_holiday_ru": d.month == 1 and d.day < 8})
    return pd.DataFrame(rows)


def small_cfg(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["topics"] = {t: {} for t in TOPICS}
    cfg["model"] = {**cfg["model"], "holdout_days": 30, "cv_folds": 3, "cv_fold_days": 20,
                    "lightgbm": {**cfg["model"]["lightgbm"], "n_estimators": 60},
                    "random_forest": {**cfg["model"]["random_forest"], "n_estimators": 30}}
    return cfg
