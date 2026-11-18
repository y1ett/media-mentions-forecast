"""Отсутствие утечки, схема валидации, воспроизводимость, метрики, мониторинг (PSI), реестр версий."""
import time

import numpy as np
import pandas as pd
import pytest

from src import registry
from src.config import load_config
from src.monitor import psi
from src.features import build_features, feature_columns
from src.models import cv_folds, evaluate, holdout_start, make_model, metrics, split
from tests.conftest import TOPICS, make_mart, small_cfg

H = [1, 2, 3, 4, 5, 6, 7]


@pytest.fixture(scope="module")
def mart():
    return make_mart()


@pytest.fixture(scope="module")
def feats(mart):
    return build_features(mart, TOPICS, H)


@pytest.fixture(scope="module")
def scfg():
    return small_cfg(load_config())


# ---------------- утечка целевой переменной
def test_target_is_strictly_in_future(feats):
    assert ((feats["target_date"] - feats["feature_date"]).dt.days == feats["horizon"]).all()


def test_features_do_not_change_when_future_is_removed(mart, feats):
    cut = pd.Timestamp("2025-06-15")
    f_trunc = build_features(mart[pd.to_datetime(mart["date"]) <= cut], TOPICS, H)
    cols = [c for c in feature_columns(TOPICS) if c != "horizon"] + ["base"]
    a = feats[feats["feature_date"] <= cut].set_index(["topic", "feature_date", "horizon"])[cols].sort_index()
    b = f_trunc.set_index(["topic", "feature_date", "horizon"])[cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_changing_future_target_does_not_change_features(mart, feats):
    cut = pd.Timestamp("2025-05-01")
    m2 = mart.copy()
    m2.loc[pd.to_datetime(m2["date"]) > cut, "mentions"] *= 10
    f2 = build_features(m2, TOPICS, H)
    cols = feature_columns(TOPICS)
    pd.testing.assert_frame_equal(feats.loc[feats["feature_date"] <= cut, cols].reset_index(drop=True),
                                  f2.loc[f2["feature_date"] <= cut, cols].reset_index(drop=True))


def test_validation_windows_do_not_overlap(feats, scfg):
    for start, end in cv_folds(feats, scfg):
        train, valid = split(feats, start, end)
        assert train["target_date"].max() < start <= valid["target_date"].min()
    assert cv_folds(feats, scfg)[-1][1] <= holdout_start(feats, scfg)


def test_shuffled_target_gives_no_skill(feats, scfg):
    """Если перемешать цель, модель не должна быть заметно лучше наивного прогноза (иначе — утечка)."""
    shuffled = feats.copy()
    lab = shuffled["y_rel"].notna()
    shuffled.loc[lab, "y_rel"] = np.random.default_rng(0).permutation(shuffled.loc[lab, "y_rel"].to_numpy())
    shuffled["y_true"] = np.expm1(shuffled["y_rel"] + shuffled["log_base"])
    f = feature_columns(TOPICS)
    real, _ = evaluate(lambda: make_model("lightgbm", scfg), shuffled, f, scfg)
    naive, _ = evaluate(lambda: make_model("moving_average", scfg), shuffled, f, scfg)
    assert real["WAPE"] > 0.95 * naive["WAPE"]


# ---------------- модели
def test_training_is_reproducible_and_beats_naive(feats, scfg):
    f = feature_columns(TOPICS)
    a, _ = evaluate(lambda: make_model("lightgbm", scfg), feats, f, scfg)
    b, _ = evaluate(lambda: make_model("lightgbm", scfg), feats, f, scfg)
    naive, _ = evaluate(lambda: make_model("seasonal_naive", scfg), feats, f, scfg)
    assert a == b and a["WAPE"] < naive["WAPE"]


def test_metrics_on_known_values():
    m = metrics([100, 200], [110, 180])
    assert m["MAE"] == 15 and abs(m["WAPE"] - 0.1) < 1e-12 and m["within_15pct"] == 1.0


# ---------------- мониторинг и реестр
def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 2000))
    assert psi(ref, pd.Series(rng.normal(0, 1, 500))) < 0.1
    assert psi(ref, pd.Series(rng.normal(1.5, 1, 500))) > 0.25


def test_registry_promotes_only_better_and_rolls_back(cfg):
    meta = lambda w: {"model": "ridge", "holdout": {"WAPE": w}, "baseline_holdout_WAPE": 0.3}
    results = []
    for w in (0.20, 0.25, 0.15, 0.35):        # хуже текущей / лучше / хуже базовой
        results.append(registry.register(cfg, {"w": w}, meta(w)))
        time.sleep(1.1)                         # версия = отметка времени до секунды
    assert [a for _, a in results] == [True, False, True, False]
    assert registry.load_active(cfg)[0] == results[2][0]
    assert registry.rollback(cfg) == results[0][0]
