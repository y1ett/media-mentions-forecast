"""Контроль качества данных -> meta.dq_results.

Критичные (critical) проверки при провале останавливают пайплайн (код 1), предупреждения (warning) только фиксируются.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def run_checks(con, cfg: dict) -> pd.DataFrame:
    q = cfg["quality"]
    checks = []

    def add(layer, name, value, threshold, passed, severity):
        checks.append((layer, name, float(value), float(threshold), bool(passed), severity))

    start = pd.Timestamp(str(cfg["sources"]["start_date"]))
    # --- raw: полнота дней HN по каждому запросу
    hn = con.execute("select topic, query, count(distinct date) n, min(date) lo, max(date) hi, "
                     "sum((nb_hits < 0)::int) neg, avg((not exhaustive)::int) approx from raw.hn_daily group by 1, 2").df()
    for t, tcfg in cfg["topics"].items():
        for qy in tcfg["hn_queries"]:
            r = hn[(hn["topic"] == t) & (hn["query"] == qy)]
            if r.empty:
                add("raw", f"hn_present[{t}|{qy}]", 0, 1, False, "critical")
                continue
            r = r.iloc[0]
            expected = (pd.Timestamp(r["hi"]) - start).days + 1
            missing = 1 - r["n"] / expected
            add("raw", f"hn_missing_days[{t}|{qy}]", missing, q["max_missing_target_share"],
                missing <= q["max_missing_target_share"], "critical")
            add("raw", f"hn_negative[{t}|{qy}]", r["neg"], 0, r["neg"] == 0, "critical")
            add("raw", f"hn_approx_counts[{t}|{qy}]", r["approx"], 0.1, r["approx"] <= 0.1, "warning")

    # --- свежесть данных
    last = con.execute("select max(date) from raw.hn_daily").fetchall()[0][0]
    lag = (datetime.now(timezone.utc).date() - last).days if last else 999
    add("raw", "hn_freshness_days", lag, 3, lag <= 3, "warning")

    # --- staging: документы
    d = con.execute("select count(*) n, avg((coalesce(text_clean, '') = '')::int) empty, "
                    "count(*) - count(distinct (doc_id, topic)) dup from staging.documents").fetchall()[0]
    add("staging", "documents_count", d[0], 1, d[0] > 0, "critical")
    add("staging", "documents_empty_text_share", d[1] or 0, 0.2, (d[1] or 0) <= 0.2, "warning")
    add("staging", "documents_duplicates", d[2], 0, d[2] == 0, "critical")

    # --- mart: целевая переменная и ряды
    m = con.execute("select * from mart.topic_daily order by topic, date").df()
    if m.empty:
        add("mart", "mart_rows", 0, 1, False, "critical")
        return pd.DataFrame(checks, columns=["layer", "check_name", "value", "threshold", "passed", "severity"])
    m["date"] = pd.to_datetime(m["date"])
    for t, g in m.groupby("topic"):
        span = (g["date"].max() - g["date"].min()).days + 1
        add("mart", f"date_gaps[{t}]", span - len(g), 0, span == len(g), "critical")
        miss = g["mentions"].isna().mean()
        add("mart", f"target_missing[{t}]", miss, q["max_missing_target_share"],
            miss <= q["max_missing_target_share"], "critical")
        add("mart", f"target_imputed[{t}]", g["mentions_imputed"].mean(), 0.05,
            g["mentions_imputed"].mean() <= 0.05, "warning")
        add("mart", f"target_negative[{t}]", (g["mentions"] < 0).sum(), 0, (g["mentions"] < 0).sum() == 0, "critical")
        wmiss = g["wiki_views"].isna().mean()
        add("mart", f"wiki_missing[{t}]", wmiss, q["max_missing_wiki_share"], wmiss <= q["max_missing_wiki_share"],
            "warning")
        y = np.log1p(g["mentions"].astype(float))
        med = y.rolling(28, min_periods=14).median()
        mad = (y - med).abs().rolling(28, min_periods=14).median()
        z = ((y - med) / (1.4826 * mad.replace(0, np.nan))).abs()
        spikes = int((z > q["spike_robust_z"]).sum())
        add("mart", f"target_spikes[{t}]", spikes, 5, spikes <= 5, "warning")

    # --- журнал: сбои загрузок за последние сутки
    fails = con.execute("select count(*) from meta.load_log where status = 'failed' "
                        "and started_at > now() - interval 1 day").fetchall()[0][0]
    add("meta", "load_failures_24h", fails, 0, fails == 0, "warning")
    return pd.DataFrame(checks, columns=["layer", "check_name", "value", "threshold", "passed", "severity"])


def run_dq(con, cfg: dict) -> int:
    """Проверки -> meta.dq_results; код 1, если провалена критичная проверка."""
    res = run_checks(con, cfg)
    res.insert(0, "checked_at", datetime.now())
    con.register("dq", res)
    con.execute("insert into meta.dq_results by name select * from dq")
    con.unregister("dq")
    bad = res[~res["passed"]]
    print(f"Проверок: {len(res)}, провалено: {len(bad)} "
          f"(критичных {int((bad['severity'] == 'critical').sum())}, предупреждений {int((bad['severity'] == 'warning').sum())})")
    if len(bad):
        print(bad[["layer", "check_name", "value", "threshold", "severity"]].to_string(index=False))
    return 1 if (bad["severity"] == "critical").any() else 0
