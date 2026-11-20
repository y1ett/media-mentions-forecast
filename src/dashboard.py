"""Дашборд: streamlit run src/dashboard.py (или make dashboard)

Каждая вкладка отвечает на один вопрос пользователя (SMM-аналитика / редактора).
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.processing import top_terms  # noqa: E402

# понятные названия признаков для объяснений
LABELS = {
    "lag0_rel": "упоминания вчера (день t) к среднему за 7 дней",
    "lag1_rel": "упоминания t-1 к среднему", "lag2_rel": "упоминания t-2 к среднему",
    "lag3_rel": "упоминания t-3 к среднему", "lag6_rel": "упоминания неделю назад к среднему",
    "lag13_rel": "упоминания 2 недели назад к среднему", "lag27_rel": "упоминания 4 недели назад к среднему",
    "same_dow_rel": "тот же день недели неделю назад", "same_dow2_rel": "тот же день недели 2 недели назад",
    "log_base": "уровень темы (среднее за 7 дней, лог)", "ma28_rel": "среднее за 28 дней к среднему за 7",
    "std7_rel": "разброс за 7 дней", "max7_rel": "максимум за 7 дней", "min7_rel": "минимум за 7 дней",
    "ewm3_rel": "сглаженный тренд (EWM 3)", "horizon": "горизонт прогноза, дней",
    "target_dow": "день недели прогноза", "target_dow_sin": "день недели (sin)", "target_dow_cos": "день недели (cos)",
    "target_weekend": "выходной", "target_holiday_us": "праздник в США", "target_holiday_ru": "праздник в России",
    "target_month": "месяц", "wiki_rel": "просмотры Википедии к среднему за 7 дней",
    "wiki_trend": "тренд просмотров Википедии (7 к 28 дням)", "wiki_log": "уровень просмотров Википедии",
    "wiki_lag1_rel": "просмотры Википедии t-1", "txt_share_story": "доля постов среди текстов",
    "txt_mean_points": "средний рейтинг постов", "txt_mean_comments": "среднее число комментариев к постам",
    "txt_mean_len": "средняя длина текста", "txt_sentiment": "тональность текстов",
    "txt_neg_share": "доля негативных текстов", "txt_event_share": "доля текстов со словами-событиями",
    "txt_novelty": "новизна лексики", "txt_docs": "число текстов в выборке",
    "txt_novelty_ma3": "новизна лексики за 3 дня", "txt_event_ma3": "слова-события за 3 дня",
    "txt_sentiment_ma7": "тональность за 7 дней",
}


def label(name: str) -> str:
    if name.startswith("topic_"):
        return f"тема: {name[6:]}"
    return LABELS.get(name, name)


st.set_page_config(page_title="Медиа-активность по темам", layout="wide")
CFG = load_config()
TOPICS = CFG["topics"]


@st.cache_data(ttl=300)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    con = duckdb.connect(CFG["paths"]["db"], read_only=True)
    try:
        return con.execute(sql, list(params)).df()
    finally:
        con.close()


@st.cache_resource(ttl=600)
def context():
    from src.explain import load_context
    return load_context(CFG)


def registry() -> dict:
    f = Path(CFG["paths"]["models"]) / "registry.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"active": None, "versions": {}}


def report(name: str) -> pd.DataFrame | None:
    f = ROOT / "reports" / name
    return pd.read_csv(f) if f.exists() else None


reg = registry()
active = reg.get("active")
if not active:
    st.error("Нет обученной модели. Выполните `make all`.")
    st.stop()
meta = json.loads((Path(CFG["paths"]["models"]) / active / "meta.json").read_text(encoding="utf-8"))

# ---------------- боковая панель: фильтры
st.sidebar.header("Фильтры")
topic = st.sidebar.selectbox("Тема", list(TOPICS), format_func=lambda t: TOPICS[t]["title"])
mart = q("select * from mart.topic_daily order by date")
mart["date"] = pd.to_datetime(mart["date"])
dmin, dmax = mart["date"].min().date(), mart["date"].max().date()
period = st.sidebar.date_input("Период", (max(dmin, dmax - pd.Timedelta(days=120)), dmax), min_value=dmin, max_value=dmax)
p_from, p_to = (pd.Timestamp(period[0]), pd.Timestamp(period[1])) if len(period) == 2 else (pd.Timestamp(dmin), pd.Timestamp(dmax))
st.sidebar.caption(f"Данные по {dmax:%d.%m.%Y}. Активная модель: **{meta['model']}** ({active}).")

fc = q("select * from mart.forecasts")
fc["target_date"] = pd.to_datetime(fc["target_date"])
fc["feature_date"] = pd.to_datetime(fc["feature_date"])
future = fc[(fc["kind"] == "future") & (fc["model_version"] == active)]
future = future[future["feature_date"] == future["feature_date"].max()]
hold_all = fc[fc["kind"] == "holdout"].copy()
hold_all["model"] = hold_all["model_version"].str.split(":").str[-1]
hold_version = hold_all["model_version"].str.split(":").str[0].max() if len(hold_all) else None
hold_all = hold_all[hold_all["model_version"].str.startswith(str(hold_version))]
ho = meta["holdout"]
by_h = report("holdout_by_horizon.csv")

tabs = st.tabs(["📈 Прогноз", "🗂 Все темы", "🎯 Качество модели", "🔍 Почему такой прогноз", "💬 О чём пишут",
                "🛠 Данные и эксперименты"])

# ================= 1. Прогноз
with tabs[0]:
    st.subheader(f"Сколько упоминаний ждать по теме «{TOPICS[topic]['title']}» в ближайшие 7 дней?")
    m = mart[mart["topic"] == topic]
    f = future[future["topic"] == topic].sort_values("target_date")
    last7 = m.tail(7)["mentions"].sum()
    if len(f):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Прогноз на завтра", f"{f.iloc[0]['y_pred']:.0f}",
                  f"{100 * (f.iloc[0]['y_pred'] / max(m.tail(7)['mentions'].mean(), 1) - 1):+.0f}% к среднему за неделю")
        c2.metric("Сумма за 7 дней (прогноз)", f"{f['y_pred'].sum():.0f}", f"{100 * (f['y_pred'].sum() / max(last7, 1) - 1):+.0f}% к прошлой неделе")
        c3.metric("Типичная ошибка", f"±{ho['MAE']:.0f} в день", f"WAPE {ho['WAPE']:.0%}", delta_color="off")
        c4.metric("Точнее наивного прогноза", f"{100 * (1 - ho['WAPE'] / meta['baseline_holdout_WAPE']):.0f}%",
                  "на отложенной выборке", delta_color="off")
    show = m[(m["date"] >= p_from) & (m["date"] <= p_to)]
    fig = go.Figure()
    fig.add_scatter(x=show["date"], y=show["mentions"], name="факт", line=dict(color="#1f77b4"))
    h1 = hold_all[(hold_all["topic"] == topic) & (hold_all["horizon"] == 1) & (hold_all["model"] == meta["model"])]
    h1 = h1[(h1["target_date"] >= p_from) & (h1["target_date"] <= p_to)].sort_values("target_date")
    if len(h1):
        fig.add_scatter(x=h1["target_date"], y=h1["y_pred"], name="прогноз на день вперёд (проверка)",
                        line=dict(color="#ff7f0e", dash="dot"))
    if len(f):
        err = f["horizon"].map(by_h.set_index(by_h.columns[0])["MAE"]) if by_h is not None else ho["MAE"]
        fig.add_scatter(x=f["target_date"], y=f["y_pred"] + err, line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=f["target_date"], y=(f["y_pred"] - err).clip(lower=0), fill="tonexty", line=dict(width=0),
                        fillcolor="rgba(214,39,40,0.15)", name="типичная ошибка")
        fig.add_scatter(x=f["target_date"], y=f["y_pred"], name="прогноз на 7 дней", mode="lines+markers",
                        line=dict(color="#d62728"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15),
                      yaxis_title="упоминаний в день")
    st.plotly_chart(fig, width="stretch")
    st.caption("Факт — сумма упоминаний по запросам темы на Hacker News за сутки (UTC). Пунктир — как модель "
               "прогнозировала на день вперёд в период проверки, который не участвовал в обучении.")

# ================= 2. Все темы
with tabs[1]:
    st.subheader("Какие темы вырастут сильнее всего на следующей неделе?")
    rows = []
    for t in TOPICS:
        mt, ft = mart[mart["topic"] == t], future[future["topic"] == t]
        prev = mt.tail(7)["mentions"].sum()
        nxt = ft["y_pred"].sum()
        rows.append({"тема": TOPICS[t]["title"], "за прошлые 7 дней": prev, "прогноз на 7 дней": nxt,
                     "изменение, %": 100 * (nxt / prev - 1) if prev else np.nan,
                     "пик (день)": ft.loc[ft["y_pred"].idxmax(), "target_date"].strftime("%d.%m") if len(ft) else ""})
    tbl = pd.DataFrame(rows).sort_values("изменение, %", ascending=False)
    c1, c2 = st.columns([3, 2])
    c1.dataframe(tbl.round(1), hide_index=True, width="stretch")
    c2.plotly_chart(px.bar(tbl, x="изменение, %", y="тема", orientation="h", color="изменение, %",
                           color_continuous_scale="RdYlGn", range_color=(-30, 30)).update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False), width="stretch")

# ================= 3. Качество
with tabs[2]:
    st.subheader("Насколько можно доверять прогнозу?")
    comp = report("model_comparison.csv")
    if comp is not None:
        names = {"seasonal_naive": "Наивный (неделю назад) — база", "moving_average": "Среднее за 7 дней — база",
                 "ridge": "Ridge (линейная)", "random_forest": "Случайный лес", "lightgbm": "LightGBM (бустинг)"}
        t = comp.assign(модель=comp["model"].map(names))[
            ["модель", "cv_MAE", "cv_WAPE", "holdout_MAE", "holdout_WAPE", "holdout_sMAPE", "holdout_within_15pct",
             "holdout_gain_vs_naive_pct"]].rename(columns={
                "cv_MAE": "MAE (CV)", "cv_WAPE": "WAPE (CV)", "holdout_MAE": "MAE (проверка)",
                "holdout_WAPE": "WAPE (проверка)", "holdout_sMAPE": "sMAPE (проверка)",
                "holdout_within_15pct": "дней с ошибкой <15%", "holdout_gain_vs_naive_pct": "лучше наивного, %"})
        st.dataframe(t.round(3), hide_index=True, width="stretch")
        st.info(f"Выбранная модель **{meta['model']}** ошибается в среднем на **±{ho['MAE']:.0f} упоминаний в день** "
                f"(WAPE {ho['WAPE']:.1%}); в {ho['within_15pct']:.0%} случаев ошибка меньше 15%. "
                f"Это на {100 * (1 - ho['WAPE'] / meta['baseline_holdout_WAPE']):.0f}% точнее, чем «как неделю назад».")
    c1, c2 = st.columns(2)
    if by_h is not None:
        by_h = by_h.rename(columns={by_h.columns[0]: "горизонт"})
        c1.plotly_chart(px.bar(by_h, x="горизонт", y="WAPE", title="Ошибка по горизонту прогноза (WAPE)",
                               text_auto=".1%").update_layout(height=320, yaxis_tickformat=".0%"), width="stretch")
    by_t = report("holdout_by_topic.csv")
    if by_t is not None:
        by_t = by_t.rename(columns={by_t.columns[0]: "тема"})
        c2.plotly_chart(px.bar(by_t, x="тема", y="MAE", title="Средняя ошибка по темам, упоминаний в день",
                               text_auto=".0f").update_layout(height=320), width="stretch")
    if len(hold_all):
        e = hold_all[(hold_all["horizon"] == 1)].copy()
        e["abs_err"] = (e["y_pred"] - e["y_true"]).abs()
        daily = e.groupby(["target_date", "model"])["abs_err"].sum().reset_index()
        tot = e.groupby(["target_date", "model"])["y_true"].sum().reset_index()
        daily["WAPE"] = daily["abs_err"] / tot["y_true"]
        st.plotly_chart(px.line(daily, x="target_date", y="WAPE", color="model",
                                title="Ошибка во времени на проверочном периоде (горизонт 1 день)").update_layout(
            height=320, yaxis_tickformat=".0%"), width="stretch")

# ================= 4. Объяснение
with tabs[3]:
    st.subheader("Почему модель дала такой прогноз?")
    try:
        cfg, version, model, meta_c, df = context()
        from src.explain import explain_row, gain_importance, partial_dependence, shap_importance_by_period, shap_values
        c1, c2, c3 = st.columns(3)
        dates = sorted(df.loc[df["topic"] == topic, "feature_date"].unique())
        d = c1.date_input("Дата, по которую известны данные", pd.Timestamp(dates[-1]).date(),
                          min_value=pd.Timestamp(dates[0]).date(), max_value=pd.Timestamp(dates[-1]).date())
        h = c2.selectbox("Горизонт, дней", CFG["model"]["horizons"])
        top = c3.slider("Сколько признаков показать", 5, 25, 12)
        contrib, info = explain_row(model, meta_c, df, topic, d, h)
        st.markdown(f"Прогноз на **{info['target_date']:%d.%m.%Y}**: **{info['prediction']:.0f}** упоминаний "
                    f"(среднее за прошлые 7 дней — {info['base_ma7']:.0f}; факт — "
                    f"{'%.0f' % info['actual'] if info['actual'] is not None else 'ещё неизвестен'}).")
        cc = contrib.head(top).iloc[::-1].copy()
        cc["признак"] = [f"{label(i)} = {v:.2f}" for i, v in zip(cc.index, cc["value"])]
        st.plotly_chart(px.bar(cc, x="effect_pct", y="признак", orientation="h",
                               color=np.where(cc["effect_pct"] > 0, "увеличивает", "уменьшает"),
                               color_discrete_map={"увеличивает": "#2ca02c", "уменьшает": "#d62728"},
                               labels={"effect_pct": "влияние на прогноз, %"}).update_layout(
            height=40 + 28 * top, margin=dict(l=10, r=10, t=10, b=10), legend_title=""), width="stretch")
        st.caption("Вклады — значения SHAP: насколько признак сдвинул прогноз относительно среднего за 7 дней.")

        st.markdown("#### Какие признаки важнее всего (два независимых способа + SHAP)")
        perm = report("permutation_importance.csv")
        gain = gain_importance(model, meta_c["features"])
        imp = pd.DataFrame({"встроенная (gain)": gain / gain.sum()})
        if perm is not None:
            pi = perm.set_index("feature")["importance"].clip(lower=0)
            imp["перестановки"] = pi / pi.sum()
        lab = df[df["y_rel"].notna()].sample(3000, random_state=0)
        sv = shap_values(model, lab[meta_c["features"]]).drop(columns="bias").abs().mean()
        imp["SHAP"] = sv / sv.sum()
        imp = imp.sort_values("SHAP", ascending=False).head(15)
        imp.index = [label(i) for i in imp.index]
        st.plotly_chart(px.bar(imp.iloc[::-1], orientation="h", barmode="group",
                               labels={"value": "доля важности", "index": ""}).update_layout(
            height=520, legend_title="способ"), width="stretch")

        c1, c2 = st.columns(2)
        feat = c1.selectbox("Как прогноз зависит от признака", meta_c["features"], format_func=label,
                            index=meta_c["features"].index("wiki_rel") if "wiki_rel" in meta_c["features"] else 0)
        pdp = partial_dependence(model, meta_c, df, feat)
        c1.plotly_chart(px.line(pdp, x="value", y="effect_pct", markers=True,
                                labels={"value": label(feat), "effect_pct": "прогноз относительно среднего, %"}
                                ).update_layout(height=340), width="stretch")
        by_p, corr = shap_importance_by_period(model, meta_c, df)
        topf = by_p.mean(axis=1).sort_values(ascending=False).head(12).index
        heat = by_p.loc[topf]
        heat.index = [label(i) for i in heat.index]
        c2.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                                  title="Важность признаков по кварталам (среднее |SHAP|)").update_layout(height=340),
                        width="stretch")
        c2.caption(f"Ранговая корреляция важностей соседних кварталов (Спирмен): "
                   f"{', '.join(f'{corr.iloc[i, i + 1]:.2f}' for i in range(len(corr) - 1))}")
    except Exception as exc:
        st.warning(f"Объяснения недоступны: {exc}")

# ================= 5. О чём пишут
with tabs[4]:
    st.subheader(f"О чём сейчас пишут по теме «{TOPICS[topic]['title']}»?")
    docs = q("select source, date, kind, title, text_clean, url from staging.documents where topic = ? "
             "and date between ? and ?", (topic, p_from.date(), p_to.date()))
    c1, c2 = st.columns(2)
    stop = {w.lower() for qq in TOPICS[topic]["hn_queries"] for w in qq.split()}
    for col, src, name in ((c1, "hn", "Hacker News (англ.)"), (c2, "habr", "Habr (рус.)")):
        part = docs[docs["source"] == src]
        col.markdown(f"**{name}** — текстов за период: {len(part)}")
        if len(part):
            tt = top_terms(part["text_clean"], 20, stop)
            col.plotly_chart(px.bar(tt.iloc[::-1], x="count", y="term", orientation="h").update_layout(
                height=480, margin=dict(l=10, r=10, t=10, b=10)), width="stretch")
    m = mart[(mart["topic"] == topic) & (mart["date"] >= p_from) & (mart["date"] <= p_to)]
    st.plotly_chart(px.line(m, x="date", y=["txt_sentiment", "txt_event_share", "txt_novelty"],
                            title="Тональность, доля слов-событий и новизна лексики по дням").update_layout(
        height=300, legend_title=""), width="stretch")
    habr = docs[docs["source"] == "habr"].sort_values("date", ascending=False).head(10)
    if len(habr):
        st.markdown("**Свежие статьи Habr:**")
        for r in habr.itertuples():
            st.markdown(f"- {r.date:%d.%m} [{r.title}]({r.url})")

# ================= 6. Данные и эксперименты
with tabs[5]:
    st.subheader("Всё ли в порядке с данными и моделью?")
    c1, c2 = st.columns(2)
    ll = q("select source, coalesce(topic, '-') topic, status, rows_fetched, rows_inserted, started_at, message "
           "from meta.load_log order by started_at desc limit 40")
    c1.markdown("**Журнал загрузок**")
    c1.dataframe(ll, hide_index=True, width="stretch", height=320)
    dq = q("select * from meta.dq_results where checked_at = (select max(checked_at) from meta.dq_results) "
           "order by passed, severity")
    c2.markdown(f"**Проверки качества данных** — провалено {int((~dq['passed']).sum()) if len(dq) else 0} из {len(dq)}")
    c2.dataframe(dq[["layer", "check_name", "value", "threshold", "passed", "severity"]], hide_index=True,
                 width="stretch", height=320)
    st.markdown("**Мониторинг модели** (дрейф признаков, ошибка на новых фактах, правило переобучения)")
    mon = q("select * from meta.monitoring order by checked_at")
    if len(mon):
        last = mon[mon["checked_at"] == mon["checked_at"].max()]
        dec = last[last["metric"] == "decision_retrain"]
        if len(dec):
            (st.error if bool(dec["alert"].iloc[0]) else st.success)(
                f"Последняя проверка {last['checked_at'].iloc[0]:%d.%m.%Y %H:%M}: {dec['details'].iloc[0]}")
        c1, c2 = st.columns(2)
        psi_last = last[last["metric"].str.startswith("psi:")].assign(
            признак=lambda x: x["metric"].str[4:].map(label))
        if len(psi_last):
            c1.plotly_chart(px.bar(psi_last, x="value", y="признак", orientation="h", color="alert",
                                   color_discrete_map={True: "#d62728", False: "#1f77b4"},
                                   title="Дрейф ключевых признаков (PSI, порог 0.25)").update_layout(
                height=360, showlegend=False), width="stretch")
        else:
            note = last.loc[last["metric"] == "psi", "details"]
            c1.info("Дрейф признаков: " + (note.iloc[0] if len(note) else "ещё не считался") +
                    ". Дрейф считается по данным, пришедшим после обучения модели.")
        rw = mon[mon["metric"] == "recent_WAPE"].dropna(subset=["value"])
        if len(rw):
            c2.plotly_chart(px.line(rw, x="checked_at", y=["value", "threshold"], markers=True,
                                    title="Ошибка на новых фактах и порог переобучения").update_layout(
                height=360, yaxis_tickformat=".0%", legend_title=""), width="stretch")
        else:
            c2.info("Ошибка на новых фактах появится, когда наступят даты после обучения модели.")
        st.caption("Правило: переобучить, если WAPE на новых данных > 1.2 x WAPE на проверке, или PSI > 0.25 "
                   "у 3+ ключевых признаков, или модели больше 30 дней. Откат: `make rollback`.")
    else:
        st.info("Мониторинг ещё не запускался: `make monitor`.")

    st.markdown("**Журнал экспериментов**")
    ex = q("select run_id, created_at, kind, model, data_version, cast(cv_metrics->>'WAPE' as double) cv_wape, "
           "cast(holdout_metrics->>'WAPE' as double) holdout_wape, note, params from meta.experiments order by created_at desc")
    kinds = st.multiselect("Тип", sorted(ex["kind"].unique()), default=sorted(ex["kind"].unique()))
    exs = ex[ex["kind"].isin(kinds)]
    st.dataframe(exs, hide_index=True, width="stretch", height=280)
    tune = ex[ex["kind"] == "tune"].dropna(subset=["cv_wape"]).sort_values("created_at")
    if len(tune):
        tune["лучший на момент"] = tune["cv_wape"].cummin()
        st.plotly_chart(px.line(tune, x="created_at", y=["cv_wape", "лучший на момент"], markers=True,
                                title="Автоподбор гиперпараметров: WAPE на CV по попыткам").update_layout(
            height=300, legend_title=""), width="stretch")
    st.caption("Воспроизвести эксперимент: `python -m src.train reproduce <run_id>`")
    st.markdown("**Реестр версий моделей**")
    rv = pd.DataFrame(reg["versions"]).T.reset_index().rename(columns={"index": "версия"})
    rv["активная"] = rv["версия"] == active
    st.dataframe(rv, hide_index=True, width="stretch")
