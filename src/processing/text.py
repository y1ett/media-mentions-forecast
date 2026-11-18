"""Обработка слабоструктурированного источника (тексты HN и Habr).

Все преобразования не обучаются на данных (очистка, словарь тональности VADER, HashingVectorizer),
поэтому их можно применять ко всей истории без риска утечки: признак дня D зависит только от текстов за D и раньше.
"""
import html
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, HashingVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-zа-яё][a-zа-яё0-9+#.-]{2,}")


def clean_text(text: str | None) -> str:
    """HTML-сущности и теги, ссылки, регистр, лишние пробелы."""
    if not text:
        return ""
    text = html.unescape(html.unescape(text))      # в HN двойное экранирование (&amp;#x27;)
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip().lower()


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


_vader = None


def sentiment(texts: pd.Series) -> pd.Series:
    """Тональность VADER (compound от -1 до 1) — словарный метод, без обучения."""
    global _vader
    _vader = _vader or SentimentIntensityAnalyzer()
    return texts.map(lambda t: _vader.polarity_scores(t[:1000])["compound"] if t else 0.0)


def daily_text_features(docs: pd.DataFrame, event_words: list[str], n_features: int) -> pd.DataFrame:
    """Агрегаты текстов по (тема, день). docs: topic, date, kind, text, points, num_comments."""
    docs = docs.copy()
    docs["len"] = docs["text"].str.len()
    docs["sent"] = sentiment(docs["text"])
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, event_words)) + r")\b")
    docs["event"] = docs["text"].str.contains(pattern)
    docs["is_story"] = docs["kind"].eq("story")
    stories = docs[docs["is_story"]]

    g = docs.groupby(["topic", "date"])
    out = pd.DataFrame({
        "txt_docs": g.size(),
        "txt_share_story": g["is_story"].mean(),
        "txt_mean_len": g["len"].mean(),
        "txt_sentiment": g["sent"].mean(),
        "txt_neg_share": g["sent"].apply(lambda s: (s < -0.05).mean()),
        "txt_event_share": g["event"].mean(),
    })
    gs = stories.groupby(["topic", "date"])
    out["txt_mean_points"] = np.log1p(gs["points"].mean())
    out["txt_mean_comments"] = np.log1p(gs["num_comments"].mean())
    out["txt_novelty"] = novelty(docs, n_features)
    return out.reset_index()


def novelty(docs: pd.DataFrame, n_features: int, window: int = 7) -> pd.Series:
    """Новизна лексики: 1 - косинусная близость словаря дня D и словаря дней D-7..D-1.

    Тексты векторизуются HashingVectorizer (фиксированное хэширование, без обучения).
    Высокая новизна = в теме появилась новая подтема (часто предвестник всплеска).
    """
    vec = HashingVectorizer(n_features=n_features, alternate_sign=False, norm=None,
                            stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#-]{2,}\b")
    result = {}
    for topic, part in docs.groupby("topic"):
        days = part.groupby("date")["text"].apply(lambda s: " ".join(s)).sort_index()
        m = vec.transform(days.values).tocsr()
        idx = pd.to_datetime(days.index)
        for i, day in enumerate(days.index):
            lo = idx.searchsorted(idx[i] - pd.Timedelta(days=window))
            if i - lo < window:            # мало истории — признак не считаем
                continue
            prev = sparse.csr_matrix(m[lo:i].sum(axis=0))
            cur = m[i]
            num = cur.multiply(prev).sum()
            den = np.sqrt(cur.multiply(cur).sum() * prev.multiply(prev).sum())
            result[(topic, day)] = 1 - num / den if den else np.nan
    return pd.Series(result, dtype=float).rename_axis(["topic", "date"])


def top_terms(texts: pd.Series, n: int = 20, stop: set[str] | None = None) -> pd.DataFrame:
    """Частые слова для дашборда («о чём пишут»)."""
    stop = (stop or set()) | set(ENGLISH_STOP_WORDS) | RU_STOP
    counts = pd.Series([t for text in texts for t in tokens(text) if t not in stop and not t[0].isdigit()])
    return counts.value_counts().head(n).rename_axis("term").reset_index(name="count")


RU_STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас
нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам
чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один
почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше
тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть
том нельзя такой им более всегда конечно всю между это как также которые который которая которых читать далее
""".split()) | {"https", "http", "www", "com", "just", "like", "don", "think", "people", "really", "doesn",
                "didn", "isn", "time", "way", "make", "use", "used", "using", "want", "good", "know", "thing",
                "things", "lot", "going", "work", "need", "actually", "pretty", "probably", "sure", "say",
                "said", "yes", "does", "did", "got", "get", "much", "even", "still", "also", "one", "would",
                "could", "should", "can", "will", "new", "year", "years", "the", "and", "for", "that", "this"}
