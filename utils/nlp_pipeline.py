import re, numpy as np, pandas as pd
from typing import List, Optional

try:
    from textblob import TextBlob
    _TB = True
except ImportError:
    _TB = False

_NEGATIVE = [
    "Management was dismissive and the workload unsustainable.",
    "Salary has not kept up with market rates despite good reviews.",
    "No clear career path — felt ignored in promotion cycles.",
    "Toxic team dynamics and poor communication from leadership.",
    "Felt undervalued and overworked, burnout was the main reason.",
]
_NEUTRAL = ["Contract ended as expected.", "Retirement.", "Personal reasons."]
_POSITIVE = [
    "Left for a better opportunity, overall great experience.",
    "Personal relocation, enjoyed working here very much.",
]

def generate_synthetic_feedback(df, seed=42):
    rng = np.random.default_rng(seed)
    df = df.copy()
    def _pick(row):
        if row.get("Termd", 0) != 1:
            return ""
        r = rng.random()
        if r < 0.50: return str(rng.choice(_NEGATIVE))
        elif r < 0.75: return str(rng.choice(_NEUTRAL))
        else: return str(rng.choice(_POSITIVE))
    df["exit_feedback"] = df.apply(_pick, axis=1)
    print(f"[NLP] Generated feedback for {(df['exit_feedback'] != '').sum()} employees.")
    return df

def _clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

class _Keyword:
    _POS = {"great","good","excellent","happy","satisfied","love","positive","enjoy"}
    _NEG = {"bad","poor","toxic","stress","burnout","unfair","lack","underpaid"}
    def score(self, texts):
        out = []
        for t in texts:
            w = set(_clean(t).split())
            p, n = len(w & self._POS), len(w & self._NEG)
            out.append((p - n) / (p + n) if (p + n) else 0.0)
        return out

class _TB_Backend:
    def score(self, texts):
        return [TextBlob(t).sentiment.polarity for t in texts]

def add_sentiment_features(df, text_cols=None, force_lightweight=False):
    df = df.copy()
    text_cols = text_cols or ["exit_feedback"]
    present = [c for c in text_cols if c in df.columns]
    if not present:
        print("[NLP] No text columns found — adding neutral placeholder.")
        df["feedback_sentiment"] = 0.0
        return df
    backend = _TB_Backend() if _TB else _Keyword()
    print(f"[NLP] Using {'TextBlob' if _TB else 'keyword'} backend.")
    for col in present:
        texts = df[col].fillna("").apply(_clean).tolist()
        scores = backend.score(texts)
        df[f"{col}_sentiment"] = np.clip(scores, -1.0, 1.0)
        print(f"[NLP] '{col}' -> '{col}_sentiment' | mean={np.mean(scores):.3f}")
    return df
