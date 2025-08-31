from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from .config import Config
from .utils import setup_logging

logger = setup_logging("analysis")

# Multilingual Twitter sentiment model (no paid API)
MODEL_NAME = (
    "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # [-1,0,1] ≈ [neg, neu, pos]
)


@torch.inference_mode()
def batched_sentiment(
    texts: list[str], batch_size: int = 32, device: str = None
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="sentiment"):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch, padding=True, truncation=True, max_length=96, return_tensors="pt"
        ).to(device)
        out = model(**enc).logits.softmax(dim=-1).detach().cpu().numpy()
        # sentiment score ∈ [-1,1]: pos - neg
        s = out[:, 2] - out[:, 0]
        scores.append(s)
    return np.concatenate(scores, axis=0)


def build_tfidf(texts: pd.Series, max_features=2000):
    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),  # char-ngrams handle Hinglish/Devanagari mixed text
        max_features=max_features,
        dtype=np.float32,
    )
    X = vec.fit_transform(texts.tolist())
    return vec, X


def composite_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine:
      - sentiment_s: [-1,1]
      - engagement z-score from likes/retweets/replies
      - bullish/bearish keyword flags
    """
    eng = df[["like_count", "retweet_count", "reply_count"]].fillna(0)
    z = (eng - eng.mean()) / eng.std(ddof=0).replace(0, 1)
    df["eng_z"] = z.mean(axis=1).astype("float32")

    # weights (tuneable hyperparams)
    w_sent, w_eng, w_kw = 0.6, 0.3, 0.1
    kw = df["has_bullish_kw"].astype(float) - df["has_bearish_kw"].astype(float)
    df["signal_raw"] = w_sent * df["sentiment_s"] + w_eng * df["eng_z"] + w_kw * kw

    # aggregate per minute for memory-light curves
    g = (
        df.groupby("t_minute", as_index=False)["signal_raw"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    g.rename(
        columns={"mean": "signal_mean", "count": "n", "std": "signal_std"}, inplace=True
    )

    # confidence interval (normal approx)
    g["se"] = (g["signal_std"] / np.sqrt(g["n"].clip(lower=1))).fillna(0)
    g["ci_lo"] = g["signal_mean"] - 1.96 * g["se"]
    g["ci_hi"] = g["signal_mean"] + 1.96 * g["se"]
    return g[["t_minute", "signal_mean", "ci_lo", "ci_hi", "n"]]


def plot_signal(minute_df: pd.DataFrame, parquet_processed: str, logger):
    import matplotlib.pyplot as plt

    dd = minute_df.sort_values("t_minute")
    if len(dd) > 1500:
        dd = dd.iloc[:: max(1, len(dd) // 1500)]  # simple stride sampling

    plt.figure()
    plt.plot(dd["t_minute"], dd["signal_mean"], label="signal")
    plt.fill_between(
        dd["t_minute"], dd["ci_lo"], dd["ci_hi"], alpha=0.2, label="95% CI"
    )
    plt.legend()
    plt.title("Composite Market Sentiment Signal (minute, last 24h)")
    plt.xlabel("IST minute")
    plt.ylabel("signal")
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig_path = parquet_processed.replace(".parquet", "_signal_plot.png")
    plt.savefig(fig_path, dpi=140)
    logger.info(f"Saved plot: {fig_path}")


def run_analysis(parquet_processed: str, cfg: Config):
    df = pd.read_parquet(parquet_processed)

    # 1) TF-IDF matrix (kept sparse; don’t densify!)
    vec, X = build_tfidf(df["content_norm"])

    # 2) Multilingual sentiment
    df["sentiment_s"] = batched_sentiment(df["content_norm"].tolist())

    # 3) Composite signal + CI
    minute_df = composite_signal(df)

    # 4) Save outputs (compact CSV for quick looks, Parquet for fidelity)
    out_csv = parquet_processed.replace(".parquet", "_signals.csv")
    out_parq = parquet_processed.replace(".parquet", "_signals.parquet")
    minute_df.to_csv(out_csv, index=False)
    minute_df.to_parquet(out_parq, index=False)
    logger.info(f"Wrote signals: {out_csv} & {out_parq}")

    # 5) Lightweight plotting (downsample if large)
    try:
        plot_signal(minute_df, parquet_processed, logger)
    except Exception as e:
        logger.warning(f"Plotting skipped: {e}")


if __name__ == "__main__":
    from .processor import process
    from .scrapper import run_scrape

    cfg = Config()
    raw = run_scrape(cfg)
    proc = process(raw, cfg)
    run_analysis(proc, cfg)
