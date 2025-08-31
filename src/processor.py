from __future__ import annotations
import pandas as pd, regex as re, unicodedata, emoji
import pyarrow.parquet as pq, pyarrow as pa
from datetime import timezone
from .config import Config
from .utils import setup_logging

logger = setup_logging("processor")

URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")

BULLISH = {"buy", "long", "bull", "breakout", "uptrend", "support", "accumulate"}
BEARISH = {"sell", "short", "bear", "breakdown", "downtrend", "resistance", "dump"}


def normalize_text(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    # Keep emojis (useful signals), strip URLs, normalize unicode, squeeze spaces
    txt = unicodedata.normalize("NFKC", txt)
    txt = URL_RE.sub("", txt)
    txt = WS_RE.sub(" ", txt).strip()
    return txt


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    txt = df["content_norm"].str.lower().fillna("")
    df["len_chars"] = txt.str.len()
    df["has_emoji"] = txt.apply(lambda s: int(any(ch in emoji.EMOJI_DATA for ch in s)))
    df["has_question"] = txt.str.contains(r"\?", regex=True).astype("int8")
    df["has_number"] = txt.str.contains(r"\d").astype("int8")
    df["has_bullish_kw"] = txt.apply(
        lambda s: int(any(w in s for w in BULLISH))
    ).astype("int8")
    df["has_bearish_kw"] = txt.apply(
        lambda s: int(any(w in s for w in BEARISH))
    ).astype("int8")
    df["hashtag_count"] = (
        df["hashtags"]
        .apply(lambda x: len(x) if isinstance(x, list) else 0)
        .astype("int16")
    )
    df["mention_count"] = (
        df["mentions"]
        .apply(lambda x: len(x) if isinstance(x, list) else 0)
        .astype("int16")
    )
    return df


def process(parquet_in: str, cfg: Config) -> str:
    logger.info(f"Reading: {parquet_in}")
    table = pq.read_table(parquet_in)
    df = table.to_pandas()
    # Dedup again belt-and-suspenders
    df = df.drop_duplicates(subset=["tweet_id"])

    # normalize text
    df["content_norm"] = df["content"].apply(normalize_text)
    # restrict to >0 length
    df = df[df["content_norm"].str.len() > 0].copy()

    # timestamps to IST + rounded minute (useful for aggregation)
    df["timestamp_ist"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert(
        cfg.IST
    )
    df["t_minute"] = df["timestamp_ist"].dt.floor("min")

    # simple language harmonization (keep original `lang` too)
    # (optional) could use langdetect per row; keep it light for speed

    df = engineer_features(df)

    # Column order / dtypes for compact Parquet
    sel = [
        "tweet_id",
        "username",
        "displayname",
        "timestamp_utc",
        "timestamp_ist",
        "t_minute",
        "lang",
        "content",
        "content_norm",
        "reply_count",
        "retweet_count",
        "like_count",
        "quote_count",
        "hashtags",
        "mentions",
        "source_label",
        "len_chars",
        "has_emoji",
        "has_question",
        "has_number",
        "has_bullish_kw",
        "has_bearish_kw",
        "hashtag_count",
        "mention_count",
        "link",
    ]
    df = df[sel]

    # write processed parquet
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        cfg.parquet_processed,
        compression="zstd",
    )
    logger.info(f"Wrote processed parquet: {cfg.parquet_processed} | rows={len(df)}")
    return cfg.parquet_processed


if __name__ == "__main__":
    from .scrapper import run_scrape

    cfg = Config()
    raw = run_scrape(cfg)
    process(raw, cfg)
