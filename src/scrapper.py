from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import snscrape.modules.twitter as sntwitter
from tqdm import tqdm
from typing import Iterable, Dict, Any
from .config import Config
from .utils import setup_logging

logger = setup_logging("scraper")


def _normalize_row(t) -> Dict[str, Any]:
    # hashtags & mentions can be None; convert to safe lists of strings
    hashtags = list(t.hashtags or [])
    hashtags = [h.lower() for h in hashtags] if hashtags else []
    mentions = (
        [m.username.lower() for m in (t.mentionedUsers or [])]
        if t.mentionedUsers
        else []
    )

    return {
        "tweet_id": int(t.id),
        "username": t.user.username,
        "displayname": t.user.displayname,
        "timestamp_utc": t.date.replace(tzinfo=timezone.utc),
        "content": t.content,
        "reply_count": getattr(t, "replyCount", None),
        "retweet_count": getattr(t, "retweetCount", None),
        "like_count": getattr(t, "likeCount", None),
        "quote_count": getattr(t, "quoteCount", None),
        "hashtags": hashtags,
        "mentions": mentions,
        "lang": t.lang,
        "link": f"https://x.com/{t.user.username}/status/{t.id}",
        "source_label": getattr(t, "sourceLabel", None),
    }


def scrape_hashtag(tag: str, cfg: Config) -> Iterable[Dict[str, Any]]:
    # Use date floor in query; do strict timestamp filtering ourselves
    query = f"{tag} since:{cfg.since_date_query}"
    logger.info(f"Scraping: {query}")
    count = 0
    for t in sntwitter.TwitterSearchScraper(query).get_items():
        row = _normalize_row(t)
        # filter strictly by IST window
        if row["timestamp_utc"].astimezone(cfg.IST) < cfg.since_dt_ist:
            continue
        yield row
        count += 1
        if count >= cfg.max_items_per_query:
            break


def run_scrape(cfg: Config) -> str:
    # concurrent across hashtags, writing to Parquet in batches with dedup by tweet_id
    schema = pa.schema(
        [
            pa.field("tweet_id", pa.int64()),
            pa.field("username", pa.string()),
            pa.field("displayname", pa.string()),
            pa.field("timestamp_utc", pa.timestamp("us", tz="UTC")),
            pa.field("content", pa.string()),
            pa.field("reply_count", pa.int64()),
            pa.field("retweet_count", pa.int64()),
            pa.field("like_count", pa.int64()),
            pa.field("quote_count", pa.int64()),
            pa.field("hashtags", pa.list_(pa.string())),
            pa.field("mentions", pa.list_(pa.string())),
            pa.field("lang", pa.string()),
            pa.field("link", pa.string()),
            pa.field("source_label", pa.string()),
        ]
    )

    writer = pq.ParquetWriter(cfg.parquet_raw, schema=schema, compression="snappy")
    seen = set()
    try:
        with ThreadPoolExecutor(max_workers=min(8, len(cfg.hashtags))) as ex:
            futures = [
                ex.submit(list, scrape_hashtag(tag, cfg)) for tag in cfg.hashtags
            ]
            all_rows = []
            for fut in as_completed(futures):
                all_rows.extend(fut.result())

        # stream in batches with dedup
        logger.info(f"Fetched {len(all_rows)} rows before dedup")
        batch = []
        for row in tqdm(all_rows, desc="writing"):
            if row["tweet_id"] in seen:
                continue
            seen.add(row["tweet_id"])
            batch.append(row)
            if len(batch) >= cfg.batch_size:
                table = pa.Table.from_pandas(
                    pd.DataFrame(batch), schema=schema, preserve_index=False
                )
                writer.write_table(table)
                batch.clear()
        if batch:
            table = pa.Table.from_pandas(
                pd.DataFrame(batch), schema=schema, preserve_index=False
            )
            writer.write_table(table)

    finally:
        writer.close()

    logger.info(f"Wrote raw parquet: {cfg.parquet_raw} | unique: {len(seen)}")
    if len(seen) < cfg.min_count:
        logger.warning(f"WARNING: collected {len(seen)} < target {cfg.min_count}")
    return cfg.parquet_raw


if __name__ == "__main__":
    cfg = Config()
    run_scrape(cfg)
