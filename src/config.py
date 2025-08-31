from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import os

IST = timezone(timedelta(hours=5, minutes=30))

@dataclass
class Config:
    hashtags: tuple[str, ...] = ("#nifty50", "#sensex", "#intraday", "#banknifty")
    min_count: int = 2000
    hours_back: int = 24
    out_dir: str = "data"
    raw_dir: str = field(init=False)
    processed_dir: str = field(init=False)
    parquet_raw: str = field(init=False)
    parquet_processed: str = field(init=False)
    batch_size: int = 500  # write to disk in chunks to save memory
    max_items_per_query: int = 5000  # safety cap per hashtag

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.raw_dir = os.path.join(self.out_dir, "raw")
        self.processed_dir = os.path.join(self.out_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        ts = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        self.parquet_raw = os.path.join(self.raw_dir, f"tweets_raw_{ts}.parquet")
        self.parquet_processed = os.path.join(self.processed_dir, f"tweets_processed_{ts}.parquet")

    @property
    def since_dt_ist(self) -> datetime:
        return datetime.now(IST) - timedelta(hours=self.hours_back)

    @property
    def since_date_query(self) -> str:
        # snscrape supports date-only, will filter by full timestamp in code
        return self.since_dt_ist.date().isoformat()
