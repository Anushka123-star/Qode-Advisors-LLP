# Twitter Market Sentiment Signals (Qode Assignment)
I built a real-time data collection and analysis pipeline that scrapes Twitter/X for Indian stock market discussions and converts them into quantitative sentiment signals.

---

## Overview

The goal of this project was to:
- Collect at least **2000 tweets from the last 24 hours** related to Indian stock markets (`#nifty50`, `#sensex`, `#intraday`, `#banknifty`) without using paid APIs.
- Process and normalize the data, handling Unicode, Hinglish, and emojis.
- Store tweets efficiently in **Parquet format** with deduplication.
- Convert raw text into **numerical signals** using TF-IDF, multilingual sentiment analysis, and custom feature engineering.
- Aggregate signals into a time series with confidence intervals for potential use in algorithmic trading.

I structured the solution as a production-ready pipeline with modular components:  
`scraper → processor → analysis`.

---

## Project Structure

```

qode-twitter-signals/
├─ src/
│  ├─ config.py        # Configs (hashtags, paths, time window)
│  ├─ scraper.py       # Tweet scraper (snscrape + concurrency + parquet writer)
│  ├─ processor.py     # Cleaning, normalization, feature engineering
│  ├─ analysis.py      # TF-IDF, sentiment, composite signals, visualization
│  └─ utils.py         # Logging utilities
├─ data/
│  ├─ raw/             # Raw parquet files (scraped tweets)
│  └─ processed/       # Processed parquet + signals
├─ run\_all.sh          # One-command runner
├─ requirements.txt    # Dependencies
└─ README.md           # Documentation

````

---

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/YOURUSERNAME/qode-twitter-signals.git
cd qode-twitter-signals
````

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Demo

To run the entire pipeline (scrape → process → analysis):

```bash
bash run_all.sh
```

This will:

1. Scrape 2000+ tweets from the last 24h across selected hashtags.
2. Write raw parquet data into `data/raw/`.
3. Clean and normalize text, engineer features, and save to `data/processed/`.
4. Run TF-IDF, multilingual sentiment, and composite signal analysis.
5. Save:

   * `tweets_processed_*.parquet` (processed dataset)
   * `tweets_processed_*_signals.csv` (aggregated signals)
   * `tweets_processed_*_signal_plot.png` (visualization)

---

### Example Logs

```
2025-09-01 10:32:41 | INFO | scraper | Scraping: #nifty50 since:2025-08-31
2025-09-01 10:32:44 | INFO | scraper | Scraping: #sensex since:2025-08-31
2025-09-01 10:33:20 | INFO | scraper | Wrote raw parquet: data/raw/tweets_raw_20250901_1032.parquet | unique: 2845
2025-09-01 10:33:31 | INFO | processor | Wrote processed parquet: data/processed/tweets_processed_20250901_1032.parquet | rows=2701
2025-09-01 10:34:50 | INFO | analysis  | Wrote signals: data/processed/tweets_processed_20250901_1032_signals.csv
2025-09-01 10:34:52 | INFO | analysis  | Saved plot: data/processed/tweets_processed_20250901_1032_signal_plot.png
```

---

### Example Outputs

#### Processed Tweet Features

```
content_norm        | has_bullish_kw | has_bearish_kw | len_chars | has_emoji
--------------------|----------------|----------------|-----------|----------
nifty breakout soon | 1              | 0              | 21        | 0
banknifty support   | 1              | 0              | 18        | 1
```

#### Aggregated Signal (per-minute)

```
t_minute              | signal_mean | ci_lo | ci_hi | n
----------------------|-------------|-------|-------|---
2025-09-01 10:32:00   |  0.42       | 0.33  | 0.51  | 12
2025-09-01 10:33:00   | -0.15       | -0.24 | -0.06 | 10
```

#### Visualization

Automatically generated and saved under `data/processed/`.
---

## Technical Highlights

* **Scraping**: `snscrape` with concurrency, rate-limit safe, deduplication by tweet ID.
* **Processing**: Unicode normalization, emoji preservation, Hinglish/Devanagari safe cleaning.
* **Features**:

  * TF-IDF (char n-grams, robust for code-mixed text).
  * Multilingual transformer-based sentiment (`cardiffnlp/twitter-xlm-roberta-base-sentiment`).
  * Custom signals (bullish/bearish keywords, engagement).
* **Composite Signal**: Weighted mix of sentiment, engagement, and keywords with 95% confidence intervals.
* **Performance**: Batch parquet writes, sparse TF-IDF matrices, downsampled plotting for memory efficiency.

---

## Scalability

This pipeline is designed to scale:

* 10x more tweets can be handled with the same approach.
* Can be extended to streaming (Kafka + Spark/Flink).
* Replace snscrape with Selenium + rotating user agents if scraping restrictions tighten.
