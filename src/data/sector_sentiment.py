# src/data/sector_sentiment.py

"""
Sector-Level Sentiment Pipeline (v2 — Fixed)

v1 Problem: Google/Yahoo RSS only return RECENT articles (~last week).
            For a 2019-2024 backtest, RSS feeds give almost nothing.

v2 Solution:
  1. Use Finnhub for HISTORICAL news (needs free API key, covers years)
  2. Use RSS feeds for RECENT/LIVE news only
  3. Generate smart synthetic sentiment for gaps using market data
     (NOT random — derived from actual price patterns and volatility)
  4. Clearly separate real vs synthetic in the output

The key insight: for backtesting, you need Finnhub or a similar
historical news API. RSS feeds are for paper/live trading only.
"""

import os
import re
import time
import json
import torch
import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote


class SectorSentimentPipeline:
    """
    Full pipeline: fetch sector news → classify relevance →
    score sentiment → aggregate to daily features.
    """

    SECTORS = {
        'technology': {
            'tickers': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                'CRM', 'ORCL', 'ADBE', 'INTC', 'AMD', 'CSCO', 'IBM',
            ],
            'names': {
                'AAPL': ['apple', 'iphone', 'ipad', 'mac', 'tim cook',
                          'app store', 'apple inc'],
                'MSFT': ['microsoft', 'windows', 'azure', 'satya nadella',
                          'xbox', 'office 365', 'linkedin', 'copilot'],
                'GOOGL': ['google', 'alphabet', 'youtube', 'android',
                           'sundar pichai', 'waymo', 'deepmind', 'gemini'],
                'AMZN': ['amazon', 'aws', 'jeff bezos', 'andy jassy',
                          'prime', 'alexa', 'whole foods'],
                'NVDA': ['nvidia', 'jensen huang', 'geforce', 'cuda',
                          'gpu', 'h100', 'a100'],
                'META': ['meta', 'facebook', 'instagram', 'whatsapp',
                          'zuckerberg', 'metaverse', 'threads'],
                'TSLA': ['tesla', 'elon musk', 'cybertruck', 'model 3',
                          'model y', 'supercharger', 'gigafactory'],
                'CRM': ['salesforce', 'marc benioff'],
                'ORCL': ['oracle', 'larry ellison'],
                'ADBE': ['adobe', 'photoshop', 'creative cloud'],
                'INTC': ['intel', 'pat gelsinger'],
                'AMD': ['amd', 'advanced micro', 'lisa su', 'ryzen'],
                'CSCO': ['cisco', 'webex'],
                'IBM': ['ibm', 'red hat', 'watson'],
            },
            'keywords': [
                'tech stocks', 'technology sector', 'big tech', 'faang',
                'magnificent seven', 'mag 7', 'tech earnings',
                'semiconductor', 'chip', 'artificial intelligence',
                'machine learning', 'cloud computing', 'saas',
                'software', 'data center', 'cybersecurity',
            ],
        },
    }

    MARKET_KEYWORDS = [
        'fed', 'federal reserve', 'interest rate', 'inflation',
        'gdp', 'unemployment', 'jobs report', 'cpi', 'ppi',
        'treasury', 'bond yield', 's&p 500', 'nasdaq',
        'dow jones', 'stock market', 'wall street', 'bull market',
        'bear market', 'recession', 'rally', 'selloff', 'sell-off',
        'earnings season', 'ipo', 'sec', 'regulation',
        'trade war', 'tariff', 'geopolitical',
    ]

    RELEVANCE_WEIGHTS = {
        'DIRECT':  1.0,
        'SECTOR':  0.5,
        'MARKET':  0.2,
        'DISCARD': 0.0,
    }

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.target_ticker = self.config['data']['ticker']
        self.sector = 'technology'
        self.sector_config = self.SECTORS[self.sector]

        self.model = None
        self.tokenizer = None
        self.device = None

        self.output_dir = Path('data/raw/news')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
            ),
        })
        self._last_request_time = 0

    def _rate_limit(self, interval: float = 1.5):
        elapsed = time.time() - self._last_request_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request_time = time.time()

    def _clean_headline(self, text: str) -> str:
        if not text:
            return ''
        text = re.sub(r'<[^>]+>', '', text)
        text = ' '.join(text.split()).strip()
        return text

    # ================================================================
    # NEWS FETCHING
    # ================================================================

    def fetch_finnhub_sector(self, start_date: str,
                              end_date: str) -> pd.DataFrame:
        """
        Fetch historical news from Finnhub for ALL sector tickers.
        This is the PRIMARY source for backtesting (has years of history).
        """
        finnhub_key = os.environ.get('FINNHUB_API_KEY', '')
        if not finnhub_key:
            print(f"    [Finnhub] SKIPPED — set FINNHUB_API_KEY for "
                  f"historical news")
            print(f"    [Finnhub] Get free key: https://finnhub.io")
            return pd.DataFrame()

        print(f"    [Finnhub] Fetching historical news for sector ...")

        # Fetch for target + top peers
        tickers_to_fetch = [self.target_ticker] + [
            t for t in self.sector_config['tickers']
            if t != self.target_ticker
        ][:7]  # target + 7 peers = 8 total

        all_articles = []
        seen_headlines = set()

        for ticker_idx, ticker in enumerate(tickers_to_fetch):
            print(f"      [{ticker_idx + 1}/{len(tickers_to_fetch)}] "
                  f"{ticker} ...", end=" ", flush=True)

            ticker_count = 0
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            current = start

            while current < end:
                chunk_end = min(current + timedelta(days=60), end)
                self._rate_limit(1.2)

                params = {
                    'symbol': ticker,
                    'from': current.strftime('%Y-%m-%d'),
                    'to': chunk_end.strftime('%Y-%m-%d'),
                    'token': finnhub_key,
                }

                try:
                    resp = requests.get(
                        'https://finnhub.io/api/v1/company-news',
                        params=params, timeout=15,
                    )

                    if resp.status_code == 429:
                        print(f"rate limited, waiting 60s...", end=" ")
                        time.sleep(60)
                        continue

                    if resp.status_code != 200:
                        current = chunk_end + timedelta(days=1)
                        continue

                    articles = resp.json()

                    if isinstance(articles, list):
                        for art in articles:
                            headline = self._clean_headline(
                                art.get('headline', '')
                            )
                            if len(headline) < 15:
                                continue

                            # Dedup by first 80 chars lowercase
                            hkey = headline.lower()[:80]
                            if hkey in seen_headlines:
                                continue
                            seen_headlines.add(hkey)

                            ts = art.get('datetime', 0)
                            if ts == 0:
                                current = chunk_end + timedelta(days=1)
                                continue

                            dt = datetime.fromtimestamp(ts)

                            all_articles.append({
                                'date': dt.strftime('%Y-%m-%d'),
                                'datetime': dt.isoformat(),
                                'headline': headline,
                                'source': art.get('source', ''),
                                'url': art.get('url', ''),
                                'api_source': 'finnhub',
                                'fetched_for_ticker': ticker,
                            })
                            ticker_count += 1

                except Exception as e:
                    pass

                current = chunk_end + timedelta(days=1)

            print(f"{ticker_count} articles")

        df = pd.DataFrame(all_articles)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        print(f"      Finnhub total: {len(df)} unique articles")
        return df

    def fetch_rss_recent(self) -> pd.DataFrame:
        """
        Fetch recent news from Yahoo + Google RSS.
        Only useful for the last ~7 days. Used to supplement
        Finnhub and for paper/live trading.
        """
        print(f"    [RSS Feeds] Fetching recent news ...")

        all_articles = []
        seen_headlines = set()

        # Yahoo RSS for target + top peers
        tickers_rss = [self.target_ticker] + [
            t for t in self.sector_config['tickers'][:5]
            if t != self.target_ticker
        ]

        for ticker in tickers_rss:
            url = (f"https://feeds.finance.yahoo.com/rss/2.0/"
                   f"headline?s={ticker}&region=US&lang=en-US")
            self._rate_limit(2.0)

            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code != 200:
                    continue

                root = ET.fromstring(resp.content)
                for item in root.findall('.//item'):
                    title = self._clean_headline(
                        item.findtext('title', '')
                    )
                    pub_date = item.findtext('pubDate', '')
                    link = item.findtext('link', '')

                    if len(title) < 15:
                        continue

                    hkey = title.lower()[:80]
                    if hkey in seen_headlines:
                        continue
                    seen_headlines.add(hkey)

                    try:
                        dt = pd.Timestamp(pub_date)
                    except Exception:
                        continue

                    all_articles.append({
                        'date': dt.strftime('%Y-%m-%d'),
                        'datetime': dt.isoformat(),
                        'headline': title,
                        'source': 'Yahoo Finance',
                        'url': link,
                        'api_source': 'yahoo_rss',
                        'fetched_for_ticker': ticker,
                    })
            except Exception:
                pass

        # Google RSS — a few targeted queries only
        google_queries = [
            f'{self.target_ticker} stock',
            'technology stocks earnings',
            'big tech stocks',
        ]

        for query in google_queries:
            encoded = quote(query)
            url = (f"https://news.google.com/rss/search?"
                   f"q={encoded}&hl=en-US&gl=US&ceid=US:en")
            self._rate_limit(3.0)

            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code != 200:
                    continue

                root = ET.fromstring(resp.content)
                for item in root.findall('.//item'):
                    title = self._clean_headline(
                        item.findtext('title', '')
                    )
                    pub_date = item.findtext('pubDate', '')

                    if len(title) < 15:
                        continue

                    # Strip source suffix
                    if ' - ' in title:
                        parts = title.rsplit(' - ', 1)
                        if len(parts[1]) < 30:
                            title = parts[0].strip()

                    hkey = title.lower()[:80]
                    if hkey in seen_headlines:
                        continue
                    seen_headlines.add(hkey)

                    try:
                        dt = pd.Timestamp(pub_date)
                    except Exception:
                        continue

                    all_articles.append({
                        'date': dt.strftime('%Y-%m-%d'),
                        'datetime': dt.isoformat(),
                        'headline': title,
                        'source': 'Google News',
                        'url': '',
                        'api_source': 'google_rss',
                        'fetched_for_ticker': '',
                    })
            except Exception:
                pass

        df = pd.DataFrame(all_articles)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])

        print(f"      RSS total: {len(df)} recent articles")
        return df

    def load_kaggle_news(self) -> pd.DataFrame:
        """Load from local Kaggle dataset if available."""
        kaggle_path = Path(
            self.config.get('news_sources', {}).get(
                'kaggle_path', 'data/raw/raw_analyst_ratings.csv'
            )
        )

        if not kaggle_path.exists():
            print(f"    [Kaggle] Skipped (not found: {kaggle_path})")
            return pd.DataFrame()

        print(f"    [Kaggle] Loading ...")

        try:
            df = pd.read_csv(kaggle_path)
            df.columns = df.columns.str.lower().str.strip()

            headline_col = None
            date_col = None
            ticker_col = None

            for col in df.columns:
                if 'headline' in col or 'title' in col:
                    headline_col = col
                elif 'date' in col or 'time' in col:
                    date_col = col
                elif 'stock' in col or 'ticker' in col or 'symbol' in col:
                    ticker_col = col

            if headline_col is None or date_col is None:
                print(f"      Cannot identify columns: {list(df.columns)}")
                return pd.DataFrame()

            # Filter for sector tickers
            sector_tickers = set(
                t.upper() for t in self.sector_config['tickers']
            )

            if ticker_col:
                df = df[df[ticker_col].str.upper().isin(sector_tickers)]

            records = []
            seen = set()
            for _, row in df.iterrows():
                try:
                    headline = self._clean_headline(str(row[headline_col]))
                    if len(headline) < 15:
                        continue

                    hkey = headline.lower()[:80]
                    if hkey in seen:
                        continue
                    seen.add(hkey)

                    dt = pd.Timestamp(row[date_col])
                    fetched_ticker = (
                        row[ticker_col].upper() if ticker_col else ''
                    )

                    records.append({
                        'date': dt.strftime('%Y-%m-%d'),
                        'datetime': dt.isoformat(),
                        'headline': headline,
                        'source': 'kaggle',
                        'url': '',
                        'api_source': 'kaggle',
                        'fetched_for_ticker': fetched_ticker,
                    })
                except Exception:
                    pass

            result = pd.DataFrame(records)
            if not result.empty:
                result['date'] = pd.to_datetime(result['date'])

            print(f"      Kaggle: {len(result)} sector articles")
            return result

        except Exception as e:
            print(f"      Kaggle error: {e}")
            return pd.DataFrame()

    
    def fetch_all_sector_news(self) -> pd.DataFrame:
        """Fetch from all available sources and combine."""
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']

        print(f"\n  Fetching sector-wide news")
        print(f"  Target: {self.target_ticker}")
        print(f"  Sector: {self.sector} "
              f"({len(self.sector_config['tickers'])} companies)")
        print(f"  Date range: {start_date} to {end_date}")

        dfs = []

        # Historical source (primary for backtesting)
        finnhub_df = self.fetch_finnhub_sector(start_date, end_date)
        if not finnhub_df.empty:
            dfs.append(finnhub_df)

        # Kaggle (historical backup)
        kaggle_df = self.load_kaggle_news()
        if not kaggle_df.empty:
            kaggle_filtered = kaggle_df[
                (kaggle_df['date'] >= start_date) &
                (kaggle_df['date'] <= end_date)
            ]
            if not kaggle_filtered.empty:
                dfs.append(kaggle_filtered)

        # RSS (recent only — supplements recent end of data)
        rss_df = self.fetch_rss_recent()
        if not rss_df.empty:
            rss_filtered = rss_df[
                (rss_df['date'] >= start_date) &
                (rss_df['date'] <= end_date)
            ]
            if not rss_filtered.empty:
                dfs.append(rss_filtered)

        if not dfs:
            print(f"\n    [!] No news fetched from any source.")
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])

        # Deduplicate on first 80 chars of headline lowercase
        combined['_dedup_key'] = (
            combined['headline'].str.lower().str[:80]
        )
        before_dedup = len(combined)
        combined = combined.drop_duplicates(
            subset=['_dedup_key']
        ).drop(columns=['_dedup_key'])
        combined = combined.sort_values('date').reset_index(drop=True)

        # Remove too-short headlines
        combined = combined[
            combined['headline'].str.strip().str.len() > 15
        ]

        after_dedup = len(combined)
        print(f"\n    Before dedup: {before_dedup}")
        print(f"    After dedup:  {after_dedup}")
        print(f"    Removed:      {before_dedup - after_dedup} duplicates")

        # Source breakdown
        if not combined.empty:
            print(f"\n    Sources:")
            for src, cnt in combined['api_source'].value_counts().items():
                print(f"      {src:<15s}: {cnt:>5d} articles")

            # Date coverage
            unique_dates = combined['date'].dt.date.nunique()
            trading_days = pd.bdate_range(start_date, end_date)
            coverage = unique_dates / max(len(trading_days), 1)
            print(f"\n    Unique news days: {unique_dates}")
            print(f"    Trading days:     {len(trading_days)}")
            print(f"    Raw coverage:     {coverage:.0%}")

        return combined

    # ================================================================
    # RELEVANCE CLASSIFICATION
    # ================================================================

    def classify_relevance(self, headline: str) -> Tuple[str, float]:
        headline_lower = headline.lower()

        target_names = self.sector_config['names'].get(
            self.target_ticker, []
        )
        target_all = target_names + [self.target_ticker.lower()]

        for name in target_all:
            if name.lower() in headline_lower:
                return 'DIRECT', self.RELEVANCE_WEIGHTS['DIRECT']

        for ticker, names in self.sector_config['names'].items():
            if ticker == self.target_ticker:
                continue
            for name in names + [ticker.lower()]:
                if name.lower() in headline_lower:
                    return 'SECTOR', self.RELEVANCE_WEIGHTS['SECTOR']

        for keyword in self.sector_config['keywords']:
            if keyword.lower() in headline_lower:
                return 'SECTOR', self.RELEVANCE_WEIGHTS['SECTOR']

        for keyword in self.MARKET_KEYWORDS:
            if keyword.lower() in headline_lower:
                return 'MARKET', self.RELEVANCE_WEIGHTS['MARKET']

        return 'DISCARD', self.RELEVANCE_WEIGHTS['DISCARD']

    def classify_all(self, news_df: pd.DataFrame) -> pd.DataFrame:
        levels = []
        weights = []

        for _, row in news_df.iterrows():
            level, weight = self.classify_relevance(row['headline'])
            levels.append(level)
            weights.append(weight)

        news_df = news_df.copy()
        news_df['relevance_level'] = levels
        news_df['relevance_weight'] = weights

        counts = news_df['relevance_level'].value_counts()
        print(f"\n    Relevance classification:")
        for level in ['DIRECT', 'SECTOR', 'MARKET', 'DISCARD']:
            count = counts.get(level, 0)
            pct = count / max(len(news_df), 1) * 100
            print(f"      {level:<8s}: {count:>5d} ({pct:>5.1f}%)  "
                  f"weight={self.RELEVANCE_WEIGHTS[level]}")

        before = len(news_df)
        news_df = news_df[news_df['relevance_level'] != 'DISCARD']
        print(f"    Kept {len(news_df)}/{before} relevant articles")

        return news_df.reset_index(drop=True)

    # ================================================================
    # SENTIMENT SCORING
    # ================================================================

    def _load_finbert(self):
        if self.model is not None:
            return

        print(f"    Loading FinBERT ...")
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
        )

        model_name = self.config.get('sentiment', {}).get(
            'model_name', 'ProsusAI/finbert'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model.eval()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        print(f"    FinBERT loaded on {self.device}")

    def _load_vader(self):
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def score_sentiment(self, headlines: List[str],
                        use_finbert: bool = True,
                        batch_size: int = 16) -> List[float]:
        if use_finbert:
            return self._score_finbert(headlines, batch_size)
        else:
            return self._score_vader(headlines)

    def _score_finbert(self, headlines: List[str],
                        batch_size: int = 16) -> List[float]:
        self._load_finbert()
        scores = []
        total = len(headlines)

        for i in range(0, total, batch_size):
            batch = headlines[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors='pt', truncation=True,
                max_length=512, padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            for j in range(len(batch)):
                scores.append(float(probs[j][0] - probs[j][1]))

            if (i + batch_size) % (batch_size * 20) == 0 and i > 0:
                print(f"      Scored {min(i + batch_size, total)}/{total}")

        return scores

    def _score_vader(self, headlines: List[str]) -> List[float]:
        if not hasattr(self, 'vader'):
            self._load_vader()
        return [self.vader.polarity_scores(h)['compound'] for h in headlines]

    # ================================================================
    # DAILY AGGREGATION
    # ================================================================
    # ================================================================
    # DAILY AGGREGATION (FIXED)
    # ================================================================
    # ================================================================
    # DAILY AGGREGATION (FIXED)
    # ================================================================

    def aggregate_daily(self, news_df: pd.DataFrame,
                        trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Aggregate to daily features using relevance weights.
        Uses PREVIOUS day's news (look-ahead avoidance).
        """
        print(f"\n    Aggregating to daily features ...")

        news_df = news_df.copy()
        news_df['trading_date'] = (
            pd.to_datetime(news_df['date']) + pd.Timedelta(days=1)
        )
        news_df['weighted_score'] = (
            news_df['sentiment_score'] * news_df['relevance_weight']
        )

        daily_records = []

        for date in trading_dates:
            day_news = news_df[
                news_df['trading_date'].dt.date == date.date()
            ]

            if len(day_news) == 0:
                daily_records.append({
                    'date': date,
                    'direct_sentiment': np.nan,
                    'sector_sentiment': np.nan,
                    'market_sentiment': np.nan,
                    'weighted_sentiment': np.nan,
                    'sentiment_dispersion': np.nan,
                    'news_volume': 0.0,
                    'n_articles': 0,
                    'n_direct': 0,
                    'is_real_news': False,
                })
                continue

            direct = day_news[day_news['relevance_level'] == 'DIRECT']
            sector = day_news[day_news['relevance_level'] == 'SECTOR']
            market = day_news[day_news['relevance_level'] == 'MARKET']

            direct_sent = (
                direct['sentiment_score'].mean() if len(direct) > 0
                else np.nan
            )
            sector_sent = (
                sector['sentiment_score'].mean() if len(sector) > 0
                else np.nan
            )
            market_sent = (
                market['sentiment_score'].mean() if len(market) > 0
                else np.nan
            )

            total_weight = day_news['relevance_weight'].sum()
            weighted_sent = (
                day_news['weighted_score'].sum() / total_weight
                if total_weight > 0 else np.nan
            )

            dispersion = (
                day_news['weighted_score'].std()
                if len(day_news) > 1 else 0.0
            )

            daily_records.append({
                'date': date,
                'direct_sentiment': direct_sent,
                'sector_sentiment': sector_sent,
                'market_sentiment': market_sent,
                'weighted_sentiment': weighted_sent,
                'sentiment_dispersion': dispersion,
                'news_volume': float(len(day_news)),
                'n_articles': len(day_news),
                'n_direct': len(direct),
                'is_real_news': True,
            })

        daily = pd.DataFrame(daily_records).set_index('date')

        real_days = daily['is_real_news'].sum()
        total_days = len(daily)
        print(f"    Real news days: {real_days}/{total_days} "
              f"({real_days / total_days:.0%})")

        return daily
    # ================================================================
    # GAP FILLING (FIXED)
    # ================================================================

    def fill_gaps_with_proxy(self, daily: pd.DataFrame,
                              trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        FIX #1: Do NOT use market-proxy sentiment.
        Instead, forward-fill real sentiment with DECAY toward zero.

        Why: Market proxy = price-derived = redundant with technicals.
        Zero signal is better than fake signal.

        Approach:
          - Days with real news: actual sentiment (untouched)
          - Days without news: carry forward last real sentiment
            but decay it toward zero (half-life = 3 days)
          - news_volume = 0 tells the agent "this is stale"
        """
        print(f"\n    Filling gaps with decayed forward-fill ...")

        real_mask = daily['is_real_news'].astype(bool)
        real_count = int(real_mask.sum())
        gap_count = int((~real_mask).sum())

        sentiment_cols = [
            'direct_sentiment', 'sector_sentiment',
            'market_sentiment', 'weighted_sentiment',
            'sentiment_dispersion',
        ]

        # Step 1: Forward-fill NaN from real news days
        for col in sentiment_cols:
            daily.ffill(inplace=True)



        # Step 2: Apply exponential decay on non-real days
        # Half-life = 3 days: sentiment halves every 3 days without news
        decay_half_life = 3
        days_since_news = 0

        for i in range(len(daily)):
            if daily.iloc[i]['is_real_news']:
                days_since_news = 0
            else:
                days_since_news += 1
                decay = 0.5 ** (days_since_news / decay_half_life)
                for col in sentiment_cols:
                    col_idx = daily.columns.get_loc(col)
                    daily.iloc[i, col_idx] = daily.iloc[i, col_idx] * decay

        # Fill any remaining NaN at start with 0
        daily = daily.fillna(0.0)

        print(f"    Gap-fill results:")
        print(f"      Real news days:     {real_count}")
        print(f"      Gap days (decayed): {gap_count}")
        print(f"      Decay half-life:    {decay_half_life} days")
        print(f"      Method:             forward-fill + exponential decay")
        print(f"      Coverage:           100%")

        return daily
    # ================================================================
    # FEATURE SELECTION (FIXED — ALL 5 ISSUES)
    # ================================================================

    def select_features(self, daily: pd.DataFrame,
                        n_features: int = 5) -> pd.DataFrame:
        """
        Select final 5 sentiment features for the RL agent.

        All 5 fixes applied:
          #1: Proxy days decayed toward zero (done in fill_gaps)
          #2: news_volume = confidence indicator
          #3: EMA smoothing reduces noise
          #4: EMA captures multi-day impact, momentum captures trend
          #5: Confidence gate zeroes low-quality signal
        """
        print(f"\n    Selecting and processing final features ...")

        # FIX #3: Smooth with EMA (span=3)
        ema_span = 3
        smoothed_sentiment = (
            daily['weighted_sentiment']
            .ewm(span=ema_span, adjust=False)
            .mean()
        )
        smoothed_dispersion = (
            daily['sentiment_dispersion']
            .ewm(span=ema_span, adjust=False)
            .mean()
        )

        # Normalize news volume
        avg_vol = daily['news_volume'].replace(0, np.nan).mean()
        if avg_vol and avg_vol > 0:
            news_volume_norm = daily['news_volume'] / avg_vol
        else:
            news_volume_norm = pd.Series(0.0, index=daily.index)

        # FIX #5: Confidence gate — zero out weak signals
        confidence_threshold = 0.3
        has_confidence = news_volume_norm >= confidence_threshold

        gated_sentiment = np.where(
            has_confidence, smoothed_sentiment, 0.0
        )
        gated_dispersion = np.where(
            has_confidence, smoothed_dispersion, 0.0
        )

        # FIX #4: Momentum from gated signal
        gated_series = pd.Series(gated_sentiment, index=daily.index)
        sentiment_momentum = gated_series.diff(5).fillna(0.0)

        # FIX #2: news_volume as confidence indicator
        news_volume_clipped = news_volume_norm.clip(0, 3)

        # Build final features
        selected = pd.DataFrame(index=daily.index)
        selected['sentiment_score'] = gated_sentiment
        selected['sentiment_std'] = gated_dispersion
        selected['sentiment_momentum'] = sentiment_momentum.values
        selected['news_volume'] = news_volume_clipped.values
        selected['sentiment_ma5'] = (
            gated_series.rolling(5).mean().fillna(0.0).values
        )

        # Report
        active_days = int((selected['sentiment_score'] != 0).sum())
        total_days = len(selected)
        silent_days = total_days - active_days

        print(f"\n    Final feature stats:")
        print(f"      Active sentiment days:  {active_days} "
              f"({active_days / total_days:.0%})")
        print(f"      Silent days (zeroed):   {silent_days} "
              f"({silent_days / total_days:.0%})")
        print(f"      EMA smoothing span:     {ema_span}")
        print(f"      Confidence threshold:   {confidence_threshold}")
        print(f"")

        if active_days > 0:
            active_vals = selected.loc[
                selected['sentiment_score'] != 0, 'sentiment_score'
            ]
            print(f"      Active sentiment range: "
                  f"[{active_vals.min():.3f}, {active_vals.max():.3f}]")
            print(f"      Active sentiment mean:  {active_vals.mean():.4f}")

        print(f"")
        print(f"    Agent behavior:")
        print(f"      news_volume > 0  →  trust sentiment signal")
        print(f"      news_volume = 0  →  rely on technicals only")

        return selected

    def run(self, use_finbert: bool = True,
            save_intermediate: bool = True) -> pd.DataFrame:
        """Run the complete sector sentiment pipeline."""
        print("=" * 60)
        print("  SECTOR SENTIMENT PIPELINE v2")
        print(f"  Target: {self.target_ticker}")
        print(f"  Sector: {self.sector}")
        print(f"  Scorer: {'FinBERT' if use_finbert else 'VADER'}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Step 1: Fetch
        raw_news = self.fetch_all_sector_news()

        if raw_news.empty:
            print("\n  [!] No news fetched. Using market-proxy only.")
            prices = pd.read_csv(
                self.config['paths']['raw_prices'],
                index_col=0, parse_dates=True,
            )
            trading_dates = prices.index

            proxy = self._generate_market_proxy_sentiment(
                trading_dates, set()
            )
            if proxy.empty:
                print("  [!] Cannot generate proxy either. Aborting.")
                return pd.DataFrame()

            proxy['sentiment_momentum'] = (
                proxy['weighted_sentiment'].diff(5).fillna(0)
            )
            final = self.select_features(proxy)
            output_path = Path(self.config['paths']['sentiment_scores'])
            final.to_csv(output_path)
            print(f"\n    Proxy-only features saved to {output_path}")
            return final

        if save_intermediate:
            raw_path = (self.output_dir /
                        f'{self.target_ticker}_sector_raw.csv')
            raw_news.to_csv(raw_path, index=False)

        # Step 2: Classify
        classified = self.classify_all(raw_news)

        if classified.empty:
            print("\n  [!] No relevant articles after classification.")
            return pd.DataFrame()

        # Step 3: Score
        print(f"\n    Scoring {len(classified)} articles ...")
        scores = self.score_sentiment(
            classified['headline'].tolist(),
            use_finbert=use_finbert,
        )
        classified['sentiment_score'] = scores

        if save_intermediate:
            scored_path = (self.output_dir /
                           f'{self.target_ticker}_sector_scored.csv')
            classified.to_csv(scored_path, index=False)

        # Step 4: Load trading dates
        prices = pd.read_csv(
            self.config['paths']['raw_prices'],
            index_col=0, parse_dates=True,
        )
        trading_dates = prices.index

        # Step 5: Aggregate
        daily = self.aggregate_daily(classified, trading_dates)

        # Step 6: Fill gaps
        daily = self.fill_gaps_with_proxy(daily, trading_dates)

        if save_intermediate:
            full_path = (self.output_dir /
                         f'{self.target_ticker}_sector_daily.csv')
            daily.to_csv(full_path)

        # Step 7: Select final features
        final = self.select_features(daily)

        # Step 8: Save
        output_path = Path(self.config['paths']['sentiment_scores'])
        final.to_csv(output_path)
        print(f"\n    Sentiment features saved to {output_path}")

        # Summary
        real_pct = daily['is_real_news'].mean() * 100
        print(f"\n    ╔══════════════════════════════════════════════╗")
        print(f"    ║  SECTOR SENTIMENT PIPELINE COMPLETE           ║")
        print(f"    ╠══════════════════════════════════════════════╣")
        print(f"    ║  Articles fetched:   {len(raw_news):<22d} ║")
        print(f"    ║  Relevant articles:  {len(classified):<22d} ║")
        print(f"    ║  Real news days:     {int(daily['is_real_news'].sum()):<22d} ║")
        print(f"    ║  Real news coverage: {real_pct:<21.0f}% ║")
        print(f"    ║  Total trading days: {len(final):<22d} ║")
        print(f"    ║  Final features:     {final.shape[1]:<22d} ║")
        print(f"    ║  Gap-fill method:    {'market proxy':<22s} ║")
        print(f"    ╚══════════════════════════════════════════════╝")

        return final


# ======================================================================
# CLI
# ======================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Sector-level sentiment pipeline'
    )
    parser.add_argument(
        '--vader', action='store_true',
        help='Use VADER instead of FinBERT',
    )
    parser.add_argument(
        '--ticker', type=str, default=None,
        help='Target ticker (default: from config)',
    )
    args = parser.parse_args()

    pipeline = SectorSentimentPipeline()

    if args.ticker:
        pipeline.target_ticker = args.ticker

    pipeline.run(use_finbert=not args.vader)


if __name__ == '__main__':
    main()