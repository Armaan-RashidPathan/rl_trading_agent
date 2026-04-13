# src/data/news_fetcher.py

"""
Real News Data Fetcher

Downloads actual financial news for any ticker from multiple sources:
  - Finnhub.io (primary: company news, press releases)
  - NewsAPI.org (backup: broader coverage)
  - Kaggle datasets (historical: analyst ratings)

News is aligned to trading days and stored for sentiment extraction.
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import yaml


class NewsFetcher:
    """
    Fetches real financial news from multiple APIs.
    
    Usage:
        fetcher = NewsFetcher()
        news_df = fetcher.fetch_all("AAPL", "2022-01-01", "2024-12-01")
        fetcher.save(news_df, "AAPL")
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        news_cfg = self.config.get('news_sources', {})

        # API keys from environment variables
        self.finnhub_key = os.environ.get(
            news_cfg.get('finnhub_key_env', 'FINNHUB_API_KEY'), ''
        )
        self.newsapi_key = os.environ.get(
            news_cfg.get('newsapi_key_env', 'NEWSAPI_KEY'), ''
        )

        # Output directory
        self.output_dir = Path('data/raw/news')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 1.0  # seconds between requests

    def _rate_limit(self):
        """Simple rate limiter to avoid API throttling."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    # ==================================================================
    # FINNHUB (Primary source)
    # ==================================================================

    def fetch_finnhub(self, ticker: str, start_date: str,
                      end_date: str) -> pd.DataFrame:
        """
        Fetch company news from Finnhub.

        Finnhub returns news articles with:
          - headline, summary, source, url
          - datetime (unix timestamp)
          - category, related tickers

        Free tier: 60 calls/minute, 1 year history
        """
        if not self.finnhub_key:
            print("    [!] FINNHUB_API_KEY not set. Skipping Finnhub.")
            return pd.DataFrame()

        print(f"    Fetching from Finnhub ...")

        all_articles = []
        base_url = "https://finnhub.io/api/v1/company-news"

        # Finnhub limits to ~1 year per request, so chunk if needed
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Process in 3-month chunks to stay within limits
        current_start = start
        chunk_count = 0

        while current_start < end:
            current_end = min(current_start + timedelta(days=90), end)

            self._rate_limit()

            params = {
                'symbol': ticker,
                'from': current_start.strftime('%Y-%m-%d'),
                'to': current_end.strftime('%Y-%m-%d'),
                'token': self.finnhub_key,
            }

            try:
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                articles = response.json()

                if isinstance(articles, list):
                    all_articles.extend(articles)
                    chunk_count += 1

                    if chunk_count % 5 == 0:
                        print(f"      ... {len(all_articles)} articles so far "
                              f"(through {current_end.strftime('%Y-%m-%d')})")

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    print(f"      Rate limited. Waiting 60s ...")
                    time.sleep(60)
                    continue
                else:
                    print(f"      HTTP error: {e}")
            except Exception as e:
                print(f"      Error: {e}")

            current_start = current_end + timedelta(days=1)

        if not all_articles:
            print(f"      No articles found from Finnhub")
            return pd.DataFrame()

        # Parse into DataFrame
        records = []
        for article in all_articles:
            dt = datetime.fromtimestamp(article.get('datetime', 0))
            records.append({
                'date': dt.strftime('%Y-%m-%d'),
                'datetime': dt.isoformat(),
                'headline': article.get('headline', ''),
                'summary': article.get('summary', ''),
                'source': article.get('source', ''),
                'url': article.get('url', ''),
                'category': article.get('category', ''),
                'ticker': ticker,
                'api_source': 'finnhub',
            })

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])

        # Remove duplicates
        df = df.drop_duplicates(subset=['headline', 'date'])

        print(f"      Finnhub: {len(df)} unique articles")
        return df

    # ==================================================================
    # NEWSAPI (Backup source)
    # ==================================================================

    def fetch_newsapi(self, ticker: str, company_name: str = None,
                      start_date: str = None,
                      end_date: str = None) -> pd.DataFrame:
        """
        Fetch news from NewsAPI.org.

        Free tier limitations:
          - Only last 30 days of articles
          - 100 requests/day
          - No commercial use

        For historical data, use Finnhub or Kaggle instead.
        """
        if not self.newsapi_key:
            print("    [!] NEWSAPI_KEY not set. Skipping NewsAPI.")
            return pd.DataFrame()

        print(f"    Fetching from NewsAPI ...")

        # Build search query
        if company_name is None:
            # Map common tickers to company names
            ticker_to_name = {
                'AAPL': 'Apple',
                'MSFT': 'Microsoft',
                'GOOGL': 'Google OR Alphabet',
                'AMZN': 'Amazon',
                'TSLA': 'Tesla',
                'NVDA': 'NVIDIA',
                'META': 'Meta OR Facebook',
                'JPM': 'JPMorgan',
                'NFLX': 'Netflix',
                'DIS': 'Disney',
            }
            company_name = ticker_to_name.get(ticker, ticker)

        query = f'"{company_name}" stock OR shares OR market'

        base_url = "https://newsapi.org/v2/everything"

        params = {
            'q': query,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100,
            'apiKey': self.newsapi_key,
        }

        # NewsAPI free tier only allows 1 month back
        if start_date:
            one_month_ago = (datetime.now() - timedelta(days=29)).strftime('%Y-%m-%d')
            params['from'] = max(start_date, one_month_ago)
        if end_date:
            params['to'] = end_date

        all_articles = []
        page = 1
        max_pages = 5  # Limit to avoid burning through daily quota

        while page <= max_pages:
            params['page'] = page
            self._rate_limit()

            try:
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                articles = data.get('articles', [])
                if not articles:
                    break

                all_articles.extend(articles)
                page += 1

                total_results = data.get('totalResults', 0)
                if len(all_articles) >= total_results:
                    break

            except Exception as e:
                print(f"      NewsAPI error: {e}")
                break

        if not all_articles:
            print(f"      No articles found from NewsAPI")
            return pd.DataFrame()

        records = []
        for article in all_articles:
            pub_date = article.get('publishedAt', '')
            if pub_date:
                try:
                    dt = pd.Timestamp(pub_date)
                    records.append({
                        'date': dt.strftime('%Y-%m-%d'),
                        'datetime': dt.isoformat(),
                        'headline': article.get('title', ''),
                        'summary': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', ''),
                        'category': 'general',
                        'ticker': ticker,
                        'api_source': 'newsapi',
                    })
                except Exception:
                    pass

        df = pd.DataFrame(records)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates(subset=['headline', 'date'])

        print(f"      NewsAPI: {len(df)} unique articles")
        return df

    # ==================================================================
    # KAGGLE (Historical backup)
    # ==================================================================

    def load_kaggle_news(self, ticker: str) -> pd.DataFrame:
        """
        Load news from Kaggle dataset if available.

        Expected file: data/raw/raw_analyst_ratings.csv
        Columns: headline, date, stock (ticker)
        """
        kaggle_path = Path('data/raw/raw_analyst_ratings.csv')

        if not kaggle_path.exists():
            print(f"    [!] Kaggle file not found at {kaggle_path}")
            return pd.DataFrame()

        print(f"    Loading Kaggle dataset ...")

        try:
            df = pd.read_csv(kaggle_path)

            # Normalize column names
            df.columns = df.columns.str.lower().str.strip()

            # Find the relevant columns
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
                print(f"      Could not find headline/date columns in Kaggle data")
                print(f"      Available columns: {list(df.columns)}")
                return pd.DataFrame()

            # Filter for this ticker if ticker column exists
            if ticker_col:
                df = df[df[ticker_col].str.upper() == ticker.upper()]

            records = []
            for _, row in df.iterrows():
                try:
                    dt = pd.Timestamp(row[date_col])
                    records.append({
                        'date': dt.strftime('%Y-%m-%d'),
                        'datetime': dt.isoformat(),
                        'headline': str(row[headline_col]),
                        'summary': '',
                        'source': 'kaggle_analyst_ratings',
                        'url': '',
                        'category': 'analyst',
                        'ticker': ticker,
                        'api_source': 'kaggle',
                    })
                except Exception:
                    pass

            result = pd.DataFrame(records)
            if not result.empty:
                result['date'] = pd.to_datetime(result['date'])
                result = result.drop_duplicates(subset=['headline', 'date'])

            print(f"      Kaggle: {len(result)} articles for {ticker}")
            return result

        except Exception as e:
            print(f"      Kaggle load error: {e}")
            return pd.DataFrame()

    # ==================================================================
    # COMBINED FETCH
    # ==================================================================

    def fetch_all(self, ticker: str, start_date: str,
                  end_date: str) -> pd.DataFrame:
        """
        Fetch news from ALL available sources and combine.

        Priority: Finnhub > Kaggle > NewsAPI
        Deduplicates by headline similarity.
        """
        print(f"\n  Fetching news for {ticker} ({start_date} to {end_date})")

        all_dfs = []

        # Source 1: Finnhub (primary)
        finnhub_df = self.fetch_finnhub(ticker, start_date, end_date)
        if not finnhub_df.empty:
            all_dfs.append(finnhub_df)

        # Source 2: Kaggle (historical backup)
        kaggle_df = self.load_kaggle_news(ticker)
        if not kaggle_df.empty:
            # Filter to date range
            kaggle_df = kaggle_df[
                (kaggle_df['date'] >= start_date) &
                (kaggle_df['date'] <= end_date)
            ]
            if not kaggle_df.empty:
                all_dfs.append(kaggle_df)

        # Source 3: NewsAPI (recent backup)
        newsapi_df = self.fetch_newsapi(ticker, start_date=start_date,
                                         end_date=end_date)
        if not newsapi_df.empty:
            all_dfs.append(newsapi_df)

        if not all_dfs:
            print(f"    [!] No news found from any source for {ticker}")
            return pd.DataFrame()

        # Combine all sources
        combined = pd.concat(all_dfs, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])

        # Remove duplicates (same headline on same day)
        combined = combined.drop_duplicates(
            subset=['headline', 'date']
        ).sort_values('date').reset_index(drop=True)

        # Remove empty headlines
        combined = combined[
            combined['headline'].str.strip().str.len() > 10
        ]

        print(f"\n    TOTAL: {len(combined)} unique articles for {ticker}")
        print(f"    Date range: {combined['date'].min().date()} to "
              f"{combined['date'].max().date()}")

        # Coverage report
        trading_days = pd.bdate_range(start_date, end_date)
        news_dates = combined['date'].dt.date.unique()
        coverage = len(set(news_dates) & set(trading_days.date)) / len(trading_days)
        print(f"    Coverage: {coverage:.1%} of trading days have news")

        # Source breakdown
        print(f"    Sources:")
        for source, count in combined['api_source'].value_counts().items():
            print(f"      {source}: {count} articles")

        return combined

    def save(self, news_df: pd.DataFrame, ticker: str):
        """Save fetched news to CSV."""
        if news_df.empty:
            print(f"    No news to save for {ticker}")
            return

        output_path = self.output_dir / f'{ticker}_news.csv'
        news_df.to_csv(output_path, index=False)
        print(f"    Saved to {output_path}")

        # Also save a summary
        summary = {
            'ticker': ticker,
            'total_articles': len(news_df),
            'date_range': [
                str(news_df['date'].min().date()),
                str(news_df['date'].max().date()),
            ],
            'sources': news_df['api_source'].value_counts().to_dict(),
            'articles_per_day': round(
                len(news_df) / max(news_df['date'].nunique(), 1), 1
            ),
            'fetched_at': datetime.now().isoformat(),
        }

        summary_path = self.output_dir / f'{ticker}_news_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def load(self, ticker: str) -> pd.DataFrame:
        """Load previously saved news for a ticker."""
        news_path = self.output_dir / f'{ticker}_news.csv'
        if not news_path.exists():
            raise FileNotFoundError(
                f"No saved news for {ticker}. "
                f"Run fetch_all() first."
            )
        df = pd.read_csv(news_path, parse_dates=['date'])
        return df


# ======================================================================
# CLI
# ======================================================================

def main():
    """Fetch news for all configured tickers."""
    print("=" * 60)
    print("  NEWS DATA FETCHER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    fetcher = NewsFetcher()

    # Check which API keys are available
    print("\n  API Key Status:")
    print(f"    Finnhub: {'✓ Set' if fetcher.finnhub_key else '✗ Not set (set FINNHUB_API_KEY)'}")
    print(f"    NewsAPI: {'✓ Set' if fetcher.newsapi_key else '✗ Not set (set NEWSAPI_KEY)'}")
    print(f"    Kaggle:  {'✓ Found' if Path('data/raw/raw_analyst_ratings.csv').exists() else '✗ Not found'}")

    ticker = config['data']['ticker']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']

    # Fetch
    news_df = fetcher.fetch_all(ticker, start_date, end_date)

    if not news_df.empty:
        fetcher.save(news_df, ticker)

        # Show sample headlines
        print(f"\n  Sample headlines:")
        for _, row in news_df.head(5).iterrows():
            headline = row['headline'][:80]
            print(f"    [{row['date'].strftime('%Y-%m-%d')}] {headline}")
    else:
        print("\n  [!] No news data retrieved.")
        print("      Set at least one API key:")
        print("      $env:FINNHUB_API_KEY = 'your_key'")
        print("      $env:NEWSAPI_KEY = 'your_key'")


if __name__ == '__main__':
    main()