# src/data/real_sentiment.py

"""
Real Sentiment Extraction Pipeline

Takes actual news headlines and produces daily sentiment features
using FinBERT. Replaces the synthetic sentiment generation.

Flow:
  1. Load real news (from news_fetcher.py output)
  2. Score each headline with FinBERT
  3. Aggregate to daily sentiment features
  4. Align to trading days
  5. Save as sentiment_scores.csv
"""

import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional


class RealSentimentExtractor:
    """
    Extract sentiment from real news headlines using FinBERT.
    """

    def __init__(self, config_path: str = "configs/config.yaml",
                 use_finbert: bool = True):
        """
        Args:
            config_path: Path to config
            use_finbert:  True = FinBERT (accurate), False = VADER (fast)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.use_finbert = use_finbert
        self.model = None
        self.tokenizer = None

    def _load_finbert(self):
        """Load FinBERT model (lazy loading — only when needed)."""
        if self.model is not None:
            return

        print("    Loading FinBERT model ...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = self.config.get('sentiment', {}).get(
            'model_name', 'ProsusAI/finbert'
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        print(f"    FinBERT loaded on {self.device}")

    def _load_vader(self):
        """Load VADER sentiment analyzer."""
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)

        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def score_headline_finbert(self, headline: str) -> dict:
        """
        Score a single headline with FinBERT.

        Returns:
            {'positive': float, 'negative': float, 'neutral': float,
             'score': float}  # score = positive - negative
        """
        self._load_finbert()

        inputs = self.tokenizer(
            headline,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        # FinBERT labels: positive, negative, neutral
        result = {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2]),
            'score': float(probs[0] - probs[1]),  # net sentiment
        }

        return result

    def score_headline_vader(self, headline: str) -> dict:
        """Score a single headline with VADER."""
        if not hasattr(self, 'vader'):
            self._load_vader()

        scores = self.vader.polarity_scores(headline)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'score': scores['compound'],
        }

    def score_headlines_batch(self, headlines: list,
                               batch_size: int = 16) -> list:
        """
        Score a batch of headlines efficiently.

        Args:
            headlines:  List of headline strings
            batch_size: Batch size for FinBERT

        Returns:
            List of score dicts
        """
        if self.use_finbert:
            return self._score_batch_finbert(headlines, batch_size)
        else:
            return [self.score_headline_vader(h) for h in headlines]

    def _score_batch_finbert(self, headlines: list,
                              batch_size: int = 16) -> list:
        """Batch scoring with FinBERT for efficiency."""
        self._load_finbert()

        all_scores = []

        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            for j in range(len(batch)):
                all_scores.append({
                    'positive': float(probs[j][0]),
                    'negative': float(probs[j][1]),
                    'neutral': float(probs[j][2]),
                    'score': float(probs[j][0] - probs[j][1]),
                })

            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"      Scored {min(i + batch_size, len(headlines))}"
                      f"/{len(headlines)} headlines")

        return all_scores

    def process_news_to_daily_features(
        self,
        news_df: pd.DataFrame,
        trading_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Process raw news into daily sentiment features aligned
        to trading days.

        This is the main method. It:
          1. Scores each headline
          2. Groups by date
          3. Aggregates to daily features
          4. Aligns to trading calendar
          5. Forward-fills gaps (weekends, holidays)

        Args:
            news_df:       DataFrame with 'date' and 'headline' columns
            trading_dates: DatetimeIndex of trading days

        Returns:
            DataFrame with 5 sentiment features per trading day:
              sentiment_score, sentiment_std, sentiment_momentum,
              news_volume, sentiment_ma5
        """
        print(f"\n    Processing {len(news_df)} headlines into daily features ...")

        # Score all headlines
        headlines = news_df['headline'].tolist()
        scores = self.score_headlines_batch(headlines)

        # Add scores to dataframe
        news_scored = news_df.copy()
        news_scored['sentiment_score'] = [s['score'] for s in scores]
        news_scored['positive'] = [s['positive'] for s in scores]
        news_scored['negative'] = [s['negative'] for s in scores]

        # Ensure date is datetime
        news_scored['date'] = pd.to_datetime(news_scored['date'])

        # IMPORTANT: Use previous day's news to avoid look-ahead bias
        # News published on day T is used for trading on day T+1
        news_scored['trading_date'] = news_scored['date'] + pd.Timedelta(days=1)

        # Aggregate to daily features
        daily = news_scored.groupby('trading_date').agg(
            sentiment_score=('sentiment_score', 'mean'),
            sentiment_std=('sentiment_score', 'std'),
            news_count=('sentiment_score', 'count'),
            positive_pct=('positive', 'mean'),
            negative_pct=('negative', 'mean'),
        ).fillna(0)

        # Compute additional features
        daily['sentiment_momentum'] = daily['sentiment_score'].diff(5)
        avg_count = daily['news_count'].mean()
        daily['news_volume'] = daily['news_count'] / max(avg_count, 1)
        daily['sentiment_ma5'] = daily['sentiment_score'].rolling(5).mean()

        # Select final 5 features (matches your existing feature structure)
        feature_cols = [
            'sentiment_score',
            'sentiment_std',
            'sentiment_momentum',
            'news_volume',
            'sentiment_ma5',
        ]

        # Ensure all columns exist
        for col in feature_cols:
            if col not in daily.columns:
                daily[col] = 0.0

        daily_features = daily[feature_cols]

        # Align to trading calendar
        aligned = daily_features.reindex(trading_dates)

        # Forward-fill gaps (weekends, holidays with no news)
        aligned = aligned.ffill()

        # Fill any remaining NaN at the start
        aligned = aligned.fillna(0.0)

        print(f"    Daily features computed:")
        print(f"      Trading days: {len(aligned)}")
        print(f"      Days with news: {daily_features.index.isin(trading_dates).sum()}")
        print(f"      Mean sentiment: {aligned['sentiment_score'].mean():.4f}")
        print(f"      Sentiment std:  {aligned['sentiment_score'].std():.4f}")
        print(f"      Mean news/day:  {daily['news_count'].mean():.1f}")

        return aligned

    def run(self, ticker: str = None) -> pd.DataFrame:
        """
        Full pipeline: load news -> score -> daily features -> save.

        Args:
            ticker: Stock ticker (default from config)

        Returns:
            DataFrame with daily sentiment features
        """
        if ticker is None:
            ticker = self.config['data']['ticker']

        print(f"\n  Running real sentiment extraction for {ticker}")

        # Load news
        news_path = Path(f'data/raw/news/{ticker}_news.csv')
        if not news_path.exists():
            raise FileNotFoundError(
                f"No news file found at {news_path}.\n"
                f"Run the news fetcher first:\n"
                f"  python -m src.data.news_fetcher"
            )

        news_df = pd.read_csv(news_path, parse_dates=['date'])
        print(f"    Loaded {len(news_df)} articles")

        # Load trading dates from price data
        prices_path = self.config['paths']['raw_prices']
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        trading_dates = prices.index

        # Process
        daily_features = self.process_news_to_daily_features(
            news_df, trading_dates
        )

        # Save
        output_path = Path(self.config['paths']['sentiment_scores'])
        daily_features.to_csv(output_path)
        print(f"    Saved to {output_path}")

        return daily_features


# ======================================================================
# CLI
# ======================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract sentiment from real news data'
    )
    parser.add_argument(
        '--ticker', type=str, default=None,
        help='Stock ticker (default: from config)',
    )
    parser.add_argument(
        '--vader', action='store_true',
        help='Use VADER instead of FinBERT (faster)',
    )
    args = parser.parse_args()

    extractor = RealSentimentExtractor(use_finbert=not args.vader)
    extractor.run(ticker=args.ticker)


if __name__ == '__main__':
    main()