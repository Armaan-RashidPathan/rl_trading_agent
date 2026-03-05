"""
Sentiment Extraction Module
Supports both VADER (fast) and FinBERT (accurate)
"""

import os
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class VADERSentimentExtractor:
    """
    Fast rule-based sentiment using VADER
    Good for testing pipeline quickly
    Runs in seconds on CPU
    """

    def __init__(self):
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            self.analyzer = SentimentIntensityAnalyzer()

            # Add financial words to VADER lexicon
            self._add_financial_words()
            print("  VADER loaded successfully")

        except ImportError:
            raise ImportError("Run: pip install nltk")

    def _add_financial_words(self):
        """Add finance-specific words to VADER"""
        financial_lexicon = {
            # Positive financial terms
            'bullish': 3.0,
            'outperform': 2.5,
            'upgrade': 2.5,
            'beat': 2.0,
            'exceeded': 2.0,
            'record': 1.5,
            'growth': 1.5,
            'profit': 1.5,
            'surge': 2.0,
            'rally': 2.0,
            'gain': 1.5,
            'buy': 1.5,
            'strong': 1.5,

            # Negative financial terms
            'bearish': -3.0,
            'underperform': -2.5,
            'downgrade': -2.5,
            'miss': -2.0,
            'missed': -2.0,
            'below': -1.5,
            'loss': -1.5,
            'decline': -1.5,
            'drop': -1.5,
            'fall': -1.5,
            'weak': -1.5,
            'sell': -1.5,
            'lawsuit': -2.0,
            'investigation': -2.0,
            'recall': -2.0,
            'warning': -1.5,
        }

        self.analyzer.lexicon.update(financial_lexicon)

    def get_sentiment(self, text):
        """
        Get sentiment score for a single text

        Returns:
            float: Score between -1 (negative) and +1 (positive)
        """
        if not text or len(str(text)) < 3:
            return 0.0

        scores = self.analyzer.polarity_scores(str(text))
        return scores['compound']  # Already in [-1, 1]

    def get_batch_sentiment(self, texts):
        """Get sentiment for list of texts"""
        return [self.get_sentiment(text) for text in texts]


class FinBERTSentimentExtractor:
    """
    Accurate financial sentiment using FinBERT
    More accurate but slower on CPU
    Recommended for final results
    """

    def __init__(self):
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification
            )

            self.device = 'cpu'
            print(f"  Using device: {self.device}")
            print("  Loading FinBERT model (first time downloads ~500MB)...")

            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            self.model.to(self.device)
            self.model.eval()

            self.torch = torch
            print("  FinBERT loaded successfully")

        except ImportError:
            raise ImportError("Run: pip install transformers torch")

    def get_sentiment(self, text):
        """
        Get sentiment score for a single text

        Returns:
            float: Score between -1 (negative) and +1 (positive)
        """
        if not text or len(str(text)) < 3:
            return 0.0

        inputs = self.tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            probs = self.torch.softmax(outputs.logits, dim=1)

        # FinBERT: [positive, negative, neutral]
        pos = probs[0][0].item()
        neg = probs[0][1].item()

        return pos - neg  # Range [-1, 1]

    def get_batch_sentiment(self, texts, batch_size=8):
        """Get sentiment for list of texts in batches"""
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []

            for text in batch:
                score = self.get_sentiment(text)
                batch_results.append(score)

            results.extend(batch_results)

        return results


class SentimentFeatureCreator:
    """
    Creates daily sentiment features from news headlines
    Works with either VADER or FinBERT
    """

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.save_path = config['paths']['sentiment_scores']
        self.aggregation = config.get('sentiment', {}).get('aggregation', 'mean')

    def create_features(self, aligned_news, extractor):
        """
        Create daily sentiment features

        Args:
            aligned_news: DataFrame with headlines per trading day
            extractor: VADER or FinBERT extractor

        Returns:
            DataFrame with daily sentiment features
        """
        print(f"\nExtracting sentiment from {len(aligned_news)} trading days...")

        daily_sentiment = []

        for date, row in tqdm(aligned_news.iterrows(), total=len(aligned_news)):
            headlines = row['headlines']

            if not headlines or len(headlines) == 0:
                # No news - use neutral
                daily_sentiment.append({
                    'date': date,
                    'sentiment_score': 0.0,
                    'sentiment_std': 0.0,
                    'positive_ratio': 0.5,
                    'news_count': 0
                })
                continue

            # Get scores for all headlines
            scores = extractor.get_batch_sentiment(headlines)

            # Aggregate
            agg_score = np.mean(scores)

            # Count positive headlines
            pos_count = sum(1 for s in scores if s > 0.1)
            pos_ratio = pos_count / len(scores)

            daily_sentiment.append({
                'date': date,
                'sentiment_score': agg_score,
                'sentiment_std': np.std(scores) if len(scores) > 1 else 0.0,
                'positive_ratio': pos_ratio,
                'news_count': len(headlines)
            })

        # Create DataFrame
        sentiment_df = pd.DataFrame(daily_sentiment)
        sentiment_df.set_index('date', inplace=True)

        # Add sentiment momentum (5-day change)
        sentiment_df['sentiment_momentum'] = (
            sentiment_df['sentiment_score'].diff(5)
        )

        # Forward fill missing values
        sentiment_df['sentiment_score'] = (
            sentiment_df['sentiment_score'].ffill()
        )
        sentiment_df['sentiment_momentum'] = (
            sentiment_df['sentiment_momentum'].fillna(0)
        )

        return sentiment_df

    def save(self, sentiment_df):
        """Save sentiment features to CSV"""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        sentiment_df.to_csv(self.save_path)
        print(f"Saved to: {self.save_path}")

    def load(self):
        """Load existing sentiment features"""
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(
                f"No sentiment data at {self.save_path}. "
                "Run create_features() first."
            )
        df = pd.read_csv(self.save_path, index_col=0, parse_dates=True)
        print(f"Loaded sentiment features: {df.shape}")
        return df


def print_sentiment_summary(sentiment_df):
    """Print summary of sentiment features"""

    print("\n" + "=" * 50)
    print("SENTIMENT FEATURES SUMMARY")
    print("=" * 50)
    print(f"Total rows:  {len(sentiment_df)}")
    print(f"Date range:  {sentiment_df.index.min().date()} to {sentiment_df.index.max().date()}")

    print("\nFeatures Created:")
    for col in sentiment_df.columns:
        print(f"  ✓ {col}")

    print("\nSentiment Statistics:")
    print(f"  Mean score:  {sentiment_df['sentiment_score'].mean():.3f}")
    print(f"  Std score:   {sentiment_df['sentiment_score'].std():.3f}")
    print(f"  Min score:   {sentiment_df['sentiment_score'].min():.3f}")
    print(f"  Max score:   {sentiment_df['sentiment_score'].max():.3f}")

    # Distribution
    scores = sentiment_df['sentiment_score']
    positive = (scores > 0.1).sum()
    negative = (scores < -0.1).sum()
    neutral  = ((scores >= -0.1) & (scores <= 0.1)).sum()
    total    = len(scores)

    print(f"\nSentiment Distribution:")
    print(f"  Positive: {positive} days ({positive/total*100:.1f}%)")
    print(f"  Neutral:  {neutral}  days ({neutral/total*100:.1f}%)")
    print(f"  Negative: {negative} days ({negative/total*100:.1f}%)")

    print("=" * 50)


def run_sentiment_extraction(use_finbert=False):
    """
    Run complete sentiment extraction pipeline

    Args:
        use_finbert: If True use FinBERT, else use VADER
    """
    from src.data.data_loader import (
        PriceDataLoader,
        NewsDataLoader,
        NewsAligner
    )

    print("=" * 50)
    print("SENTIMENT EXTRACTION PIPELINE")
    print("=" * 50)

    # Load price data
    print("\n[1/4] Loading price data...")
    price_loader = PriceDataLoader()
    prices = price_loader.load()

    # Load news data
    print("\n[2/4] Loading news data...")
    news_loader = NewsDataLoader()
    news = news_loader.load()

    # Align news to trading days
    print("\n[3/4] Aligning news to trading days...")
    aligner = NewsAligner()
    aligned = aligner.align(prices, news)

    # Extract sentiment
    print("\n[4/4] Extracting sentiment...")

    creator = SentimentFeatureCreator()

    if use_finbert:
        print("  Using FinBERT (accurate but slow on CPU)")
        print("  Estimated time: 45-60 minutes on CPU")
        print("  TIP: Let it run and come back later!")
        extractor = FinBERTSentimentExtractor()
    else:
        print("  Using VADER (fast, good for testing)")
        extractor = VADERSentimentExtractor()

    # Create features
    sentiment_df = creator.create_features(aligned, extractor)

    # Save
    creator.save(sentiment_df)

    # Summary
    print_sentiment_summary(sentiment_df)

    print("\n" + "=" * 50)
    print("SENTIMENT EXTRACTION COMPLETE!")
    print("=" * 50)

    return sentiment_df


if __name__ == "__main__":
    import sys

    # Check if user passed --finbert flag
    use_finbert = '--finbert' in sys.argv

    if use_finbert:
        print("Running with FinBERT (slow but accurate)...")
    else:
        print("Running with VADER (fast, for testing)...")
        print("To use FinBERT: python -m src.data.sentiment_extractor --finbert")

    sentiment = run_sentiment_extraction(use_finbert=use_finbert)