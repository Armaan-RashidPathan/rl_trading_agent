# src/data/data_pipeline.py
"""
Data Pipeline - Thin wrapper around existing modules
Provides unified interface for training
"""

import os
import numpy as np
import pandas as pd
import yaml


class DataPipeline:
    """
    Loads the already-processed combined features from feature_combiner.
    Provides the interface that the trainer expects.
    """

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path

    def load_data(self):
        """
        Load pre-processed combined features.
        
        Assumes you have already run:
            1. python -m src.data.data_loader
            2. python -m src.data.feature_engineering
            3. python -m src.data.sentiment_extractor
            4. python -m src.data.feature_combiner
        
        Returns:
            features:      np.array (n_days, n_features)
            prices:        np.array of close prices
            dates:         pd.DatetimeIndex
            feature_names: list of feature column names
        """
        # Paths from config
        combined_path = self.config['paths']['combined_features']
        prices_path = self.config['paths']['raw_prices']

        # Check files exist
        if not os.path.exists(combined_path):
            raise FileNotFoundError(
                f"Combined features not found at {combined_path}.\n"
                "Run the pipeline steps first:\n"
                "  1. python -m src.data.data_loader\n"
                "  2. python -m src.data.feature_engineering\n"
                "  3. python -m src.data.sentiment_extractor\n"
                "  4. python -m src.data.feature_combiner"
            )

        if not os.path.exists(prices_path):
            raise FileNotFoundError(
                f"Price data not found at {prices_path}.\n"
                "Run: python -m src.data.data_loader"
            )

        # Load combined features
        print("Loading combined features...")
        combined_df = pd.read_csv(
            combined_path, index_col=0, parse_dates=True
        )
        print(f"  ✓ Combined features: {combined_df.shape}")

        # Load price data
        print("Loading price data...")
        prices_df = pd.read_csv(
            prices_path, index_col=0, parse_dates=True
        )
        print(f"  ✓ Price data: {prices_df.shape}")

        # Align prices to combined feature dates
        common_dates = combined_df.index.intersection(prices_df.index)
        combined_df = combined_df.loc[common_dates]
        prices_df = prices_df.loc[common_dates]

        # Extract arrays
        feature_names = list(combined_df.columns)
        features = combined_df.values.astype(np.float32)
        prices = prices_df['Close'].values.astype(np.float32)
        dates = combined_df.index

        # Validate
        assert len(features) == len(prices) == len(dates)
        assert not np.isnan(features).any(), "NaN in features!"
        assert not np.isinf(features).any(), "Inf in features!"

        print(f"\n  Final dataset:")
        print(f"    Days:     {len(dates)}")
        print(f"    Features: {len(feature_names)}")
        print(f"    Date range: {dates[0].date()} to {dates[-1].date()}")

        # Show feature breakdown
        sentiment_cols = [
            'sentiment_score', 'sentiment_std',
            'positive_ratio', 'news_count', 'sentiment_momentum'
        ]
        n_sent = len([f for f in feature_names if f in sentiment_cols])
        n_tech = len(feature_names) - n_sent
        print(f"    Technical: {n_tech}")
        print(f"    Sentiment: {n_sent}")

        return features, prices, dates, feature_names

    def get_feature_split(self, feature_names):
        """
        Split feature names into price/technical vs sentiment.
        Useful for creating the SentimentPPOAgent.
        
        Returns:
            price_feature_names: list
            sentiment_feature_names: list
        """
        sentiment_cols = [
            'sentiment_score', 'sentiment_std',
            'positive_ratio', 'news_count', 'sentiment_momentum'
        ]

        sent_features = [f for f in feature_names if f in sentiment_cols]
        price_features = [f for f in feature_names if f not in sentiment_cols]

        return price_features, sent_features


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DATA PIPELINE")
    print("=" * 60)

    pipeline = DataPipeline()

    try:
        features, prices, dates, feature_names = pipeline.load_data()

        price_feats, sent_feats = pipeline.get_feature_split(feature_names)
        print(f"\n  Price features ({len(price_feats)}): {price_feats}")
        print(f"  Sentiment features ({len(sent_feats)}): {sent_feats}")

        print(f"\n  Sample (day 1): {dates[0].date()}")
        print(f"    Price: ${prices[0]:.2f}")
        print(f"    Features: {features[0][:5]}...")

        print("\n" + "=" * 60)
        print("DATA PIPELINE TEST PASSED!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        print("\n  Run the pipeline steps first!")