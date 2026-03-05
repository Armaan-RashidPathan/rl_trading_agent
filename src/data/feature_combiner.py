"""
Feature Combiner Module
Combines technical and sentiment features into final matrix
"""

import os
import pandas as pd
import numpy as np
import yaml


class FeatureCombiner:
    """
    Combines all features into final matrix for the RL agent
    """

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config      = config
        self.save_path   = config['paths']['combined_features']
        self.tech_path   = config['paths']['technical_features']
        self.sent_path   = config['paths']['sentiment_scores']

    def load_features(self):
        """Load technical and sentiment features"""

        print("Loading feature files...")

        # Load technical features
        if not os.path.exists(self.tech_path):
            raise FileNotFoundError(
                f"Technical features not found at {self.tech_path}. "
                "Run feature_engineering.py first."
            )

        technical = pd.read_csv(
            self.tech_path,
            index_col=0,
            parse_dates=True
        )
        print(f"  ✓ Technical features: {technical.shape}")

        # Load sentiment features
        if not os.path.exists(self.sent_path):
            raise FileNotFoundError(
                f"Sentiment features not found at {self.sent_path}. "
                "Run sentiment_extractor.py first."
            )

        sentiment = pd.read_csv(
            self.sent_path,
            index_col=0,
            parse_dates=True
        )
        print(f"  ✓ Sentiment features: {sentiment.shape}")

        return technical, sentiment

    def combine(self, technical, sentiment):
        """
        Combine technical and sentiment features

        Args:
            technical: Technical features DataFrame
            sentiment: Sentiment features DataFrame

        Returns:
            Combined DataFrame
        """
        print("\nCombining features...")

        # Find common dates
        common_dates = technical.index.intersection(sentiment.index)
        print(f"  Technical dates:  {len(technical)}")
        print(f"  Sentiment dates:  {len(sentiment)}")
        print(f"  Common dates:     {len(common_dates)}")

        # Filter to common dates
        technical = technical.loc[common_dates]
        sentiment = sentiment.loc[common_dates]

        # Combine
        combined = pd.concat([technical, sentiment], axis=1)

        # Remove any remaining NaN
        before = len(combined)
        combined = combined.dropna()
        after  = len(combined)

        if before - after > 0:
            print(f"  Removed {before - after} rows with NaN")

        print(f"  Final shape: {combined.shape}")

        return combined

    def validate(self, combined):
        """
        Validate the combined feature matrix

        Checks:
        1. No NaN values
        2. No infinite values
        3. Reasonable value ranges
        4. Correct feature count
        """
        print("\nValidating combined features...")
        issues = []

        # Check NaN
        nan_count = combined.isnull().sum().sum()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values")
        else:
            print("  ✓ No NaN values")

        # Check infinite values
        inf_count = np.isinf(combined.values).sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values")
        else:
            print("  ✓ No infinite values")

        # Check feature count
        expected_features = 24  # 19 technical + 5 sentiment
        if len(combined.columns) < expected_features:
            issues.append(
                f"Expected {expected_features}+ features, "
                f"got {len(combined.columns)}"
            )
        else:
            print(f"  ✓ Feature count: {len(combined.columns)}")

        # Check date order
        if not combined.index.is_monotonic_increasing:
            issues.append("Dates are not in order")
        else:
            print("  ✓ Dates in correct order")

        # Check minimum rows
        if len(combined) < 500:
            issues.append(f"Too few rows: {len(combined)}")
        else:
            print(f"  ✓ Sufficient data: {len(combined)} rows")

        if issues:
            print("\n  ⚠ Issues found:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("\n  ✓ All validation checks passed!")

        return len(issues) == 0

    def save(self, combined):
        """Save combined features"""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        combined.to_csv(self.save_path)
        print(f"\nSaved to: {self.save_path}")

    def load(self):
        """Load existing combined features"""
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(
                f"No combined features at {self.save_path}. "
                "Run combine() first."
            )
        df = pd.read_csv(self.save_path, index_col=0, parse_dates=True)
        print(f"Loaded combined features: {df.shape}")
        return df

    def print_summary(self, combined):
        """Print detailed summary"""

        print("\n" + "=" * 50)
        print("COMBINED FEATURES SUMMARY")
        print("=" * 50)
        print(f"Total rows:     {len(combined)}")
        print(f"Total features: {len(combined.columns)}")
        print(f"Date range:     {combined.index.min().date()} "
              f"to {combined.index.max().date()}")

        # Technical features
        tech_cols = [
            'returns', 'volatility_20d', 'rsi', 'macd',
            'macd_signal', 'macd_histogram', 'price_to_sma20',
            'price_to_sma50', 'sma_crossover', 'volume_ratio',
            'bb_position', 'atr_ratio'
        ]

        # Regime features
        regime_cols = [
            'vol_regime', 'trend_strength', 'trend_direction',
            'regime_sideways', 'regime_bear',
            'regime_bull', 'regime_highvol'
        ]

        # Sentiment features
        sent_cols = [
            'sentiment_score', 'sentiment_std',
            'positive_ratio', 'news_count',
            'sentiment_momentum'
        ]

        print("\n  Technical Features:")
        for col in combined.columns:
            if col in tech_cols:
                min_v = combined[col].min()
                max_v = combined[col].max()
                print(f"    ✓ {col:25s} [{min_v:+.3f}, {max_v:+.3f}]")

        print("\n  Regime Features:")
        for col in combined.columns:
            if col in regime_cols:
                min_v = combined[col].min()
                max_v = combined[col].max()
                print(f"    ✓ {col:25s} [{min_v:+.3f}, {max_v:+.3f}]")

        print("\n  Sentiment Features:")
        for col in combined.columns:
            if col in sent_cols:
                min_v = combined[col].min()
                max_v = combined[col].max()
                print(f"    ✓ {col:25s} [{min_v:+.3f}, {max_v:+.3f}]")

        print("=" * 50)

    def print_walk_forward_info(self, combined):
        """Show walk-forward validation split info"""

        val_config   = self.config.get('validation', {})
        train_days   = val_config.get('train_days', 504)
        val_days     = val_config.get('val_days', 126)
        test_days    = val_config.get('test_days', 126)
        total        = len(combined)
        window       = train_days + val_days + test_days
        n_folds      = (total - train_days - val_days) // test_days

        print("\n" + "=" * 50)
        print("WALK-FORWARD VALIDATION INFO")
        print("=" * 50)
        print(f"Total trading days:  {total}")
        print(f"Train window:        {train_days} days (~{train_days/252:.1f} years)")
        print(f"Validation window:   {val_days} days (~{val_days/252:.1f} years)")
        print(f"Test window:         {test_days} days (~{test_days/252:.1f} years)")
        print(f"Expected folds:      {n_folds}")

        print("\nFold Structure:")
        start = 0
        for fold in range(n_folds):
            train_end = start + train_days
            val_end   = train_end + val_days
            test_end  = val_end + test_days

            if test_end > total:
                break

            train_start_date = combined.index[start].date()
            test_end_date    = combined.index[min(test_end-1, total-1)].date()

            print(f"  Fold {fold+1}: {train_start_date} → {test_end_date}")
            start += test_days

        print("=" * 50)


def run_feature_combination(config_path="configs/config.yaml"):
    """Run complete feature combination pipeline"""

    print("=" * 50)
    print("FEATURE COMBINATION PIPELINE")
    print("=" * 50)

    combiner = FeatureCombiner(config_path)

    # Load features
    technical, sentiment = combiner.load_features()

    # Combine
    combined = combiner.combine(technical, sentiment)

    # Validate
    is_valid = combiner.validate(combined)

    if not is_valid:
        print("\n⚠ Warning: Validation failed. Check issues above.")
    else:
        print("\n✓ Features are ready for training!")

    # Save
    combiner.save(combined)

    # Print summary
    combiner.print_summary(combined)

    # Print walk-forward info
    combiner.print_walk_forward_info(combined)

    print("\n" + "=" * 50)
    print("FEATURE COMBINATION COMPLETE!")
    print("Week 2 Done! Ready for Week 3: Trading Environment")
    print("=" * 50)

    return combined


if __name__ == "__main__":
    run_feature_combination()