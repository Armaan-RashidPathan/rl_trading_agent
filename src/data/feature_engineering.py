"""
Feature Engineering Module
Creates technical indicators and regime features from price data
"""

import os
import pandas as pd
import numpy as np
import yaml


class TechnicalFeatureEngineer:
    """
    Creates technical analysis features from price data
    """

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        tech = config.get('technical', {})

        # Parameters from config
        self.rsi_period      = tech.get('rsi_period', 14)
        self.macd_fast       = tech.get('macd_fast', 12)
        self.macd_slow       = tech.get('macd_slow', 26)
        self.macd_signal     = tech.get('macd_signal', 9)
        self.sma_short       = tech.get('sma_short', 20)
        self.sma_long        = tech.get('sma_long', 50)
        self.vol_short       = tech.get('volatility_short', 20)
        self.vol_long        = tech.get('volatility_long', 60)
        self.bb_period       = tech.get('bb_period', 20)
        self.atr_period      = tech.get('atr_period', 14)

    def create_features(self, df):
        """
        Create all technical features

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with technical features
        """
        features = pd.DataFrame(index=df.index)

        close  = df['Close']
        high   = df['High']
        low    = df['Low']
        volume = df['Volume']

        print("  Creating technical features...")

        # 1. Daily Returns
        features['returns'] = close.pct_change()
        print("    ✓ Returns")

        # 2. Volatility (20-day rolling std)
        features['volatility_20d'] = features['returns'].rolling(self.vol_short).std()
        print("    ✓ Volatility 20d")

        # 3. RSI
        features['rsi'] = self._calculate_rsi(close, self.rsi_period)
        print("    ✓ RSI")

        # 4-6. MACD
        macd, signal, histogram = self._calculate_macd(close)
        features['macd']           = macd
        features['macd_signal']    = signal
        features['macd_histogram'] = histogram
        print("    ✓ MACD")

        # 7-8. SMA Ratios
        sma_short = close.rolling(self.sma_short).mean()
        sma_long  = close.rolling(self.sma_long).mean()
        features['price_to_sma20'] = (close - sma_short) / sma_short
        features['price_to_sma50'] = (close - sma_long)  / sma_long
        print("    ✓ SMA Ratios")

        # 9. SMA Crossover
        features['sma_crossover'] = (sma_short - sma_long) / sma_long
        print("    ✓ SMA Crossover")

        # 10. Volume Ratio
        vol_sma = volume.rolling(20).mean()
        features['volume_ratio'] = volume / vol_sma
        print("    ✓ Volume Ratio")

        # 11. Bollinger Band Position
        bb_mid = close.rolling(self.bb_period).mean()
        bb_std = close.rolling(self.bb_period).std()
        features['bb_position'] = (close - bb_mid) / (2 * bb_std + 1e-8)
        print("    ✓ Bollinger Band Position")

        # 12. ATR Ratio
        features['atr_ratio'] = self._calculate_atr(
            high, low, close, self.atr_period
        ) / close
        print("    ✓ ATR Ratio")

        return features

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI normalized to [-1, 1]"""
        delta = prices.diff()
        gain  = delta.where(delta > 0, 0).rolling(period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs  = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        # Normalize: 0-100 → -1 to 1
        return (rsi - 50) / 50

    def _calculate_macd(self, prices):
        """Calculate MACD, Signal, and Histogram"""
        ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()

        macd      = (ema_fast - ema_slow) / prices
        signal    = macd.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd - signal

        return macd, signal, histogram

    def _calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low  - close.shift()).abs()

        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr


class RegimeFeatureEngineer:
    """
    Creates market regime features
    These help single PPO agent approximate Hierarchical RL behavior
    """

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        regime = config.get('regime', {})

        self.vol_threshold   = regime.get('volatility_threshold', 1.5)
        self.trend_threshold = regime.get('trend_threshold', 0.3)
        self.lookback        = regime.get('lookback', 20)

    def create_features(self, df, technical_features):
        """
        Create regime features

        Args:
            df: Original price DataFrame
            technical_features: Technical features DataFrame

        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=df.index)

        close   = df['Close']
        returns = technical_features['returns']
        vol_20  = technical_features['volatility_20d']

        print("  Creating regime features...")

        # 1. Volatility Regime
        vol_60 = returns.rolling(60).std()
        features['vol_regime'] = (vol_20 / (vol_60 + 1e-8)).clip(0, 3)
        print("    ✓ Volatility Regime")

        # 2. Trend Strength
        momentum = close / close.shift(self.lookback) - 1
        features['trend_strength'] = (momentum / 0.1).clip(-1, 1)
        print("    ✓ Trend Strength")

        # 3. Trend Direction
        sma_5  = close.rolling(5).mean()
        sma_20 = close.rolling(20).mean()
        features['trend_direction'] = np.sign(sma_5 - sma_20)
        print("    ✓ Trend Direction")

        # 4. Detect Regime Label
        regime_label = self._detect_regime(
            features['vol_regime'],
            features['trend_strength']
        )

        # 5-8. One-Hot Encode Regime
        features['regime_sideways'] = (regime_label == 0).astype(float)
        features['regime_bear']     = (regime_label == 1).astype(float)
        features['regime_bull']     = (regime_label == 2).astype(float)
        features['regime_highvol']  = (regime_label == 3).astype(float)
        print("    ✓ Regime One-Hot Encoding")

        return features

    def _detect_regime(self, vol_regime, trend_strength):
        """
        Detect market regime

        Labels:
            0: Sideways
            1: Bear
            2: Bull
            3: High Volatility
        """
        regime = pd.Series(0, index=vol_regime.index)

        # High volatility overrides everything
        regime[vol_regime > self.vol_threshold] = 3

        # Trend regimes (only if not high vol)
        bull_mask = (
            (trend_strength > self.trend_threshold) &
            (vol_regime <= self.vol_threshold)
        )
        bear_mask = (
            (trend_strength < -self.trend_threshold) &
            (vol_regime <= self.vol_threshold)
        )

        regime[bull_mask] = 2
        regime[bear_mask] = 1

        return regime


class FeatureNormalizer:
    """
    Normalize features using rolling z-score
    Prevents look-ahead bias in normalization
    """

    def __init__(self, lookback=252):
        self.lookback = lookback

    def normalize(self, df, skip_cols=None):
        """
        Normalize all features using rolling z-score

        Args:
            df: Feature DataFrame
            skip_cols: Columns to skip normalization

        Returns:
            Normalized DataFrame
        """
        skip_cols = skip_cols or []
        normalized = df.copy()

        for col in df.columns:
            if col in skip_cols:
                continue

            rolling_mean = df[col].rolling(
                self.lookback, min_periods=60
            ).mean()

            rolling_std = df[col].rolling(
                self.lookback, min_periods=60
            ).std()

            normalized[col] = (
                (df[col] - rolling_mean) / (rolling_std + 1e-8)
            ).clip(-3, 3)

        return normalized


def create_all_technical_features(config_path="configs/config.yaml"):
    """
    Main function to create all technical and regime features

    Returns:
        Combined feature DataFrame
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load price data
    print("Loading price data...")
    prices = pd.read_csv(
        config['paths']['raw_prices'],
        index_col=0,
        parse_dates=True
    )
    print(f"  Loaded {len(prices)} rows")

    print("\nCreating features...")

    # Technical features
    tech_engineer = TechnicalFeatureEngineer(config_path)
    technical = tech_engineer.create_features(prices)

    # Regime features
    regime_engineer = RegimeFeatureEngineer(config_path)
    regime = regime_engineer.create_features(prices, technical)

    # Combine
    all_features = pd.concat([technical, regime], axis=1)

    # Remove warmup rows
    initial_len = len(all_features)
    all_features = all_features.dropna()
    removed = initial_len - len(all_features)
    print(f"\n  Removed {removed} warmup rows (NaN)")

    # Save
    save_path = config['paths']['technical_features']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_features.to_csv(save_path)

    return all_features, prices


def print_feature_summary(features):
    """Print summary of created features"""

    print("\n" + "=" * 50)
    print("TECHNICAL FEATURES SUMMARY")
    print("=" * 50)
    print(f"Total rows:     {len(features)}")
    print(f"Total features: {len(features.columns)}")
    print(f"Date range:     {features.index.min().date()} to {features.index.max().date()}")

    print("\nFeature List:")
    technical_cols = [
        'returns', 'volatility_20d', 'rsi', 'macd',
        'macd_signal', 'macd_histogram', 'price_to_sma20',
        'price_to_sma50', 'sma_crossover', 'volume_ratio',
        'bb_position', 'atr_ratio'
    ]
    regime_cols = [
        'vol_regime', 'trend_strength', 'trend_direction',
        'regime_sideways', 'regime_bear', 'regime_bull', 'regime_highvol'
    ]

    print("\n  Technical Indicators:")
    for col in features.columns:
        if col in technical_cols:
            print(f"    ✓ {col}")

    print("\n  Regime Features:")
    for col in features.columns:
        if col in regime_cols:
            print(f"    ✓ {col}")

    print("\nFeature Statistics:")
    print(features.describe().round(3))

    # Check for remaining NaN
    nan_count = features.isnull().sum().sum()
    if nan_count == 0:
        print("\n✓ No NaN values in features")
    else:
        print(f"\n⚠ Warning: {nan_count} NaN values found")

    print("=" * 50)


def test_feature_engineering():
    """Test the feature engineering pipeline"""

    print("=" * 50)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 50)

    # Create features
    features, prices = create_all_technical_features()

    # Print summary
    print_feature_summary(features)

    # Show sample
    print("\nFirst 3 rows:")
    print(features.head(3).to_string())

    print("\nLast 3 rows:")
    print(features.tail(3).to_string())

    # Quick regime analysis
    print("\nRegime Distribution:")
    regime_counts = {
        'Sideways':       features['regime_sideways'].sum(),
        'Bear':           features['regime_bear'].sum(),
        'Bull':           features['regime_bull'].sum(),
        'High Volatility':features['regime_highvol'].sum()
    }
    total = len(features)
    for regime, count in regime_counts.items():
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {regime:20s}: {count:4.0f} days ({pct:5.1f}%) {bar}")

    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING COMPLETE!")
    print(f"Saved to: data/processed/technical_features.csv")
    print("Next Step: Sentiment Extraction")
    print("=" * 50)

    return features


if __name__ == "__main__":
    test_feature_engineering()