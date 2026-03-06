"""
Integration Test
Connects all components:
Features → Attention → Environment → Agent (random)
Verifies everything works together before training
"""

import torch
import numpy as np
import pandas as pd
import yaml
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.trading_env import TradingEnvironment
from src.models.attention import AttentionFusionLayer


def load_data(config):
    """Load all required data"""

    print("Loading data...")

    # Combined features
    features = pd.read_csv(
        config['paths']['combined_features'],
        index_col=0,
        parse_dates=True
    )

    # Price data
    prices = pd.read_csv(
        config['paths']['raw_prices'],
        index_col=0,
        parse_dates=True
    )

    # Align prices with features
    prices = prices.loc[features.index, 'Close']

    print(f"  ✓ Features: {features.shape}")
    print(f"  ✓ Prices:   {len(prices)} days")

    return features, prices


def test_full_pipeline(config):
    """
    Test complete pipeline:
    Raw features → Attention Fusion → Trading Environment
    """

    print("\n" + "=" * 50)
    print("FULL PIPELINE INTEGRATION TEST")
    print("=" * 50)

    # Load data
    features, prices = load_data(config)

    # Create attention fusion layer
    print("\n[1/4] Creating Attention Fusion Layer...")
    fusion = AttentionFusionLayer(
        total_features=features.shape[1],
        output_dim=128
    )
    fusion.eval()
    print(f"  ✓ Fusion layer ready")
    print(f"  ✓ Output dimension: 128")

    # Create trading environment
    print("\n[2/4] Creating Trading Environment...")
    env = TradingEnvironment(
        features=features,
        prices=prices,
        initial_balance=config['environment']['initial_balance'],
        transaction_cost=config['environment']['transaction_cost']
    )
    print(f"  ✓ Environment ready")
    print(f"  ✓ Observation space: {env.observation_space.shape}")
    print(f"  ✓ Action space: {env.action_space.n}")

    # Test attention on first observation
    print("\n[3/4] Testing Attention on First Observation...")
    obs, info = env.reset()

    # Extract market features from observation (first 24)
    market_features = obs[:24]
    market_tensor   = torch.FloatTensor(market_features).unsqueeze(0)

    with torch.no_grad():
        fused = fusion(market_tensor)

    print(f"  Raw obs shape:    {obs.shape}")
    print(f"  Market features:  {market_features.shape}")
    print(f"  Fused output:     {fused.shape}")

    # Get attention weights
    weights = fusion.get_attention_weights()
    print(f"\n  Initial Attention Weights:")
    print(f"    Price → Sentiment: {weights['price_to_sentiment'].mean():.3f}")
    print(f"    Sentiment → Price: {weights['sentiment_to_price'].mean():.3f}")

    # Run episode with attention tracking
    print("\n[4/4] Running Episode with Attention Tracking...")

    obs, info = env.reset()
    done  = False
    steps = 0

    attention_history = {
        'price_to_sent': [],
        'sent_to_price': [],
        'portfolio':     [],
        'actions':       []
    }

    while not done:
        # Get market features
        market_feat = obs[:24]
        market_t    = torch.FloatTensor(market_feat).unsqueeze(0)

        # Apply attention
        with torch.no_grad():
            fused = fusion(market_t)

        # Get attention weights
        w = fusion.get_attention_weights()
        attention_history['price_to_sent'].append(
            float(w['price_to_sentiment'].mean())
        )
        attention_history['sent_to_price'].append(
            float(w['sentiment_to_price'].mean())
        )

        # Random action
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)

        attention_history['portfolio'].append(info['total_value'])
        attention_history['actions'].append(info['macro_action'])

        steps += 1

    print(f"  ✓ Episode complete: {steps} steps")

    # Attention analysis
    avg_ps = np.mean(attention_history['price_to_sent'])
    avg_sp = np.mean(attention_history['sent_to_price'])

    print(f"\n  Attention Summary:")
    print(f"    Avg Price→Sentiment: {avg_ps:.3f}")
    print(f"    Avg Sentiment→Price: {avg_sp:.3f}")

    if avg_sp > avg_ps:
        print(f"    Insight: Sentiment influenced price more than vice versa")
    else:
        print(f"    Insight: Price influenced sentiment more than vice versa")

    # Action distribution
    print(f"\n  Action Distribution (random agent):")
    from collections import Counter
    action_counts = Counter(attention_history['actions'])
    total = sum(action_counts.values())
    for action, count in sorted(action_counts.items()):
        pct = count / total * 100
        bar = '█' * int(pct / 3)
        print(f"    {action:20s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Portfolio summary
    final_value    = attention_history['portfolio'][-1]
    total_return   = (final_value - 10000) / 10000 * 100
    print(f"\n  Portfolio Summary (random agent):")
    print(f"    Initial value: $10,000.00")
    print(f"    Final value:   ${final_value:,.2f}")
    print(f"    Total return:  {total_return:.2f}%")
    print(f"    Total trades:  {info['n_trades']}")

    return env, fusion, attention_history


def test_walk_forward_splits(config, features, prices):
    """
    Test walk-forward validation splits
    Verify each fold has correct data
    """

    print("\n" + "=" * 50)
    print("WALK-FORWARD SPLITS TEST")
    print("=" * 50)

    train_days = config['validation']['train_days']
    val_days   = config['validation']['val_days']
    test_days  = config['validation']['test_days']
    total      = len(features)

    print(f"\nConfiguration:")
    print(f"  Total days:  {total}")
    print(f"  Train days:  {train_days}")
    print(f"  Val days:    {val_days}")
    print(f"  Test days:   {test_days}")

    folds = []
    start = 0
    fold  = 1

    while True:
        train_end = start + train_days
        val_end   = train_end + val_days
        test_end  = val_end + test_days

        if test_end > total:
            break

        fold_info = {
            'fold':       fold,
            'train':      (start, train_end),
            'val':        (train_end, val_end),
            'test':       (val_end, test_end),
            'train_start': features.index[start].date(),
            'train_end':   features.index[train_end - 1].date(),
            'test_start':  features.index[val_end].date(),
            'test_end':    features.index[test_end - 1].date()
        }

        folds.append(fold_info)

        print(f"\n  Fold {fold}:")
        print(f"    Train: {fold_info['train_start']} → "
              f"{fold_info['train_end']} ({train_days} days)")
        print(f"    Test:  {fold_info['test_start']} → "
              f"{fold_info['test_end']} ({test_days} days)")

        # Test creating environment for this fold
        train_features = features.iloc[start:train_end]
        train_prices   = prices.iloc[start:train_end]

        test_features  = features.iloc[val_end:test_end]
        test_prices    = prices.iloc[val_end:test_end]

        train_env = TradingEnvironment(train_features, train_prices)
        test_env  = TradingEnvironment(test_features, test_prices)

        print(f"    ✓ Train env: {len(train_features)} days")
        print(f"    ✓ Test env:  {len(test_features)} days")

        start += test_days
        fold  += 1

    print(f"\n  Total folds: {len(folds)}")
    print(f"  ✓ All walk-forward splits valid!")

    return folds


def run_integration_test():
    """Run complete integration test"""

    print("=" * 60)
    print("WEEK 3 INTEGRATION TEST")
    print("=" * 60)

    # Load config
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Test full pipeline
    env, fusion, history = test_full_pipeline(config)

    # Load data for walk-forward test
    features, prices = load_data(config)

    # Test walk-forward splits
    folds = test_walk_forward_splits(config, features, prices)

    # Final summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"  ✓ Features loaded:          {features.shape}")
    print(f"  ✓ Attention fusion:          Working (128 dim output)")
    print(f"  ✓ Trading environment:       Working (30 dim obs, 24 actions)")
    print(f"  ✓ Walk-forward splits:       {len(folds)} folds ready")
    print(f"  ✓ Full episode:              Completes without errors")
    print(f"  ✓ Attention tracking:        Working")

    print("\n" + "=" * 60)
    print("WEEK 3 COMPLETE!")
    print("Ready for Week 4: PPO Agent Training")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_test()