# src/training/trainer.py
"""
Training Pipeline for PPO Trading Agent
Uses Stable-Baselines3 PPO with custom attention feature extractor.
Supports both sentiment-aware and baseline (price-only) agents.
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
import json
import time
from pathlib import Path
from collections import defaultdict

# Add project root
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data_pipeline import DataPipeline
from src.models.ppo_agent import HierarchicalPPOAgent, BaselinePPOAgent
from src.models.trading_env import TradingEnvironment


# ============================================
# EVALUATION
# ============================================

def evaluate_agent(agent_wrapper, features, prices, n_episodes=1):
    """
    Evaluate a trained SB3 agent.

    Args:
        agent_wrapper: HierarchicalPPOAgent or BaselinePPOAgent (must be trained)
        features:      Feature DataFrame
        prices:        Price Series
        n_episodes:    Number of evaluation episodes

    Returns:
        List of result dicts per episode
    """
    results = agent_wrapper.evaluate(features, prices, n_episodes)

    # Compute additional metrics from portfolio curve
    portfolio = results['portfolio']

    # Daily returns
    daily_returns = []
    for i in range(1, len(portfolio)):
        dr = (portfolio[i] - portfolio[i - 1]) / portfolio[i - 1]
        daily_returns.append(dr)

    # Sharpe ratio (annualized)
    if len(daily_returns) > 1:
        sharpe = (
            np.mean(daily_returns) /
            (np.std(daily_returns) + 1e-8) *
            np.sqrt(252)
        )
    else:
        sharpe = 0.0

    # Max drawdown
    peak = portfolio[0]
    max_dd = 0.0
    for pv in portfolio:
        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak
        if dd > max_dd:
            max_dd = dd

    results['sharpe_ratio'] = sharpe
    results['max_drawdown'] = max_dd
    results['daily_returns'] = daily_returns

    return results


def compute_buy_and_hold(prices, initial_balance=10000, transaction_cost=0.001):
    """
    Compute buy-and-hold baseline.

    Buy on day 1, hold until end.
    """
    if isinstance(prices, pd.Series):
        price_array = prices.values
    else:
        price_array = prices

    # Buy on day 1
    shares = (initial_balance * (1 - transaction_cost)) / price_array[0]
    portfolio = [shares * p for p in price_array]

    final_value = portfolio[-1]
    total_return = (final_value - initial_balance) / initial_balance * 100

    # Daily returns
    daily_returns = []
    for i in range(1, len(portfolio)):
        dr = (portfolio[i] - portfolio[i - 1]) / portfolio[i - 1]
        daily_returns.append(dr)

    # Sharpe
    if len(daily_returns) > 1:
        sharpe = (
            np.mean(daily_returns) /
            (np.std(daily_returns) + 1e-8) *
            np.sqrt(252)
        )
    else:
        sharpe = 0.0

    # Max drawdown
    peak = portfolio[0]
    max_dd = 0.0
    for pv in portfolio:
        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        'mean_return': total_return,
        'final_value': final_value,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'portfolio': portfolio,
        'n_trades': 1,
    }


# ============================================
# TRAINING PIPELINE
# ============================================

class TrainingPipeline:
    """
    Complete training pipeline.

    Steps:
    1. Load data
    2. Split into train/test
    3. Train sentiment agent
    4. Train baseline agent
    5. Evaluate both
    6. Compare with buy-and-hold
    7. Save results
    """

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self.results = {}

    def run(self, total_timesteps=None, test_ratio=0.2):
        """
        Run the full training pipeline.

        Args:
            total_timesteps: Override training steps (None = use config)
            test_ratio:      Fraction of data for testing

        Returns:
            Dictionary of all results
        """
        print("=" * 60)
        print("TRAINING PIPELINE")
        print("=" * 60)

        # 1. Load data
        print("\n[1/6] Loading data...")
        pipeline = DataPipeline(self.config_path)
        features, prices, dates, feature_names = pipeline.load_data()

        # Convert to DataFrame/Series for agent compatibility
        features_df = pd.DataFrame(
            features, index=dates, columns=feature_names
        )
        prices_series = pd.Series(prices, index=dates, name='Close')

        print(f"  Features: {features_df.shape}")
        print(f"  Prices:   {len(prices_series)} days")

        # 2. Train/test split
        print("\n[2/6] Splitting data...")
        split_idx = int(len(features_df) * (1 - test_ratio))

        train_features = features_df.iloc[:split_idx]
        train_prices = prices_series.iloc[:split_idx]
        test_features = features_df.iloc[split_idx:]
        test_prices = prices_series.iloc[split_idx:]

        print(f"  Train: {len(train_features)} days "
              f"({train_features.index[0].date()} to "
              f"{train_features.index[-1].date()})")
        print(f"  Test:  {len(test_features)} days "
              f"({test_features.index[0].date()} to "
              f"{test_features.index[-1].date()})")

        # 3. Train sentiment agent
        print("\n[3/6] Training SENTIMENT agent...")
        print("-" * 60)
        sentiment_agent = HierarchicalPPOAgent(self.config_path)
        sentiment_agent.train(
            train_features,
            train_prices,
            total_timesteps=total_timesteps,
            tag="sentiment"
        )

        # 4. Train baseline agent
        print("\n[4/6] Training BASELINE agent (price-only)...")
        print("-" * 60)
        baseline_agent = BaselinePPOAgent(self.config_path)
        baseline_agent.train(
            train_features,
            train_prices,
            total_timesteps=total_timesteps,
            tag="baseline"
        )

        # 5. Evaluate both on test data
        print("\n[5/6] Evaluating on TEST data...")
        print("-" * 60)

        print("  Evaluating sentiment agent...")
        sentiment_results = evaluate_agent(
            sentiment_agent, test_features, test_prices
        )

        print("  Evaluating baseline agent...")
        baseline_results = evaluate_agent(
            baseline_agent, test_features, test_prices
        )

        # 6. Buy and hold baseline
        print("\n[6/6] Computing buy-and-hold baseline...")
        env_config = self.config['environment']
        bh_results = compute_buy_and_hold(
            test_prices,
            initial_balance=env_config['initial_balance'],
            transaction_cost=env_config['transaction_cost'],
        )

        # Print comparison
        self._print_comparison(
            sentiment_results, baseline_results, bh_results
        )

        # Save everything
        self.results = {
            'sentiment': sentiment_results,
            'baseline': baseline_results,
            'buy_and_hold': bh_results,
            'train_days': len(train_features),
            'test_days': len(test_features),
        }

        # Save models
        save_dir = self.config.get('paths', {}).get(
            'models', 'models/saved'
        )
        os.makedirs(save_dir, exist_ok=True)
        sentiment_agent.save(f"{save_dir}/sentiment_agent")
        baseline_agent.save(f"{save_dir}/baseline_agent")

        # Save results
        self._save_results()

        return self.results, sentiment_agent, baseline_agent

    def _print_comparison(self, sentiment, baseline, buy_hold):
        """Print side-by-side comparison."""
        print("\n" + "=" * 60)
        print("RESULTS COMPARISON (TEST DATA)")
        print("=" * 60)

        header = f"{'Metric':<25} {'Sentiment':>12} {'Baseline':>12} {'Buy&Hold':>12}"
        print(header)
        print("-" * 65)

        metrics = [
            ('Total Return (%)', 'mean_return'),
            ('Final Value ($)', 'final_value'),
            ('Sharpe Ratio', 'sharpe_ratio'),
            ('Max Drawdown (%)', 'max_drawdown'),
            ('Number of Trades', 'n_trades'),
        ]

        for label, key in metrics:
            s_val = sentiment.get(key, 0)
            b_val = baseline.get(key, 0)
            bh_val = buy_hold.get(key, 0)

            if key == 'max_drawdown':
                s_val *= 100
                b_val *= 100
                bh_val *= 100

            if key in ('final_value',):
                print(f"{label:<25} ${s_val:>11,.0f} ${b_val:>11,.0f} ${bh_val:>11,.0f}")
            elif key in ('n_trades',):
                print(f"{label:<25} {s_val:>12.0f} {b_val:>12.0f} {bh_val:>12.0f}")
            else:
                print(f"{label:<25} {s_val:>12.2f} {b_val:>12.2f} {bh_val:>12.2f}")

        print("=" * 60)

        # Verdict
        s_ret = sentiment.get('mean_return', 0)
        b_ret = baseline.get('mean_return', 0)
        bh_ret = buy_hold.get('mean_return', 0)

        print("\nVERDICT:")
        if s_ret > b_ret and s_ret > bh_ret:
            print("  ★ Sentiment agent WINS! Sentiment adds value.")
        elif s_ret > b_ret:
            print("  ✓ Sentiment beats baseline, but not buy-and-hold.")
        elif b_ret > bh_ret:
            print("  ~ Baseline beats buy-and-hold, but sentiment didn't help.")
        else:
            print("  ✗ Buy-and-hold wins. More training may help.")

    def _save_results(self):
        """Save results to JSON."""
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "training_results.json")

        # Make serializable (remove portfolio lists for brevity)
        serializable = {}
        for agent_name, res in self.results.items():
            if isinstance(res, dict):
                serializable[agent_name] = {
                    k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in res.items()
                    if k not in ('portfolio', 'daily_returns', 'all_returns')
                }
            else:
                serializable[agent_name] = res

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"\nResults saved to: {path}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TRAINING PIPELINE")
    print("=" * 60)

    # Quick test with small number of timesteps
    pipeline = TrainingPipeline()

    # Use small timesteps for testing (increase for real training)
    # Real training: 50000-200000 timesteps
    results, sentiment_agent, baseline_agent = pipeline.run(
        total_timesteps=5000,  # Small for testing
        test_ratio=0.2,
    )

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE TEST COMPLETE!")
    print("=" * 60)
    print("\nFor full training, increase total_timesteps:")
    print("  pipeline.run(total_timesteps=100000)")