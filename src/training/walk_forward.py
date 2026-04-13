# src/training/walk_forward.py

"""
Walk-Forward Validation for RL Trading Agents

Train on expanding/rolling window, test on next unseen window.
Mimics real-world deployment where you never peek at future data.

Example with 5 years of data, 1-year test windows:
  Fold 1: Train [Y1-Y3] -> Test [Y4]
  Fold 2: Train [Y1-Y4] -> Test [Y5]
"""

import numpy as np
import pandas as pd
import json
import yaml
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from src.models.trading_env import TradingEnvironment


class WalkForwardValidator:
    """
    Walk-forward validation engine.

    Splits data into expanding (or rolling) train windows
    followed by fixed-size test windows, trains fresh agents
    per fold, and collects out-of-sample metrics.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        prices_series: pd.Series,
        n_folds: int = 3,
        test_window_days: int = 252,
        min_train_days: int = 504,
        expanding: bool = True,
        config: Optional[dict] = None,
    ):
        """
        Args:
            features_df:      DataFrame with combined features (index = dates)
            prices_series:    Series of close prices (same index)
            n_folds:          Number of walk-forward folds
            test_window_days: Trading days per test fold (~252 = 1 year)
            min_train_days:   Minimum training days required
            expanding:        True = expanding window, False = rolling
            config:           Full project config dict
        """
        self.features_df = features_df
        self.prices_series = prices_series
        self.n_folds = n_folds
        self.test_window_days = test_window_days
        self.min_train_days = min_train_days
        self.expanding = expanding
        self.config = config or {}
        self.fold_results: List[Dict] = []

    # ------------------------------------------------------------------
    # Fold generation
    # ------------------------------------------------------------------

    def generate_folds(self) -> List[Dict]:
        """Generate train/test index splits for walk-forward."""
        total_days = len(self.features_df)
        folds = []

        for fold_idx in range(self.n_folds):
            test_end = total_days - (self.n_folds - fold_idx - 1) * self.test_window_days
            test_start = test_end - self.test_window_days

            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - self.min_train_days)

            train_end = test_start

            if train_end - train_start < self.min_train_days:
                print(f"  [!] Skipping fold {fold_idx}: only "
                      f"{train_end - train_start} train days "
                      f"(need {self.min_train_days})")
                continue

            folds.append({
                'fold': fold_idx,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_days': train_end - train_start,
                'test_days': test_end - test_start,
                'train_date_range': (
                    str(self.features_df.index[train_start].date()),
                    str(self.features_df.index[train_end - 1].date()),
                ),
                'test_date_range': (
                    str(self.features_df.index[test_start].date()),
                    str(self.features_df.index[min(test_end - 1, total_days - 1)].date()),
                ),
            })

        return folds

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, config: dict) -> dict:
        """
        Run walk-forward validation for sentiment and baseline agents.

        Uses HierarchicalPPOAgent and BaselinePPOAgent with their
        actual interface: __init__(config_path) then train(features, prices).

        Args:
            config: Hyperparameter dict (used to read timesteps)

        Returns:
            Aggregated metrics dict across all folds
        """
        from src.models.ppo_agent import HierarchicalPPOAgent, BaselinePPOAgent

        folds = self.generate_folds()
        if not folds:
            raise ValueError(
                "No valid folds could be generated. "
                "Check data length vs test_window_days / min_train_days."
            )

        all_fold_metrics = []

        # Determine timesteps
        timesteps = config.get('ppo', {}).get('total_timesteps_quick',
                    config.get('ppo', {}).get('total_timesteps', 50000))

        for fold_info in folds:
            fold_idx = fold_info['fold']

            print(f"\n{'=' * 60}")
            print(f"  FOLD {fold_idx + 1}/{len(folds)}")
            print(f"  Train: {fold_info['train_date_range'][0]} to "
                  f"{fold_info['train_date_range'][1]}  "
                  f"({fold_info['train_days']} days)")
            print(f"  Test:  {fold_info['test_date_range'][0]} to "
                  f"{fold_info['test_date_range'][1]}  "
                  f"({fold_info['test_days']} days)")
            print(f"{'=' * 60}")

            # ---- Slice data as DataFrames / Series ----
            train_features = self.features_df.iloc[
                fold_info['train_start']:fold_info['train_end']
            ]
            test_features = self.features_df.iloc[
                fold_info['test_start']:fold_info['test_end']
            ]
            train_prices = self.prices_series.iloc[
                fold_info['train_start']:fold_info['train_end']
            ]
            test_prices = self.prices_series.iloc[
                fold_info['test_start']:fold_info['test_end']
            ]

            # ---- Train sentiment agent ----
            print("  Training sentiment agent ...")
            sentiment_agent = HierarchicalPPOAgent()
            sentiment_agent.train(
                features=train_features,
                prices=train_prices,
                total_timesteps=timesteps,
                tag=f"wf_fold{fold_idx}_sentiment"
            )

            # ---- Train baseline agent ----
            print("  Training baseline agent ...")
            baseline_agent = BaselinePPOAgent()
            baseline_agent.train(
                features=train_features,
                prices=train_prices,
                total_timesteps=timesteps,
                tag=f"wf_fold{fold_idx}_baseline"
            )

            # ---- Evaluate on test set ----
            print("  Evaluating sentiment agent on test set ...")
            sent_eval = sentiment_agent.evaluate(
                features=test_features,
                prices=test_prices,
                n_episodes=1
            )

            print("  Evaluating baseline agent on test set ...")
            base_eval = baseline_agent.evaluate(
                features=test_features,
                prices=test_prices,
                n_episodes=1
            )

            # ---- Convert agent results to our metrics format ----
            sent_metrics = self._compute_metrics_from_eval(sent_eval, test_prices)
            base_metrics = self._compute_metrics_from_eval(base_eval, test_prices)
            bh_metrics = self._buy_and_hold(test_prices.values)

            fold_result = {
                'fold': fold_idx,
                'fold_info': {
                    k: v for k, v in fold_info.items()
                    if k not in ('train_date_range', 'test_date_range')
                },
                'train_dates': fold_info['train_date_range'],
                'test_dates': fold_info['test_date_range'],
                'sentiment': sent_metrics,
                'baseline': base_metrics,
                'buy_and_hold': bh_metrics,
            }
            all_fold_metrics.append(fold_result)

            print(f"\n  Fold {fold_idx + 1} Results:")
            print(f"    Sentiment  — Return: {sent_metrics['total_return']:+.2%}  "
                  f"Sharpe: {sent_metrics['sharpe']:.3f}  "
                  f"MaxDD: {sent_metrics['max_drawdown']:.2%}")
            print(f"    Baseline   — Return: {base_metrics['total_return']:+.2%}  "
                  f"Sharpe: {base_metrics['sharpe']:.3f}  "
                  f"MaxDD: {base_metrics['max_drawdown']:.2%}")
            print(f"    Buy & Hold — Return: {bh_metrics['total_return']:+.2%}")

        # Aggregate
        summary = self._aggregate_results(all_fold_metrics)
        self.fold_results = all_fold_metrics
        return summary

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_metrics_from_eval(self, eval_result: dict,
                                    test_prices: pd.Series) -> dict:
        """
        Convert the dict returned by HierarchicalPPOAgent.evaluate()
        into our standardized metrics format.

        eval_result has keys:
          mean_return (percentage), final_value, n_trades,
          portfolio (list of values), all_returns
        """
        portfolio_values = np.array(eval_result['portfolio'], dtype=np.float64)

        # Daily returns from portfolio values
        daily_returns = np.diff(portfolio_values) / (portfolio_values[:-1] + 1e-10)

        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1.0
        mean_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns) + 1e-10

        sharpe = (mean_daily / std_daily) * np.sqrt(252)
        max_drawdown = self._compute_max_drawdown(portfolio_values)

        # Sortino
        downside = daily_returns[daily_returns < 0]
        downside_std = np.std(downside) + 1e-10 if len(downside) > 0 else 1e-10
        sortino = (mean_daily / downside_std) * np.sqrt(252)

        # Calmar
        n_days = len(daily_returns)
        ann_return = total_return * (252 / max(n_days, 1))
        calmar = ann_return / (abs(max_drawdown) + 1e-10)

        return {
            'total_return': float(total_return),
            'sharpe': float(sharpe),
            'sortino': float(sortino),
            'calmar': float(calmar),
            'max_drawdown': float(max_drawdown),
            'volatility': float(std_daily * np.sqrt(252)),
            'n_trades': int(eval_result.get('n_trades', 0)),
            'n_days': n_days,
            'final_value': float(eval_result.get('final_value', portfolio_values[-1])),
            'portfolio_values': portfolio_values.tolist(),
            'daily_returns': daily_returns.tolist(),
        }

    def _buy_and_hold(self, prices: np.ndarray) -> dict:
        """Compute buy-and-hold metrics."""
        prices = np.asarray(prices, dtype=np.float64)
        daily_returns = np.diff(prices) / (prices[:-1] + 1e-10)
        total_return = (prices[-1] / prices[0]) - 1.0
        std_daily = np.std(daily_returns) + 1e-10
        sharpe = (np.mean(daily_returns) / std_daily) * np.sqrt(252)
        max_drawdown = self._compute_max_drawdown(prices)
        return {
            'total_return': float(total_return),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'daily_returns': daily_returns.tolist(),
        }

    @staticmethod
    def _compute_max_drawdown(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / (peak + 1e-10)
        return float(np.min(drawdown))

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate_results(self, all_fold_metrics: List[Dict]) -> dict:
        """Mean +/- std across folds for each agent type."""
        summary = {}

        for agent_type in ['sentiment', 'baseline', 'buy_and_hold']:
            metric_keys = ['total_return', 'sharpe', 'max_drawdown']
            agg = {}
            for key in metric_keys:
                vals = [f[agent_type][key] for f in all_fold_metrics]
                agg[key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'values': [float(v) for v in vals],
                }
            summary[agent_type] = agg

        # Win rate: sentiment vs baseline on total_return
        sent_rets = [f['sentiment']['total_return'] for f in all_fold_metrics]
        base_rets = [f['baseline']['total_return'] for f in all_fold_metrics]
        wins = sum(1 for s, b in zip(sent_rets, base_rets) if s > b)
        summary['sentiment_win_rate'] = float(wins / len(all_fold_metrics))
        summary['n_folds'] = len(all_fold_metrics)

        return summary


# ======================================================================
# CLI entry point
# ======================================================================

def main():
    """Run walk-forward validation from the command line."""
    print("=" * 60)
    print("  WALK-FORWARD VALIDATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    wf_cfg = config.get('walk_forward', {})

    # Load data
    features_df = pd.read_csv(
        config['paths']['combined_features'],
        index_col=0,
        parse_dates=True,
    )
    prices_df = pd.read_csv(
        config['paths']['raw_prices'],
        index_col=0,
        parse_dates=True,
    )

    # Align on common dates
    common_idx = features_df.index.intersection(prices_df.index)
    features_df = features_df.loc[common_idx]
    prices_series = prices_df.loc[common_idx, 'Close']

    print(f"\n  Data loaded: {len(features_df)} trading days")
    print(f"  Date range:  {features_df.index[0].date()} -> "
          f"{features_df.index[-1].date()}")

    # Build validator
    validator = WalkForwardValidator(
        features_df=features_df,
        prices_series=prices_series,
        n_folds=wf_cfg.get('n_folds', 3),
        test_window_days=wf_cfg.get('test_window_days', 252),
        min_train_days=wf_cfg.get('min_train_days', 504),
        expanding=wf_cfg.get('expanding', True),
        config=config,
    )

    # Run — no agent classes passed; run() creates them internally
    results = validator.run(config)

    # Save
    output_dir = Path(config.get('paths', {}).get('results_dir', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / 'walk_forward_results.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  WALK-FORWARD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Results saved to {results_file}")
    print(f"  Sentiment win rate: {results['sentiment_win_rate']:.0%}")
    print(f"  Sentiment Sharpe (mean): "
          f"{results['sentiment']['sharpe']['mean']:.3f} "
          f"+/- {results['sentiment']['sharpe']['std']:.3f}")
    print(f"  Baseline  Sharpe (mean): "
          f"{results['baseline']['sharpe']['mean']:.3f} "
          f"+/- {results['baseline']['sharpe']['std']:.3f}")


if __name__ == '__main__':
    main()