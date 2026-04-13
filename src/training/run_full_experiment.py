# src/training/run_full_experiment.py

"""
Master Experiment Script

Orchestrates the full experiment pipeline:
  1. Walk-forward validation (PPO sentiment vs PPO baseline)
  2. Statistical significance tests on collected daily returns
  3. Generate all evaluation + walk-forward + explainability plots
  4. Produce a final JSON summary
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def run_full_experiment(mode: str = 'quick'):
    """
    Run the complete experiment suite.

    Args:
        mode: 'quick' (~10 min), 'full' (~60 min), or 'extensive' (~3 hrs)
    """
    start_time = datetime.now()

    print("=" * 70)
    print("  DRL TRADING AGENT — FULL EXPERIMENT SUITE")
    print(f"  Mode:    {mode}")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- Load config ----
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mode_cfg = config.get('experiment', {}).get('modes', {}).get(mode, {})
    timesteps = mode_cfg.get('timesteps',
                             config.get('ppo', {}).get('total_timesteps_quick', 50000))
    wf_folds = mode_cfg.get('walk_forward_folds',
                            config.get('walk_forward', {}).get('n_folds', 3))

    # Set timesteps so walk-forward picks it up
    config.setdefault('ppo', {})['total_timesteps_quick'] = timesteps

    # ---- Load data ----
    features_df = pd.read_csv(
        config['paths']['combined_features'],
        index_col=0, parse_dates=True,
    )
    prices_df = pd.read_csv(
        config['paths']['raw_prices'],
        index_col=0, parse_dates=True,
    )

    common_idx = features_df.index.intersection(prices_df.index)
    features_df = features_df.loc[common_idx]
    prices_series = prices_df.loc[common_idx, 'Close']

    print(f"\n  Data: {len(features_df)} trading days "
          f"({features_df.index[0].date()} -> {features_df.index[-1].date()})")
    print(f"  Timesteps per agent per fold: {timesteps:,}")
    print(f"  Walk-forward folds: {wf_folds}")

    results_dir = Path(config.get('paths', {}).get('results_dir', 'results'))
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(config.get('paths', {}).get('plots_dir', 'results/plots'))
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Step 1: Walk-Forward Validation
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("  [1/4] WALK-FORWARD VALIDATION")
    print(f"{'=' * 70}")

    from src.training.walk_forward import WalkForwardValidator

    wf_cfg = config.get('walk_forward', {})
    validator = WalkForwardValidator(
        features_df=features_df,
        prices_series=prices_series,
        n_folds=wf_folds,
        test_window_days=wf_cfg.get('test_window_days', 252),
        min_train_days=wf_cfg.get('min_train_days', 504),
        expanding=wf_cfg.get('expanding', True),
        config=config,
    )

    wf_results = validator.run(config)

    wf_file = results_dir / 'walk_forward_results.json'
    with open(wf_file, 'w') as f:
        json.dump(wf_results, f, indent=2, default=str)
    print(f"\n  Saved walk-forward results to {wf_file}")

    # ==================================================================
    # Step 2: Statistical Significance Tests
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("  [2/4] STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'=' * 70}")

    from src.evaluation.statistical_tests import StatisticalAnalyzer

    # Collect daily returns from all folds
    all_sent_returns = []
    all_base_returns = []
    for fold in validator.fold_results:
        all_sent_returns.extend(fold['sentiment']['daily_returns'])
        all_base_returns.extend(fold['baseline']['daily_returns'])

    sent_returns = np.array(all_sent_returns, dtype=np.float64)
    base_returns = np.array(all_base_returns, dtype=np.float64)

    print(f"  Collected {len(sent_returns)} daily returns across all folds")

    stat_cfg = config.get('statistical_tests', {})
    stat_results = StatisticalAnalyzer.run_all_tests(
        sent_returns,
        base_returns,
        n_bootstrap=stat_cfg.get('bootstrap_samples', 10000),
        confidence=stat_cfg.get('confidence_level', 0.95),
    )

    stat_file = results_dir / 'statistical_tests.json'
    with open(stat_file, 'w') as f:
        json.dump(stat_results, f, indent=2)
    print(f"  Saved statistical tests to {stat_file}")
    print(f"  Verdict: {stat_results['verdict']['conclusion']}")

    # ==================================================================
    # Step 3: Generate All Plots
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("  [3/4] GENERATING PLOTS")
    print(f"{'=' * 70}")

    # 3a: Standard evaluation plots
    try:
        print("\n  Running standard evaluation plots ...")
        from src.evaluation.evaluator import TradingEvaluator
        evaluator = TradingEvaluator()
        evaluator.run()
    except Exception as e:
        print(f"  [!] Evaluator error (non-fatal): {e}")

    # 3b: Walk-forward plots
    try:
        print("\n  Running walk-forward plots ...")
        from src.evaluation.walk_forward_plots import plot_walk_forward_summary
        plot_walk_forward_summary(
            results_path=str(wf_file),
            output_dir=str(plots_dir),
        )
    except Exception as e:
        print(f"  [!] Walk-forward plots error (non-fatal): {e}")

    # 3c: Attention explainability
    try:
        print("\n  Running attention explainability ...")
        from src.explainability.attention_viz import AttentionExplainer
        explainer = AttentionExplainer()
        explainer.run()
    except Exception as e:
        print(f"  [!] Explainability error (non-fatal): {e}")

    # ==================================================================
    # Step 4: Final Summary
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("  [4/4] FINAL SUMMARY")
    print(f"{'=' * 70}")

    elapsed = datetime.now() - start_time
    final_summary = {
        'experiment_date': datetime.now().isoformat(),
        'mode': mode,
        'elapsed_seconds': elapsed.total_seconds(),
        'ticker': config.get('data', {}).get('ticker', 'AAPL'),
        'data_range': {
            'start': str(features_df.index[0].date()),
            'end': str(features_df.index[-1].date()),
            'total_days': len(features_df),
        },
        'training': {
            'timesteps_per_agent': timesteps,
            'algorithm': 'PPO',
        },
        'walk_forward': {
            'n_folds': wf_results.get('n_folds', wf_folds),
            'sentiment_sharpe_mean': wf_results['sentiment']['sharpe']['mean'],
            'sentiment_sharpe_std': wf_results['sentiment']['sharpe']['std'],
            'baseline_sharpe_mean': wf_results['baseline']['sharpe']['mean'],
            'baseline_sharpe_std': wf_results['baseline']['sharpe']['std'],
            'sentiment_return_mean': wf_results['sentiment']['total_return']['mean'],
            'baseline_return_mean': wf_results['baseline']['total_return']['mean'],
            'buy_hold_return_mean': wf_results['buy_and_hold']['total_return']['mean'],
            'sentiment_win_rate': wf_results['sentiment_win_rate'],
        },
        'statistical_significance': stat_results['verdict'],
        'conclusion': stat_results['verdict']['conclusion'],
    }

    summary_file = results_dir / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2)

    # Print final report
    wf = final_summary['walk_forward']
    print(f"\n  Ticker:              {final_summary['ticker']}")
    print(f"  Data range:          {final_summary['data_range']['start']} -> "
          f"{final_summary['data_range']['end']} "
          f"({final_summary['data_range']['total_days']} days)")
    print(f"  Mode:                {mode}")
    print(f"  Timesteps/agent:     {timesteps:,}")
    print(f"  Walk-forward folds:  {wf['n_folds']}")
    print(f"")
    print(f"  Sentiment Sharpe:    {wf['sentiment_sharpe_mean']:.3f} "
          f"+/- {wf['sentiment_sharpe_std']:.3f}")
    print(f"  Baseline  Sharpe:    {wf['baseline_sharpe_mean']:.3f} "
          f"+/- {wf['baseline_sharpe_std']:.3f}")
    print(f"  Sentiment return:    {wf['sentiment_return_mean']:+.2%}")
    print(f"  Baseline  return:    {wf['baseline_return_mean']:+.2%}")
    print(f"  Buy & Hold return:   {wf['buy_hold_return_mean']:+.2%}")
    print(f"  Sentiment win rate:  {wf['sentiment_win_rate']:.0%}")
    print(f"")
    print(f"  Statistical verdict: {final_summary['conclusion']}")
    print(f"  Elapsed time:        {elapsed}")
    print(f"  Summary saved to:    {summary_file}")

    print(f"\n{'=' * 70}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")

    return final_summary


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run full DRL trading experiment suite'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='quick',
        choices=['quick', 'full', 'extensive'],
        help='Experiment mode: quick (~10min), full (~60min), extensive (~3hrs)',
    )
    args = parser.parse_args()
    run_full_experiment(mode=args.mode)