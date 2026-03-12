# src/evaluation/evaluator.py
"""
Evaluation and Visualization Module
Creates all plots and metrics for the final report.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TradingEvaluator:
    """
    Comprehensive evaluation of trading agents.
    Generates all plots and metrics for the report.
    """

    def __init__(self, save_dir="results/plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.figsize': (12, 6),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
        })

    def full_evaluation(
        self,
        sentiment_portfolio,
        baseline_portfolio,
        buy_hold_portfolio,
        test_dates,
        test_prices,
        sentiment_results,
        baseline_results,
        buy_hold_results,
    ):
        """
        Run complete evaluation and generate all plots.
        
        Args:
            sentiment_portfolio: list of portfolio values (sentiment agent)
            baseline_portfolio:  list of portfolio values (baseline agent)
            buy_hold_portfolio:  list of portfolio values (buy & hold)
            test_dates:          DatetimeIndex for test period
            test_prices:         array of test prices
            sentiment_results:   dict with metrics
            baseline_results:    dict with metrics
            buy_hold_results:    dict with metrics
        """
        print("=" * 60)
        print("GENERATING EVALUATION REPORT")
        print("=" * 60)

        # 1. Portfolio comparison plot
        print("\n[1/5] Portfolio comparison plot...")
        self.plot_portfolio_comparison(
            sentiment_portfolio,
            baseline_portfolio,
            buy_hold_portfolio,
            test_dates,
        )

        # 2. Returns distribution
        print("[2/5] Returns distribution plot...")
        self.plot_returns_distribution(
            sentiment_portfolio,
            baseline_portfolio,
            buy_hold_portfolio,
        )

        # 3. Drawdown comparison
        print("[3/5] Drawdown comparison plot...")
        self.plot_drawdowns(
            sentiment_portfolio,
            baseline_portfolio,
            buy_hold_portfolio,
            test_dates,
        )

        # 4. Performance metrics table
        print("[4/5] Performance metrics table...")
        self.plot_metrics_table(
            sentiment_results,
            baseline_results,
            buy_hold_results,
        )

        # 5. Price with trades overlay
        print("[5/5] Price and portfolio overlay...")
        self.plot_price_portfolio_overlay(
            test_prices,
            sentiment_portfolio,
            test_dates,
        )

        # Print text summary
        self.print_summary(
            sentiment_results,
            baseline_results,
            buy_hold_results,
        )

        print(f"\n✓ All plots saved to: {self.save_dir}/")
        print("=" * 60)

    def plot_portfolio_comparison(
        self, sentiment, baseline, buy_hold, dates
    ):
        """Plot portfolio value over time for all strategies."""
        fig, ax = plt.subplots(figsize=(14, 7))

        # Align lengths
        n = min(len(sentiment), len(baseline), len(buy_hold), len(dates))
        dates_plot = dates[:n]

        ax.plot(
            dates_plot, sentiment[:n],
            label='Sentiment Agent', color='#2196F3',
            linewidth=2, alpha=0.9
        )
        ax.plot(
            dates_plot, baseline[:n],
            label='Baseline Agent', color='#FF9800',
            linewidth=2, alpha=0.9
        )
        ax.plot(
            dates_plot, buy_hold[:n],
            label='Buy & Hold', color='#4CAF50',
            linewidth=2, linestyle='--', alpha=0.7
        )

        # Initial value line
        ax.axhline(
            y=10000, color='gray', linestyle=':',
            alpha=0.5, label='Initial ($10,000)'
        )

        ax.set_title(
            'Portfolio Value Comparison: Sentiment vs Baseline vs Buy & Hold',
            fontsize=16, fontweight='bold'
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend(loc='upper left', fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'portfolio_comparison.png', dpi=150)
        plt.close()
        print("    ✓ Saved portfolio_comparison.png")

    def plot_returns_distribution(self, sentiment, baseline, buy_hold):
        """Plot histogram of daily returns for each strategy."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        portfolios = {
            'Sentiment Agent': (sentiment, '#2196F3'),
            'Baseline Agent': (baseline, '#FF9800'),
            'Buy & Hold': (buy_hold, '#4CAF50'),
        }

        for ax, (name, (portfolio, color)) in zip(axes, portfolios.items()):
            # Calculate daily returns
            returns = []
            for i in range(1, len(portfolio)):
                r = (portfolio[i] - portfolio[i-1]) / portfolio[i-1]
                returns.append(r * 100)  # Percentage

            ax.hist(
                returns, bins=50, color=color, alpha=0.7,
                edgecolor='white', linewidth=0.5
            )
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.axvline(
                x=np.mean(returns), color='red',
                linestyle='--', alpha=0.8,
                label=f'Mean: {np.mean(returns):.3f}%'
            )
            ax.set_title(name, fontsize=13)
            ax.set_xlabel('Daily Return (%)')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=9)

        plt.suptitle(
            'Daily Returns Distribution',
            fontsize=16, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        plt.savefig(self.save_dir / 'returns_distribution.png', dpi=150)
        plt.close()
        print("    ✓ Saved returns_distribution.png")

    def plot_drawdowns(self, sentiment, baseline, buy_hold, dates):
        """Plot drawdown over time."""
        fig, ax = plt.subplots(figsize=(14, 6))

        portfolios = {
            'Sentiment Agent': (sentiment, '#2196F3'),
            'Baseline Agent': (baseline, '#FF9800'),
            'Buy & Hold': (buy_hold, '#4CAF50'),
        }

        n = min(len(dates), min(len(p) for _, (p, _) in portfolios.items()))
        dates_plot = dates[:n]

        for name, (portfolio, color) in portfolios.items():
            dd = self._compute_drawdown_series(portfolio[:n])
            ax.fill_between(
                dates_plot, dd, 0,
                alpha=0.3, color=color, label=name
            )
            ax.plot(dates_plot, dd, color=color, linewidth=1, alpha=0.7)

        ax.set_title(
            'Drawdown Comparison',
            fontsize=16, fontweight='bold'
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend(loc='lower left', fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'drawdown_comparison.png', dpi=150)
        plt.close()
        print("    ✓ Saved drawdown_comparison.png")

    def plot_metrics_table(self, sentiment_res, baseline_res, bh_res):
        """Create a visual metrics comparison table."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')

        metrics = [
            ['Metric', 'Sentiment', 'Baseline', 'Buy & Hold'],
            [
                'Total Return (%)',
                f"{sentiment_res.get('mean_return', 0):.2f}",
                f"{baseline_res.get('mean_return', 0):.2f}",
                f"{bh_res.get('mean_return', 0):.2f}",
            ],
            [
                'Final Value ($)',
                f"${sentiment_res.get('final_value', 0):,.0f}",
                f"${baseline_res.get('final_value', 0):,.0f}",
                f"${bh_res.get('final_value', 0):,.0f}",
            ],
            [
                'Sharpe Ratio',
                f"{sentiment_res.get('sharpe_ratio', 0):.2f}",
                f"{baseline_res.get('sharpe_ratio', 0):.2f}",
                f"{bh_res.get('sharpe_ratio', 0):.2f}",
            ],
            [
                'Max Drawdown (%)',
                f"{sentiment_res.get('max_drawdown', 0)*100:.2f}",
                f"{baseline_res.get('max_drawdown', 0)*100:.2f}",
                f"{bh_res.get('max_drawdown', 0)*100:.2f}",
            ],
            [
                'Number of Trades',
                f"{sentiment_res.get('n_trades', 0):.0f}",
                f"{baseline_res.get('n_trades', 0):.0f}",
                f"{bh_res.get('n_trades', 0):.0f}",
            ],
        ]

        # Determine best values for highlighting
        table = ax.table(
            cellText=metrics[1:],
            colLabels=metrics[0],
            cellLoc='center',
            loc='center',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)

        # Style header
        for j in range(4):
            table[0, j].set_facecolor('#37474F')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # Style rows
        for i in range(1, len(metrics)):
            for j in range(4):
                if i % 2 == 0:
                    table[i, j].set_facecolor('#E3F2FD')
                else:
                    table[i, j].set_facecolor('#FFFFFF')

        ax.set_title(
            'Performance Metrics Comparison',
            fontsize=16, fontweight='bold', pad=20
        )

        plt.tight_layout()
        plt.savefig(self.save_dir / 'metrics_table.png', dpi=150)
        plt.close()
        print("    ✓ Saved metrics_table.png")

    def plot_price_portfolio_overlay(self, prices, portfolio, dates):
        """Plot stock price and portfolio value on dual axes."""
        fig, ax1 = plt.subplots(figsize=(14, 7))

        n = min(len(prices), len(portfolio), len(dates))
        dates_plot = dates[:n]

        # Price
        color1 = '#37474F'
        ax1.plot(
            dates_plot, prices[:n],
            color=color1, linewidth=1.5, alpha=0.7,
            label='Stock Price'
        )
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price ($)', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        # Portfolio on second axis
        ax2 = ax1.twinx()
        color2 = '#2196F3'
        ax2.plot(
            dates_plot, portfolio[:n],
            color=color2, linewidth=2, alpha=0.9,
            label='Sentiment Portfolio'
        )
        ax2.set_ylabel('Portfolio Value ($)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax1.set_title(
            'Stock Price vs Portfolio Value (Sentiment Agent)',
            fontsize=16, fontweight='bold'
        )
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'price_portfolio_overlay.png', dpi=150)
        plt.close()
        print("    ✓ Saved price_portfolio_overlay.png")

    def _compute_drawdown_series(self, portfolio):
        """Compute drawdown percentage at each timestep."""
        peak = portfolio[0]
        drawdowns = []
        for val in portfolio:
            if val > peak:
                peak = val
            dd = -((peak - val) / peak) * 100
            drawdowns.append(dd)
        return drawdowns

    def print_summary(self, sentiment_res, baseline_res, bh_res):
        """Print text summary for report."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        s_ret = sentiment_res.get('mean_return', 0)
        b_ret = baseline_res.get('mean_return', 0)
        bh_ret = bh_res.get('mean_return', 0)

        print(f"\nTotal Returns:")
        print(f"  Sentiment Agent: {s_ret:.2f}%")
        print(f"  Baseline Agent:  {b_ret:.2f}%")
        print(f"  Buy & Hold:      {bh_ret:.2f}%")

        print(f"\nSharpe Ratios:")
        print(f"  Sentiment: {sentiment_res.get('sharpe_ratio', 0):.2f}")
        print(f"  Baseline:  {baseline_res.get('sharpe_ratio', 0):.2f}")
        print(f"  Buy&Hold:  {bh_res.get('sharpe_ratio', 0):.2f}")

        print(f"\nMax Drawdowns:")
        print(f"  Sentiment: {sentiment_res.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Baseline:  {baseline_res.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Buy&Hold:  {bh_res.get('max_drawdown', 0)*100:.2f}%")

        # Key findings
        print(f"\nKEY FINDINGS:")
        if s_ret > b_ret:
            improvement = s_ret - b_ret
            print(f"  ✓ Sentiment improves returns by {improvement:.2f}% over baseline")
        else:
            print(f"  ✗ Baseline outperforms sentiment by {b_ret - s_ret:.2f}%")

        s_sharpe = sentiment_res.get('sharpe_ratio', 0)
        b_sharpe = baseline_res.get('sharpe_ratio', 0)
        if s_sharpe > b_sharpe:
            print(f"  ✓ Sentiment has better risk-adjusted returns (Sharpe: {s_sharpe:.2f} vs {b_sharpe:.2f})")

        s_dd = sentiment_res.get('max_drawdown', 0)
        b_dd = baseline_res.get('max_drawdown', 0)
        bh_dd = bh_res.get('max_drawdown', 0)
        if s_dd < b_dd and s_dd < bh_dd:
            print(f"  ✓ Sentiment has lowest drawdown ({s_dd*100:.2f}%)")

        print("=" * 60)


# ============================================================
# MAIN: Run evaluation on saved results
# ============================================================

if __name__ == "__main__":
    from src.data.data_pipeline import DataPipeline
    from src.models.ppo_agent import HierarchicalPPOAgent, BaselinePPOAgent
    from src.training.trainer import evaluate_agent, compute_buy_and_hold

    print("=" * 60)
    print("RUNNING FULL EVALUATION")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    pipeline = DataPipeline()
    features, prices, dates, feature_names = pipeline.load_data()

    features_df = pd.DataFrame(features, index=dates, columns=feature_names)
    prices_series = pd.Series(prices, index=dates, name='Close')

    # Split same as training
    test_ratio = 0.2
    split_idx = int(len(features_df) * (1 - test_ratio))
    test_features = features_df.iloc[split_idx:]
    test_prices = prices_series.iloc[split_idx:]
    test_dates = test_features.index

    print(f"  Test period: {test_dates[0].date()} to {test_dates[-1].date()}")
    print(f"  Test days:   {len(test_dates)}")

    # Load trained models
    print("\n[2/4] Loading trained models...")
    import yaml
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    sentiment_agent = HierarchicalPPOAgent()
    sentiment_agent.load(
        "models/saved/sentiment_agent",
        test_features, test_prices
    )

    baseline_agent = BaselinePPOAgent()
    baseline_agent.load(
        "models/saved/baseline_agent",
        test_features, test_prices
    )

    # Evaluate
    print("\n[3/4] Evaluating agents...")
    sentiment_results = evaluate_agent(sentiment_agent, test_features, test_prices)
    baseline_results = evaluate_agent(baseline_agent, test_features, test_prices)

    env_config = config['environment']
    buy_hold_results = compute_buy_and_hold(
        test_prices,
        initial_balance=env_config['initial_balance'],
        transaction_cost=env_config['transaction_cost'],
    )

    # Generate all plots
    print("\n[4/4] Generating evaluation plots...")
    evaluator = TradingEvaluator()
    evaluator.full_evaluation(
        sentiment_portfolio=sentiment_results['portfolio'],
        baseline_portfolio=baseline_results['portfolio'],
        buy_hold_portfolio=buy_hold_results['portfolio'],
        test_dates=test_dates,
        test_prices=test_prices.values,
        sentiment_results=sentiment_results,
        baseline_results=baseline_results,
        buy_hold_results=buy_hold_results,
    )

    print("\nDone! Check results/plots/ for all visualizations.")