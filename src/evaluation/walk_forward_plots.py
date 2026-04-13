# src/evaluation/walk_forward_plots.py

"""
Walk-Forward Validation Plots

Generates publication-quality figures from walk_forward_results.json:
  1. Fold-by-fold bar chart (return, Sharpe, drawdown)
  2. Aggregate Sharpe with error bars
  3. Sentiment win-rate pie chart
"""

import json
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def plot_walk_forward_summary(
    results_path: str = None,
    output_dir: str = None,
):
    """
    Read walk_forward_results.json and produce three PNG plots.
    """
    # Resolve paths from config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if results_path is None:
        results_dir = Path(config.get('paths', {}).get('results_dir', 'results'))
        results_path = results_dir / 'walk_forward_results.json'
    else:
        results_path = Path(results_path)

    if output_dir is None:
        output_dir = Path(config.get('paths', {}).get('plots_dir', 'results/plots'))
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"  [!] {results_path} not found.  Run walk-forward validation first:")
        print(f"      python -m src.training.walk_forward")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    # ------------------------------------------------------------------ #
    # Plot 1 — Fold-by-fold comparison
    # ------------------------------------------------------------------ #
    metrics_to_plot = ['total_return', 'sharpe', 'max_drawdown']
    titles = ['Total Return', 'Sharpe Ratio', 'Max Drawdown']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        sent_vals = results['sentiment'][metric]['values']
        base_vals = results['baseline'][metric]['values']
        bh_vals = results['buy_and_hold'][metric]['values']

        n_folds = len(sent_vals)
        x = np.arange(n_folds)
        width = 0.25

        ax.bar(x - width, sent_vals, width,
               label='Sentiment', color='#2196F3', alpha=0.85)
        ax.bar(x, base_vals, width,
               label='Baseline', color='#FF9800', alpha=0.85)
        ax.bar(x + width, bh_vals, width,
               label='Buy & Hold', color='#4CAF50', alpha=0.85)

        ax.set_xlabel('Fold')
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Fold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i + 1}' for i in range(n_folds)])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        if metric == 'total_return':
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f'{y:.0%}')
            )

    plt.tight_layout()
    path1 = output_dir / 'walk_forward_comparison.png'
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path1}")

    # ------------------------------------------------------------------ #
    # Plot 2 — Aggregate Sharpe with error bars
    # ------------------------------------------------------------------ #
    agents = ['Sentiment Agent', 'Baseline Agent', 'Buy & Hold']
    agent_keys = ['sentiment', 'baseline', 'buy_and_hold']
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    sharpe_means = [results[k]['sharpe']['mean'] for k in agent_keys]
    sharpe_stds = [results[k]['sharpe']['std'] for k in agent_keys]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        agents, sharpe_means, yerr=sharpe_stds, capsize=8,
        color=colors, alpha=0.85, edgecolor='black', linewidth=0.5,
    )

    ax.set_ylabel('Sharpe Ratio (annualized)', fontsize=12)
    ax.set_title('Walk-Forward Validation: Sharpe Ratio (Mean ± Std)',
                 fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5,
               label='Sharpe = 1.0')
    ax.legend()

    for bar, mean, std in zip(bars, sharpe_means, sharpe_stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.05,
            f'{mean:.2f}±{std:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
        )

    plt.tight_layout()
    path2 = output_dir / 'walk_forward_sharpe_summary.png'
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path2}")

    # ------------------------------------------------------------------ #
    # Plot 3 — Win-rate pie chart
    # ------------------------------------------------------------------ #
    win_rate = results.get('sentiment_win_rate', 0.5)
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [win_rate, 1.0 - win_rate]
    labels = ['Sentiment Wins', 'Baseline Wins']
    colors_pie = ['#2196F3', '#FF9800']

    wedges, texts, autotexts = ax.pie(
        sizes, explode=(0.05, 0), labels=labels,
        colors=colors_pie, autopct='%1.0f%%',
        shadow=True, startangle=90,
        textprops={'fontsize': 12},
    )
    ax.set_title(
        'Sentiment Agent Win Rate\n(Across Walk-Forward Folds)',
        fontsize=14,
    )

    plt.tight_layout()
    path3 = output_dir / 'walk_forward_win_rate.png'
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path3}")

    print(f"\n  All walk-forward plots saved to {output_dir}/")


if __name__ == '__main__':
    plot_walk_forward_summary()