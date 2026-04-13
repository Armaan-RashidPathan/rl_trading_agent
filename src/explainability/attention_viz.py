# src/explainability/attention_viz.py
"""
Explainability Module
Extracts and visualizes attention weights from the trained model.
Shows HOW the agent uses sentiment information for trading decisions.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data_pipeline import DataPipeline
from src.models.ppo_agent import HierarchicalPPOAgent
from src.models.trading_env import TradingEnvironment


class AttentionExplainer:
    """
    Extracts attention weights from the trained sentiment agent
    and creates interpretable visualizations.
    """

    def __init__(self, save_dir="results/plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def extract_attention_weights(self, agent, features_df, prices_series):
        """
        Run the trained agent through test data and extract
        attention weights at each timestep.

        Returns:
            DataFrame with attention weights aligned to dates
        """
        print("  Extracting attention weights from model...")

        import yaml
        with open("configs/config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        env_config = config['environment']

        # Create environment
        env = TradingEnvironment(
            features=features_df,
            prices=prices_series,
            initial_balance=env_config['initial_balance'],
            transaction_cost=env_config['transaction_cost'],
            mode='test'
        )

        # Get the attention layer from the model
        attn_layer = agent.model.policy.features_extractor.attention

        # Run episode and collect weights
        obs, info = env.reset()
        done = False

        price_to_sent_weights = []
        sent_to_price_weights = []
        actions_taken = []
        portfolio_values = [env_config['initial_balance']]
        macro_actions = []
        step_dates = []

        step = 0
        while not done:
            # Get action from model
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                # Forward pass through the policy (triggers attention)
                action, _ = agent.model.predict(obs, deterministic=True)

            # Now extract stored attention weights
            weights = attn_layer.get_attention_weights()

            if 'price_to_sentiment' in weights:
                p2s = weights['price_to_sentiment'].flatten()[0]
                price_to_sent_weights.append(float(p2s))
            else:
                price_to_sent_weights.append(0.5)

            if 'sentiment_to_price' in weights:
                s2p = weights['sentiment_to_price'].flatten()[0]
                sent_to_price_weights.append(float(s2p))
            else:
                sent_to_price_weights.append(0.5)

            # Step
            obs, reward, done, _, info = env.step(int(action))
            actions_taken.append(int(action))
            portfolio_values.append(info['total_value'])
            macro_actions.append(info.get('macro_action', 'HOLD'))

            if step < len(features_df.index):
                step_dates.append(features_df.index[step])

            step += 1

        # Build DataFrame
        n = min(len(step_dates), len(price_to_sent_weights))
        attn_df = pd.DataFrame({
            'price_to_sentiment': price_to_sent_weights[:n],
            'sentiment_to_price': sent_to_price_weights[:n],
            'action': actions_taken[:n],
            'macro_action': macro_actions[:n],
            'portfolio_value': portfolio_values[1:n+1],
        }, index=step_dates[:n])

        print(f"  ✓ Extracted {len(attn_df)} timesteps of attention weights")
        print(f"    Price→Sentiment attention: mean={attn_df['price_to_sentiment'].mean():.3f}")
        print(f"    Sentiment→Price attention: mean={attn_df['sentiment_to_price'].mean():.3f}")

        return attn_df

    def plot_attention_timeline(self, attn_df, features_df, prices_series):
        """
        Plot 1: Attention weights over time with price and sentiment.
        Shows WHEN the model pays attention to sentiment.
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

        dates = attn_df.index
        n = len(dates)

        # Panel 1: Stock Price
        ax1 = axes[0]
        price_vals = prices_series.loc[dates].values
        ax1.plot(dates, price_vals, color='#37474F', linewidth=1.5)
        ax1.set_ylabel('Stock Price ($)')
        ax1.set_title('Attention Analysis: How the Agent Uses Sentiment',
                      fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Mark buy/sell points
        for i, row in attn_df.iterrows():
            if row['macro_action'] in ('AGGRESSIVE_LONG', 'CONSERVATIVE_LONG'):
                ax1.axvline(x=i, color='green', alpha=0.15, linewidth=1)
            elif row['macro_action'] == 'EXIT':
                ax1.axvline(x=i, color='red', alpha=0.15, linewidth=1)

        # Panel 2: Sentiment Score
        ax2 = axes[1]
        if 'sentiment_score' in features_df.columns:
            sent = features_df.loc[dates, 'sentiment_score']
            colors = ['#4CAF50' if v > 0 else '#F44336' for v in sent]
            ax2.bar(dates, sent, color=colors, alpha=0.6, width=1.5)
        ax2.set_ylabel('Sentiment Score')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Price→Sentiment Attention (how much price cares about sentiment)
        ax3 = axes[2]
        p2s = attn_df['price_to_sentiment']
        p2s_smooth = p2s.rolling(window=5, min_periods=1).mean()

        ax3.fill_between(dates, p2s_smooth, alpha=0.3, color='#2196F3')
        ax3.plot(dates, p2s_smooth, color='#2196F3', linewidth=1.5,
                 label='Price → Sentiment Attention')
        ax3.set_ylabel('Attention Weight')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        # Highlight high attention periods
        threshold = p2s.mean() + p2s.std()
        high_attn = p2s > threshold
        for i, (date, is_high) in enumerate(zip(dates, high_attn)):
            if is_high:
                ax3.axvline(x=date, color='#FF5722', alpha=0.1, linewidth=1)

        # Panel 4: Sentiment→Price Attention
        ax4 = axes[3]
        s2p = attn_df['sentiment_to_price']
        s2p_smooth = s2p.rolling(window=5, min_periods=1).mean()

        ax4.fill_between(dates, s2p_smooth, alpha=0.3, color='#FF9800')
        ax4.plot(dates, s2p_smooth, color='#FF9800', linewidth=1.5,
                 label='Sentiment → Price Attention')
        ax4.set_ylabel('Attention Weight')
        ax4.set_xlabel('Date')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'attention_timeline.png', dpi=150)
        plt.close()
        print("    ✓ Saved attention_timeline.png")

    def plot_attention_vs_volatility(self, attn_df, features_df):
        """
        Plot 2: Does the model pay MORE attention to sentiment
        during volatile periods?
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        dates = attn_df.index

        # Left: Scatter plot
        ax1 = axes[0]
        if 'volatility_20d' in features_df.columns:
            vol = features_df.loc[dates, 'volatility_20d'].values
            p2s = attn_df['price_to_sentiment'].values

            scatter = ax1.scatter(vol, p2s, alpha=0.4, c=p2s,
                                  cmap='coolwarm', s=20)
            plt.colorbar(scatter, ax=ax1, label='Attention Weight')

            # Add trend line
            z = np.polyfit(vol, p2s, 1)
            p = np.poly1d(z)
            vol_sorted = np.sort(vol)
            ax1.plot(vol_sorted, p(vol_sorted), "r--", alpha=0.8,
                     linewidth=2, label=f'Trend (slope={z[0]:.3f})')

            corr = np.corrcoef(vol, p2s)[0, 1]
            ax1.set_title(f'Attention vs Volatility (corr={corr:.3f})',
                         fontsize=13)
            ax1.set_xlabel('Volatility (20d)')
            ax1.set_ylabel('Price→Sentiment Attention')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Right: Scatter with sentiment magnitude
        ax2 = axes[1]
        if 'sentiment_score' in features_df.columns:
            sent_abs = features_df.loc[dates, 'sentiment_score'].abs().values
            p2s = attn_df['price_to_sentiment'].values

            scatter2 = ax2.scatter(sent_abs, p2s, alpha=0.4, c=p2s,
                                   cmap='coolwarm', s=20)
            plt.colorbar(scatter2, ax=ax2, label='Attention Weight')

            z2 = np.polyfit(sent_abs, p2s, 1)
            p2 = np.poly1d(z2)
            sent_sorted = np.sort(sent_abs)
            ax2.plot(sent_sorted, p2(sent_sorted), "r--", alpha=0.8,
                     linewidth=2, label=f'Trend (slope={z2[0]:.3f})')

            corr2 = np.corrcoef(sent_abs, p2s)[0, 1]
            ax2.set_title(f'Attention vs |Sentiment| (corr={corr2:.3f})',
                         fontsize=13)
            ax2.set_xlabel('|Sentiment Score|')
            ax2.set_ylabel('Price→Sentiment Attention')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.suptitle('What Drives Sentiment Attention?',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'attention_drivers.png', dpi=150)
        plt.close()
        print("    ✓ Saved attention_drivers.png")

    def plot_attention_by_action(self, attn_df):
        """
        Plot 3: How does attention differ by action type?
        Shows whether the agent uses sentiment more for BUY vs SELL decisions.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Group by macro action
        action_groups = attn_df.groupby('macro_action')

        # Left: Price→Sentiment by action
        ax1 = axes[0]
        action_means = {}
        action_stds = {}
        for name, group in action_groups:
            if len(group) >= 2:
                action_means[name] = group['price_to_sentiment'].mean()
                action_stds[name] = group['price_to_sentiment'].std()

        if action_means:
            names = list(action_means.keys())
            means = [action_means[n] for n in names]
            stds = [action_stds[n] for n in names]

            colors = []
            for n in names:
                if 'LONG' in n or 'TREND' in n or 'MEAN' in n:
                    colors.append('#4CAF50')
                elif 'EXIT' in n:
                    colors.append('#F44336')
                else:
                    colors.append('#FF9800')

            bars = ax1.bar(range(len(names)), means, yerr=stds,
                          color=colors, alpha=0.7, capsize=5,
                          edgecolor='white', linewidth=1)
            ax1.set_xticks(range(len(names)))
            ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
            ax1.set_ylabel('Mean Attention Weight')
            ax1.set_title('Price→Sentiment Attention by Action',
                         fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')

        # Right: Action distribution pie chart
        ax2 = axes[1]
        action_counts = attn_df['macro_action'].value_counts()
        pie_colors = []
        for name in action_counts.index:
            if 'LONG' in name or 'TREND' in name or 'MEAN' in name:
                pie_colors.append('#4CAF50')
            elif 'EXIT' in name:
                pie_colors.append('#F44336')
            elif 'HOLD' in name:
                pie_colors.append('#FF9800')
            else:
                pie_colors.append('#9E9E9E')

        ax2.pie(action_counts.values, labels=action_counts.index,
                autopct='%1.1f%%', colors=pie_colors, startangle=90,
                textprops={'fontsize': 9})
        ax2.set_title('Action Distribution', fontsize=13, fontweight='bold')

        plt.suptitle('Attention Weights by Trading Decision',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'attention_by_action.png', dpi=150)
        plt.close()
        print("    ✓ Saved attention_by_action.png")

    def plot_attention_heatmap(self, attn_df, features_df):
        """
        Plot 4: Correlation heatmap - which features correlate with attention?
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        dates = attn_df.index

        # Build correlation dataframe
        corr_data = pd.DataFrame(index=dates)
        corr_data['P→S Attention'] = attn_df['price_to_sentiment'].values
        corr_data['S→P Attention'] = attn_df['sentiment_to_price'].values

        # Add key features
        feature_map = {
            'returns': 'Daily Returns',
            'volatility_20d': 'Volatility',
            'rsi': 'RSI',
            'macd': 'MACD',
            'trend_strength': 'Trend Strength',
            'sentiment_score': 'Sentiment Score',
            'sentiment_std': 'Sentiment Std',
            'sentiment_momentum': 'Sentiment Momentum',
            'news_count': 'News Count',
            'vol_regime': 'Vol Regime',
        }

        for feat_col, display_name in feature_map.items():
            if feat_col in features_df.columns:
                corr_data[display_name] = features_df.loc[dates, feat_col].values

        # Compute correlation matrix
        corr_matrix = corr_data.corr()

        # Plot heatmap
        import matplotlib.colors as mcolors
        cmap = plt.cm.RdBu_r

        im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1,
                       aspect='auto')

        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right',
                          fontsize=9)
        ax.set_yticklabels(corr_matrix.columns, fontsize=9)

        # Add correlation values
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                val = corr_matrix.values[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=8)

        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title('Feature-Attention Correlation Heatmap',
                     fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'attention_correlation_heatmap.png', dpi=150)
        plt.close()
        print("    ✓ Saved attention_correlation_heatmap.png")

    def generate_text_report(self, attn_df, features_df):
        """
        Generate text-based explainability findings.
        """
        print("\n" + "=" * 60)
        print("EXPLAINABILITY FINDINGS")
        print("=" * 60)

        dates = attn_df.index
        p2s = attn_df['price_to_sentiment']
        s2p = attn_df['sentiment_to_price']

        print(f"\n1. ATTENTION WEIGHT STATISTICS:")
        print(f"   Price→Sentiment: mean={p2s.mean():.3f}, std={p2s.std():.3f}")
        print(f"   Sentiment→Price: mean={s2p.mean():.3f}, std={s2p.std():.3f}")

        # High attention events
        threshold = p2s.mean() + 1.5 * p2s.std()
        high_attn_days = attn_df[p2s > threshold]
        print(f"\n2. HIGH ATTENTION EVENTS ({len(high_attn_days)} days):")
        for date, row in high_attn_days.head(5).iterrows():
            sent = features_df.loc[date, 'sentiment_score'] if 'sentiment_score' in features_df.columns else 0
            print(f"   {date.date()}: attention={row['price_to_sentiment']:.3f}, "
                  f"sentiment={sent:.3f}, action={row['macro_action']}")

        # Correlation with volatility
        if 'volatility_20d' in features_df.columns:
            vol = features_df.loc[dates, 'volatility_20d']
            corr_vol = p2s.corr(vol)
            print(f"\n3. ATTENTION-VOLATILITY CORRELATION: {corr_vol:.3f}")
            if corr_vol > 0.1:
                print("   → Model pays MORE attention to sentiment during volatile periods")
            elif corr_vol < -0.1:
                print("   → Model pays LESS attention to sentiment during volatile periods")
            else:
                print("   → Attention is independent of volatility")

        # Attention by action type
        print(f"\n4. ATTENTION BY ACTION TYPE:")
        for action, group in attn_df.groupby('macro_action'):
            if len(group) >= 2:
                print(f"   {action:25s}: attention={group['price_to_sentiment'].mean():.3f} "
                      f"(n={len(group)})")

        # Sentiment correlation
        if 'sentiment_score' in features_df.columns:
            sent = features_df.loc[dates, 'sentiment_score']
            corr_sent = p2s.corr(sent.abs())
            print(f"\n5. ATTENTION vs |SENTIMENT| CORRELATION: {corr_sent:.3f}")
            if corr_sent > 0.1:
                print("   → Model attends more when sentiment signal is strong")
            else:
                print("   → Attention is not driven by sentiment magnitude alone")

        print("\n" + "=" * 60)


def run_explainability():
    """Run complete explainability analysis."""
    print("=" * 60)
    print("RUNNING EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    pipeline = DataPipeline()
    features, prices, dates, feature_names = pipeline.load_data()

    features_df = pd.DataFrame(features, index=dates, columns=feature_names)
    prices_series = pd.Series(prices, index=dates, name='Close')

    # Use test split
    test_ratio = 0.2
    split_idx = int(len(features_df) * (1 - test_ratio))
    test_features = features_df.iloc[split_idx:]
    test_prices = prices_series.iloc[split_idx:]

    print(f"  Test period: {test_features.index[0].date()} to {test_features.index[-1].date()}")

    # Load model
    print("\n[2/4] Loading trained model...")
    agent = HierarchicalPPOAgent()
    agent.load("models/saved/sentiment_agent", test_features, test_prices)

    # Extract attention
    print("\n[3/4] Extracting and analyzing attention weights...")
    explainer = AttentionExplainer()
    attn_df = explainer.extract_attention_weights(
        agent, test_features, test_prices
    )

    # Generate all plots
    print("\n[4/4] Generating explainability plots...")

    explainer.plot_attention_timeline(attn_df, test_features, test_prices)
    explainer.plot_attention_vs_volatility(attn_df, test_features)
    explainer.plot_attention_by_action(attn_df)
    explainer.plot_attention_heatmap(attn_df, test_features)
    explainer.generate_text_report(attn_df, test_features)

    print("\n✓ All explainability plots saved to results/plots/")
    print("=" * 60)

    return attn_df


if __name__ == "__main__":
    run_explainability()