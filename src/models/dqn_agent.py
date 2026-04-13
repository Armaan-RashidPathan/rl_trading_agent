# src/models/dqn_agent.py

"""
DQN Trading Agents

Provides DQN-based agents using Stable-Baselines3 for comparison
with the PPO agents. Follows the same interface pattern as
HierarchicalPPOAgent: __init__(config_path), train(features, prices),
evaluate(features, prices).
"""

import os
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path

from src.models.trading_env import TradingEnvironment


class DQNTradingAgent:
    """
    DQN trading agent wrapping Stable-Baselines3.

    Interface matches HierarchicalPPOAgent:
      agent = DQNTradingAgent()
      agent.train(features, prices, total_timesteps=50000)
      results = agent.evaluate(features, prices)
      agent.save("models/saved/dqn_agent.zip")
    """

    def __init__(self, config_path="configs/config.yaml"):
        """
        Args:
            config_path: Path to project config YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.dqn_config = self.config.get('dqn', {})
        self.model = None
        self.env = None

    def create_env(self, features, prices, mode='train'):
        """
        Create wrapped trading environment.

        Args:
            features: Feature DataFrame
            prices:   Price Series
            mode:     'train' or 'test'

        Returns:
            DummyVecEnv wrapped environment
        """
        env_config = self.config['environment']

        def make_env():
            return TradingEnvironment(
                features=features,
                prices=prices,
                initial_balance=env_config['initial_balance'],
                transaction_cost=env_config['transaction_cost'],
                mode=mode,
            )

        return DummyVecEnv([make_env])

    def build(self, env):
        """
        Build DQN model.

        Args:
            env: Vectorized environment

        Returns:
            DQN model
        """
        net_arch = self.dqn_config.get('net_arch', [256, 128, 64])

        model = DQN(
            policy='MlpPolicy',
            env=env,
            learning_rate=self.dqn_config.get('learning_rate', 1e-4),
            buffer_size=self.dqn_config.get('buffer_size', 50000),
            learning_starts=self.dqn_config.get('learning_starts', 1000),
            batch_size=self.dqn_config.get('batch_size', 64),
            tau=self.dqn_config.get('tau', 0.005),
            gamma=self.dqn_config.get('gamma', 0.99),
            train_freq=self.dqn_config.get('train_freq', 4),
            target_update_interval=self.dqn_config.get('target_update_interval', 1000),
            exploration_fraction=self.dqn_config.get('exploration_fraction', 0.1),
            exploration_initial_eps=self.dqn_config.get('exploration_initial_eps', 1.0),
            exploration_final_eps=self.dqn_config.get('exploration_final_eps', 0.02),
            policy_kwargs=dict(net_arch=net_arch),
            verbose=self.config.get('verbose', 1),
            seed=self.config.get('seed', 42),
        )

        return model

    def train(self, features, prices, total_timesteps=None, tag=""):
        """
        Train the DQN agent.

        Args:
            features:         Feature DataFrame
            prices:           Price Series
            total_timesteps:  Training steps
            tag:              Label for this run

        Returns:
            Trained model
        """
        if total_timesteps is None:
            total_timesteps = self.dqn_config.get('total_timesteps_quick', 50000)

        print(f"\n  Creating DQN environment ...")
        self.env = self.create_env(features, prices, mode='train')

        print(f"  Building DQN model ...")
        self.model = self.build(self.env)

        total_params = sum(p.numel() for p in self.model.policy.parameters())
        print(f"  Total parameters: {total_params:,}")

        print(f"  Training DQN for {total_timesteps:,} timesteps ...")
        self.model.learn(total_timesteps=total_timesteps)

        print(f"  DQN training complete!")
        return self.model

    def evaluate(self, features, prices, n_episodes=1):
        """
        Evaluate trained agent.

        Args:
            features:   Feature DataFrame
            prices:     Price Series
            n_episodes: Number of evaluation episodes

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Run train() first.")

        env_config = self.config['environment']

        all_returns = []
        all_values = []
        all_trades = []
        all_portfolios = []

        for episode in range(n_episodes):
            env = TradingEnvironment(
                features=features,
                prices=prices,
                initial_balance=env_config['initial_balance'],
                transaction_cost=env_config['transaction_cost'],
                mode='test',
            )

            obs, _ = env.reset()
            done = False
            portfolio = [env_config['initial_balance']]

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(int(action))
                portfolio.append(info['total_value'])

            final_value = info['total_value']
            total_return = (
                final_value - env_config['initial_balance']
            ) / env_config['initial_balance'] * 100

            all_returns.append(total_return)
            all_values.append(final_value)
            all_trades.append(info['n_trades'])
            all_portfolios.append(portfolio)

        results = {
            'mean_return': np.mean(all_returns),
            'final_value': np.mean(all_values),
            'n_trades': np.mean(all_trades),
            'portfolio': all_portfolios[0],
            'all_returns': all_returns,
        }

        return results

    def predict(self, obs, deterministic=True):
        """Predict action given observation."""
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path):
        """Save trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"  DQN model saved to {path}")

    def load(self, path, features, prices):
        """Load saved model."""
        self.env = self.create_env(features, prices)
        self.model = DQN.load(path, env=self.env)
        print(f"  DQN model loaded from {path}")
        return self.model


class DQNBaselineAgent(DQNTradingAgent):
    """
    DQN agent using only price/technical features (no sentiment).
    Strips sentiment columns the same way BaselinePPOAgent does.
    """

    def create_env(self, features, prices, mode='train'):
        """Create environment with price-only features (first 19 columns)."""
        if isinstance(features, pd.DataFrame):
            price_only_features = features.iloc[:, :19].copy()
        else:
            price_only_features = features[:, :19].copy()

        env_config = self.config['environment']

        def make_env():
            return TradingEnvironment(
                features=price_only_features,
                prices=prices,
                initial_balance=env_config['initial_balance'],
                transaction_cost=env_config['transaction_cost'],
                mode=mode,
            )

        return DummyVecEnv([make_env])

    def evaluate(self, features, prices, n_episodes=1):
        """Evaluate baseline — strips sentiment features first."""
        if self.model is None:
            raise ValueError("Model not trained yet. Run train() first.")

        if isinstance(features, pd.DataFrame):
            price_only_features = features.iloc[:, :19].copy()
        else:
            price_only_features = features[:, :19].copy()

        env_config = self.config['environment']

        all_returns = []
        all_values = []
        all_trades = []
        all_portfolios = []

        for episode in range(n_episodes):
            env = TradingEnvironment(
                features=price_only_features,
                prices=prices,
                initial_balance=env_config['initial_balance'],
                transaction_cost=env_config['transaction_cost'],
                mode='test',
            )

            obs, _ = env.reset()
            done = False
            portfolio = [env_config['initial_balance']]

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(int(action))
                portfolio.append(info['total_value'])

            final_value = info['total_value']
            total_return = (
                final_value - env_config['initial_balance']
            ) / env_config['initial_balance'] * 100

            all_returns.append(total_return)
            all_values.append(final_value)
            all_trades.append(info['n_trades'])
            all_portfolios.append(portfolio)

        results = {
            'mean_return': np.mean(all_returns),
            'final_value': np.mean(all_values),
            'n_trades': np.mean(all_trades),
            'portfolio': all_portfolios[0],
            'all_returns': all_returns,
        }

        return results


# ======================================================================
# Quick smoke test
# ======================================================================

def test_dqn_agent():
    """Smoke test: create env, train DQN briefly, predict."""
    print("=" * 50)
    print("DQN AGENT SMOKE TEST")
    print("=" * 50)

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    features = pd.read_csv(
        config['paths']['combined_features'],
        index_col=0, parse_dates=True,
    )
    prices = pd.read_csv(
        config['paths']['raw_prices'],
        index_col=0, parse_dates=True,
    )
    prices = prices.loc[features.index, 'Close']

    # Use a small slice for speed
    features_small = features.iloc[:200]
    prices_small = prices.iloc[:200]

    # Test sentiment DQN
    print("\n[1/2] Testing DQN Sentiment Agent ...")
    agent = DQNTradingAgent()
    agent.train(features_small, prices_small, total_timesteps=2000)

    metrics = agent.evaluate(features_small, prices_small, n_episodes=1)
    print(f"  Total return: {metrics['mean_return']:.2f}%")
    print(f"  Trades:       {metrics['n_trades']:.0f}")

    # Save / load round-trip
    save_path = 'models/saved/dqn_test.zip'
    agent.save(save_path)
    agent.load(save_path, features_small, prices_small)

    # Test baseline DQN
    print("\n[2/2] Testing DQN Baseline Agent ...")
    baseline = DQNBaselineAgent()
    baseline.train(features_small, prices_small, total_timesteps=2000)

    base_metrics = baseline.evaluate(features_small, prices_small, n_episodes=1)
    print(f"  Total return: {base_metrics['mean_return']:.2f}%")
    print(f"  Trades:       {base_metrics['n_trades']:.0f}")

    print("\n  ✓ DQN agent smoke test passed!")


if __name__ == '__main__':
    test_dqn_agent()