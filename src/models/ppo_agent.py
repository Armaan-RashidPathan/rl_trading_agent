"""
Hierarchical-Like PPO Agent
Custom PPO with:
- Cross-modal attention fusion
- Regime-conditioned behavior
- Action commitment mechanism
- Macro-action space
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

from src.models.attention import AttentionFusionLayer
from src.models.trading_env import TradingEnvironment


# ============================================
# CUSTOM FEATURE EXTRACTOR
# ============================================

class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for PPO
    Uses cross-modal attention to fuse price and sentiment

    Input:  30-dim observation
            [24 market features + 6 position/commitment features]
    Output: 128 + 6 = 134 dim representation
    """

    def __init__(self, observation_space, features_dim=134):
        super(AttentionFeatureExtractor, self).__init__(
            observation_space,
            features_dim=features_dim
        )

        # Market feature dimensions
        self.n_market    = 24  # Technical + Sentiment + Regime
        self.n_position  = 6   # Position info + commitment state

        # Cross-modal attention for market features
        self.attention = AttentionFusionLayer(
            total_features=self.n_market,
            output_dim=128
        )

        # Process position features separately
        self.position_net = nn.Sequential(
            nn.Linear(self.n_position, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

        # Final output dimension
        # 128 (attention) + 6 (position) = 134
        self._features_dim = 134

    def forward(self, observations):
        """
        Forward pass

        Args:
            observations: Tensor [batch, 30]

        Returns:
            features: Tensor [batch, 134]
        """
        # Split observation
        market_features   = observations[:, :self.n_market]   # [batch, 24]
        position_features = observations[:, self.n_market:]   # [batch, 6]

        # Apply cross-modal attention to market features
        attended = self.attention(market_features)             # [batch, 128]

        # Process position features
        position_out = self.position_net(position_features)    # [batch, 6]

        # Combine
        combined = torch.cat([attended, position_out], dim=1) # [batch, 134]

        return combined


# ============================================
# TRAINING CALLBACK
# ============================================

class TrainingCallback(BaseCallback):
    """
    Callback to track training progress
    Logs rewards and portfolio values
    """

    def __init__(self, check_freq=1000, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq    = check_freq
        self.episode_rewards = []
        self.best_mean_reward = -np.inf

    def _on_step(self):
        """Called at every step"""

        # Log episode info when available
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info:
                self.episode_rewards.append(ep_info['r'])

        # Print progress every check_freq steps
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                print(f"  Step {self.n_calls:6d} | "
                      f"Mean reward: {mean_reward:.4f} | "
                      f"Episodes: {len(self.episode_rewards)}")

        return True


# ============================================
# PPO AGENT WRAPPER
# ============================================

class HierarchicalPPOAgent:
    """
    Hierarchical-Like PPO Trading Agent

    Combines:
    1. Cross-modal attention (price + sentiment fusion)
    2. Regime-conditioned behavior (via regime features in state)
    3. Action commitment (via macro-actions)
    4. Stable-Baselines3 PPO algorithm
    """

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.ppo_config = self.config['ppo']
        self.model      = None
        self.env        = None

    def create_env(self, features, prices, mode='train'):
        """
        Create wrapped trading environment

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
                mode=mode
            )

        return DummyVecEnv([make_env])

    def build(self, env):
        """
        Build PPO model with custom attention network

        Args:
            env: Vectorized environment

        Returns:
            PPO model
        """
        # Custom policy kwargs
        policy_kwargs = dict(
            features_extractor_class=AttentionFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=134),
            net_arch=dict(
                pi=[128, 64],   # Policy network layers
                vf=[128, 64]    # Value network layers
            ),
            activation_fn=nn.ReLU
        )

        # Create PPO model
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=self.ppo_config['learning_rate'],
            gamma=self.ppo_config['gamma'],
            gae_lambda=self.ppo_config['gae_lambda'],
            clip_range=self.ppo_config['clip_range'],
            ent_coef=self.ppo_config['entropy_coef'],
            vf_coef=self.ppo_config['value_coef'],
            max_grad_norm=self.ppo_config['max_grad_norm'],
            batch_size=self.ppo_config['batch_size'],
            n_epochs=self.ppo_config['n_epochs'],
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=self.config['seed']
        )

        return model

    def train(self, features, prices, total_timesteps=None, tag=""):
        """
        Train the PPO agent

        Args:
            features:         Feature DataFrame
            prices:           Price Series
            total_timesteps:  Training steps
            tag:              Label for this run

        Returns:
            Trained model
        """
        if total_timesteps is None:
            total_timesteps = self.ppo_config['total_timesteps']

        print(f"\nCreating environment...")
        self.env = self.create_env(features, prices, mode='train')

        print(f"Building PPO model...")
        self.model = self.build(self.env)

        # Count parameters
        total_params = sum(
            p.numel()
            for p in self.model.policy.parameters()
        )
        print(f"Total parameters: {total_params:,}")

        # Training callback
        callback = TrainingCallback(check_freq=2000, verbose=1)

        print(f"\nTraining for {total_timesteps:,} timesteps...")
        print(f"{'─' * 50}")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

        print(f"{'─' * 50}")
        print(f"Training complete!")

        return self.model

    def evaluate(self, features, prices, n_episodes=1):
        """
        Evaluate trained agent

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

        all_returns      = []
        all_values       = []
        all_trades       = []
        all_portfolios   = []

        for episode in range(n_episodes):
            env = TradingEnvironment(
                features=features,
                prices=prices,
                initial_balance=env_config['initial_balance'],
                transaction_cost=env_config['transaction_cost'],
                mode='test'
            )

            obs, _   = env.reset()
            done     = False
            portfolio = [env_config['initial_balance']]

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(int(action))
                portfolio.append(info['total_value'])

            final_value  = info['total_value']
            total_return = (
                final_value - env_config['initial_balance']
            ) / env_config['initial_balance'] * 100

            all_returns.append(total_return)
            all_values.append(final_value)
            all_trades.append(info['n_trades'])
            all_portfolios.append(portfolio)

        results = {
            'mean_return':    np.mean(all_returns),
            'final_value':    np.mean(all_values),
            'n_trades':       np.mean(all_trades),
            'portfolio':      all_portfolios[0],
            'all_returns':    all_returns
        }

        return results

    def save(self, path):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to: {path}")

    def load(self, path, features, prices):
        """Load saved model"""
        self.env   = self.create_env(features, prices)
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from: {path}")
        return self.model


# ============================================
# BASELINE AGENT (Price Only - No Sentiment)
# ============================================

class BaselinePPOAgent(HierarchicalPPOAgent):
    """
    Baseline PPO Agent without sentiment features
    Used for comparison with full model

    Removes sentiment features from observation:
    Uses only first 19 features (technical + regime)
    """

    def create_env(self, features, prices, mode='train'):
        """Create environment with price-only features"""

        # Drop sentiment features (last 5 columns)
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
                mode=mode
            )

        return DummyVecEnv([make_env])

    def build(self, env):
        """Build PPO with smaller network (no attention needed)"""

        policy_kwargs = dict(
            net_arch=dict(
                pi=[128, 64],
                vf=[128, 64]
            ),
            activation_fn=nn.ReLU
        )

        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=self.ppo_config['learning_rate'],
            gamma=self.ppo_config['gamma'],
            gae_lambda=self.ppo_config['gae_lambda'],
            clip_range=self.ppo_config['clip_range'],
            ent_coef=self.ppo_config['entropy_coef'],
            vf_coef=self.ppo_config['value_coef'],
            max_grad_norm=self.ppo_config['max_grad_norm'],
            batch_size=self.ppo_config['batch_size'],
            n_epochs=self.ppo_config['n_epochs'],
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=self.config['seed']
        )

        return model

    def evaluate(self, features, prices, n_episodes=1):
        """
        Evaluate baseline agent.
        IMPORTANT: Must strip sentiment features, same as training.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Run train() first.")

        # Strip sentiment features - same as create_env
        if isinstance(features, pd.DataFrame):
            price_only_features = features.iloc[:, :19].copy()
        else:
            price_only_features = features[:, :19].copy()

        env_config = self.config['environment']

        all_returns    = []
        all_values     = []
        all_trades     = []
        all_portfolios = []

        for episode in range(n_episodes):
            env = TradingEnvironment(
                features=price_only_features,
                prices=prices,
                initial_balance=env_config['initial_balance'],
                transaction_cost=env_config['transaction_cost'],
                mode='test'
            )

            obs, _    = env.reset()
            done      = False
            portfolio = [env_config['initial_balance']]

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(int(action))
                portfolio.append(info['total_value'])

            final_value  = info['total_value']
            total_return = (
                final_value - env_config['initial_balance']
            ) / env_config['initial_balance'] * 100

            all_returns.append(total_return)
            all_values.append(final_value)
            all_trades.append(info['n_trades'])
            all_portfolios.append(portfolio)

        results = {
            'mean_return':  np.mean(all_returns),
            'final_value':  np.mean(all_values),
            'n_trades':     np.mean(all_trades),
            'portfolio':    all_portfolios[0],
            'all_returns':  all_returns
        }

        return results


# ============================================
# QUICK TEST
# ============================================

def test_agent():
    """Quick test of PPO agent"""

    print("=" * 50)
    print("TESTING PPO AGENT")
    print("=" * 50)

    # Load config
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    print("\nLoading data...")
    features = pd.read_csv(
        config['paths']['combined_features'],
        index_col=0,
        parse_dates=True
    )
    prices = pd.read_csv(
        config['paths']['raw_prices'],
        index_col=0,
        parse_dates=True
    )
    prices = prices.loc[features.index, 'Close']

    # Use small subset for quick test
    test_features = features.iloc[:200]
    test_prices   = prices.iloc[:200]

    print(f"  Features: {test_features.shape}")
    print(f"  Prices:   {len(test_prices)} days")

    # Test feature extractor
    print("\n[1/3] Testing Feature Extractor...")
    obs_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(30,),
        dtype=np.float32
    )
    extractor = AttentionFeatureExtractor(obs_space)
    dummy_obs = torch.randn(4, 30)
    out       = extractor(dummy_obs)
    print(f"  Input:  {dummy_obs.shape}")
    print(f"  Output: {out.shape}")
    print(f"  ✓ Feature extractor working")

    # Test full sentiment agent
    print("\n[2/3] Testing Sentiment PPO Agent...")
    agent = HierarchicalPPOAgent()
    env   = agent.create_env(test_features, test_prices)
    model = agent.build(env)

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  ✓ Sentiment agent built successfully")

    # Test baseline agent
    print("\n[3/3] Testing Baseline PPO Agent...")
    baseline = BaselinePPOAgent()
    base_env  = baseline.create_env(test_features, test_prices)
    base_model = baseline.build(base_env)

    base_params = sum(p.numel() for p in base_model.policy.parameters())
    print(f"  Total parameters: {base_params:,}")
    print(f"  ✓ Baseline agent built successfully")

    print("\n" + "=" * 50)
    print("PPO AGENT TEST COMPLETE!")
    print("Ready for training!")
    print("=" * 50)


if __name__ == "__main__":
    test_agent()