"""
Trading Environment Module
Gymnasium-compatible trading environment with:
- Macro actions (hierarchical-like behavior)
- Action commitment mechanism
- Transaction costs
- Portfolio tracking
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for RL agent

    STATE:
        - 24 market features (technical + sentiment + regime)
        - Current position (0=cash, 1=holding)
        - Balance ratio (cash / total value)
        - Committed action (one-hot: 3 dims)
        - Remaining commitment (normalized)
        Total: 24 + 2 + 3 + 1 = 30 dimensions

    ACTIONS (Macro-Actions × Commitment):
        Macro-Actions:
            0: EXIT           - Close position
            1: HOLD           - Maintain position
            2: AGGRESSIVE_LONG- Buy immediately
            3: CONSERVATIVE_LONG - Buy only on dips
            4: TREND_FOLLOW   - Follow momentum
            5: MEAN_REVERT    - Counter-trend

        Commitment: [1, 3, 5, 10] days
        Total: 6 × 4 = 24 discrete actions

    REWARD:
        Portfolio return per step minus transaction costs
    """

    # Macro action names for interpretability
    MACRO_ACTIONS = {
        0: 'EXIT',
        1: 'HOLD',
        2: 'AGGRESSIVE_LONG',
        3: 'CONSERVATIVE_LONG',
        4: 'TREND_FOLLOW',
        5: 'MEAN_REVERT'
    }

    COMMITMENT_OPTIONS = [1, 3, 5, 10]

    def __init__(
        self,
        features,
        prices,
        initial_balance=10000,
        transaction_cost=0.001,
        mode='train'
    ):
        """
        Initialize Trading Environment

        Args:
            features: DataFrame with all features (24 cols)
            prices: Series with Close prices
            initial_balance: Starting capital
            transaction_cost: Cost per trade (0.1%)
            mode: 'train' or 'test'
        """
        super().__init__()

        # Store data
        self.features         = features.values.astype(np.float32)
        self.prices           = prices.values.astype(np.float32)
        self.dates            = features.index
        self.initial_balance  = initial_balance
        self.transaction_cost = transaction_cost
        self.mode             = mode
        self.n_steps          = len(features)

        # Feature dimensions
        self.n_features       = features.shape[1]  # 24

        # Action space: 6 macro × 4 commitment = 24
        self.n_macro          = len(self.MACRO_ACTIONS)
        self.n_commitment     = len(self.COMMITMENT_OPTIONS)
        self.n_actions        = self.n_macro * self.n_commitment

        # Observation space: features + position info
        # 24 features + 1 position + 1 balance + 3 committed_onehot + 1 remaining
        self.obs_dim = self.n_features + 6

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        self.current_step      = 0
        self.balance           = float(self.initial_balance)
        self.shares            = 0.0
        self.total_value       = float(self.initial_balance)
        self.prev_value        = float(self.initial_balance)

        # Commitment state
        self.committed_action  = 1  # Default: HOLD
        self.remaining_commit  = 0

        # Tracking
        self.trade_history     = []
        self.portfolio_history = [self.initial_balance]
        self.action_history    = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """
        Execute one step in the environment

        Args:
            action: Integer (0-23) representing macro×commitment

        Returns:
            observation, reward, done, truncated, info
        """
        # Decode action
        macro_action, commitment = self._decode_action(action)

        # Handle commitment
        if self.remaining_commit > 0:
            # Still committed - use previous action
            macro_action = self.committed_action
            self.remaining_commit -= 1
        else:
            # New action - set commitment
            self.committed_action = macro_action
            self.remaining_commit = commitment - 1

        # Get current market state
        current_price = self.prices[self.current_step]
        current_features = self.features[self.current_step]

        # Convert macro to primitive action
        primitive = self._macro_to_primitive(
            macro_action,
            current_price,
            current_features
        )

        # Execute primitive action
        self._execute_trade(primitive, current_price)

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Calculate new portfolio value
        new_price = self.prices[self.current_step]
        new_value = self.balance + self.shares * new_price

        # Calculate reward
        reward = self._calculate_reward(new_value)

        # Update tracking
        self.prev_value = new_value
        self.total_value = new_value
        self.portfolio_history.append(new_value)
        self.action_history.append(macro_action)

        # Get observation and info
        obs  = self._get_observation()
        info = self._get_info()
        info['macro_action']  = self.MACRO_ACTIONS[macro_action]
        info['primitive']     = primitive
        info['current_price'] = new_price

        return obs, reward, done, False, info

    def _decode_action(self, action):
        """
        Decode combined action into macro + commitment

        Action encoding:
            action = macro_idx * n_commitment + commitment_idx
        """
        macro_idx      = action // self.n_commitment
        commitment_idx = action % self.n_commitment

        macro_action = macro_idx
        commitment   = self.COMMITMENT_OPTIONS[commitment_idx]

        return macro_action, commitment

    def _macro_to_primitive(self, macro_action, price, features):
        """
        Convert macro-action to primitive (Buy/Sell/Hold)

        Args:
            macro_action: Integer (0-5)
            price: Current price
            features: Current feature vector

        Returns:
            0: Sell
            1: Hold
            2: Buy
        """
        is_holding = self.shares > 0

        # Get relevant features for decision
        # Feature indices based on our feature matrix order
        rsi_idx      = 2   # RSI feature
        macd_idx     = 3   # MACD feature
        trend_idx    = 13  # trend_strength
        trend_dir    = 14  # trend_direction

        rsi          = features[rsi_idx]       # normalized [-1, 1]
        macd         = features[macd_idx]      # normalized
        trend_str    = features[trend_idx]     # [-1, 1]
        trend_d      = features[trend_dir]     # -1 or 1

        if macro_action == 0:  # EXIT
            return 0 if is_holding else 1

        elif macro_action == 1:  # HOLD
            return 1

        elif macro_action == 2:  # AGGRESSIVE_LONG
            # Buy immediately if not holding
            return 2 if not is_holding else 1

        elif macro_action == 3:  # CONSERVATIVE_LONG
            # Buy only on dips (RSI oversold)
            if not is_holding and rsi < -0.2:  # Oversold
                return 2
            return 1

        elif macro_action == 4:  # TREND_FOLLOW
            # Follow the trend
            if trend_d > 0 and trend_str > 0.2 and not is_holding:
                return 2  # Buy in uptrend
            elif trend_d < 0 and trend_str < -0.2 and is_holding:
                return 0  # Sell in downtrend
            return 1

        elif macro_action == 5:  # MEAN_REVERT
            # Counter-trend trading
            if rsi > 0.4 and is_holding:
                return 0  # Sell overbought
            elif rsi < -0.4 and not is_holding:
                return 2  # Buy oversold
            return 1

        return 1  # Default: Hold

    def _execute_trade(self, primitive, price):
        """
        Execute primitive trade action

        Args:
            primitive: 0=Sell, 1=Hold, 2=Buy
            price: Current price
        """
        if primitive == 2 and self.balance > 0:
            # BUY: Use all available cash
            cost          = self.balance * self.transaction_cost
            invest_amount = self.balance - cost
            self.shares   = invest_amount / price
            self.balance  = 0.0

            self.trade_history.append({
                'step':   self.current_step,
                'action': 'BUY',
                'price':  price,
                'shares': self.shares
            })

        elif primitive == 0 and self.shares > 0:
            # SELL: Sell all shares
            revenue       = self.shares * price
            cost          = revenue * self.transaction_cost
            self.balance  = revenue - cost
            self.shares   = 0.0

            self.trade_history.append({
                'step':   self.current_step,
                'action': 'SELL',
                'price':  price,
                'shares': 0
            })

        # primitive == 1: HOLD - do nothing

    def _calculate_reward(self, new_value):
        """
        Calculate step reward

        Reward = portfolio return for this step
        """
        if self.prev_value > 0:
            reward = (new_value - self.prev_value) / self.prev_value
        else:
            reward = 0.0

        return float(reward)

    def _get_observation(self):
        """
        Build observation vector

        Returns:
            numpy array of shape (obs_dim,)
        """
        # Market features
        market_features = self.features[self.current_step]

        # Position info
        is_holding    = 1.0 if self.shares > 0 else 0.0
        balance_ratio = self.balance / (self.total_value + 1e-8)

        # Commitment state (one-hot encode committed action)
        # Only track: EXIT(0), HOLD(1), LONG-type(2)
        # Simplified: sell(0), hold(1), buy(2)
        committed_onehot = np.zeros(3, dtype=np.float32)
        if self.committed_action == 0:    # EXIT
            committed_onehot[0] = 1.0
        elif self.committed_action == 1:  # HOLD
            committed_onehot[1] = 1.0
        else:                             # Any long action
            committed_onehot[2] = 1.0

        # Remaining commitment (normalized)
        remaining_norm = self.remaining_commit / max(self.COMMITMENT_OPTIONS)

        # Combine all
        obs = np.concatenate([
            market_features,
            [is_holding, balance_ratio],
            committed_onehot,
            [remaining_norm]
        ]).astype(np.float32)

        return obs

    def _get_info(self):
        """Get current environment info"""
        return {
            'step':          self.current_step,
            'balance':       self.balance,
            'shares':        self.shares,
            'total_value':   self.total_value,
            'position':      'holding' if self.shares > 0 else 'cash',
            'n_trades':      len(self.trade_history),
            'committed':     self.MACRO_ACTIONS[self.committed_action],
            'remaining':     self.remaining_commit
        }

    def get_portfolio_value(self):
        """Get current portfolio value"""
        if self.current_step < self.n_steps:
            price = self.prices[self.current_step]
        else:
            price = self.prices[-1]
        return self.balance + self.shares * price

    def get_trade_history(self):
        """Get list of all trades made"""
        return self.trade_history

    def get_portfolio_history(self):
        """Get portfolio value over time"""
        return self.portfolio_history

    def render(self):
        """Print current state"""
        print(f"Step: {self.current_step}/{self.n_steps}")
        print(f"Price: ${self.prices[self.current_step]:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares: {self.shares:.4f}")
        print(f"Total Value: ${self.total_value:.2f}")
        print(f"Position: {'Holding' if self.shares > 0 else 'Cash'}")
        print(f"Committed: {self.MACRO_ACTIONS[self.committed_action]}")


def test_environment():
    """Test the trading environment with random actions"""
    import yaml

    print("=" * 50)
    print("TESTING TRADING ENVIRONMENT")
    print("=" * 50)

    # Load config
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Load combined features
    print("\nLoading features...")
    features = pd.read_csv(
        config['paths']['combined_features'],
        index_col=0,
        parse_dates=True
    )

    # Load prices
    prices = pd.read_csv(
        config['paths']['raw_prices'],
        index_col=0,
        parse_dates=True
    )

    # Align prices with features
    prices = prices.loc[features.index, 'Close']

    print(f"  Features shape: {features.shape}")
    print(f"  Price points:   {len(prices)}")

    # Create environment
    print("\nCreating environment...")
    env = TradingEnvironment(
        features=features,
        prices=prices,
        initial_balance=config['environment']['initial_balance'],
        transaction_cost=config['environment']['transaction_cost']
    )

    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space:      {env.action_space.n}")
    print(f"  Macro actions:     {env.n_macro}")
    print(f"  Commitment opts:   {env.COMMITMENT_OPTIONS}")

    # Test with random actions
    print("\nRunning random episode...")
    obs, info = env.reset()

    print(f"\nInitial State:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Portfolio value:   ${info['total_value']:,.2f}")

    # Run for 50 steps
    total_reward = 0
    n_steps_test = 50

    for step in range(n_steps_test):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        if step % 10 == 0:
            macro = info['macro_action']
            prim  = info['primitive']
            val   = info['total_value']
            pos   = info['position']
            print(f"  Step {step:3d}: {macro:20s} → {['SELL','HOLD','BUY'][prim]:4s} | "
                  f"${val:8,.2f} | {pos}")

        if done:
            break

    print(f"\nAfter {n_steps_test} steps:")
    print(f"  Final value:  ${info['total_value']:,.2f}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Trades made:  {info['n_trades']}")

    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"  ✓ Reset successful")
    print(f"  Portfolio value: ${info['total_value']:,.2f}")

    # Run full episode
    print("\nRunning full episode...")
    obs, info = env.reset()
    done = False
    steps = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        steps += 1

    final_value = info['total_value']
    n_trades    = info['n_trades']
    returns     = (final_value - 10000) / 10000 * 100

    print(f"  Steps completed: {steps}")
    print(f"  Final value:     ${final_value:,.2f}")
    print(f"  Total return:    {returns:.2f}%")
    print(f"  Total trades:    {n_trades}")

    print("\n" + "=" * 50)
    print("ENVIRONMENT TEST COMPLETE!")
    print("Next: Cross-Modal Attention Module")
    print("=" * 50)

    return env


if __name__ == "__main__":
    test_environment()