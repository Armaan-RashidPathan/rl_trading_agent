"""
Trading Environment Module
Gymnasium-compatible trading environment with:
- Macro actions (hierarchical-like behavior)
- Action commitment mechanism
- Transaction costs + slippage + spread modeling
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

    ACTIONS (Macro-Actions x Commitment):
        Macro-Actions:
            0: EXIT           - Close position
            1: HOLD           - Maintain position
            2: AGGRESSIVE_LONG- Buy immediately
            3: CONSERVATIVE_LONG - Buy only on dips
            4: TREND_FOLLOW   - Follow momentum
            5: MEAN_REVERT    - Counter-trend

        Commitment: [1, 3, 5, 10] days
        Total: 6 x 4 = 24 discrete actions

    REWARD:
        Portfolio return per step minus transaction costs

    EXECUTION MODEL (NEW):
        - Transaction cost:  flat fee per trade (default 0.1%)
        - Slippage:          price impact proportional to volatility
        - Spread:            bid-ask spread (half-spread applied each side)
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
        slippage_model='proportional',
        slippage_base_bps=5.0,
        spread_bps=2.0,
        volatility_slippage_mult=1.0,
        mode='train'
    ):
        """
        Initialize Trading Environment

        Args:
            features:                  DataFrame with all features (24 cols)
            prices:                    Series with Close prices
            initial_balance:           Starting capital
            transaction_cost:          Cost per trade as fraction (0.001 = 0.1%)
            slippage_model:            'none', 'fixed', or 'proportional'
            slippage_base_bps:         Base slippage in basis points (1bp = 0.01%)
            spread_bps:                Half-spread in basis points
            volatility_slippage_mult:  How much volatility amplifies slippage
            mode:                      'train' or 'test'
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

        # ============================================
        # EXECUTION MODEL (NEW)
        # ============================================
        self.slippage_model          = slippage_model
        self.slippage_base_bps       = slippage_base_bps
        self.spread_bps              = spread_bps
        self.volatility_slippage_mult = volatility_slippage_mult

        # Pre-compute rolling volatility for slippage calculation
        # Use 20-day rolling std of returns
        price_series = pd.Series(self.prices)
        returns = price_series.pct_change()
        self.rolling_volatility = returns.rolling(
            window=20, min_periods=5
        ).std().fillna(returns.std()).values.astype(np.float32)

        # Tracking for execution costs
        self.total_transaction_costs = 0.0
        self.total_slippage_costs    = 0.0
        self.total_spread_costs      = 0.0
        self.execution_log           = []

        # Feature dimensions
        self.n_features       = features.shape[1]  # 24

        # Action space: 6 macro x 4 commitment = 24
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

        # Reset execution cost tracking
        self.total_transaction_costs = 0.0
        self.total_slippage_costs    = 0.0
        self.total_spread_costs      = 0.0
        self.execution_log           = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    # ============================================
    # EXECUTION MODEL (NEW)
    # ============================================

    def _calculate_slippage(self, price, is_buy):
        """
        Calculate slippage based on the selected model.

        Slippage represents the price impact of executing a trade.
        When buying, you pay more than the quoted price.
        When selling, you receive less.

        Args:
            price:  Quoted mid-price
            is_buy: True for buy, False for sell

        Returns:
            Adjusted execution price (worse than quoted)
        """
        if self.slippage_model == 'none':
            return price

        # Base slippage in decimal (5 bps = 0.0005)
        base_slip = self.slippage_base_bps / 10000.0

        if self.slippage_model == 'fixed':
            # Fixed slippage: same amount every trade
            slippage_pct = base_slip

        elif self.slippage_model == 'proportional':
            # Proportional to current volatility
            # High volatility = more slippage (wider spreads, less liquidity)
            current_vol = self.rolling_volatility[self.current_step]
            median_vol = np.median(self.rolling_volatility[
                self.rolling_volatility > 0
            ])

            if median_vol > 0:
                vol_ratio = current_vol / median_vol
            else:
                vol_ratio = 1.0

            # Slippage scales with volatility
            slippage_pct = base_slip * (
                1.0 + self.volatility_slippage_mult * max(0, vol_ratio - 1.0)
            )

        else:
            slippage_pct = base_slip

        # Apply direction: buys get worse (higher) price,
        #                   sells get worse (lower) price
        if is_buy:
            execution_price = price * (1.0 + slippage_pct)
        else:
            execution_price = price * (1.0 - slippage_pct)

        return execution_price

    def _calculate_spread_cost(self, price, is_buy):
        """
        Calculate the bid-ask spread cost.

        The quoted price is the mid-price. Actual execution happens at:
          - Ask price (mid + half_spread) for buys
          - Bid price (mid - half_spread) for sells

        Args:
            price:  Mid-price
            is_buy: True for buy, False for sell

        Returns:
            Price adjusted for spread
        """
        half_spread = self.spread_bps / 10000.0

        if is_buy:
            return price * (1.0 + half_spread)
        else:
            return price * (1.0 - half_spread)

    def _get_execution_price(self, quoted_price, is_buy):
        """
        Get the realistic execution price including all friction.

        Order of application:
          1. Spread (bid-ask)
          2. Slippage (market impact)

        The transaction cost (commission) is applied separately
        to the total trade value in _execute_trade.

        Args:
            quoted_price: The Close price (mid-price)
            is_buy:       True for buy, False for sell

        Returns:
            execution_price, spread_cost, slippage_cost
        """
        # Step 1: Apply spread
        after_spread = self._calculate_spread_cost(quoted_price, is_buy)
        spread_cost = abs(after_spread - quoted_price)

        # Step 2: Apply slippage on top of spread-adjusted price
        execution_price = self._calculate_slippage(after_spread, is_buy)
        slippage_cost = abs(execution_price - after_spread)

        return execution_price, spread_cost, slippage_cost

    def _execute_trade(self, primitive, price):
        """
        Execute primitive trade action with realistic execution model.

        Args:
            primitive: 0=Sell, 1=Hold, 2=Buy
            price: Current quoted (mid) price
        """
        if primitive == 2 and self.balance > 0:
            # BUY -------------------------------------------------------
            # Get realistic execution price
            exec_price, spread_cost, slippage_cost = (
                self._get_execution_price(price, is_buy=True)
            )

            # Commission on total trade value
            commission = self.balance * self.transaction_cost

            # Amount available after commission
            invest_amount = self.balance - commission

            # Buy shares at the execution price (worse than quoted)
            self.shares = invest_amount / exec_price
            self.balance = 0.0

            # Track costs
            # Spread cost in dollar terms
            dollar_spread = spread_cost * self.shares
            dollar_slippage = slippage_cost * self.shares

            self.total_transaction_costs += commission
            self.total_spread_costs      += dollar_spread
            self.total_slippage_costs    += dollar_slippage

            self.execution_log.append({
                'step':            self.current_step,
                'action':          'BUY',
                'quoted_price':    float(price),
                'execution_price': float(exec_price),
                'shares':          float(self.shares),
                'commission':      float(commission),
                'spread_cost':     float(dollar_spread),
                'slippage_cost':   float(dollar_slippage),
                'total_friction':  float(commission + dollar_spread + dollar_slippage),
            })

            self.trade_history.append({
                'step':            self.current_step,
                'action':          'BUY',
                'price':           float(exec_price),
                'quoted_price':    float(price),
                'shares':          float(self.shares),
                'commission':      float(commission),
                'spread_cost':     float(dollar_spread),
                'slippage_cost':   float(dollar_slippage),
            })

        elif primitive == 0 and self.shares > 0:
            # SELL ------------------------------------------------------
            # Get realistic execution price
            exec_price, spread_cost, slippage_cost = (
                self._get_execution_price(price, is_buy=False)
            )

            # Revenue at execution price (worse than quoted)
            revenue = self.shares * exec_price

            # Commission on revenue
            commission = revenue * self.transaction_cost

            # Net proceeds
            self.balance = revenue - commission

            # Track costs
            dollar_spread = spread_cost * self.shares
            dollar_slippage = slippage_cost * self.shares

            self.total_transaction_costs += commission
            self.total_spread_costs      += dollar_spread
            self.total_slippage_costs    += dollar_slippage

            self.execution_log.append({
                'step':            self.current_step,
                'action':          'SELL',
                'quoted_price':    float(price),
                'execution_price': float(exec_price),
                'shares':          float(self.shares),
                'commission':      float(commission),
                'spread_cost':     float(dollar_spread),
                'slippage_cost':   float(dollar_slippage),
                'total_friction':  float(commission + dollar_spread + dollar_slippage),
            })

            self.trade_history.append({
                'step':            self.current_step,
                'action':          'SELL',
                'price':           float(exec_price),
                'quoted_price':    float(price),
                'shares':          0,
                'commission':      float(commission),
                'spread_cost':     float(dollar_spread),
                'slippage_cost':   float(dollar_slippage),
            })

            self.shares = 0.0

        # primitive == 1: HOLD - do nothing

    def step(self, action):
        """
        Execute one step in the environment

        Args:
            action: Integer (0-23) representing macro x commitment

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

        # Execute primitive action (now with slippage + spread)
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
        rsi_idx      = min(2, len(features) - 1)
        macd_idx     = min(3, len(features) - 1)
        trend_idx    = min(13, len(features) - 1)
        trend_dir    = min(14, len(features) - 1)

        rsi          = features[rsi_idx]
        macd         = features[macd_idx]
        trend_str    = features[trend_idx]
        trend_d      = features[trend_dir]

        if macro_action == 0:  # EXIT
            return 0 if is_holding else 1

        elif macro_action == 1:  # HOLD
            return 1

        elif macro_action == 2:  # AGGRESSIVE_LONG
            return 2 if not is_holding else 1

        elif macro_action == 3:  # CONSERVATIVE_LONG
            if not is_holding and rsi < -0.2:
                return 2
            return 1

        elif macro_action == 4:  # TREND_FOLLOW
            if trend_d > 0 and trend_str > 0.2 and not is_holding:
                return 2
            elif trend_d < 0 and trend_str < -0.2 and is_holding:
                return 0
            return 1

        elif macro_action == 5:  # MEAN_REVERT
            if rsi > 0.4 and is_holding:
                return 0
            elif rsi < -0.4 and not is_holding:
                return 2
            return 1

        return 1

    def _calculate_reward(self, new_value):
        """
        Calculate step reward

        Enhanced reward with:
        - Portfolio return (base signal)
        - Risk penalty (drawdown awareness)
        - Trade efficiency bonus
        """
        if self.prev_value > 0:
            portfolio_return = (new_value - self.prev_value) / self.prev_value
        else:
            portfolio_return = 0.0

        # Risk penalty: penalize drawdowns from peak
        peak_value = max(self.portfolio_history) if self.portfolio_history else self.initial_balance
        if peak_value > 0:
            drawdown = (peak_value - new_value) / peak_value
            risk_penalty = -0.1 * max(0, drawdown - 0.05)
        else:
            risk_penalty = 0.0

        # Scale reward
        reward = portfolio_return * 100 + risk_penalty

        return float(reward)

    def _get_observation(self):
        """
        Build observation vector

        Returns:
            numpy array of shape (obs_dim,)
        """
        market_features = self.features[self.current_step]

        is_holding    = 1.0 if self.shares > 0 else 0.0
        balance_ratio = self.balance / (self.total_value + 1e-8)

        committed_onehot = np.zeros(3, dtype=np.float32)
        if self.committed_action == 0:
            committed_onehot[0] = 1.0
        elif self.committed_action == 1:
            committed_onehot[1] = 1.0
        else:
            committed_onehot[2] = 1.0

        remaining_norm = self.remaining_commit / max(self.COMMITMENT_OPTIONS)

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
            'step':                    self.current_step,
            'balance':                 self.balance,
            'shares':                  self.shares,
            'total_value':             self.total_value,
            'position':                'holding' if self.shares > 0 else 'cash',
            'n_trades':                len(self.trade_history),
            'committed':               self.MACRO_ACTIONS[self.committed_action],
            'remaining':               self.remaining_commit,
            'total_transaction_costs':  self.total_transaction_costs,
            'total_slippage_costs':     self.total_slippage_costs,
            'total_spread_costs':       self.total_spread_costs,
            'total_execution_costs':   (
                self.total_transaction_costs +
                self.total_slippage_costs +
                self.total_spread_costs
            ),
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

    def get_execution_summary(self):
        """
        Get a summary of all execution costs incurred.

        Returns:
            Dict with cost breakdown
        """
        total_exec = (
            self.total_transaction_costs +
            self.total_slippage_costs +
            self.total_spread_costs
        )

        return {
            'n_trades':              len(self.trade_history),
            'transaction_costs':     round(self.total_transaction_costs, 2),
            'slippage_costs':        round(self.total_slippage_costs, 2),
            'spread_costs':          round(self.total_spread_costs, 2),
            'total_execution_costs': round(total_exec, 2),
            'cost_as_pct_initial':   round(total_exec / self.initial_balance * 100, 4),
            'avg_cost_per_trade':    round(
                total_exec / max(len(self.trade_history), 1), 2
            ),
            'execution_log':         self.execution_log,
        }

    def render(self):
        """Print current state"""
        print(f"Step: {self.current_step}/{self.n_steps}")
        print(f"Price: ${self.prices[self.current_step]:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares: {self.shares:.4f}")
        print(f"Total Value: ${self.total_value:.2f}")
        print(f"Position: {'Holding' if self.shares > 0 else 'Cash'}")
        print(f"Committed: {self.MACRO_ACTIONS[self.committed_action]}")
        exec_summary = self.get_execution_summary()
        print(f"Execution costs: ${exec_summary['total_execution_costs']:.2f} "
              f"({exec_summary['cost_as_pct_initial']:.2f}% of initial)")


def test_environment():
    """Test the trading environment with execution model"""
    import yaml

    print("=" * 60)
    print("TESTING TRADING ENVIRONMENT")
    print("  (with transaction costs + slippage + spread)")
    print("=" * 60)

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

    # ============================================
    # Test 1: Environment with NO slippage (baseline)
    # ============================================
    print("\n" + "=" * 60)
    print("TEST 1: No slippage (old behavior)")
    print("=" * 60)

    env_no_slip = TradingEnvironment(
        features=features,
        prices=prices,
        initial_balance=config['environment']['initial_balance'],
        transaction_cost=config['environment']['transaction_cost'],
        slippage_model='none',
        spread_bps=0.0,
    )

    obs, _ = env_no_slip.reset()
    done = False
    while not done:
        action = env_no_slip.action_space.sample()
        obs, reward, done, _, info = env_no_slip.step(action)

    summary_no_slip = env_no_slip.get_execution_summary()
    final_no_slip = info['total_value']

    print(f"  Final value:        ${final_no_slip:,.2f}")
    print(f"  Trades:             {summary_no_slip['n_trades']}")
    print(f"  Transaction costs:  ${summary_no_slip['transaction_costs']:.2f}")
    print(f"  Slippage costs:     ${summary_no_slip['slippage_costs']:.2f}")
    print(f"  Spread costs:       ${summary_no_slip['spread_costs']:.2f}")
    print(f"  Total exec costs:   ${summary_no_slip['total_execution_costs']:.2f}")

    # ============================================
    # Test 2: Environment with FIXED slippage
    # ============================================
    print("\n" + "=" * 60)
    print("TEST 2: Fixed slippage (5 bps) + spread (2 bps)")
    print("=" * 60)

    env_fixed = TradingEnvironment(
        features=features,
        prices=prices,
        initial_balance=config['environment']['initial_balance'],
        transaction_cost=config['environment']['transaction_cost'],
        slippage_model='fixed',
        slippage_base_bps=5.0,
        spread_bps=2.0,
    )

    # Use SAME random seed for fair comparison
    np.random.seed(42)
    obs, _ = env_fixed.reset()
    done = False
    while not done:
        action = env_fixed.action_space.sample()
        obs, reward, done, _, info = env_fixed.step(action)

    summary_fixed = env_fixed.get_execution_summary()
    final_fixed = info['total_value']

    print(f"  Final value:        ${final_fixed:,.2f}")
    print(f"  Trades:             {summary_fixed['n_trades']}")
    print(f"  Transaction costs:  ${summary_fixed['transaction_costs']:.2f}")
    print(f"  Slippage costs:     ${summary_fixed['slippage_costs']:.2f}")
    print(f"  Spread costs:       ${summary_fixed['spread_costs']:.2f}")
    print(f"  Total exec costs:   ${summary_fixed['total_execution_costs']:.2f}")
    print(f"  Avg cost/trade:     ${summary_fixed['avg_cost_per_trade']:.2f}")

    # ============================================
    # Test 3: Environment with PROPORTIONAL slippage
    # ============================================
    print("\n" + "=" * 60)
    print("TEST 3: Proportional slippage (volatility-scaled)")
    print("=" * 60)

    env_prop = TradingEnvironment(
        features=features,
        prices=prices,
        initial_balance=config['environment']['initial_balance'],
        transaction_cost=config['environment']['transaction_cost'],
        slippage_model='proportional',
        slippage_base_bps=5.0,
        spread_bps=2.0,
        volatility_slippage_mult=1.5,
    )

    np.random.seed(42)
    obs, _ = env_prop.reset()
    done = False
    while not done:
        action = env_prop.action_space.sample()
        obs, reward, done, _, info = env_prop.step(action)

    summary_prop = env_prop.get_execution_summary()
    final_prop = info['total_value']

    print(f"  Final value:        ${final_prop:,.2f}")
    print(f"  Trades:             {summary_prop['n_trades']}")
    print(f"  Transaction costs:  ${summary_prop['transaction_costs']:.2f}")
    print(f"  Slippage costs:     ${summary_prop['slippage_costs']:.2f}")
    print(f"  Spread costs:       ${summary_prop['spread_costs']:.2f}")
    print(f"  Total exec costs:   ${summary_prop['total_execution_costs']:.2f}")
    print(f"  Avg cost/trade:     ${summary_prop['avg_cost_per_trade']:.2f}")
    print(f"  Cost % of initial:  {summary_prop['cost_as_pct_initial']:.2f}%")

    # ============================================
    # Comparison
    # ============================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<25} {'Final Value':>12} {'Exec Costs':>12} {'Cost %':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*8}")
    print(f"  {'No slippage':<25} ${final_no_slip:>10,.2f} "
          f"${summary_no_slip['total_execution_costs']:>10.2f} "
          f"{summary_no_slip['cost_as_pct_initial']:>6.2f}%")
    print(f"  {'Fixed (5bp + 2bp spread)':<25} ${final_fixed:>10,.2f} "
          f"${summary_fixed['total_execution_costs']:>10.2f} "
          f"{summary_fixed['cost_as_pct_initial']:>6.2f}%")
    print(f"  {'Proportional (vol-scaled)':<25} ${final_prop:>10,.2f} "
          f"${summary_prop['total_execution_costs']:>10.2f} "
          f"{summary_prop['cost_as_pct_initial']:>6.2f}%")

    # Show first few execution log entries
    if env_prop.execution_log:
        print(f"\n  Sample execution log (proportional model):")
        for entry in env_prop.execution_log[:3]:
            slip_pct = (
                abs(entry['execution_price'] - entry['quoted_price']) /
                entry['quoted_price'] * 100
            )
            print(f"    Step {entry['step']:4d}: {entry['action']:4s} "
                  f"quoted=${entry['quoted_price']:.2f} "
                  f"exec=${entry['execution_price']:.2f} "
                  f"slip={slip_pct:.3f}% "
                  f"friction=${entry['total_friction']:.2f}")

    print("\n" + "=" * 60)
    print("ENVIRONMENT TEST COMPLETE!")
    print("=" * 60)

    return env_prop


if __name__ == "__main__":
    test_environment()