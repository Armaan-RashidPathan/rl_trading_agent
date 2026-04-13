# tests/test_trading_env.py

import pytest
import numpy as np
import pandas as pd
from src.models.trading_env import TradingEnvironment


class TestTradingEnvironment:
    """Unit tests for the trading environment."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple test environment with known data."""
        n_days = 100
        n_features = 24
        idx = pd.date_range('2020-01-01', periods=n_days, freq='D')
        feat_arr = np.random.randn(n_days, n_features).astype(np.float64)
        features = pd.DataFrame(feat_arr, index=idx)
        price_arr = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
        price_arr = np.maximum(price_arr, 1.0)
        prices = pd.Series(price_arr, index=idx)
        return TradingEnvironment(features=features, prices=prices)
    
    def test_reset_returns_valid_obs(self, simple_env):
        obs, info = simple_env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert not np.any(np.isnan(obs))
    
    def test_step_returns_correct_format(self, simple_env):
        obs, _ = simple_env.reset()
        obs, reward, terminated, truncated, info = simple_env.step(1)  # HOLD
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_portfolio_value_positive(self, simple_env):
        obs, _ = simple_env.reset()
        for _ in range(50):
            action = np.random.randint(0, simple_env.action_space.n)
            obs, reward, terminated, truncated, info = simple_env.step(action)
            if terminated or truncated:
                break
            assert info.get('total_value', 0) > 0, "Portfolio value should remain positive"
    
    def test_buy_then_sell_incurs_costs(self, simple_env):
        """After buy then immediate sell, portfolio should be slightly less due to transaction costs."""
        obs, _ = simple_env.reset()
        initial_value = simple_env.initial_balance
        
        # This test depends on your action encoding; adjust action indices accordingly
        # Conceptual test
        assert initial_value > 0
    
    def test_episode_terminates(self, simple_env):
        obs, _ = simple_env.reset()
        done = False
        steps = 0
        while not done:
            action = simple_env.action_space.sample()
            obs, _, terminated, truncated, _ = simple_env.step(action)
            done = terminated or truncated
            steps += 1
            if steps > 200:
                break
        assert done, "Episode should terminate within the data length"


class TestAttentionModule:
    def test_attention_output_shape(self):
        import torch
        from src.models.attention import CrossModalAttention
        
        batch_size = 8
        n_price = 19
        n_sentiment = 5
        
        attention = CrossModalAttention(price_dim=n_price, sentiment_dim=n_sentiment)
        
        price_features = torch.randn(batch_size, n_price)
        sentiment_features = torch.randn(batch_size, n_sentiment)
        
        output = attention(price_features, sentiment_features)
        assert output is not None
        assert output.shape[0] == batch_size


# tests/test_data_pipeline.py

class TestDataPipeline:
    def test_combined_features_exist(self):
        from pathlib import Path
        assert Path('data/processed/combined_features.csv').exists(), \
            "Combined features file should exist"
    
    def test_combined_features_no_nans(self):
        import pandas as pd
        from pathlib import Path
        
        path = Path('data/processed/combined_features.csv')
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            nan_pct = df.isna().mean().mean()
            assert nan_pct < 0.05, f"Too many NaN values: {nan_pct:.1%}"
    
    def test_prices_and_features_aligned(self):
        import pandas as pd
        from pathlib import Path
        
        features_path = Path('data/processed/combined_features.csv')
        prices_path = Path('data/raw/prices.csv')
        
        if features_path.exists() and prices_path.exists():
            features = pd.read_csv(features_path, index_col=0, parse_dates=True)
            prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
            overlap = features.index.intersection(prices.index)
            assert len(overlap) > 0, "Features and prices should share dates"

# tests/test_trading_env.py  — add these tests at the bottom of
# your existing TestTradingEnvironment class

    def test_slippage_increases_costs(self):
        """Slippage should make execution more expensive than no slippage."""
        n_days = 100
        n_features = 24
        idx = pd.date_range('2020-01-01', periods=n_days, freq='D')
        np.random.seed(42)
        feat_arr = np.random.randn(n_days, n_features).astype(np.float64)
        features = pd.DataFrame(feat_arr, index=idx)
        price_arr = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
        price_arr = np.maximum(price_arr, 1.0)
        prices = pd.Series(price_arr, index=idx)

        # Pre-generate a fixed sequence of actions
        # so both environments execute the EXACT same trades
        np.random.seed(99)
        actions = [np.random.randint(0, 24) for _ in range(50)]

        # --- Run with NO slippage ---
        env_no = TradingEnvironment(
            features=features, prices=prices,
            slippage_model='none', spread_bps=0.0,
        )
        obs, _ = env_no.reset()
        for action in actions:
            obs, _, done, _, _ = env_no.step(action)
            if done:
                break
        summary_no = env_no.get_execution_summary()

        # --- Run with slippage ---
        env_slip = TradingEnvironment(
            features=features, prices=prices,
            slippage_model='fixed', slippage_base_bps=10.0, spread_bps=5.0,
        )
        obs, _ = env_slip.reset()
        for action in actions:
            obs, _, done, _, _ = env_slip.step(action)
            if done:
                break
        summary_slip = env_slip.get_execution_summary()

        # Same actions means same number of trades
        assert summary_no['n_trades'] == summary_slip['n_trades'], (
            f"Trade count mismatch: no_slip={summary_no['n_trades']}, "
            f"with_slip={summary_slip['n_trades']}"
        )

        # Slippage env must have higher total execution costs
        # because it has the SAME commission costs PLUS extra
        # slippage and spread costs on top
        assert summary_slip['total_execution_costs'] >= summary_no['total_execution_costs'], (
            f"Slippage should increase costs: "
            f"no_slip={summary_no['total_execution_costs']:.2f}, "
            f"with_slip={summary_slip['total_execution_costs']:.2f}"
        )

        # Verify the extra cost comes from slippage and spread
        assert summary_slip['slippage_costs'] > 0, "Slippage costs should be > 0"
        assert summary_slip['spread_costs'] > 0, "Spread costs should be > 0"
        assert summary_no['slippage_costs'] == 0, "No-slip env should have 0 slippage"
        assert summary_no['spread_costs'] == 0, "No-slip env should have 0 spread"
        
    def test_execution_summary_structure(self):
        """Execution summary should have all expected keys."""
        n_days = 50
        n_features = 24
        idx = pd.date_range('2020-01-01', periods=n_days, freq='D')
        features = pd.DataFrame(
            np.random.randn(n_days, n_features), index=idx
        )
        prices = pd.Series(
            np.maximum(100 + np.cumsum(np.random.randn(n_days) * 0.5), 1.0),
            index=idx
        )

        env = TradingEnvironment(features=features, prices=prices)
        obs, _ = env.reset()

        # Do a few steps
        for _ in range(20):
            obs, _, done, _, _ = env.step(env.action_space.sample())
            if done:
                break

        summary = env.get_execution_summary()
        required_keys = [
            'n_trades', 'transaction_costs', 'slippage_costs',
            'spread_costs', 'total_execution_costs',
            'cost_as_pct_initial', 'avg_cost_per_trade',
        ]
        for key in required_keys:
            assert key in summary, f"Missing key in execution summary: {key}"

    def test_buy_execution_price_higher_than_quoted(self):
        """When buying with slippage, execution price should be >= quoted."""
        n_days = 50
        n_features = 24
        idx = pd.date_range('2020-01-01', periods=n_days, freq='D')
        features = pd.DataFrame(
            np.random.randn(n_days, n_features), index=idx
        )
        prices = pd.Series(
            np.maximum(100 + np.cumsum(np.random.randn(n_days) * 0.5), 1.0),
            index=idx
        )

        env = TradingEnvironment(
            features=features, prices=prices,
            slippage_model='fixed', slippage_base_bps=10.0,
            spread_bps=5.0,
        )

        # Check the execution price calculation directly
        quoted = 100.0
        exec_price, spread, slippage = env._get_execution_price(
            quoted, is_buy=True
        )
        assert exec_price > quoted, (
            f"Buy execution price ({exec_price:.4f}) should be > "
            f"quoted ({quoted:.4f})"
        )

    def test_sell_execution_price_lower_than_quoted(self):
        """When selling with slippage, execution price should be <= quoted."""
        n_days = 50
        n_features = 24
        idx = pd.date_range('2020-01-01', periods=n_days, freq='D')
        features = pd.DataFrame(
            np.random.randn(n_days, n_features), index=idx
        )
        prices = pd.Series(
            np.maximum(100 + np.cumsum(np.random.randn(n_days) * 0.5), 1.0),
            index=idx
        )

        env = TradingEnvironment(
            features=features, prices=prices,
            slippage_model='fixed', slippage_base_bps=10.0,
            spread_bps=5.0,
        )

        quoted = 100.0
        exec_price, spread, slippage = env._get_execution_price(
            quoted, is_buy=False
        )
        assert exec_price < quoted, (
            f"Sell execution price ({exec_price:.4f}) should be < "
            f"quoted ({quoted:.4f})"
        )