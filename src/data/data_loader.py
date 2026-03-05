"""
Data Loader Module
Downloads and manages price and news data
"""

import os
import pandas as pd
import yfinance as yf
import yaml
from datetime import datetime


class Config:
    """Load and access configuration"""
    
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, *keys):
        """Get nested config value"""
        value = self._config
        for key in keys:
            value = value[key]
        return value
    
    @property
    def ticker(self):
        return self._config['data']['ticker']
    
    @property
    def start_date(self):
        return self._config['data']['start_date']
    
    @property
    def end_date(self):
        return self._config['data']['end_date']


class PriceDataLoader:
    """
    Download and load stock price data
    """
    
    def __init__(self, config_path="configs/config.yaml"):
        self.config = Config(config_path)
        self.ticker = self.config.ticker
        self.start_date = self.config.start_date
        self.end_date = self.config.end_date
        self.save_path = self.config.get('paths', 'raw_prices')
        
    def download(self, force=False):
        """
        Download price data from Yahoo Finance
        
        Args:
            force: If True, download even if file exists
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check if already exists
        if os.path.exists(self.save_path) and not force:
            print(f"Data already exists at {self.save_path}")
            print("Use force=True to re-download")
            return self.load()
        
        print(f"Downloading {self.ticker}...")
        print(f"Period: {self.start_date} to {self.end_date}")
        
        # Download from Yahoo Finance
        df = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            progress=True
        )
        
        # Check if data was downloaded
        if df.empty:
            raise ValueError(f"No data downloaded for {self.ticker}")
        
        # Fix column names (yfinance returns MultiIndex for single ticker)
        df = self._fix_columns(df)
        
        # Basic info
        print(f"\nDownload complete!")
        print(f"Total rows: {len(df)}")
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(self.save_path)
        print(f"Saved to: {self.save_path}")
        
        return df
    
    def _fix_columns(self, df):
        """
        Fix column names from yfinance
        Converts MultiIndex columns to simple column names
        """
        # Check if columns are MultiIndex (tuples)
        if isinstance(df.columns, pd.MultiIndex):
            # Take only the first level (e.g., 'Close' from ('Close', 'AAPL'))
            df.columns = df.columns.get_level_values(0)
        
        # Ensure standard column names
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check for Adj Close and rename if needed
        if 'Adj Close' in df.columns:
            # Use Adj Close as Close for accuracy
            df['Close'] = df['Adj Close']
        
        # Keep only needed columns
        cols_to_keep = [col for col in expected_cols if col in df.columns]
        df = df[cols_to_keep]
        
        return df
    
    def load(self):
        """
        Load existing price data from CSV
        
        Returns:
            DataFrame with OHLCV data
        """
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(
                f"No data found at {self.save_path}. "
                "Run download() first."
            )
        
        df = pd.read_csv(self.save_path, index_col=0, parse_dates=True)
        print(f"Loaded {len(df)} rows from {self.save_path}")
        
        return df
    
    def get_info(self, df=None):
        """
        Print information about the data
        
        Args:
            df: DataFrame (loads from file if not provided)
        """
        if df is None:
            df = self.load()
        
        print("\n" + "=" * 50)
        print("PRICE DATA SUMMARY")
        print("=" * 50)
        print(f"Ticker: {self.ticker}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"Trading days: {len(df)}")
        
        print("\nPrice Statistics:")
        print(f"  Open  - Min: ${df['Open'].min():.2f}, Max: ${df['Open'].max():.2f}")
        print(f"  Close - Min: ${df['Close'].min():.2f}, Max: ${df['Close'].max():.2f}")
        print(f"  Volume - Avg: {df['Volume'].mean():,.0f}")
        
        print("\nMissing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("  None")
        else:
            for col, count in missing.items():
                if count > 0:
                    print(f"  {col}: {count}")
        
        print("=" * 50)
        
        return df


def test_loader():
    """Test the data loader"""
    print("Testing PriceDataLoader...")
    
    loader = PriceDataLoader()
    
    # Download data (force=True to re-download with fixed columns)
    df = loader.download(force=True)
    
    # Show info
    loader.get_info(df)
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Show last few rows
    print("\nLast 5 rows:")
    print(df.tail())
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE - Data loader working!")
    print("=" * 50)


if __name__ == "__main__":
    test_loader()