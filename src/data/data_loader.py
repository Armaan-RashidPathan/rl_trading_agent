"""
Data Loader Module
Downloads and manages price and news data
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import yaml


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


# ============================================
# PRICE DATA
# ============================================

class PriceDataLoader:
    """Download and load stock price data"""

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
        if os.path.exists(self.save_path) and not force:
            print(f"Data already exists at {self.save_path}")
            print("Use force=True to re-download")
            return self.load()

        print(f"Downloading {self.ticker}...")
        print(f"Period: {self.start_date} to {self.end_date}")

        df = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            progress=True
        )

        if df.empty:
            raise ValueError(f"No data downloaded for {self.ticker}")

        # Fix column names
        df = self._fix_columns(df)

        print(f"\nDownload complete!")
        print(f"Total rows: {len(df)}")
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        df.to_csv(self.save_path)
        print(f"Saved to: {self.save_path}")

        return df

    def _fix_columns(self, df):
        """Fix MultiIndex columns from yfinance"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']

        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        cols_to_keep = [col for col in expected_cols if col in df.columns]
        df = df[cols_to_keep]

        return df

    def load(self):
        """Load existing price data from CSV"""
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(
                f"No data found at {self.save_path}. "
                "Run download() first."
            )

        df = pd.read_csv(self.save_path, index_col=0, parse_dates=True)
        print(f"Loaded {len(df)} rows from {self.save_path}")
        return df

    def get_info(self, df=None):
        """Print information about the price data"""
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


# ============================================
# NEWS DATA
# ============================================

class NewsDataLoader:
    """
    Load and process financial news data
    Combines real Kaggle data with sample data for missing periods
    """

    def __init__(self, config_path="configs/config.yaml"):
        self.config = Config(config_path)
        self.ticker = self.config.ticker
        self.start_date = pd.to_datetime(self.config.start_date)
        self.end_date = pd.to_datetime(self.config.end_date)
        self.save_path = self.config.get('paths', 'raw_news')

    def load_and_combine(self, price_df):
        """
        Main function:
        1. Load real Kaggle data
        2. Generate sample for missing periods
        3. Combine and save

        Args:
            price_df: Price DataFrame (for date alignment)

        Returns:
            Combined news DataFrame
        """
        print("\n" + "=" * 50)
        print("LOADING NEWS DATA")
        print("=" * 50)

        # Step 1: Load real Kaggle data
        real_news = self._load_kaggle_data()

        # Step 2: Generate sample for missing dates
        sample_news = self._generate_sample_data(price_df, real_news)

        # Step 3: Combine
        combined = self._combine(real_news, sample_news)

        # Step 4: Save
        self._save(combined)

        return combined

    def _load_kaggle_data(self):
        """Load real AAPL headlines from Kaggle dataset"""

        kaggle_path = "data/raw/raw_analyst_ratings.csv"

        if not os.path.exists(kaggle_path):
            print("Kaggle file not found, will use sample data only")
            return pd.DataFrame(columns=['date', 'headline', 'source'])

        print("\nStep 1: Loading real Kaggle headlines...")

        df = pd.read_csv(kaggle_path)

        # Filter for our ticker
        df = df[df['stock'] == self.ticker].copy()
        print(f"  Found {len(df)} real {self.ticker} headlines")

        # Fix dates
        df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True)
        df['date'] = df['date'].dt.tz_localize(None)

        # Keep only needed columns
        df = df[['date', 'headline']].copy()
        df['source'] = 'real'

        # Filter to our date range
        df = df[
            (df['date'] >= self.start_date) &
            (df['date'] <= self.end_date)
        ]

        # Clean headlines
        df['headline'] = df['headline'].astype(str).str.strip()
        df = df[df['headline'].str.len() > 10]

        # Sort
        df = df.sort_values('date').reset_index(drop=True)

        print(f"  After filtering: {len(df)} headlines")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

        return df

    def _generate_sample_data(self, price_df, real_news):
        """Generate sample headlines for dates not in real data"""

        print("\nStep 2: Generating sample data for missing periods...")

        # Find dates already covered
        if len(real_news) > 0:
            real_dates = set(real_news['date'].dt.date)
        else:
            real_dates = set()

        # Find missing trading dates
        all_dates = price_df.index
        missing_dates = [d for d in all_dates if d.date() not in real_dates]

        print(f"  Total trading days: {len(all_dates)}")
        print(f"  Days with real news: {len(real_dates)}")
        print(f"  Days needing sample: {len(missing_dates)}")

        if not missing_dates:
            print("  No sample data needed!")
            return pd.DataFrame(columns=['date', 'headline', 'source'])

        # News templates
        positive_news = [
            f"{self.ticker} reports record quarterly earnings",
            f"{self.ticker} beats revenue expectations",
            f"Analysts upgrade {self.ticker} to strong buy",
            f"{self.ticker} announces major product launch",
            f"{self.ticker} expands into new markets",
            f"{self.ticker} stock hits all time high",
            f"Strong demand drives {self.ticker} sales growth",
            f"{self.ticker} raises full year guidance",
            f"Investors bullish on {self.ticker} outlook",
            f"{self.ticker} secures major new contract",
            f"Wall Street optimistic about {self.ticker} growth",
            f"{self.ticker} crushes earnings estimates again",
            f"Analysts raise price target on {self.ticker}",
            f"{self.ticker} sees record user growth"
        ]

        negative_news = [
            f"{self.ticker} misses earnings estimates",
            f"Analysts downgrade {self.ticker} stock",
            f"{self.ticker} faces regulatory investigation",
            f"{self.ticker} warns of supply chain disruptions",
            f"{self.ticker} revenue falls short of expectations",
            f"Concerns mount over {self.ticker} competition",
            f"{self.ticker} lowers annual guidance",
            f"Investors worried about {self.ticker} margins",
            f"{self.ticker} faces class action lawsuit",
            f"{self.ticker} stock drops on weak outlook",
            f"{self.ticker} loses market share to rivals",
            f"Analysts cut price target on {self.ticker}"
        ]

        neutral_news = [
            f"{self.ticker} CEO speaks at industry conference",
            f"{self.ticker} announces board meeting date",
            f"Analysts maintain hold rating on {self.ticker}",
            f"{self.ticker} releases software update",
            f"Market watches {self.ticker} closely",
            f"{self.ticker} files regular SEC report",
            f"Trading volume normal for {self.ticker}",
            f"Investors await {self.ticker} earnings report",
            f"{self.ticker} management presents at investor day",
            f"{self.ticker} announces share buyback program",
            f"Institutional investors increase {self.ticker} holdings",
            f"{self.ticker} partners with leading technology firm"
        ]

        # Get price returns for sentiment bias
        returns = price_df['Close'].pct_change()

        np.random.seed(42)
        sample_data = []

        for date in missing_dates:
            # Number of headlines per day
            n_headlines = np.random.randint(1, 5)

            # Get daily return for sentiment bias
            if date in returns.index:
                daily_return = returns.loc[date]
                if pd.isna(daily_return):
                    daily_return = 0
            else:
                daily_return = 0

            # Bias based on price movement
            if daily_return > 0.02:
                weights = [0.60, 0.15, 0.25]
            elif daily_return < -0.02:
                weights = [0.15, 0.60, 0.25]
            else:
                weights = [0.33, 0.27, 0.40]

            for _ in range(n_headlines):
                category = np.random.choice(
                    ['positive', 'negative', 'neutral'],
                    p=weights
                )

                if category == 'positive':
                    headline = np.random.choice(positive_news)
                elif category == 'negative':
                    headline = np.random.choice(negative_news)
                else:
                    headline = np.random.choice(neutral_news)

                sample_data.append({
                    'date': pd.Timestamp(date),
                    'headline': headline,
                    'source': 'sample'
                })

        sample_df = pd.DataFrame(sample_data)
        print(f"  Generated {len(sample_df)} sample headlines")

        return sample_df

    def _combine(self, real_news, sample_news):
        """Combine real and sample news"""

        print("\nStep 3: Combining real and sample data...")

        combined = pd.concat([real_news, sample_news], ignore_index=True)
        combined = combined.sort_values('date').reset_index(drop=True)

        real_count = len(combined[combined['source'] == 'real'])
        sample_count = len(combined[combined['source'] == 'sample'])

        print(f"  Real headlines:   {real_count}")
        print(f"  Sample headlines: {sample_count}")
        print(f"  Total headlines:  {len(combined)}")

        return combined

    def _save(self, df):
        """Save to CSV"""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        df.to_csv(self.save_path, index=False)
        print(f"  Saved to: {self.save_path}")

    def load(self):
        """Load existing news data"""
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(
                f"No news data at {self.save_path}. "
                "Run load_and_combine() first."
            )
        df = pd.read_csv(self.save_path, parse_dates=['date'])
        print(f"Loaded {len(df)} headlines from {self.save_path}")
        return df

    def get_info(self, df=None):
        """Print news data summary"""
        if df is None:
            df = self.load()

        print("\n" + "=" * 50)
        print("NEWS DATA SUMMARY")
        print("=" * 50)
        print(f"Ticker: {self.ticker}")
        print(f"Total headlines: {len(df)}")

        if 'source' in df.columns:
            real = len(df[df['source'] == 'real'])
            sample = len(df[df['source'] == 'sample'])
            print(f"Real headlines:   {real} ({real/len(df)*100:.1f}%)")
            print(f"Sample headlines: {sample} ({sample/len(df)*100:.1f}%)")

        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

        print(f"\nHeadlines per year:")
        yearly = df.groupby(df['date'].dt.year).size()
        for year, count in yearly.items():
            if 'source' in df.columns:
                year_real = len(df[
                    (df['date'].dt.year == year) &
                    (df['source'] == 'real')
                ])
                tag = f" ({year_real} real)" if year_real > 0 else " (sample)"
            else:
                tag = ""
            print(f"  {year}: {count} headlines{tag}")

        print(f"\nSample real headlines:")
        if 'source' in df.columns:
            real_samples = df[df['source'] == 'real']['headline'].head(3).tolist()
        else:
            real_samples = df['headline'].head(3).tolist()

        for h in real_samples:
            print(f"  - {h}")

        print("=" * 50)


# ============================================
# NEWS ALIGNER
# ============================================

class NewsAligner:
    """Align news headlines to trading days"""

    def align(self, price_df, news_df):
        """
        Align news to trading days
        Uses PREVIOUS day news to prevent look-ahead bias

        Args:
            price_df: Price DataFrame
            news_df: News DataFrame

        Returns:
            DataFrame with headlines per trading day
        """
        print("\nAligning news to trading days...")
        print("(Using previous day news - prevents look-ahead bias)")

        trading_days = price_df.index
        aligned_data = []

        for trade_date in trading_days:
            prev_date = trade_date - pd.Timedelta(days=1)

            day_news = news_df[
                news_df['date'].dt.date == prev_date.date()
            ]['headline'].tolist()

            aligned_data.append({
                'date': trade_date,
                'headlines': day_news,
                'headline_count': len(day_news)
            })

        aligned_df = pd.DataFrame(aligned_data)
        aligned_df.set_index('date', inplace=True)

        total = len(aligned_df)
        with_news = (aligned_df['headline_count'] > 0).sum()
        without = (aligned_df['headline_count'] == 0).sum()

        print(f"  Total trading days:  {total}")
        print(f"  Days with news:      {with_news} ({with_news/total*100:.1f}%)")
        print(f"  Days without news:   {without} ({without/total*100:.1f}%)")
        print(f"  Avg headlines/day:   {aligned_df['headline_count'].mean():.1f}")

        return aligned_df


# ============================================
# TEST FUNCTION
# ============================================

def test_all():
    """Test complete data pipeline"""

    print("=" * 50)
    print("TESTING COMPLETE DATA PIPELINE")
    print("=" * 50)

    # Step 1: Price Data
    print("\n[1/3] PRICE DATA")
    price_loader = PriceDataLoader()
    prices = price_loader.load()
    price_loader.get_info(prices)

    # Step 2: News Data
    print("\n[2/3] NEWS DATA")
    news_loader = NewsDataLoader()
    news = news_loader.load_and_combine(prices)
    news_loader.get_info(news)

    # Step 3: Align
    print("\n[3/3] ALIGNING NEWS TO TRADING DAYS")
    aligner = NewsAligner()
    aligned = aligner.align(prices, news)

    print("\nSample aligned data:")
    for date, row in aligned.head(5).iterrows():
        print(f"\n  Date: {date.date()}")
        print(f"  Headlines: {row['headline_count']}")
        if row['headlines']:
            print(f"  Example: {row['headlines'][0]}")

    print("\n" + "=" * 50)
    print("DATA PIPELINE COMPLETE!")
    print("Next Step: Feature Engineering")
    print("=" * 50)

    return prices, news, aligned


if __name__ == "__main__":
    test_all()