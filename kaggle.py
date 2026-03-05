import pandas as pd

# Load both files
print("Loading files...")
df2 = pd.read_csv('data/raw/raw_analyst_ratings.csv')
df3 = pd.read_csv('data/raw/raw_partner_headlines.csv')

# Filter AAPL
aapl2 = df2[df2['stock'] == 'AAPL'].copy()
aapl3 = df3[df3['stock'] == 'AAPL'].copy()

print(f"AAPL rows in File 2: {len(aapl2)}")
print(f"AAPL rows in File 3: {len(aapl3)}")

# Fix date parsing (use format='mixed' to handle different formats)
aapl2['date'] = pd.to_datetime(aapl2['date'], format='mixed', utc=True)
aapl3['date'] = pd.to_datetime(aapl3['date'], format='mixed', utc=True)

# Remove timezone info for simplicity
aapl2['date'] = aapl2['date'].dt.tz_localize(None)
aapl3['date'] = aapl3['date'].dt.tz_localize(None)

print(f"\nFile 2 AAPL date range:")
print(f"  From: {aapl2['date'].min()}")
print(f"  To:   {aapl2['date'].max()}")

print(f"\nFile 3 AAPL date range:")
print(f"  From: {aapl3['date'].min()}")
print(f"  To:   {aapl3['date'].max()}")

# Sample headlines
print("\nSample AAPL headlines from File 2:")
for h in aapl2['headline'].head(5).tolist():
    print(f"  - {h}")

print("\nSample AAPL headlines from File 3:")
for h in aapl3['headline'].head(5).tolist():
    print(f"  - {h}")

# Check yearly distribution
print("\nFile 2 - AAPL headlines per year:")
print(aapl2['date'].dt.year.value_counts().sort_index())

exit()