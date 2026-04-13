# test_finnhub.py  (save in project root, run directly)

"""
Diagnostic script to test Finnhub API directly.
Run this BEFORE changing the pipeline.
"""

import os
import requests
import json
from datetime import datetime, timedelta

def test_finnhub():
    api_key = os.environ.get('FINNHUB_API_KEY', '')
    
    print("=" * 60)
    print("  FINNHUB API DIAGNOSTIC")
    print("=" * 60)
    
    # Check 1: API key exists
    print(f"\n  [1] API Key Check:")
    if not api_key:
        print(f"      ✗ FINNHUB_API_KEY is NOT set")
        print(f"      Set it with: $env:FINNHUB_API_KEY = 'your_key'")
        return
    else:
        print(f"      ✓ Key found: {api_key[:8]}...{api_key[-4:]}")
        print(f"      Length: {len(api_key)} chars")
    
    # Check 2: Basic API connectivity
    print(f"\n  [2] API Connectivity:")
    try:
        resp = requests.get(
            'https://finnhub.io/api/v1/quote',
            params={'symbol': 'AAPL', 'token': api_key},
            timeout=10,
        )
        print(f"      Status code: {resp.status_code}")
        print(f"      Response: {resp.text[:200]}")
        
        if resp.status_code == 401:
            print(f"      ✗ INVALID API KEY — check your key at finnhub.io")
            return
        elif resp.status_code == 429:
            print(f"      ✗ RATE LIMITED — wait a minute and try again")
            return
        elif resp.status_code == 200:
            data = resp.json()
            print(f"      ✓ AAPL current price: ${data.get('c', 'N/A')}")
        else:
            print(f"      ? Unexpected status code")
            
    except Exception as e:
        print(f"      ✗ Connection error: {e}")
        return
    
    # Check 3: Company news endpoint — RECENT (last 7 days)
    print(f"\n  [3] Company News — Last 7 days:")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    url = 'https://finnhub.io/api/v1/company-news'
    params = {
        'symbol': 'AAPL',
        'from': start_date,
        'to': end_date,
        'token': api_key,
    }
    
    print(f"      URL: {url}")
    print(f"      Params: symbol=AAPL, from={start_date}, to={end_date}")
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        print(f"      Status: {resp.status_code}")
        print(f"      Content-Type: {resp.headers.get('Content-Type', 'unknown')}")
        print(f"      Response length: {len(resp.text)} chars")
        
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                print(f"      ✓ Got {len(data)} articles")
                if len(data) > 0:
                    art = data[0]
                    print(f"      First article:")
                    print(f"        headline: {art.get('headline', '')[:80]}")
                    print(f"        datetime: {art.get('datetime', '')}")
                    print(f"        source:   {art.get('source', '')}")
                    
                    dt = datetime.fromtimestamp(art.get('datetime', 0))
                    print(f"        parsed date: {dt.strftime('%Y-%m-%d %H:%M')}")
                else:
                    print(f"      ✗ Empty list — no articles in this period")
            else:
                print(f"      ✗ Unexpected response type: {type(data)}")
                print(f"      Response: {str(data)[:300]}")
        else:
            print(f"      ✗ Error response: {resp.text[:300]}")
            
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # Check 4: Company news — 30-day chunk (the way pipeline uses it)
    print(f"\n  [4] Company News — 30-day chunk (2024-01-01 to 2024-01-31):")
    params_hist = {
        'symbol': 'AAPL',
        'from': '2024-01-01',
        'to': '2024-01-31',
        'token': api_key,
    }
    
    try:
        import time
        time.sleep(1)  # rate limit safety
        
        resp = requests.get(url, params=params_hist, timeout=15)
        print(f"      Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                print(f"      ✓ Got {len(data)} articles for Jan 2024")
                if len(data) > 0:
                    print(f"      Sample: {data[0].get('headline', '')[:80]}")
            else:
                print(f"      ✗ Not a list: {type(data)}")
                print(f"      {str(data)[:300]}")
        else:
            print(f"      ✗ Status {resp.status_code}: {resp.text[:300]}")
            
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # Check 5: Older historical data (2019)
    print(f"\n  [5] Company News — Historical (2019-06-01 to 2019-06-30):")
    params_old = {
        'symbol': 'AAPL',
        'from': '2019-06-01',
        'to': '2019-06-30',
        'token': api_key,
    }
    
    try:
        time.sleep(1)
        resp = requests.get(url, params=params_old, timeout=15)
        print(f"      Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                print(f"      ✓ Got {len(data)} articles for Jun 2019")
                if len(data) == 0:
                    print(f"      NOTE: Finnhub free tier may not have")
                    print(f"            data this far back. Check your plan.")
            else:
                print(f"      Response: {str(data)[:300]}")
                
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # Check 6: Test multiple tickers
    print(f"\n  [6] Multi-ticker test (recent 7 days):")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    for ticker in tickers:
        time.sleep(1.2)  # respect rate limit
        params_t = {
            'symbol': ticker,
            'from': start_date,
            'to': end_date,
            'token': api_key,
        }
        try:
            resp = requests.get(url, params=params_t, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                count = len(data) if isinstance(data, list) else 0
                print(f"      {ticker}: {count} articles")
            else:
                print(f"      {ticker}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"      {ticker}: Error — {e}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"  DIAGNOSTIC COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  If all checks pass but pipeline still gets 0:")
    print(f"    → The pipeline code has a bug in its chunking loop")
    print(f"    → Share the output of this script")
    print(f"\n  If check 3-5 return 0 articles:")
    print(f"    → Finnhub free tier may have limited history")
    print(f"    → Try upgrading or use Kaggle as primary source")


if __name__ == '__main__':
    test_finnhub()