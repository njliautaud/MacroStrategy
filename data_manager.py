import os
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import config

class DataManager:
    def __init__(self):
        self.fmp_key = config.FMP_API_KEY.strip()
        self.fred_key = config.FRED_API_KEY.strip()
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, source, symbol):
        return os.path.join(self.cache_dir, f"{source}_{symbol}.csv")

    def fetch_fmp_prices(self, symbol, start_date=config.START_DATE):
        """Fetch historical daily prices from FMP."""
        cache_path = self._get_cache_path("fmp_prices", symbol)
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, index_col="date", parse_dates=True)

        # Updated to use the 'stable' endpoint as per new FMP documentation
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={symbol}&from={start_date}&apikey={self.fmp_key}"
        response = requests.get(url)
        data = response.json()
        
        # The 'stable' endpoint returns a list directly
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            # Use 'close' as 'adjClose' isn't explicitly named in the stable list snippet, 
            # but usually 'close' in stable is adjusted or we can rename for consistency.
            if "close" in df.columns and "adjClose" not in df.columns:
                df["adjClose"] = df["close"]
            
            df.to_csv(cache_path)
            print(f"Successfully fetched {symbol}")
            return df
        else:
            error_msg = data.get('Error Message', 'Unknown error') if isinstance(data, dict) else 'Empty or invalid data'
            print(f"Error fetching {symbol}: {error_msg}")
        return pd.DataFrame()

    def fetch_fred_data(self, series_id, start_date=config.START_DATE):
        """Fetch economic data from FRED."""
        cache_path = self._get_cache_path("fred", series_id)
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.fred_key}&file_type=json&observation_start={start_date}"
        response = requests.get(url)
        data = response.json()
        
        if "observations" in data:
            df = pd.DataFrame(data["observations"])
            if df.empty:
                print(f"Warning: No FRED data for {series_id}")
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df.set_index("date", inplace=True)
            df = df[["value"]].rename(columns={"value": series_id})
            df.sort_index(inplace=True)
            df.to_csv(cache_path)
            print(f"Successfully fetched FRED series {series_id}")
            return df
        else:
            print(f"Error fetching FRED series {series_id}: {data.get('error_message', data)}")
        return pd.DataFrame()

    def get_consolidated_data(self):
        """Build a unified dataframe of all assets and indicators, including volumes."""
        all_prices = []
        all_volumes = []
        
        # Load Equities & ETFs
        for category, sectors in {**config.ASSET_UNIVERSE["equities"], **config.ASSET_UNIVERSE["etfs"]}.items():
            for symbol in sectors:
                df = self.fetch_fmp_prices(symbol)
                if not df.empty:
                    all_prices.append(df["adjClose"].rename(symbol))
                    if "volume" in df.columns:
                        all_volumes.append(df["volume"].rename(f"{symbol}_volume"))

        # Load Macro Indicators
        for series_id in config.ASSET_UNIVERSE["macro_indicators"].values():
            for sid in series_id:
                df = self.fetch_fred_data(sid)
                if not df.empty:
                    all_prices.append(df)

        # Merge all
        prices_df = pd.concat(all_prices, axis=1).sort_index().ffill().dropna(how="all")
        volumes_df = pd.concat(all_volumes, axis=1).sort_index().ffill().fillna(0)
        
        # Combine into one DF for easier handling in features
        final_df = pd.concat([prices_df, volumes_df], axis=1)
        return final_df

    def clean_data(self, df):
        """Basic data cleaning: handling missing values and outliers."""
        # Interpolate small gaps, then forward fill
        df = df.interpolate(method="time", limit=5).ffill().bfill()
        return df

if __name__ == "__main__":
    dm = DataManager()
    # Test fetch (will fail without keys but sets up structure)
    try:
        data = dm.get_consolidated_data()
        print(f"Data shape: {data.shape}")
        print(data.head())
    except Exception as e:
        print(f"Data fetch failed (likely API keys): {e}")
