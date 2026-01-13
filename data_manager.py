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
        self._fmp_key_invalid = False
        self._fred_key_invalid = False

    def _get_cache_path(self, source, symbol):
        return os.path.join(self.cache_dir, f"{source}_{symbol}.csv")

    def fetch_fmp_prices(self, symbol, start_date=config.START_DATE):
        """Fetch historical daily prices (OHLCV + VWAP when available) from FMP."""
        cache_path = self._get_cache_path("fmp_prices", symbol)
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, index_col="date", parse_dates=True)

        # If we've already detected an invalid key, don't spam the API (or the console).
        if self._fmp_key_invalid:
            return pd.DataFrame()

        # Use EOD full to get OHLC + volume + vwap (enables richer features).
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={symbol}&from={start_date}&apikey={self.fmp_key}"
        response = requests.get(url)
        try:
            data = response.json()
        except Exception:
            # Sometimes FMP responds with HTML (transient) or empty body; don't crash the whole run.
            snippet = (response.text or "")[:200].replace("\n", " ")
            print(f"Warning: Non-JSON response for {symbol} (status={response.status_code}). Head: {snippet}")
            return pd.DataFrame()
        
        # The 'stable' endpoint returns a list directly
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            # Normalize naming for downstream code
            if "close" in df.columns and "adjClose" not in df.columns:
                df["adjClose"] = df["close"]
            # Some symbols may not have volume/vwap; keep columns if present
            
            df.to_csv(cache_path)
            print(f"Successfully fetched {symbol}")
            return df
        else:
            error_msg = data.get('Error Message', 'Unknown error') if isinstance(data, dict) else 'Empty or invalid data'
            # Detect invalid key early and stop further calls to reduce noise
            if isinstance(error_msg, str) and "Invalid API KEY" in error_msg:
                self._fmp_key_invalid = True
                print("CRITICAL: FMP API key appears invalid. Update `config.FMP_API_KEY` (or set env var `FMP_API_KEY`) and re-run.")
            else:
                print(f"Error fetching {symbol}: {error_msg}")
        return pd.DataFrame()

    def fetch_fred_data(self, series_id, start_date=config.START_DATE):
        """Fetch economic data from FRED."""
        cache_path = self._get_cache_path("fred", series_id)
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

        if self._fred_key_invalid:
            return pd.DataFrame()

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
            msg = data.get('error_message', data)
            if isinstance(msg, str) and ("Invalid API key" in msg or "api_key" in msg and "invalid" in msg.lower()):
                self._fred_key_invalid = True
                print("CRITICAL: FRED API key appears invalid. Update `config.FRED_API_KEY` (or set env var `FRED_API_KEY`) and re-run.")
            else:
                print(f"Error fetching FRED series {series_id}: {msg}")
        return pd.DataFrame()

    def get_consolidated_data(self):
        """Build a unified dataframe of all assets and indicators, including volumes."""
        all_prices = []
        all_volumes = []
        all_aux = []  # open/high/low/vwap for better features
        seen = set()
        
        # 1. Load Equities, ETFs, and Commodities from FMP
        asset_groups = {
            **config.ASSET_UNIVERSE["equities"],
            **config.ASSET_UNIVERSE["etfs"],
            **config.ASSET_UNIVERSE["commodities_futures"],
            **config.ASSET_UNIVERSE.get("commodities_fmp", {}),
        }
        for category, symbols in asset_groups.items():
            for symbol in symbols:
                if symbol in seen:
                    continue
                seen.add(symbol)
                df = self.fetch_fmp_prices(symbol)
                if not df.empty:
                    all_prices.append(df["adjClose"].rename(symbol))
                    if "volume" in df.columns:
                        all_volumes.append(df["volume"].rename(f"{symbol}_volume"))
                    # Add OHLC/VWAP when available
                    for k in ("open", "high", "low", "vwap"):
                        if k in df.columns:
                            all_aux.append(df[k].rename(f"{symbol}_{k}"))

        # 2. Load Macro Indicators from FRED
        for category, series_ids in config.ASSET_UNIVERSE["macro_indicators"].items():
            for sid in series_ids:
                df = self.fetch_fred_data(sid)
                if not df.empty:
                    all_prices.append(df)

        if not all_prices:
            print("CRITICAL: No data fetched. Check API keys and internet connection.")
            return pd.DataFrame()

        # 3. Merge and Clean
        prices_df = pd.concat(all_prices, axis=1).sort_index().ffill().dropna(how="all")
        volumes_df = pd.concat(all_volumes, axis=1).reindex(prices_df.index).ffill().fillna(0) if all_volumes else pd.DataFrame()
        aux_df = pd.concat(all_aux, axis=1).reindex(prices_df.index).ffill().fillna(0) if all_aux else pd.DataFrame()
        
        return pd.concat([prices_df, volumes_df, aux_df], axis=1).dropna(subset=prices_df.columns, how='all')

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
