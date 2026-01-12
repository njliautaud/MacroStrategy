import os

# API Keys - USER TO PROVIDE
FMP_API_KEY = "REDACTED_FMP_API_KEY"
FRED_API_KEY = "REDACTED_FRED_API_KEY"

# Data Directories
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# Standard Asset Universe
# Level 1: Asset Classes
# Level 2: Sectors / Macro Proxies
# Level 3: Individual Tickers / Commodities

ASSET_UNIVERSE = {
    "equities": {
        "tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "NFLX", "INTC", "CSCO"],
        "financials": ["JPM", "GS", "BAC", "MS", "WFC", "V", "MA", "AXP", "BLK", "BX", "C", "PYPL", "SPGI", "SCHW"],
        "energy": ["XOM", "CVX", "SLB", "COP", "MPC", "PSX", "VLO", "HES", "OXY", "HAL", "BKR"],
        "industrials": ["CAT", "HON", "GE", "LMT", "BA", "UPS", "UNP", "DE", "RTX", "MMM", "GD", "FDX"],
        "healthcare": ["JNJ", "PFE", "UNH", "ABBV", "LLY", "MRK", "TMO", "DHR", "ISRG", "GILD", "BMY", "AMGN"],
        "consumer": ["PG", "KO", "PEP", "COST", "WMT", "NKE", "MCD", "DIS", "HD", "LOW", "SBUX", "TJX"],
        "materials": ["LIN", "APD", "SHW", "FCX", "NEM", "NUE", "ECL", "CTVA", "DOW"],
        "utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL"],
        "real_estate": ["PLD", "AMT", "CCI", "EQIX", "WY", "PSA"],
    },
    "etfs": {
        "broad": ["SPY", "QQQ", "IWM", "DIA", "EFA", "VWO", "VEA", "TLT", "LQD", "HYG"],
        "sector": ["XLK", "XLF", "XLE", "XLI", "XLV", "XLP", "XLU", "XLB", "XLRE", "XLY", "XLC"],
        "commodity_proxies": ["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBE", "UGA", "WOOD", "REMX", "COPX", "LIT"],
    },
    "commodities_futures": {
        "metals": ["GC=F", "SI=F", "HG=F", "PL=F", "PA=F"], 
        "energy": ["CL=F", "NG=F", "RB=F", "HO=F"], 
        "agri": ["ZC=F", "ZS=F", "ZW=F", "KC=F", "CT=F", "CC=F", "SB=F", "OJ=F"],
    },
    "macro_indicators": {
        "rates": ["DGS10", "DGS2", "T10Y2Y", "FEDFUNDS", "DGS30", "T10YIE"], 
        "inflation": ["CPIAUCSL", "PCEPI", "WPSFD49207"],
        "industrial": ["INDPRO", "TCU", "DGORDER", "BUSINV"],
        "employment": ["UNRATE", "PAYEMS", "ICSA", "CIVPART"],
        "sentiment": ["UMCSENT", "CSCICP03USM665S"],
    }
}

# Decision Features for GA Optimization
FEATURE_SET = [
    "momentum_1m", "momentum_3m", 
    "volatility_1m",
    "relative_to_sector", "relative_to_market",
    "causal_centrality", "upstream_pressure_1st", "upstream_pressure_2nd",
    "macro_alignment", "cost_headwind_score",
    "flow_intensity", "volume_surge", "money_flow_index",
    "net_inflow_pct", "sector_flow_pressure",
    "beta", "implied_vol_proxy"
]

# Training / Backtest Configuration
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

TRAIN_SPLIT = "2020-01-01"
VALIDATION_SPLIT = "2022-01-01"

# Technical Parameters
ROLLING_WINDOWS = [21, 63, 126, 252] # 1m, 3m, 6m, 1y
SMOOTHING_FACTOR = 0.1
