import os

# API Keys (User provided)
FMP_API_KEY = "REDACTED_FMP_API_KEY" 
FRED_API_KEY = "REDACTED_FRED_API_KEY "

# Data Settings
START_DATE = "2010-01-01"
# Walk-forward protocol:
# - Everything before TRAIN_END is "fair game" for model development.
# - HOLDOUT_START..HOLDOUT_END is the true walk-forward period shown in the dashboard.
TRAIN_END = "2022-12-31"
HOLDOUT_START = "2023-01-01"
HOLDOUT_END = "2025-12-31"
END_DATE = "2026-01-13"

# Retrain cadence for the symbolic/regime models during holdout (months).
# 1 = retrain every rebalance; 3 = quarterly; 6 = semiannual.
RETRAIN_EVERY_N_MONTHS = 3
CACHE_DIR = "cache"

# Alpha objective shaping (higher positive alpha gets extra weight)
ALPHA_POS_BONUS = 2.0
ALPHA_CLIP = 0.5

# Expanded Asset Universe
ASSET_UNIVERSE = {
    "equities": {
        "tech": [
            "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "ORCL", "CRM", "ADBE",
            "AMD", "NFLX", "INTC", "CSCO", "QCOM", "TXN", "MU", "PANW", "SNOW", "AMAT", "LRCX", "KLAC",
            # Mid-cap / high-growth tech
            "NOW", "MDB", "DDOG", "NET", "CRWD", "ZS", "OKTA", "TEAM", "SHOP", "U", "SQ", "PLTR"
        ],
        "financials": [
            "JPM", "GS", "BAC", "MS", "WFC", "V", "MA", "AXP", "BLK", "BX", "C", "PYPL", "SPGI", "SCHW",
            "CB", "MMC", "AON", "PGR", "MET", "PRU", "TROW",
            # Mid-cap / regional / insurers
            "KMX", "FITB", "HBAN", "KEY", "RF", "CMA", "PNC", "TRV", "AIG", "ALL"
        ],
        "energy": [
            "XOM", "CVX", "SLB", "COP", "MPC", "PSX", "VLO", "HES", "OXY", "HAL", "BKR", "KMI", "WMB",
            "DVN", "FANG", "EOG", "PXD", "CTRA",
            # Mid-cap / services
            "OKE", "ET", "LNG", "APA", "MRO"
        ],
        "industrials": [
            "CAT", "HON", "GE", "LMT", "BA", "UPS", "UNP", "DE", "RTX", "MMM", "GD", "FDX", "NSC", "EMR",
            "CP", "CNI", "WM", "RSG", "TT", "ETN",
            # Mid-cap industrials
            "ROK", "IR", "CMI", "PH", "ITW", "PCAR"
        ],
        "healthcare": [
            "JNJ", "PFE", "UNH", "ABBV", "LLY", "MRK", "TMO", "DHR", "ISRG", "GILD", "BMY", "AMGN",
            "CVS", "SYK", "ZTS", "BDX", "BSX", "VRTX",
            # Mid-cap biotech / medtech
            "REGN", "HCA", "IQV", "ILMN", "EW"
        ],
        "consumer": [
            "PG", "KO", "PEP", "COST", "WMT", "NKE", "MCD", "DIS", "HD", "LOW", "SBUX", "TJX", "EL",
            "CL", "MDLZ", "PM", "MO", "TGT", "DG",
            # Mid-cap / discretionary
            "ROST", "YUM", "DRI", "LULU", "ETSY", "CHWY"
        ],
        "materials": [
            "LIN", "APD", "SHW", "FCX", "NEM", "NUE", "ECL", "CTVA", "DOW", "DD", "ALB", "MLM", "VMC",
            "CF", "MOS",
            # Mid-cap materials / chemicals / miners
            "IFF", "EMN", "CE", "BLL", "LYB", "SSRM"
        ],
        "utilities": [
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "WEC", "ES", "DTE", "FE", "ETR",
            # Mid-cap utilities
            "AWK", "CMS", "PPL"
        ],
        "real_estate": [
            "PLD", "AMT", "CCI", "EQIX", "WY", "PSA", "DLR", "O", "VICI", "CBRE", "AVB", "EQR",
            # Mid-cap REITs
            "EXR", "WELL", "SPG", "ARE", "IRM"
        ],
    },
    "etfs": {
        "broad": ["SPY", "QQQ", "IWM", "DIA", "EFA", "VWO", "VEA", "TLT", "LQD", "HYG", "SHV", "BIL"],
        "sector": ["XLK", "XLF", "XLE", "XLI", "XLV", "XLP", "XLU", "XLB", "XLRE", "XLY", "XLC"],
        "industries": [
            # Tech / semis / software
            "SMH", "SOXX", "IGV", "SKYY", "SOCL", "CLOU",
            # Health / biotech
            "XBI", "IBB", "IHI", "ARKG",
            # Financials
            "KRE", "KBE", "IAI",
            # Industrials / defense / transports
            "XLI", "ITA", "IYT",
            # Consumer
            "XHB", "ITB", "XRT", "FDIS",
            # Energy / materials
            "XOP", "XME", "TAN", "PBW",
            # Real estate / infra
            "VNQ", "PAVE",
            # Telecom
            "XTL",
            # Additional industries / themes
            "BOTZ", "ROBO", "HACK", "FINX", "SKYY", "KWEB"
        ],
        "commodity_proxies": [
            # Metals
            "GLD", "SLV", "PPLT", "PALL", "GDX", "GDXJ",
            # Energy
            "USO", "BNO", "UNG", "UGA", "USL",
            # Broad baskets / industrial metals
            "DBC", "PDBC", "DBA", "DBB", "DBE",
            # Thematic commodity exposure
            "WOOD", "REMX", "COPX", "LIT", "URA",
            # Softs / ag proxies
            "CORN", "SOYB", "WEAT", "CANE", "JO", "BAL",
            # Additional commodities / themes
            "GSG", "COMB", "SGOL"
        ],
    },
    "commodities_futures": {
        "precious_metals": ["GC=F", "SI=F", "PL=F", "PA=F"],
        # Keep this set conservative (high-liquidity symbols that are most likely to exist in data feeds)
        "industrial_metals": ["HG=F", "SI=F"],
        "energy": ["CL=F", "NG=F", "RB=F", "HO=F", "BZ=F"],
        "agri_grains": ["ZC=F", "ZS=F", "ZW=F", "ZM=F", "ZL=F"],
        "agri_softs": ["KC=F", "CT=F", "CC=F", "SB=F", "OJ=F"],
        "livestock": ["LE=F", "GF=F", "HE=F"],
    },
    # FMP “Commodity Market Data” symbols (more reliable than '=F' tickers for FMP).
    # These are used for expanding commodity coverage (metals/energy/ag/rates).
    "commodities_fmp": {
        # NOTE: some symbols are subscription-gated (HTTP 402). We default to symbols observed to work,
        # and represent gated ones via liquid proxy ETFs in `etfs.commodity_proxies` (e.g., COPX for copper).
        "energy": ["BZUSD"],
        "precious_metals": ["GCUSD", "SIUSD"],
        "industrial_metals": ["HGUSD", "ALIUSD", "NIUSD", "ZNUSD", "XCUUSD"],
        "agri": ["SBUSD", "CTUSD", "CCUSD", "KCUSD", "ZCUSD", "ZSUSD", "ZWUSD"],
    },
    "macro_indicators": {
        "rates": ["DGS10", "DGS2", "T10Y2Y", "FEDFUNDS", "DGS30", "T10YIE", "T5YIE", "TEDRATE"],
        "inflation": ["CPIAUCSL", "PCEPI", "WPSFD49207", "PPIACO"],
        "industrial": ["INDPRO", "TCU", "DGORDER", "BUSINV", "MANEMP", "IPMAN"],
        "employment": ["UNRATE", "PAYEMS", "ICSA", "CIVPART", "JTSJOL"],
        # NFIB series IDs vary; keep the set to known-good FRED series to avoid breaking runs.
        "sentiment": ["UMCSENT", "CSCICP03USM665S"],
        "housing": ["HOUST", "HSN1F", "PERMIT"],
    }
}

# Commodity flow/volume proxies (used when futures volume is missing/unreliable).
# Keys can be futures symbols (e.g., "CL=F") or commodity-related tickers; values are liquid proxy ETFs.
COMMODITY_FLOW_PROXY = {
    "CL=F": "USO",
    "BZ=F": "BNO",
    "NG=F": "UNG",
    "GC=F": "GLD",
    "SI=F": "SLV",
    "HG=F": "COPX",
    "HGUSD": "COPX",
    "ZC=F": "CORN",
    "ZCUSD": "CORN",
    "ZS=F": "SOYB",
    "ZSUSD": "SOYB",
    "ZW=F": "WEAT",
    "ZWUSD": "WEAT",
    "CC=F": "CC",   # cocoa proxy is limited; may be absent in some feeds
    "KC=F": "JO",
    "CT=F": "BAL",
    "SB=F": "CANE",
    "ZM=F": "DBA",
    "ZL=F": "DBA",
    "HGUSD": "COPX",
    "ALIUSD": "DBB",
    "NIUSD": "DBB",
    "ZNUSD": "DBB",
    "XCUUSD": "COPX",
    "CCUSD": "BAL",
    "KCUSD": "JO",
}

# Commodity proxy groups (for reporting/flow aggregation)
COMMODITY_PROXY_GROUPS = {
    "metals": ["GLD", "SLV", "PPLT", "PALL", "GDX", "GDXJ", "COPX", "REMX"],
    "energy": ["USO", "BNO", "UNG", "UGA", "USL"],
    "broad": ["DBC", "PDBC", "DBB", "DBE"],
    "agri": ["DBA", "CORN", "SOYB", "WEAT", "CANE", "JO", "BAL"],
    "batteries": ["LIT"],
    "nuclear": ["URA"],
    "timber": ["WOOD"],
}

# Cost-input baskets for 2nd-order effects (commodity cost pressure -> equity sectors)
# Each sector maps to a list of (commodity_symbol, weight). These are *directional cost headwinds*.
COST_INPUT_BASKETS = {
    "tech": [("HG=F", 0.45), ("CL=F", 0.35), ("PA=F", 0.20)],
    "industrials": [("HG=F", 0.50), ("CL=F", 0.50)],
    "consumer": [("CL=F", 0.40), ("ZC=F", 0.20), ("ZS=F", 0.20), ("KC=F", 0.20)],
    "materials": [("CL=F", 0.50), ("NG=F", 0.50)],
    "utilities": [("NG=F", 0.70), ("CL=F", 0.30)],
    "energy": [("CL=F", -0.60), ("NG=F", -0.40)],  # producers benefit from higher inputs (negative headwind)
}

# Advanced Decision Features
FEATURE_SET = [
    "frac_mem", "momentum_1m", "momentum_3m", 
    "volatility_1m", "beta", "implied_vol_proxy",
    "flow_intensity", "volume_surge", "net_inflow_pct",
    "inflow_1m", "outflow_1m", "flow_acceleration",
    "causal_centrality", "upstream_pressure_1st", "upstream_pressure_2nd",
    "macro_alignment", "cost_headwind_score",
    "rsi_14", "atr_ratio", "rel_strength_idx",
    "sector_rotation_drift", "commodity_drag",
    "supply_demand_proxy",
    "flow_magnitude_divergence",
    "fundamental_pressure_index",
    "second_order_impact",
    # OHLC/VWAP-derived features (available for equities/ETFs/commodities via EOD full)
    "atr_14", "range_pct", "gap_pct", "vwap_dev"
]
