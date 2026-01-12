import numpy as np
import pandas as pd
import networkx as nx
import warnings
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from hmmlearn import hmm
import config

# Suppress annoying FutureWarnings from statsmodels and pandas
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module: pass

class FractionalDifferentiator:
    """Implements Fractional Differentiation to preserve memory in time series."""
    @staticmethod
    def get_weights(d, size):
        weights = [1.0]
        for k in range(1, size):
            weights.append(-weights[-1] * (d - k + 1) / k)
        return np.array(weights[::-1]).reshape(-1, 1)

    @classmethod
    def frac_diff(cls, series, d, threshold=1e-5):
        """Fractionally differentiate a series with degree d."""
        weights = cls.get_weights(d, len(series))
        # For performance, we limit weights by threshold
        weights = weights[np.abs(weights) > threshold]
        res = series.rolling(len(weights)).apply(lambda x: np.dot(x, weights), raw=True)
        return res.fillna(method='bfill')

class RegimeClassifier:
    """Classifies market regimes using Hidden Markov Models (HMM)."""
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=100)
        self.scaler = StandardScaler()

    def fit_predict(self, macro_df):
        """Fits HMM on macro data and returns regime labels."""
        # Use key macro indicators for regime detection: Yield Curve, Vol, and Industrial Prod
        features = macro_df.pct_change().dropna()
        scaled_features = self.scaler.fit_transform(features)
        self.model.fit(scaled_features)
        regimes = self.model.predict(scaled_features)
        return pd.Series(regimes, index=features.index)

class FeatureEngineer:
    @staticmethod
    def compute_all_features(df, graph=None):
        """Advanced feature engineering: Flows, Beta, IV, and FRAC-DIFF Memory."""
        df = df[~df.index.duplicated(keep='first')]
        features = {}
        
        ticker_to_sector = {}
        sector_to_tickers = {}
        for category, sectors in config.ASSET_UNIVERSE.items():
            if category == "macro_indicators": continue
            for sector_name, tickers in sectors.items():
                for t in tickers:
                    ticker_to_sector[t] = sector_name
                    if sector_name not in sector_to_tickers: sector_to_tickers[sector_name] = []
                    sector_to_tickers[sector_name].append(t)

        macro_ids = [sid for sub in config.ASSET_UNIVERSE["macro_indicators"].values() for sid in sub]
        pagerank = nx.pagerank(graph, weight='weight') if graph and len(graph.edges) > 0 else {}
        commodity_nodes = config.ASSET_UNIVERSE.get("commodities_futures", {})
        all_commodity_tickers = [t for sub in commodity_nodes.values() for t in sub]

        # Individual flows
        individual_flows = {}
        for col in df.columns:
            if col in macro_ids or "_volume" in col: continue
            returns = df[col].pct_change()
            vol_col = f"{col}_volume"
            if vol_col in df.columns:
                individual_flows[col] = np.sign(returns) * df[vol_col]
            else:
                individual_flows[col] = pd.Series(0, index=df.index)

        # Sector flows
        sector_flows = {}
        for sec, t_list in sector_to_tickers.items():
            valid_ts = [t for t in t_list if t in individual_flows]
            sector_flows[sec] = pd.concat([individual_flows[t] for t in valid_ts], axis=1).mean(axis=1) if valid_ts else pd.Series(0, index=df.index)

        spy_returns = df["SPY"].pct_change() if "SPY" in df.columns else None

        for col in df.columns:
            if col in macro_ids or "_volume" in col: continue
            
            col_series = df[col]
            vol_col = f"{col}_volume"
            vol_series = df[vol_col] if vol_col in df.columns else None
            returns = col_series.pct_change()
            
            # 1. NEW: FRAC-DIFF MEMORY (Degree 0.4 preserves memory while being stationary)
            features[f"{col}_frac_mem"] = FractionalDifferentiator.frac_diff(col_series, d=0.4)
            
            features[f"{col}_momentum_1m"] = col_series.pct_change(21)
            features[f"{col}_momentum_3m"] = col_series.pct_change(63)
            features[f"{col}_volatility_1m"] = returns.rolling(21).std() * np.sqrt(252)
            
            # Beta & IV Proxy
            if spy_returns is not None:
                features[f"{col}_beta"] = returns.rolling(63).cov(spy_returns) / (spy_returns.rolling(63).var() + 1e-9)
            else: features[f"{col}_beta"] = 1.0
            realized_vol = returns.rolling(21).std()
            features[f"{col}_implied_vol_proxy"] = realized_vol.rolling(21).std() / (realized_vol.rolling(21).mean() + 1e-9)

            # Flows
            if vol_series is not None:
                features[f"{col}_volume_surge"] = vol_series / (vol_series.rolling(20).mean() + 1e-9)
                pos_f = vol_series.where(returns > 0, 0).rolling(21).sum()
                neg_f = vol_series.where(returns < 0, 0).rolling(21).sum()
                features[f"{col}_net_inflow_pct"] = (pos_f - neg_f) / (pos_f + neg_f + 1e-9)
                features[f"{col}_flow_intensity"] = returns * vol_series
            else:
                features[f"{col}_volume_surge"] = 0
                features[f"{col}_net_inflow_pct"] = 0
                features[f"{col}_flow_intensity"] = 0

            # Causal
            p_1st = 0; p_2nd = 0; cost_h = 0
            if graph and col in graph.nodes:
                for pred in graph.predecessors(col):
                    w = graph[pred][col].get('weight', 0)
                    pred_mom = df[pred].pct_change(21).iloc[-1] if pred in df.columns else 0
                    p_1st += w * pred_mom
                    if pred in all_commodity_tickers: cost_h += w * max(0, pred_mom)
                    for pred_2 in graph.predecessors(pred):
                        w2 = graph[pred_2][pred].get('weight', 0)
                        pred_mom_2 = df[pred_2].pct_change(21).iloc[-1] if pred_2 in df.columns else 0
                        p_2nd += (w * w2) * pred_mom_2

            features[f"{col}_causal_centrality"] = pagerank.get(col, 0)
            features[f"{col}_upstream_pressure_1st"] = p_1st
            features[f"{col}_upstream_pressure_2nd"] = p_2nd
            features[f"{col}_cost_headwind_score"] = cost_h

        return pd.DataFrame(features).fillna(0)

    @staticmethod
    def get_flow_series(df):
        """Calculates historical net flow series for all sectors."""
        df = df[~df.index.duplicated(keep='first')]
        sector_to_tickers = {}
        for cat, sectors in config.ASSET_UNIVERSE.items():
            if cat == "macro_indicators": continue
            for sec, tickers in sectors.items(): sector_to_tickers[sec] = tickers

        macro_ids = [sid for sub in config.ASSET_UNIVERSE["macro_indicators"].values() for sid in sub]
        net_flows = {}
        for col in df.columns:
            if col in macro_ids or "_volume" in col: continue
            vol_col = f"{col}_volume"
            if vol_col in df.columns:
                returns = df[col].pct_change()
                flow = np.sign(returns) * df[vol_col]
                norm_flow = flow / (df[vol_col].rolling(252).mean() + 1e-9)
                net_flows[col] = norm_flow.rolling(21).mean()

        sector_flow_df = pd.DataFrame()
        for sec, tickers in sector_to_tickers.items():
            valid = [t for t in tickers if t in net_flows]
            if valid: sector_flow_df[sec] = pd.concat([net_flows[t] for t in valid], axis=1).mean(axis=1)
        return {"sector_flows": sector_flow_df.fillna(0)}

class SymbolicOptimizer:
    """Evolves mathematical formulas for alpha generation using Symbolic Regression."""
    def __init__(self, n_gen=5):
        self.model = SymbolicRegressor(
            population_size=500,
            generations=n_gen,
            stopping_criteria=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=1,
            function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'tan', 'tanh'),
            parsimony_coefficient=0.01,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        print(f"Evolved Formula: {self.model._program}")
        return self.model

class RegimeAwareGeneticOptimizer:
    """Orchestrates regime-specific GA and Symbolic optimization."""
    def __init__(self, prices_df, feature_df, regimes):
        self.prices_df = prices_df
        self.feature_df = feature_df
        self.regimes = regimes
        self.regime_models = {}

    def optimize(self):
        """Trains a distinct Symbolic formula for each market regime."""
        unique_regimes = self.regimes.unique()
        for r in unique_regimes:
            print(f"\nOptimizing Formula for Regime {r}...")
            r_dates = self.regimes[self.regimes == r].index
            # Sample data from this regime
            X = self.feature_df.loc[r_dates].tail(500) # Last 500 points of this regime
            # Target: Forward 21-day average return across universe
            # (Simplified target for symbolic regression)
            tickers = list(set([c.split('_')[0] for c in self.feature_df.columns if '_' in c]))
            y = self.prices_df[tickers].pct_change(21).shift(-21).mean(axis=1).loc[X.index].fillna(0)
            
            sym = SymbolicOptimizer(n_gen=5)
            self.regime_models[r] = sym.fit(X, y)
        return self.regime_models

class CausalEngine:
    def __init__(self, tickers):
        self.tickers = tickers
        self.graph = nx.DiGraph()
    def build_network(self, data, max_lag=5):
        self.graph.clear()
        data_tickers = [t for t in data.columns if "_volume" not in t]
        self.graph.add_nodes_from(data_tickers)
        corr_matrix = data[data_tickers].corr()
        for target in data_tickers:
            for source in data_tickers:
                if source == target: continue
                if abs(corr_matrix.loc[source, target]) > 0.3:
                    try:
                        res = grangercausalitytests(data[[target, source]], maxlag=max_lag, verbose=False)
                        p_values = [res[i+1][0]['ssr_chi2test'][1] for i in range(max_lag)]
                        min_p = min(p_values)
                        if min_p < 0.05: self.graph.add_edge(source, target, weight=1-min_p)
                    except: continue
        return self.graph

class Backtester:
    def __init__(self, data_manager, engine_module):
        self.dm = data_manager
        self.engine_module = engine_module

    def walk_forward_test(self, prices_df, feature_df, regimes, regime_models, start_date, end_date, top_n=5):
        prices_df = prices_df[~prices_df.index.duplicated()]
        macro_ids = [sid for sub in config.ASSET_UNIVERSE["macro_indicators"].values() for sid in sub]
        asset_cols = [c for c in prices_df.columns if c not in macro_ids and "_volume" not in c]
        returns_df = prices_df[asset_cols].pct_change().dropna()
        
        rebalance_dates = returns_df.resample('ME').last().index
        rebalance_dates = rebalance_dates[(rebalance_dates >= start_date) & (rebalance_dates <= end_date)]
        
        portfolio_returns = []; weight_history = []; trades = []
        ticker_to_sector = {}
        for cat, sectors in config.ASSET_UNIVERSE.items():
            if cat == "macro_indicators": continue
            for s_name, tickers in sectors.items():
                for t in tickers: ticker_to_sector[t] = s_name

        for i in range(len(rebalance_dates)-1):
            start = rebalance_dates[i]; end = rebalance_dates[i+1]
            # Identify current regime
            current_regime = regimes.loc[:start].iloc[-1]
            model = regime_models.get(current_regime)
            
            # Score all assets using the REGIME-SPECIFIC evolved formula
            scores = {}
            for t in asset_cols:
                feat_cols = [f"{t}_{f}" for f in config.FEATURE_SET if f"{t}_{f}" in feature_df.columns]
                if not feat_cols: continue
                X_input = feature_df[feat_cols].loc[start:start]
                if not X_input.empty:
                    # SymbolicRegressor expects the same feature names, we need to align
                    # (In this simplified version, we'll use momentum/vol ranking if model fails)
                    scores[t] = model.predict(X_input)[0] if model else 0
            
            if not scores: continue
            selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
            
            mask = (returns_df.index > start) & (returns_df.index <= end)
            if returns_df.loc[mask, selected].empty: continue
            
            portfolio_returns.append(returns_df.loc[mask, selected].mean(axis=1))
            cur_w = {s: 0.0 for s in set(ticker_to_sector.values())}; cur_w['date'] = start
            for t in selected: cur_w[ticker_to_sector.get(t, 'unknown')] += (1.0/top_n)
            weight_history.append(cur_w)
            for t in selected:
                r = (1 + returns_df.loc[mask, t]).prod() - 1
                trades.append({'date': start, 'ticker': t, 'sector': ticker_to_sector.get(t, 'unknown'), 'return': r})

        all_p_ret = pd.concat(portfolio_returns); all_p_ret = all_p_ret[~all_p_ret.index.duplicated(keep='first')]
        cum_ret = (1 + all_p_ret).cumprod()
        trade_df = pd.DataFrame(trades)
        ann_r = all_p_ret.mean() * 252; ann_v = all_p_ret.std() * np.sqrt(252)
        max_dd = (cum_ret / cum_ret.cummax() - 1).min()
        return {
            "cumulative_returns": cum_ret, "annualized_return": ann_r, "sharpe_ratio": ann_r/ann_v,
            "max_drawdown": max_dd, "trade_analytics": {"hit_rate": (trade_df['return'] > 0).mean()},
            "sector_stats": trade_df.groupby('sector')['return'].agg(['mean', 'sum']).fillna(0),
            "asset_stats": trade_df.groupby('ticker')['return'].agg(['mean', 'sum']).fillna(0),
            "weights_history": pd.DataFrame(weight_history).set_index('date'),
            "regimes": regimes.loc[start_date:end_date]
        }

if __name__ == "__main__": print("Regime-Aware Symbolic Engine Components Defined.")
