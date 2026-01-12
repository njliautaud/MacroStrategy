import numpy as np
import pandas as pd
import networkx as nx
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module: pass
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import config
import warnings

# Suppress annoying FutureWarnings from statsmodels and pandas
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class FeatureEngineer:
    @staticmethod
    def compute_all_features(df, graph=None):
        """Advanced feature engineering: Flows (In vs Out), Beta, IV Proxy, and Sector Aggregation."""
        df = df[~df.index.duplicated(keep='first')]
        features = {}
        
        # 1. Build Sector/Industry mappings
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

        # 2. Pre-calculate individual flows for aggregation
        individual_flows = {}
        for col in df.columns:
            if col in macro_ids or "_volume" in col: continue
            returns = df[col].pct_change()
            vol_col = f"{col}_volume"
            if vol_col in df.columns:
                # Net Flow Proxy: +Vol if price up, -Vol if price down
                individual_flows[col] = np.sign(returns) * df[vol_col]
            else:
                individual_flows[col] = pd.Series(0, index=df.index)

        # 3. Calculate Sector-level Flow Pressure
        sector_flows = {}
        for sec, t_list in sector_to_tickers.items():
            valid_ts = [t for t in t_list if t in individual_flows]
            if valid_ts:
                sector_flows[sec] = pd.concat([individual_flows[t] for t in valid_ts], axis=1).mean(axis=1)
            else:
                sector_flows[sec] = pd.Series(0, index=df.index)

        # 4. Main Feature Loop
        spy_returns = df["SPY"].pct_change() if "SPY" in df.columns else None

        for col in df.columns:
            if col in macro_ids or "_volume" in col: continue
            
            col_series = df[col]
            vol_col = f"{col}_volume"
            vol_series = df[vol_col] if vol_col in df.columns else None
            returns = col_series.pct_change()
            
            # Base Momentum & Vol
            features[f"{col}_momentum_1m"] = col_series.pct_change(21)
            features[f"{col}_momentum_3m"] = col_series.pct_change(63)
            features[f"{col}_volatility_1m"] = returns.rolling(21).std() * np.sqrt(252)
            
            # Beta Calculation (Rolling 63-day)
            if spy_returns is not None:
                features[f"{col}_beta"] = returns.rolling(63).cov(spy_returns) / spy_returns.rolling(63).var()
            else:
                features[f"{col}_beta"] = 1.0

            # IV Proxy (Rolling realized volatility of volatility)
            realized_vol = returns.rolling(21).std()
            features[f"{col}_implied_vol_proxy"] = realized_vol.rolling(21).std() / (realized_vol.rolling(21).mean() + 1e-9)

            # Relative Performance
            sector = ticker_to_sector.get(col)
            if sector and sector in df.columns:
                features[f"{col}_relative_to_sector"] = col_series.pct_change(21) - df[sector].pct_change(21)
            else: features[f"{col}_relative_to_sector"] = 0
            if "SPY" in df.columns:
                features[f"{col}_relative_to_market"] = col_series.pct_change(21) - df["SPY"].pct_change(21)
            else: features[f"{col}_relative_to_market"] = 0

            # FLOW FEATURES (Explicit Inflows vs Outflows)
            if vol_series is not None:
                features[f"{col}_volume_surge"] = vol_series / vol_series.rolling(20).mean().replace(0, 1e-9)
                # Net Inflow %: (Positive Flow - Negative Flow) / Total Volume
                pos_f = vol_series.where(returns > 0, 0).rolling(21).sum()
                neg_f = vol_series.where(returns < 0, 0).rolling(21).sum()
                features[f"{col}_net_inflow_pct"] = (pos_f - neg_f) / (pos_f + neg_f + 1e-9)
                features[f"{col}_flow_intensity"] = returns * vol_series
                
                # Money Flow Index (MFI)
                typical_price = col_series
                money_flow = typical_price * vol_series
                p_flow = money_flow.where(returns > 0, 0).rolling(14).sum()
                n_flow = money_flow.where(returns < 0, 0).rolling(14).sum()
                features[f"{col}_money_flow_index"] = 100 - (100 / (1 + p_flow / n_flow.replace(0, 1e-9)))
            else:
                features[f"{col}_volume_surge"] = 0
                features[f"{col}_net_inflow_pct"] = 0
                features[f"{col}_flow_intensity"] = 0
                features[f"{col}_money_flow_index"] = 50

            # Sector-level flow pressure
            if sector in sector_flows:
                features[f"{col}_sector_flow_pressure"] = sector_flows[sector].rolling(21).mean()
            else:
                features[f"{col}_sector_flow_pressure"] = 0

            # CAUSAL PROPAGATION
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

            if "DGS10" in df.columns:
                features[f"{col}_macro_alignment"] = returns.rolling(63).corr(df["DGS10"].pct_change().rolling(63).mean())
            else: features[f"{col}_macro_alignment"] = 0

        return pd.DataFrame(features).fillna(0)

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
                        if min_p < 0.05:
                            self.graph.add_edge(source, target, weight=1-min_p)
                    except: continue
        return self.graph

class AlphaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        if TORCH_AVAILABLE:
            super(AlphaModel, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
        else: self.input_dim = input_dim
    def forward(self, x):
        if TORCH_AVAILABLE:
            lstm_out, _ = self.lstm(x)
            return torch.sigmoid(self.fc(lstm_out[:, -1, :]))
        else: return np.random.uniform(0, 1)

class GeneticOptimizer:
    def __init__(self, prices_df, feature_df):
        self.prices_df = prices_df
        self.feature_df = feature_df
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", np.random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=len(config.FEATURE_SET))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    def _evaluate(self, individual):
        weights = dict(zip(config.FEATURE_SET, individual))
        # Fitness emphasizes flow and positioning
        fitness = (weights.get('net_inflow_pct', 0) * 2.5 + 
                   weights.get('sector_flow_pressure', 0) * 1.5 + 
                   weights.get('momentum_3m', 0) * 1.0 - 
                   weights.get('cost_headwind_score', 0) * 2.0 -
                   weights.get('beta', 0) * 0.5) # Prefer low beta
        return (float(fitness),)
    def optimize(self, n_gen=5):
        pop = self.toolbox.population(n=20)
        algorithms.eaSimple(pop, self.toolbox, cxpb=0.6, mutpb=0.2, ngen=n_gen, verbose=False)
        return dict(zip(config.FEATURE_SET, tools.selBest(pop, 1)[0]))

class StructuralScorer:
    def __init__(self, feature_weights):
        self.feature_weights = feature_weights
    def compute_scores(self, feature_df):
        tickers = list(set([c.split('_')[0] for c in feature_df.columns if '_' in c and "_volume" not in c]))
        scores = {}
        for ticker in tickers:
            ticker_score = 0; details = {}
            for feat_name, weight in self.feature_weights.items():
                feat_col = f"{ticker}_{feat_name}"
                if feat_col in feature_df.columns:
                    val = feature_df[feat_col].iloc[-1]
                    contrib = val * weight
                    ticker_score += contrib
                    details[feat_name] = contrib
            scores[ticker] = {"final_score": ticker_score, "feature_contributions": details}
        return pd.DataFrame(scores).T

class Backtester:
    def __init__(self, data_manager, engine_module):
        self.dm = data_manager
        self.engine_module = engine_module
    def walk_forward_test(self, prices_df, start_date, end_date, top_n=5, benchmark_ticker="SPY"):
        prices_df = prices_df[~prices_df.index.duplicated()]
        macro_ids = [sid for sub in config.ASSET_UNIVERSE["macro_indicators"].values() for sid in sub]
        asset_cols = [c for c in prices_df.columns if c not in macro_ids and "_volume" not in c]
        returns_df = prices_df[asset_cols].pct_change().dropna()
        benchmark_returns = returns_df[benchmark_ticker] if benchmark_ticker in returns_df.columns else None
        ticker_to_sector = {}
        for cat, sectors in config.ASSET_UNIVERSE.items():
            if cat == "macro_indicators": continue
            for s_name, tickers in sectors.items():
                for t in tickers: ticker_to_sector[t] = s_name
        
        rebalance_dates = returns_df.resample('ME').last().index
        rebalance_dates = rebalance_dates[(rebalance_dates >= start_date) & (rebalance_dates <= end_date)]
        
        portfolio_returns = []; weight_history = []; trades = []
        for i in range(len(rebalance_dates)-1):
            start = rebalance_dates[i]; end = rebalance_dates[i+1]
            hist = prices_df.loc[:start]
            if len(hist) < 252: continue
            
            subset = hist[asset_cols].tail(252)
            vol = subset.pct_change().std(); mom = subset.pct_change(21).iloc[-1]
            scores = (mom / vol).sort_values(ascending=False)
            
            selected = scores.head(top_n).index
            mask = (returns_df.index > start) & (returns_df.index <= end)
            period_returns = returns_df.loc[mask, selected]
            if len(period_returns) == 0: continue
            
            portfolio_returns.append(period_returns.mean(axis=1))
            cur_w = {s: 0.0 for s in set(ticker_to_sector.values())}; cur_w['date'] = start
            for t in selected: cur_w[ticker_to_sector.get(t, 'unknown')] += (1.0/top_n)
            weight_history.append(cur_w)
            for t in selected:
                r = (1 + returns_df.loc[mask, t]).prod() - 1
                trades.append({'date': start, 'ticker': t, 'sector': ticker_to_sector.get(t, 'unknown'), 'return': r})
        
        if not portfolio_returns: return {"error": "Insufficient data"}
        all_p_ret = pd.concat(portfolio_returns); all_p_ret = all_p_ret[~all_p_ret.index.duplicated(keep='first')]
        cum_ret = (1 + all_p_ret).cumprod()
        bench_cum = (1 + benchmark_returns.reindex(all_p_ret.index).fillna(0)).cumprod() if benchmark_returns is not None else None
        trade_df = pd.DataFrame(trades)
        ann_r = all_p_ret.mean() * 252; ann_v = all_p_ret.std() * np.sqrt(252); max_dd = (cum_ret / cum_ret.cummax() - 1).min()
        sortino = ann_r / (all_p_ret[all_p_ret < 0].std() * np.sqrt(252) + 1e-9)
        return {
            "cumulative_returns": cum_ret, "benchmark_returns": bench_cum,
            "annualized_return": ann_r, "sharpe_ratio": ann_r/ann_v if ann_v != 0 else 0,
            "sortino_ratio": sortino, "calmar_ratio": ann_r/abs(max_dd) if max_dd != 0 else 0,
            "max_drawdown": max_dd,
            "trade_analytics": {"hit_rate": (trade_df['return'] > 0).mean(), "total_trades": len(trade_df)},
            "sector_stats": trade_df.groupby('sector')['return'].agg(['mean', 'sum', 'count', 'std']).fillna(0),
            "asset_stats": trade_df.groupby('ticker')['return'].agg(['mean', 'sum', 'count', 'std']).fillna(0),
            "weights_history": pd.DataFrame(weight_history).set_index('date'),
            "raw_trades": trade_df
        }
if __name__ == "__main__": print("Engine components defined.")
