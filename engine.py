import numpy as np
import pandas as pd
import networkx as nx
import warnings
import operator
import math
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms, gp
from hmmlearn import hmm
from scipy.stats import ttest_1samp, spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import config

# Suppress annoying FutureWarnings
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
        # Use a fixed window for performance
        window = 50
        weights = cls.get_weights(d, window)
        weights = weights[np.abs(weights) > threshold]
        res = series.rolling(len(weights)).apply(lambda x: np.dot(x, weights), raw=True)
        return res.bfill()

class RegimeClassifier:
    """Classifies market regimes using Hidden Markov Models (HMM)."""
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=100)
        self.scaler = StandardScaler()

    def fit_predict(self, macro_df):
        features = macro_df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        scaled_features = self.scaler.fit_transform(features)
        self.model.fit(scaled_features)
        regimes = self.model.predict(scaled_features)
        return pd.Series(regimes, index=features.index)

class FeatureEngineer:
    @staticmethod
    def compute_all_features(df, graph=None):
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

        spy_returns = df["SPY"].pct_change() if "SPY" in df.columns else None

        for col in df.columns:
            if col in macro_ids or "_volume" in col or col.endswith("_open") or col.endswith("_high") or col.endswith("_low") or col.endswith("_vwap"):
                continue
            
            col_series = df[col]
            vol_col = f"{col}_volume"
            vol_series = df[vol_col] if vol_col in df.columns else None
            # If futures volume is missing/unreliable, try a liquid proxy ETF volume
            if vol_series is None and col in config.COMMODITY_FLOW_PROXY:
                proxy = config.COMMODITY_FLOW_PROXY[col]
                proxy_vol = f"{proxy}_volume"
                if proxy_vol in df.columns:
                    vol_series = df[proxy_vol]
            returns = col_series.pct_change()
            
            # 1. FRAC-DIFF MEMORY
            features[f"{col}_frac_mem"] = FractionalDifferentiator.frac_diff(col_series, d=0.4)
            
            # 2. RSI (14-day)
            delta = returns.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            features[f"{col}_rsi_14"] = 100 - (100 / (1 + rs))

            # 3. ATR RATIO (Volatility proxy)
            features[f"{col}_atr_ratio"] = returns.abs().rolling(14).mean() / (returns.abs().rolling(63).mean() + 1e-9)
            # True ATR (if OHLC present)
            hi = df[f"{col}_high"] if f"{col}_high" in df.columns else None
            lo = df[f"{col}_low"] if f"{col}_low" in df.columns else None
            op = df[f"{col}_open"] if f"{col}_open" in df.columns else None
            if hi is not None and lo is not None:
                prev_close = col_series.shift(1)
                tr = pd.concat([(hi - lo).abs(), (hi - prev_close).abs(), (lo - prev_close).abs()], axis=1).max(axis=1)
                atr14 = tr.rolling(14).mean()
                features[f"{col}_atr_14"] = (atr14 / (col_series + 1e-9)).fillna(0)
                features[f"{col}_range_pct"] = ((hi - lo) / (col_series + 1e-9)).fillna(0)
            else:
                features[f"{col}_atr_14"] = 0
                features[f"{col}_range_pct"] = 0
            # Gap feature (open vs prev close)
            if op is not None:
                features[f"{col}_gap_pct"] = ((op - col_series.shift(1)) / (col_series.shift(1) + 1e-9)).fillna(0)
            else:
                features[f"{col}_gap_pct"] = 0
            # VWAP deviation (close vs vwap)
            if f"{col}_vwap" in df.columns:
                vwap = df[f"{col}_vwap"]
                features[f"{col}_vwap_dev"] = ((col_series - vwap) / (vwap + 1e-9)).fillna(0)
            else:
                features[f"{col}_vwap_dev"] = 0

            # 4. RELATIVE STRENGTH INDEX (Ticker vs Sector)
            sector = ticker_to_sector.get(col)
            if sector and sector in df.columns:
                features[f"{col}_rel_strength_idx"] = col_series.pct_change(21) / (df[sector].pct_change(21) + 1e-9)
            else:
                features[f"{col}_rel_strength_idx"] = 1.0

            # 5. SECTOR ROTATION DRIFT (Momentum relative to ALL other sectors)
            sector_etfs = config.ASSET_UNIVERSE["etfs"]["sector"]
            if sector and sector in df.columns:
                sector_mom = df[sector].pct_change(21)
                valid_etfs = [e for e in sector_etfs if e in df.columns]
                avg_market_mom = df[valid_etfs].pct_change(21).mean(axis=1) if valid_etfs else 0
                features[f"{col}_sector_rotation_drift"] = sector_mom - avg_market_mom
            else:
                features[f"{col}_sector_rotation_drift"] = 0

            # 6. 2nd ORDER COMMODITY DRAG (Cost pressure propagation)
            # Link industrial/tech assets to energy/metals spikes
            commodity_drag = 0
            if col in config.ASSET_UNIVERSE["equities"]["tech"] or col in config.ASSET_UNIVERSE["equities"]["industrials"]:
                # Energy and Industrial Metals as cost inputs
                inputs = ["CL=F", "HG=F", "PA=F"]
                for inp in inputs:
                    if inp in df.columns:
                        commodity_drag += df[inp].pct_change(21)
            features[f"{col}_commodity_drag"] = commodity_drag

            # 6b. Structured cost input pressure by sector (explicit basket)
            # Positive values generally indicate cost headwinds (unless the basket has negative weights).
            input_pressure = pd.Series(0.0, index=df.index)
            if sector and sector in config.COST_INPUT_BASKETS:
                for sym, w in config.COST_INPUT_BASKETS[sector]:
                    if sym in df.columns:
                        input_pressure += w * df[sym].pct_change(21)
            features[f"{col}_second_order_impact"] = input_pressure.rolling(21).mean()

            features[f"{col}_momentum_1m"] = col_series.pct_change(21)
            features[f"{col}_momentum_3m"] = col_series.pct_change(63)
            features[f"{col}_volatility_1m"] = returns.rolling(21).std() * np.sqrt(252)
            
            # Beta & IV
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
                # Flow acceleration: change in flow intensity (captures catalysts / surges)
                features[f"{col}_flow_acceleration"] = features[f"{col}_flow_intensity"].diff(5).fillna(0)
                # Explicit inflow/outflow measures (requested)
                features[f"{col}_inflow_1m"] = (pos_f / (vol_series.rolling(21).sum() + 1e-9))
                features[f"{col}_outflow_1m"] = (neg_f / (vol_series.rolling(21).sum() + 1e-9))
                # Supply/demand proxy (works best on commodities; safe for all assets)
                features[f"{col}_supply_demand_proxy"] = (features[f"{col}_volume_surge"] * np.sign(returns)).fillna(0)
            else:
                features[f"{col}_volume_surge"] = 0
                features[f"{col}_net_inflow_pct"] = 0
                features[f"{col}_flow_intensity"] = 0
                features[f"{col}_flow_acceleration"] = 0
                features[f"{col}_inflow_1m"] = 0
                features[f"{col}_outflow_1m"] = 0
                features[f"{col}_supply_demand_proxy"] = 0

            # Flow magnitude divergence (price trend vs flow trend)
            if vol_series is not None:
                price_mom = returns.rolling(21).mean()
                flow_mom = (vol_series * np.sign(returns)).rolling(21).mean() / (vol_series.rolling(21).mean() + 1e-9)
                features[f"{col}_flow_magnitude_divergence"] = (price_mom - flow_mom).fillna(0)
            else:
                features[f"{col}_flow_magnitude_divergence"] = 0

            # Fundamental pressure index (rates/inflation squeeze proxy)
            fpi = 0
            if "DGS10" in df.columns and "CPIAUCSL" in df.columns:
                yield_curve = df["DGS10"] - (df["DGS2"] if "DGS2" in df.columns else df["DGS10"])
                inflation = df["CPIAUCSL"].pct_change(252)
                fpi = (yield_curve - inflation).fillna(0)
            features[f"{col}_fundamental_pressure_index"] = fpi

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
            
            if "DGS10" in df.columns:
                features[f"{col}_macro_alignment"] = returns.rolling(63).corr(df["DGS10"].pct_change().rolling(63).mean())
            else: features[f"{col}_macro_alignment"] = 0

        return pd.DataFrame(features).fillna(0)

    @staticmethod
    def get_flow_series(df):
        df = df[~df.index.duplicated(keep='first')]
        # Build group mappings (macro-first)
        sector_to_tickers = {}
        # Equity sectors
        for sec, tickers in config.ASSET_UNIVERSE.get("equities", {}).items():
            sector_to_tickers[f"eq_{sec}"] = tickers
        # ETF sector + industries (rotation instruments)
        for sec, tickers in config.ASSET_UNIVERSE.get("etfs", {}).items():
            sector_to_tickers[f"etf_{sec}"] = tickers
        # Futures commodity groups
        for grp, tickers in config.ASSET_UNIVERSE.get("commodities_futures", {}).items():
            sector_to_tickers[f"cmd_{grp}"] = tickers
        # Proxy commodity groups (metals/energy/agri/broad etc.)
        for grp, tickers in getattr(config, "COMMODITY_PROXY_GROUPS", {}).items():
            sector_to_tickers[f"proxy_{grp}"] = tickers

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

        group_flow_df = pd.DataFrame()
        for grp, tickers in sector_to_tickers.items():
            valid = [t for t in tickers if t in net_flows]
            if valid:
                group_flow_df[grp] = pd.concat([net_flows[t] for t in valid], axis=1).mean(axis=1)
        return {"group_flows": group_flow_df.fillna(0)}

def protected_div(left, right):
    return left / right if abs(right) > 1e-6 else 1.0

def protected_log(x):
    return math.log(abs(x) + 1e-9)

def protected_sqrt(x):
    return math.sqrt(abs(x))

def sigmoid(x):
    try: return 1 / (1 + math.exp(-max(min(x, 20), -20)))
    except: return 0.5

def relu(x):
    return max(0, x)

def gen_rand():
    return np.random.uniform(-1, 1)

class SymbolicOptimizer:
    """Advanced Symbolic Regression using DEAP (Genetic Programming)."""
    def __init__(self, n_features, n_gen=50, pop_size=1000, max_depth=6):
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.max_depth = max_depth
        
        self.pset = gp.PrimitiveSet("MAIN", n_features)
        
        # Standard Primitives
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(protected_div, 2)
        self.pset.addPrimitive(operator.neg, 1)
        
        # Non-linear & Advanced Math
        self.pset.addPrimitive(math.tanh, 1)
        self.pset.addPrimitive(sigmoid, 1)
        self.pset.addPrimitive(relu, 1)
        self.pset.addPrimitive(protected_log, 1, name="log")
        self.pset.addPrimitive(protected_sqrt, 1, name="sqrt")
        self.pset.addPrimitive(math.exp, 1)
        self.pset.addPrimitive(abs, 1, name="abs")
        
        # Cyclical & Seasonal Primitives (Captures rotation cycles)
        self.pset.addPrimitive(math.sin, 1)
        self.pset.addPrimitive(math.cos, 1)
        
        # Thresholding/Logic Primitives
        self.pset.addPrimitive(lambda a, b: max(a, b), 2, name="max")
        self.pset.addPrimitive(lambda a, b: min(a, b), 2, name="min")
        
        self.pset.addEphemeralConstant("rand101", gen_rand)
        
        if not hasattr(creator, "FitnessMaxGP"):
            creator.create("FitnessMaxGP", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndividualGP"):
            creator.create("IndividualGP", gp.PrimitiveTree, fitness=creator.FitnessMaxGP)

        self.toolbox = base.Toolbox()
        # Deep trees: allow depth up to self.max_depth (up to 6 branches deep)
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=self.max_depth)
        self.toolbox.register("individual", tools.initIterate, creator.IndividualGP, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def _evaluate(self, individual, X, y):
        func = self.toolbox.compile(expr=individual)
        try:
            # Vectorized evaluation
            preds = np.array([func(*record) for record in X])
            # Correlation as fitness
            # Using Spearman rank correlation for robustness against outliers
            if np.std(preds) < 1e-9: return (0.0,)
            corr = pd.Series(preds).corr(pd.Series(y), method='spearman')
            return (corr if not np.isnan(corr) else 0.0,)
        except:
            return (0.0,)

    def fit(self, X, y):
        self.toolbox.register("evaluate", self._evaluate, X=X, y=y)
        self.toolbox.register("select", tools.selTournament, tournsize=4) # More aggressive selection
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        
        # Limit height to avoid bloat
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth + 2))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth + 2))

        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(5)
        
        # Use eaMuPlusLambda for more robust evolution (Searching deeper branches)
        algorithms.eaMuPlusLambda(pop, self.toolbox, mu=self.pop_size, lambda_=self.pop_size * 2, 
                                 cxpb=0.6, mutpb=0.3, ngen=self.n_gen, 
                                 halloffame=hof, verbose=False)
        return hof[0]

class RegimeAwareGeneticOptimizer:
    def __init__(self, prices_df, feature_df, regimes):
        self.prices_df = prices_df
        self.feature_df = feature_df
        self.regimes = regimes
        self.regime_models = {}

    def optimize(self):
        unique_regimes = self.regimes.unique()
        for r in unique_regimes:
            print(f"  Evolving Alpha Formula (Unified Discovery, Depth=6, Pop=1000)..." if len(unique_regimes) == 1 else f"  Evolving Alpha Formula for Regime {r} (Advanced GA, Depth=6, Pop=1000)...")
            r_dates = self.regimes[self.regimes == r].index
            if len(r_dates) < 50: continue
            
            tickers = list(set([c.split('_')[0] for c in self.feature_df.columns if '_' in c]))
            X_data = []
            y_data = []
            
            # Use a larger sample for the unified model to capture cross-asset patterns better
            sample_count = 500 if len(unique_regimes) == 1 else 300
            sample_dates = r_dates[-sample_count:] if len(r_dates) > sample_count else r_dates
            spy_col = "SPY_close" if "SPY_close" in self.prices_df.columns else "SPY"
            market_return = self.prices_df[spy_col].pct_change(21).shift(-21) if spy_col in self.prices_df.columns else 0
            
            for date in sample_dates:
                # Pick a subset of tickers for each date to avoid overfitting to one stock
                valid_tickers = [t for t in tickers if t in self.prices_df.columns]
                if not valid_tickers:
                    continue
                # Pick more tickers for unified model to see more relationships
                pick_n = min(len(valid_tickers), 35 if len(unique_regimes) == 1 else 25)
                for t in np.random.choice(valid_tickers, pick_n, replace=False):
                    feat_vec = []
                    for f in config.FEATURE_SET:
                        col = f"{t}_{f}"
                        val = self.feature_df[col].asof(date) if col in self.feature_df.columns else 0
                        feat_vec.append(val)
                    
                    # TARGET: Individual Asset Alpha (Excess Return over Market)
                    asset_ret = self.prices_df[t].pct_change(21).shift(-21).asof(date)
                    mkt_ret = market_return.asof(date) if isinstance(market_return, pd.Series) else 0
                    
                    if np.isfinite(asset_ret) and np.isfinite(mkt_ret):
                        X_data.append(feat_vec)
                        y_data.append(asset_ret - mkt_ret)

            if not X_data: continue
            X_np = np.nan_to_num(np.array(X_data))
            y_np = np.nan_to_num(np.array(y_data))
            
            # Deep search: Larger population, more generations, deeper trees
            sym = SymbolicOptimizer(
                n_features=len(config.FEATURE_SET), 
                n_gen=50, 
                pop_size=1000, 
                max_depth=6
            )
            self.regime_models[r] = {"formula": sym.fit(X_np, y_np), "pset": sym.pset}
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

class PortfolioOptimizer:
    """Implements Hierarchical Risk Parity (HRP) for robust asset allocation."""
    @staticmethod
    def get_hrp_weights(returns_df):
        if returns_df.empty or returns_df.shape[1] < 2:
            return pd.Series(1.0/max(1, returns_df.shape[1]), index=returns_df.columns)
        
        corr = returns_df.corr().fillna(0)
        dist = np.sqrt(0.5 * (1 - corr))
        # Handle cases with very small or no variance
        try:
            link = linkage(dist, 'single')
            sort_idx = leaves_list(link)
        except:
            return pd.Series(1.0/max(1, returns_df.shape[1]), index=returns_df.columns)
            
        sorted_tickers = returns_df.columns[sort_idx]
        
        def get_cluster_var(cov, cluster_items):
            cov_c = cov.loc[cluster_items, cluster_items]
            w = 1.0 / (np.diag(cov_c) + 1e-9)
            w /= w.sum()
            return np.dot(np.dot(w, cov_c), w)

        def recursive_bisection(cov, sorted_items):
            weights = pd.Series(1.0, index=sorted_items)
            items = [sorted_items]
            while len(items) > 0:
                items = [i[j:k] for i in items for j, k in ((0, len(i)//2), (len(i)//2, len(i))) if len(i) > 1]
                for i in range(0, len(items), 2):
                    c_left = items[i]
                    c_right = items[i+1]
                    v_left = get_cluster_var(cov, c_left)
                    v_right = get_cluster_var(cov, c_right)
                    alpha = 1 - v_left / (v_left + v_right + 1e-9)
                    weights[c_left] *= alpha
                    weights[c_right] *= (1 - alpha)
            return weights

        cov = returns_df.cov().fillna(0)
        weights = recursive_bisection(cov, sorted_tickers)
        return weights

class ExecutionOptimizer:
    """A secondary GA that evolves optimal exit/management rules."""
    def __init__(self, n_gen=20, pop_size=300):
        # Exit Features: Profit, Peak DD, Days Held, Sector Mom, Mkt Mom, Vol
        self.pset = gp.PrimitiveSet("EXIT", 6)
        self.pset.renameArguments(ARG0="profit", ARG1="peak_dd", ARG2="days", ARG3="sec_mom", ARG4="mkt_mom", ARG5="vol")
        
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(protected_div, 2)
        self.pset.addPrimitive(math.tanh, 1)
        self.pset.addPrimitive(sigmoid, 1)
        self.pset.addPrimitive(lambda a, b: 1.0 if a > b else 0.0, 2, name="gt")
        self.pset.addEphemeralConstant("rand_exit", gen_rand)

        if not hasattr(creator, "FitnessMaxExit"):
            creator.create("FitnessMaxExit", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndividualExit"):
            creator.create("IndividualExit", gp.PrimitiveTree, fitness=creator.FitnessMaxExit)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.IndividualExit, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.n_gen = n_gen
        self.pop_size = pop_size

    def _evaluate(self, individual, X, y):
        func = self.toolbox.compile(expr=individual)
        try:
            # Output is a 'should exit' score. We want to maximize future returns avoided during drawdown.
            scores = np.array([func(*record) for record in X])
            # Simple fitness: Correlation between exit signal and subsequent bad returns
            if np.std(scores) < 1e-9: return (0.0,)
            # y is 'return if held' (we want exit when this is negative)
            corr, _ = spearmanr(scores, -y) 
            return (corr if np.isfinite(corr) else 0.0,)
        except: return (0.0,)

    def fit(self, X, y):
        self.toolbox.register("evaluate", self._evaluate, X=X, y=y)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=gp.genFull(min_=0, max_=2), pset=self.pset)
        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        algorithms.eaMuPlusLambda(pop, self.toolbox, mu=self.pop_size, lambda_=self.pop_size, 
                                 cxpb=0.6, mutpb=0.3, ngen=self.n_gen, halloffame=hof, verbose=False)
        return hof[0]

class Backtester:
    def __init__(self, data_manager, engine_module):
        self.dm = data_manager
        self.engine_module = engine_module

    def walk_forward_test(self, prices_df, feature_df, regimes, regime_models, start_date, end_date, top_n=8):
        prices_df = prices_df[~prices_df.index.duplicated()]
        macro_ids = [sid for sub in config.ASSET_UNIVERSE["macro_indicators"].values() for sid in sub]
        asset_cols = [c for c in prices_df.columns if c not in macro_ids and "_volume" not in c]
        returns_df = prices_df[asset_cols].pct_change().dropna()
        
        # EXECUTION PARAMETERS
        TARGET_VOL = 0.15  # 15% Annualized Vol Target
        STOP_LOSS_MULT = 2.0  # 2x Daily Vol Trailing Stop
        
        rebalance_dates = returns_df.resample('ME').last().index
        rebalance_dates = rebalance_dates[(rebalance_dates >= start_date) & (rebalance_dates <= end_date)]
        
        portfolio_returns = []; weight_history = []; trades = []
        # Store per-rebalance feature IC samples (directional contribution around 0)
        ic_samples = {f: [] for f in config.FEATURE_SET}
        ticker_to_sector = {}
        for cat, sectors in config.ASSET_UNIVERSE.items():
            if cat == "macro_indicators": continue
            for s_name, tickers in sectors.items():
                for t in tickers: ticker_to_sector[t] = s_name

        for i in range(len(rebalance_dates)-1):
            start = rebalance_dates[i]; end = rebalance_dates[i+1]
            current_regime = regimes.asof(start)
            model_data = regime_models.get(current_regime)
            if not model_data: continue
            
            func = gp.compile(model_data["formula"], model_data["pset"])
            scores = {}
            for t in asset_cols:
                input_vec = []
                for f in config.FEATURE_SET:
                    col = f"{t}_{f}"
                    val = feature_df[col].asof(start) if col in feature_df.columns else 0
                    input_vec.append(val)
                try: scores[t] = func(*input_vec)
                except: scores[t] = 0
            
            selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
            
            # 1. HRP ALLOCATION
            hist_subset = returns_df.loc[:start].tail(252)[selected]
            raw_weights = PortfolioOptimizer.get_hrp_weights(hist_subset)
            
            # 2. DYNAMIC VOLATILITY TARGETING
            # Scale exposure based on predicted portfolio vol
            port_vol_ann = np.sqrt(np.dot(np.dot(raw_weights, hist_subset.cov() * 252), raw_weights))
            exposure_scale = TARGET_VOL / (port_vol_ann + 1e-9)
            # Limit leverage to 1.5x
            exposure_scale = min(exposure_scale, 1.5)
            final_weights = raw_weights * exposure_scale
            
            mask = (returns_df.index > start) & (returns_df.index <= end)
            if returns_df.loc[mask, selected].empty: continue
            
            # 3. DAILY RISK MANAGEMENT (Stops & Catalysts)
            period_returns = returns_df.loc[mask, selected]
            daily_port_rets = []
            
            for date, row in period_returns.iterrows():
                # Tighten stops if regime is "High Vol" (Regime 2)
                current_stop = STOP_LOSS_MULT * 0.7 if current_regime == 2 else STOP_LOSS_MULT
                
                # Simple Daily Stop Logic: if asset drops > stop_mult * its own vol, exit for the day
                # (In a real system, this would be a high-frequency trigger)
                asset_vols = hist_subset.std()
                active_weights = final_weights.copy()
                
                for t in selected:
                    if row[t] < -(asset_vols[t] * current_stop):
                        active_weights[t] = 0 # Exit position on stop-out
                
                daily_port_rets.append((row * active_weights).sum())

            portfolio_returns.extend([pd.Series(daily_port_rets, index=period_returns.index)])
            
            cur_w = {s: 0.0 for s in set(ticker_to_sector.values())}; cur_w['date'] = start
            for t, w in final_weights.items():
                cur_w[ticker_to_sector.get(t, 'unknown')] += w
            weight_history.append(cur_w)
            
            for t in selected:
                r = (1 + returns_df.loc[mask, t]).prod() - 1
                trades.append({'date': start, 'ticker': t, 'sector': ticker_to_sector.get(t, 'unknown'), 'return': r})

            # --- Feature directional contribution (IC) at this rebalance ---
            # Use realized next-period excess returns across the full universe (not just selected)
            try:
                bench_period = (1 + prices_df.loc[mask, "SPY"].pct_change().fillna(0)).prod() - 1 if "SPY" in prices_df.columns else 0.0
                period_asset = (1 + returns_df.loc[mask, asset_cols]).prod() - 1
                period_alpha = (period_asset - bench_period).replace([np.inf, -np.inf], np.nan).dropna()
                if len(period_alpha) >= 10:
                    for f in config.FEATURE_SET:
                        feat_vals = {}
                        for a in period_alpha.index:
                            c = f"{a}_{f}"
                            if c in feature_df.columns:
                                feat_vals[a] = feature_df[c].asof(start)
                        if len(feat_vals) >= 10:
                            x = pd.Series(feat_vals).replace([np.inf, -np.inf], np.nan).dropna()
                            y = period_alpha.reindex(x.index).dropna()
                            if len(x) >= 10 and y.std() > 1e-12 and x.std() > 1e-12:
                                ic = spearmanr(x.values, y.values, nan_policy="omit").correlation
                                if ic is not None and not np.isnan(ic):
                                    ic_samples[f].append(float(ic))
            except Exception:
                pass

        if not portfolio_returns: return {"error": "Insufficient data"}
        all_p_ret = pd.concat(portfolio_returns); all_p_ret = all_p_ret[~all_p_ret.index.duplicated(keep='first')]
        cum_ret = (1 + all_p_ret).cumprod()
        
        benchmark_returns = prices_df['SPY'].pct_change().reindex(all_p_ret.index).fillna(0)
        bench_cum = (1 + benchmark_returns).cumprod()

        excess_returns = all_p_ret - benchmark_returns
        t_stat, p_val = ttest_1samp(excess_returns, 0)
        info_ratio = (excess_returns.mean() / (excess_returns.std() + 1e-9)) * np.sqrt(252)

        bench_ann_r = benchmark_returns.mean() * 252
        bench_ann_v = benchmark_returns.std() * np.sqrt(252)
        bench_dd = (bench_cum / bench_cum.cummax() - 1).min()

        trade_df = pd.DataFrame(trades)
        ann_r = all_p_ret.mean() * 252; ann_v = all_p_ret.std() * np.sqrt(252)
        max_dd = (cum_ret / cum_ret.cummax() - 1).min()
        
        # Feature directional contribution (IC mean). Positive = predictive of future alpha, negative = headwind.
        feature_loadings = {f: (float(np.mean(v)) if len(v) else 0.0) for f, v in ic_samples.items()}

        # Inter-market correlation focus: commodities (futures + proxies) vs equities
        comm = [t for sub in config.ASSET_UNIVERSE["commodities_futures"].values() for t in sub if t in prices_df.columns]
        comm_proxy = [t for t in config.ASSET_UNIVERSE["etfs"]["commodity_proxies"] if t in prices_df.columns]
        eq = [t for sub in config.ASSET_UNIVERSE["equities"].values() for t in sub if t in prices_df.columns]
        macro = [sid for sub in config.ASSET_UNIVERSE["macro_indicators"].values() for sid in sub if sid in prices_df.columns]
        # Full matrix in a deliberate order (macro last so commodities/equities are visually central)
        ordered = (comm + comm_proxy + [t for t in config.ASSET_UNIVERSE["etfs"]["sector"] if t in prices_df.columns] + eq + macro)
        corr_matrix = prices_df[ordered].corr() if ordered else prices_df[asset_cols].corr()
        # A tighter “macro project” view: commodities/proxies vs equities only
        corr_comm_eq = prices_df[(comm + comm_proxy + eq)].corr() if (comm or comm_proxy) and eq else pd.DataFrame()

        return {
            "cumulative_returns": cum_ret, "benchmark_returns": bench_cum,
            "annualized_return": ann_r, "annualized_vol": ann_v,
            "sharpe_ratio": ann_r/(ann_v+1e-9), "info_ratio": info_ratio,
            "feature_loadings": feature_loadings,
            "bench_stats": {"return": bench_ann_r, "vol": bench_ann_v, "max_dd": bench_dd},
            "t_stat": t_stat, "p_value": p_val,
            "max_drawdown": max_dd, 
            "trade_analytics": {"hit_rate": (trade_df['return'] > 0).mean()},
            "sector_stats": trade_df.groupby('sector')['return'].agg(['mean', 'sum']).fillna(0),
            "asset_stats": trade_df.groupby('ticker')['return'].agg(['mean', 'sum']).fillna(0),
            "weights_history": pd.DataFrame(weight_history).set_index('date'),
            "regimes": regimes.loc[start_date:end_date],
            "correlation_matrix": corr_matrix,
            "correlation_comm_eq": corr_comm_eq
        }

    def permutation_test(self, returns, benchmark_returns, n_iter=500):
        """
        Shuffles the excess returns to see if performance is due to luck.
        """
        excess = returns - benchmark_returns
        actual_sharpe = (excess.mean() / (excess.std() + 1e-9)) * np.sqrt(252)
        
        shuffled_sharpes = []
        for _ in range(n_iter):
            # SHUFFLE VERSION: Randomly scramble the daily excess returns
            # This breaks the temporal link between signals and market outcomes
            shuffled_excess = np.random.permutation(excess.values)
            s_mean = np.mean(shuffled_excess)
            s_std = np.std(shuffled_excess)
            shuffled_sharpes.append((s_mean / (s_std + 1e-9)) * np.sqrt(252))
            
        p_value = (np.abs(shuffled_sharpes) >= np.abs(actual_sharpe)).mean()
        return p_value, shuffled_sharpes

    def placebo_test(self, prices_df, feature_df, macro_df, start_date, end_date):
        """
        Intentionally shift features by 1 rebalance period to check if performance collapses.
        """
        print("  Running Placebo Test (shifted features)...")
        # Shift features forward by 21 days (roughly one month rebalance)
        shifted_features = feature_df.shift(21).fillna(0)
        results = self.walk_forward_test_expanding(
            prices_df, shifted_features, macro_df, start_date, end_date, is_placebo=True
        )
        return results.get('sharpe_ratio', 0)

    def walk_forward_test_expanding(
        self,
        prices_df,
        feature_df,
        macro_df,
        start_date,
        end_date,
        retrain_every_n_months=3,
        top_n=8,
        is_placebo=False
    ):
        """
        Expanding walk-forward:
        - At each rebalance, training expands to include all history up to that rebalance.
        - Models are retrained every `retrain_every_n_months`.
        - True holdout window is [start_date, end_date].
        """
        prices_df = prices_df[~prices_df.index.duplicated()]
        macro_df = macro_df.loc[prices_df.index] if macro_df is not None and not macro_df.empty else pd.DataFrame(index=prices_df.index)

        macro_ids = [sid for sub in config.ASSET_UNIVERSE["macro_indicators"].values() for sid in sub]
        asset_cols = [
            c for c in prices_df.columns
            if c not in macro_ids and "_volume" not in c and not c.endswith(("_open", "_high", "_low", "_vwap"))
        ]
        returns_df = prices_df[asset_cols].pct_change().dropna()

        # CONFIG & PARAMS
        TARGET_VOL = 0.15; STOP_LOSS_MULT = 2.0
        
        rebalance_dates = returns_df.resample('ME').last().index
        rebalance_dates = rebalance_dates[(rebalance_dates >= start_date) & (rebalance_dates <= end_date)]
        
        # Performance Tracking
        portfolio_returns = []; model_only_returns = []; no_vol_returns = []
        weight_history = []; trades = []; ic_samples = {f: [] for f in config.FEATURE_SET}
        
        # Dual-Model State
        current_models = None; exit_model = None; last_retrain = None
        last_final_weights = {}

        ticker_to_sector = {}
        for cat, sectors in config.ASSET_UNIVERSE.items():
            if cat == "macro_indicators": continue
            for s_name, tickers in sectors.items():
                for t in tickers: ticker_to_sector[t] = s_name

        for i in range(len(rebalance_dates) - 1):
            start = rebalance_dates[i]; end = rebalance_dates[i + 1]

            do_retrain = (last_retrain is None or (start.year - last_retrain.year)*12 + (start.month - last_retrain.month) >= retrain_every_n_months)

            if do_retrain:
                if not is_placebo: print(f"  [{start.date()}] Retraining Unified Prediction + Execution Engines...")
                hist_prices = prices_df.loc[:start]; hist_features = feature_df.loc[:start]
                
                # 1. RETRAIN UNIFIED PREDICTION ENGINE
                # We use all historical data for a single robust formula
                regimes = pd.Series(0, index=hist_prices.index)
                
                pred_optimizer = RegimeAwareGeneticOptimizer(hist_prices, hist_features, regimes)
                current_models = pred_optimizer.optimize()
                
                # 2. RETRAIN EXECUTION ENGINE (Learn from past trade outcomes)
                exec_opt = ExecutionOptimizer()
                # Dummy training data (Replace with real trade logs in production)
                X_ex = np.random.randn(100, 6); y_ex = np.random.randn(100)
                exit_model = exec_opt.fit(X_ex, y_ex)
                
                last_retrain = start

            if not current_models: continue

            # --- PREDICTION STEP ---
            current_regime = regimes.asof(start) if 'regimes' in locals() else 0
            model_data = current_models.get(current_regime, next(iter(current_models.values())))
            func = gp.compile(model_data["formula"], model_data["pset"])
            
            scores = {}
            for t in asset_cols:
                input_vec = [feature_df[f"{t}_{f}"].asof(start) if f"{t}_{f}" in feature_df.columns else 0 for f in config.FEATURE_SET]
                try: 
                    s = func(*input_vec)
                    scores[t] = s if np.isfinite(s) else 0.0
                except: scores[t] = 0.0

            selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
            if not selected: continue

            # --- ALLOCATION & VOL TARGETING STEP ---
            hist_subset = returns_df.loc[:start].tail(252)[selected]
            raw_weights = PortfolioOptimizer.get_hrp_weights(hist_subset)
            
            # Executed Weights (Vol Targeted)
            port_vol_ann = np.sqrt(np.dot(np.dot(raw_weights, hist_subset.cov() * 252), raw_weights))
            exposure_scale = TARGET_VOL / (port_vol_ann + 1e-9)
            final_weights = raw_weights * min(exposure_scale, 1.5)
            
            # Shadow Weight (No Vol Targeting / Pure Model)
            no_vol_weights = raw_weights # 100% exposure, equal to HRP relative weights

            mask = (returns_df.index > start) & (returns_df.index <= end)
            if returns_df.loc[mask, selected].empty: continue

            # --- EXECUTION LOOP (Daily) ---
            period_returns = returns_df.loc[mask, selected]
            daily_port_rets = []; daily_no_vol_rets = []; daily_model_only = []
            
            asset_vols = hist_subset.std()
            for dt, row in period_returns.iterrows():
                # 1. Model-Only (Equal Weight, No Stops, Raw Prediction)
                daily_model_only.append(row.mean())
                
                # 2. No-Vol Shadow (HRP Weights, but no scaling or exits)
                daily_no_vol_rets.append((row * no_vol_weights).sum())
                
                # 3. Executed (Vol Targeting + Daily Stop Management)
                active_weights = final_weights.copy()
                stop_mult = STOP_LOSS_MULT * (0.7 if current_regime == 2 else 1.0)
                for t in selected:
                    if row[t] < -(asset_vols[t] * stop_mult): active_weights[t] = 0
                daily_port_rets.append((row * active_weights).sum())

            portfolio_returns.append(pd.Series(daily_port_rets, index=period_returns.index))
            no_vol_returns.append(pd.Series(daily_no_vol_rets, index=period_returns.index))
            model_only_returns.append(pd.Series(daily_model_only, index=period_returns.index))
            
            last_final_weights = final_weights.to_dict()
            cur_w = {s: 0.0 for s in set(ticker_to_sector.values())}; cur_w["date"] = start
            for t, w in final_weights.items(): cur_w[ticker_to_sector.get(t, "unknown")] += float(w)
            weight_history.append(cur_w)

            # --- SIGNAL QUALITY METRICS (Information Coefficient) ---
            try:
                spy_col = "SPY_close" if "SPY_close" in prices_df.columns else "SPY"
                bench_period = (1 + prices_df.loc[mask, spy_col].pct_change().fillna(0)).prod() - 1
                asset_rets_period = (1 + returns_df.loc[mask, asset_cols]).prod() - 1
                asset_alpha_period = (asset_rets_period - bench_period).replace([np.inf, -np.inf], np.nan).dropna()
                if len(asset_alpha_period) >= 10:
                    for f in config.FEATURE_SET:
                        feat_vals = {a: feature_df[f"{a}_{f}"].asof(start) for a in asset_alpha_period.index if f"{a}_{f}" in feature_df.columns}
                        if len(feat_vals) >= 10:
                            x_ic = pd.Series(feat_vals).replace([np.inf, -np.inf], np.nan).dropna()
                            y_ic = asset_alpha_period.reindex(x_ic.index).dropna()
                            if len(x_ic) >= 10 and y_ic.std() > 1e-12:
                                ic_val, _ = spearmanr(x_ic.values, y_ic.values, nan_policy="omit")
                                if np.isfinite(ic_val): ic_samples[f].append(float(ic_val))
            except: pass

        if not portfolio_returns: return {"error": "Insufficient data"}
        
        # Final Consolidation
        all_p_ret = pd.concat(portfolio_returns).sort_index(); all_p_ret = all_p_ret[~all_p_ret.index.duplicated()]
        all_nv_ret = pd.concat(no_vol_returns).sort_index(); all_nv_ret = all_nv_ret[~all_nv_ret.index.duplicated()]
        all_mo_ret = pd.concat(model_only_returns).sort_index(); all_mo_ret = all_mo_ret[~all_mo_ret.index.duplicated()]
        
        cum_ret = (1 + all_p_ret).cumprod()
        cum_nv = (1 + all_nv_ret).cumprod()
        cum_mo = (1 + all_mo_ret).cumprod()
        
        spy_col = "SPY_close" if "SPY_close" in prices_df.columns else "SPY"
        benchmark_returns = prices_df[spy_col].pct_change().reindex(all_p_ret.index).fillna(0)
        bench_cum = (1 + benchmark_returns).cumprod()

        # --- DUAL-ENGINE ANALYTICS ---
        ann_r = all_p_ret.mean() * 252; ann_v = all_p_ret.std() * np.sqrt(252)
        ann_nv = all_nv_ret.mean() * 252; ann_mo = all_mo_ret.mean() * 252
        
        # Vol Targeting Impact: Executed Return - Shadow Return
        vol_impact = ann_r - ann_nv
        # Selection Alpha: Model-Only Return - Benchmark Return
        selection_alpha = ann_mo - (benchmark_returns.mean() * 252)

        return {
            "cumulative_returns": cum_ret, "benchmark_returns": bench_cum,
            "shadow_returns_no_vol": cum_nv, "model_only_returns": cum_mo,
            "annualized_return": ann_r, "annualized_vol": ann_v, "sharpe_ratio": ann_r/(ann_v+1e-9),
            "vol_targeting_impact": vol_impact, "selection_alpha": selection_alpha,
            "feature_loadings": {f: (float(np.mean(v)) if len(v) else 0.0) for f, v in ic_samples.items()},
            "bench_stats": {"return": benchmark_returns.mean()*252, "vol": benchmark_returns.std()*np.sqrt(252)},
            "current_holdings": last_final_weights,
            "weights_history": pd.DataFrame(weight_history).set_index("date"),
            "asset_stats": pd.DataFrame(trades).groupby('ticker')['return'].sum() if trades else pd.Series()
        }


if __name__ == "__main__": print("Regime-Aware Symbolic Engine Components Defined.")
