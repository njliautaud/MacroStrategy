import data_manager
import engine
import config
import pandas as pd
import os
import pickle

def main():
    print("Initializing Macro-Structural Alpha Engine (Symbolic & Regime-Aware)...")
    
    # 1. Data Ingestion
    dm = data_manager.DataManager()
    print("Fetching data from FMP and FRED...")
    raw_data = dm.get_consolidated_data()
    raw_data = raw_data[~raw_data.index.duplicated(keep='first')]
    clean_data = dm.clean_data(raw_data)
    
    # 2. Macro Data for Regime Detection
    macro_ids = [sid for sub in config.ASSET_UNIVERSE["macro_indicators"].values() for sid in sub]
    available_macro_ids = [sid for sid in macro_ids if sid in clean_data.columns]
    if len(available_macro_ids) < len(macro_ids):
        missing = sorted(set(macro_ids) - set(available_macro_ids))
        print(f"Warning: missing macro series in dataset (skipping): {missing[:10]}{'...' if len(missing)>10 else ''}")
    macro_data = clean_data[available_macro_ids].dropna() if available_macro_ids else pd.DataFrame()
    
    # 3. Causal Network Construction (built ONLY on pre-holdout/dev data to avoid leakage)
    print("Building causal network (dev-period only)...")
    tickers = [t for t in clean_data.columns if "_volume" not in t]
    asset_tickers = [t for t in tickers if t not in macro_ids]
    ce = engine.CausalEngine(asset_tickers)
    graph_macro = [m for m in macro_ids[:5] if m in clean_data.columns]
    dev_for_graph = clean_data.loc[:config.TRAIN_END]
    graph = ce.build_network(dev_for_graph[asset_tickers[:15] + graph_macro], max_lag=2)
    
    # 4. Advanced Feature Engineering (Frac-Diff, Flows, etc.)
    print("Engineering decision features (Including Fractional Memory)...")
    fe = engine.FeatureEngineer()
    all_features = fe.compute_all_features(clean_data, graph)
    market_flows = fe.get_flow_series(clean_data)
    
    # 5. Regime Detection (Simplified to Unified Model for Rotation Discovery)
    print("Initializing Unified Model (Discovery of Cross-Asset Rotation Patterns)...")
    # Setting n_regimes=1 forces the GA to find a robust formula that works across all conditions
    regimes = pd.Series(0, index=clean_data.index)
    
    # 6. Regime-Aware Symbolic Optimization (dev period only: 2010..2022)
    print(f"Evolving alpha formulas for each regime (Dev period: {config.START_DATE} to {config.TRAIN_END})...")
    dev_prices = clean_data.loc[:config.TRAIN_END]
    dev_features = all_features.loc[:config.TRAIN_END]
    dev_regimes = regimes.loc[:config.TRAIN_END] if isinstance(regimes, pd.Series) else pd.Series(0, index=dev_prices.index)
    
    optimizer = engine.RegimeAwareGeneticOptimizer(dev_prices, dev_features, dev_regimes)
    regime_models = optimizer.optimize()
    
    # 7. TRUE HOLDOUT: Expanding walk-forward 2023-2025 with periodic retraining (no peeking)
    print(f"\nRunning TRUE HOLDOUT expanding walk-forward ({config.HOLDOUT_START}..{config.HOLDOUT_END})...")
    bt = engine.Backtester(dm, engine)
    backtest_results = bt.walk_forward_test_expanding(
        prices_df=clean_data,
        feature_df=all_features,
        macro_df=macro_data,
        start_date=config.HOLDOUT_START,
        end_date=config.HOLDOUT_END,
        retrain_every_n_months=getattr(config, "RETRAIN_EVERY_N_MONTHS", 3),
        top_n=8,
    )
    
    if "error" not in backtest_results:
        print(f"\nBacktest Performance (OOS):")
        print(f"  Annualized Return: {backtest_results['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    
    # 8. Save Results
    results = {
        "graph": graph,
        "backtest": backtest_results,
        "clean_data": clean_data,
        "regime_models": regime_models,
        "market_flows": market_flows,
        "dev_period": {"start": config.START_DATE, "end": config.TRAIN_END},
    }
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # 9. Show Dashboard
    print("\nLaunching Interactive Symbolic Dashboard...")
    import dashboard
    dashboard.show_popup_dashboard()

if __name__ == "__main__":
    main()
