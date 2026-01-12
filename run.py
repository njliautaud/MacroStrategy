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
    macro_data = clean_data[macro_ids].dropna()
    
    # 3. Causal Network Construction
    print("Building causal network...")
    tickers = [t for t in clean_data.columns if "_volume" not in t]
    asset_tickers = [t for t in tickers if t not in macro_ids]
    ce = engine.CausalEngine(asset_tickers)
    graph = ce.build_network(clean_data[asset_tickers[:15] + macro_ids[:5]], max_lag=2)
    
    # 4. Advanced Feature Engineering (Frac-Diff, Flows, etc.)
    print("Engineering decision features (Including Fractional Memory)...")
    fe = engine.FeatureEngineer()
    all_features = fe.compute_all_features(clean_data, graph)
    market_flows = fe.get_flow_series(clean_data)
    
    # 5. Regime Detection (HMM)
    print("Detecting market regimes...")
    rc = engine.RegimeClassifier(n_regimes=3)
    regimes = rc.fit_predict(macro_data)
    
    # 6. Regime-Aware Symbolic Optimization
    print(f"Evolving alpha formulas for each regime (Training: {config.START_DATE} to {config.TRAIN_SPLIT})...")
    train_prices = clean_data.loc[:config.TRAIN_SPLIT]
    train_features = all_features.loc[:config.TRAIN_SPLIT]
    train_regimes = regimes.loc[:config.TRAIN_SPLIT]
    
    optimizer = engine.RegimeAwareGeneticOptimizer(train_prices, train_features, train_regimes)
    regime_models = optimizer.optimize()
    
    # 7. Out-of-Sample Walk-Forward Backtesting
    print(f"\nRunning Out-of-Sample regime-aware walk-forward backtest...")
    bt = engine.Backtester(dm, engine)
    backtest_results = bt.walk_forward_test(
        clean_data, all_features, regimes, regime_models, 
        config.TRAIN_SPLIT, config.END_DATE
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
        "market_flows": market_flows
    }
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # 9. Show Dashboard
    print("\nLaunching Interactive Symbolic Dashboard...")
    import dashboard
    dashboard.show_popup_dashboard()

if __name__ == "__main__":
    main()
