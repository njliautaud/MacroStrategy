import data_manager
import engine
import config
import pandas as pd
import os

def main():
    print("Initializing Macro-Structural Alpha Engine...")
    
    # 1. Data Ingestion
    dm = data_manager.DataManager()
    print("Fetching data from FMP and FRED...")
    raw_data = dm.get_consolidated_data()
    # Remove any duplicates from raw data before cleaning
    raw_data = raw_data[~raw_data.index.duplicated(keep='first')]
    clean_data = dm.clean_data(raw_data)
    
    # 2. Feature Engineering (Initial)
    # We now compute all features in step 4
    
    # 3. Causal Network Construction
    print("Building causal network (Sub-sampled for speed)...")
    tickers = [t for t in clean_data.columns if "_volume" not in t]
    macro_ids = [sid for sub in config.ASSET_UNIVERSE["macro_indicators"].values() for sid in sub]
    asset_tickers = [t for t in tickers if t not in macro_ids]
    
    ce = engine.CausalEngine(asset_tickers)
    # Build graph on a representative subset
    graph_tickers = asset_tickers[:15] + macro_ids[:5]
    graph = ce.build_network(clean_data[graph_tickers], max_lag=2)
    
    # 4. Feature Engineering (Comprehensive)
    print("Engineering decision features...")
    fe = engine.FeatureEngineer()
    all_features = fe.compute_all_features(clean_data, graph)
    
    # 5. Genetic Algorithm Optimization
    print(f"Optimizing feature weights via Genetic Algorithm (Training period: {config.START_DATE} to {config.TRAIN_SPLIT})...")
    train_prices = clean_data.loc[:config.TRAIN_SPLIT]
    train_features = all_features.loc[:config.TRAIN_SPLIT]
    
    ga = engine.GeneticOptimizer(train_prices, train_features)
    best_weights = ga.optimize(n_gen=5)
    print(f"Optimal Feature Weights: {best_weights}")
    
    # 6. Scoring
    print("Computing structural scores with GA weights...")
    scorer = engine.StructuralScorer(best_weights)
    scores = scorer.compute_scores(all_features)
    
    # 7. Backtesting
    print(f"\nRunning Out-of-Sample walk-forward backtest (OOS period: {config.TRAIN_SPLIT} to {config.END_DATE})...")
    bt = engine.Backtester(dm, engine)
    backtest_results = bt.walk_forward_test(clean_data, config.TRAIN_SPLIT, config.END_DATE)
    
    if "error" not in backtest_results:
        print(f"Backtest Performance:")
        print(f"  Annualized Return: {backtest_results['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    else:
        print(f"Backtest failed: {backtest_results['error']}")

    # 8. Save results for dashboard
    import pickle
    results = {
        "scores": scores,
        "graph": graph,
        "backtest": backtest_results,
        "clean_data": clean_data,
        "feature_weights": best_weights
    }
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nResults saved to results.pkl.")
    
    # 9. Show Popup Dashboard
    print("\nLaunching popup dashboard...")
    import dashboard
    dashboard.show_popup_dashboard()

if __name__ == "__main__":
    # Ensure project follows user memory ID: 5926025 (run in existing env)
    main()
