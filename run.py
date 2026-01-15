import data_manager
import engine
import config
import pandas as pd
import os
import pickle
import psutil
import multiprocessing
from tqdm import tqdm
import time

def print_resources():
    cores = multiprocessing.cpu_count()
    memory = psutil.virtual_memory()
    print("="*50)
    print(f"SYSTEM RESOURCES:")
    print(f"  CPU Cores: {cores}")
    print(f"  Total RAM: {memory.total / (1024**3):.2f} GB")
    print(f"  Available RAM: {memory.available / (1024**3):.2f} GB")
    print("="*50)

def main():
    print_resources()
    print("Initializing Macro-Structural Alpha Engine (Symbolic & Regime-Aware)...")
    
    # Prompt at start so user can walk away
    skip_input = input("\n  Skip retraining during holdout period (2023-2025) to save time? [y/N]: ").strip().lower()
    skip_retraining = skip_input == "y"
    lite_input = input("  Run in lightweight mode (faster, lower GA depth)? [y/N]: ").strip().lower()
    lightweight_mode = lite_input == "y"
    if lightweight_mode:
        print("  Lightweight mode enabled: smaller GA population/generations/depth.")

    with tqdm(total=9, desc="Full Pipeline", unit="step") as pbar:
        # 1. Data Ingestion
        dm = data_manager.DataManager()
        print("\n[Step 1/9] Fetching data from FMP and FRED...")
        raw_data = dm.get_consolidated_data()
        raw_data = raw_data[~raw_data.index.duplicated(keep='first')]
        clean_data = dm.clean_data(raw_data)
        print(f"  Processed {clean_data.shape[1]} series over {len(clean_data)} trading days.")
        pbar.update(1)
        
        # 2. Macro Data for Regime Detection
        print("\n[Step 2/9] Processing Macro Data...")
        macro_ids = [sid for sub in config.ASSET_UNIVERSE["macro_indicators"].values() for sid in sub]
        available_macro_ids = [sid for sid in macro_ids if sid in clean_data.columns]
        if len(available_macro_ids) < len(macro_ids):
            missing = sorted(set(macro_ids) - set(available_macro_ids))
            print(f"  Warning: missing macro series: {missing[:5]}...")
        macro_data = clean_data[available_macro_ids].dropna() if available_macro_ids else pd.DataFrame()
        print(f"  Processed {len(available_macro_ids)} macro indicators.")
        pbar.update(1)
        
        # 3. Causal Network Construction
        print("\n[Step 3/9] Building causal network (dev-period only)...")
        tickers = [t for t in clean_data.columns if "_volume" not in t]
        asset_tickers = [t for t in tickers if t not in macro_ids]
        ce = engine.CausalEngine(asset_tickers)
        graph_macro = [m for m in macro_ids[:5] if m in clean_data.columns]
        dev_for_graph = clean_data.loc[:config.TRAIN_END]
        graph = ce.build_network(dev_for_graph[asset_tickers[:15] + graph_macro], max_lag=2)
        pbar.update(1)
        
        # 4. Advanced Feature Engineering
        print("\n[Step 4/9] Engineering decision features (Including Fractional Memory)...")
        fe = engine.FeatureEngineer()
        all_features = fe.compute_all_features(clean_data, graph)
        market_flows = fe.get_flow_series(clean_data)
        print(f"  Generated {all_features.shape[1]} features across the universe.")
        pbar.update(1)
        
        # 5. Unified Model Initialization
        print("\n[Step 5/9] Initializing Unified Model...")
        regimes = pd.Series(0, index=clean_data.index)
        pbar.update(1)
        
        # 6. Symbolic Optimization
        print(f"\n[Step 6/9] Evolving unified alpha formula (Dev period: {config.START_DATE} to {config.TRAIN_END})...")
        dev_prices = clean_data.loc[:config.TRAIN_END]
        dev_features = all_features.loc[:config.TRAIN_END]
        dev_regimes = regimes.loc[:config.TRAIN_END] if isinstance(regimes, pd.Series) else pd.Series(0, index=dev_prices.index)
        
        print(f"  Training on {len(dev_prices)} history points with {all_features.shape[1]} features.")
        if lightweight_mode:
            ga_params = {"n_gen": 25, "pop_size": 300, "max_depth": 4}
        else:
            ga_params = {"n_gen": 100, "pop_size": 1500, "max_depth": 7}

        optimizer = engine.RegimeAwareGeneticOptimizer(
            dev_prices,
            dev_features,
            dev_regimes,
            n_gen=ga_params["n_gen"],
            pop_size=ga_params["pop_size"],
            max_depth=ga_params["max_depth"],
        )
        regime_models = optimizer.optimize()
        pbar.update(1)

        # Save trained model for execution testing (phase 2)
        os.makedirs("saved_models", exist_ok=True)
        model_timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_tag = "lw" if lightweight_mode else "full"
        model_path = os.path.join("saved_models", f"model_{model_timestamp}_{model_tag}.pkl")
        model_bundle = {
            "metadata": {
                "trained_at": model_timestamp,
                "train_start": config.START_DATE,
                "train_end": config.TRAIN_END,
                "holdout_start": config.HOLDOUT_START,
                "holdout_end": config.HOLDOUT_END,
                "features_count": all_features.shape[1],
                "assets_count": len([c for c in clean_data.columns if "_volume" not in c]),
                "lightweight": lightweight_mode,
                "ga_params": ga_params,
            },
            "regime_models": regime_models,
        }
        with open(model_path, "wb") as f:
            pickle.dump(model_bundle, f)
        print(f"  Saved trained model to {model_path}")
        
        # 7. TRUE HOLDOUT Walk-Forward
        print(f"\n[Step 7/9] Running TRUE HOLDOUT expanding walk-forward ({config.HOLDOUT_START}..{config.HOLDOUT_END})...")
        if skip_retraining:
            print("  (Using static pre-trained model for holdout period)")
        else:
            print("  (Using periodic walk-forward retraining - this will take a long time)")
            
        bt = engine.Backtester(dm, engine)
        backtest_results = bt.walk_forward_test_expanding(
            prices_df=clean_data,
            feature_df=all_features,
            macro_df=macro_data,
            start_date=config.HOLDOUT_START,
            end_date=config.HOLDOUT_END,
            retrain_every_n_months=getattr(config, "RETRAIN_EVERY_N_MONTHS", 3),
            top_n=8,
            skip_retraining=skip_retraining,
            initial_models=regime_models if skip_retraining else None,
            collect_predictions=True,
            ga_params=ga_params if not skip_retraining else None,
        )
        pbar.update(1)
        
        if "error" not in backtest_results:
            print(f"\nOUT-OF-SAMPLE PERFORMANCE (HOLD-OUT: {config.HOLDOUT_START} to {config.HOLDOUT_END})")
            print(f"="*50)
            print(f"Annualized Return:    {backtest_results['annualized_return']:>10.2%}")
            print(f"Annualized Vol:      {backtest_results['annualized_vol']:>10.2%}")
            print(f"Sharpe Ratio:        {backtest_results['sharpe_ratio']:>10.2f}")
            print(f"Max Drawdown:        {backtest_results['max_drawdown']:>10.2%}")
            
            avg_exp = backtest_results.get('weights_history', pd.DataFrame())
            if 'exposure' in avg_exp.columns:
                print(f"Avg Portfolio Exp:   {avg_exp['exposure'].mean():>10.2%}")
            
            print(f"Selection Alpha:     {backtest_results.get('selection_alpha', 0):>10.2%}")
            print(f"Execution Alpha:     {backtest_results.get('vol_targeting_impact', 0):>10.2%} (Risk Mgmt Boost)")
            print(f"Model Hit Rate:      {backtest_results.get('hit_rate', 0):>10.2%} (vs SPY)")
            
            print(f"\nTRADE ANALYTICS (HOLDOUT PERIOD):")
            print("-" * 50)
            trade_df = backtest_results.get('trade_log', pd.DataFrame())
            if not trade_df.empty:
                wins = trade_df[trade_df['return'] > 0]['return']
                losses = trade_df[trade_df['return'] <= 0]['return']
                print(f"  Total Trades:      {len(trade_df)}")
                print(f"  Win Rate:          {len(wins)/len(trade_df):>10.2%}")
                print(f"  Avg Win:           {wins.mean():>10.2%}")
                print(f"  Avg Loss:          {losses.mean():>10.2%}")
                p_sum = wins.sum()
                l_sum = abs(losses.sum())
                profit_factor = p_sum / l_sum if l_sum > 0 else 0
                print(f"  Profit Factor:     {profit_factor:>10.2f}")
                
                print(f"\nSECTOR ALPHA BREAKDOWN (WHICH MACRO THEMES ARE WORKING?):")
                print("-" * 50)
                sector_stats = trade_df.groupby('sector')['return'].agg(['sum', 'count'])
                sector_stats = sector_stats.sort_values('sum', ascending=False)
                for sector, row in sector_stats.iterrows():
                    print(f"  {sector:<15}: {row['sum']:>8.2%} ({int(row['count'])} trades)")

            print(f"\nGA EVOLUTION METRICS:")
            print("-"*50)
            print(f"  Final Gen Best IC: {backtest_results.get('ga_best_ic', 0):.4f}")
            
            quality = backtest_results.get("model_quality", {})
            if quality:
                print(f"\nMODEL QUALITY (OUTPERFORMANCE FOCUS):")
                print("-"*50)
                print(f"  Decile Spread:     {quality.get('decile_spread', 0):>10.4%} (Top decile - Bottom decile)")
                print(f"  Long/Short:        {quality.get('long_short', 0):>10.4%} (Top 10% - Bottom 10%)")
                print(f"  Top-N Alpha:       {quality.get('top_n_alpha', 0):>10.4%} (Selected - Unselected)")

            # Execution comparison table
            shadow_stats = backtest_results.get("shadow_stats", {})
            model_only_stats = backtest_results.get("model_only_stats", {})
            if shadow_stats or model_only_stats:
                print(f"\nEXECUTION COMPARISON (ANNUALIZED):")
                print("-"*50)
                print(f"  Executed  | Ret: {backtest_results['annualized_return']:>7.2%} | Vol: {backtest_results['annualized_vol']:>7.2%} | Sharpe: {backtest_results['sharpe_ratio']:>5.2f}")
                if shadow_stats:
                    print(f"  Shadow    | Ret: {shadow_stats.get('annualized_return', 0):>7.2%} | Vol: {shadow_stats.get('annualized_vol', 0):>7.2%} | Sharpe: {shadow_stats.get('sharpe_ratio', 0):>5.2f}")
                if model_only_stats:
                    print(f"  ModelOnly | Ret: {model_only_stats.get('annualized_return', 0):>7.2%} | Vol: {model_only_stats.get('annualized_vol', 0):>7.2%} | Sharpe: {model_only_stats.get('sharpe_ratio', 0):>5.2f}")
            
            print(f"\nTOP 15 ALPHA-POSITIVE TICKERS:")
            print("-"*50)
            asset_stats = backtest_results.get('asset_stats')
            if isinstance(asset_stats, pd.DataFrame) and not asset_stats.empty:
                top_perf = asset_stats.sort_values('sum', ascending=False).head(15)
                for ticker, row in top_perf.iterrows():
                    print(f"  {ticker:<10}: {row['sum']:>8.2%}")
            
            print(f"\nTOP 10 NEGATIVE ALPHA TICKERS (DETRACTORS):")
            print("-"*50)
            if isinstance(asset_stats, pd.DataFrame) and not asset_stats.empty:
                bot_perf = asset_stats.sort_values('sum', ascending=True).head(10)
                for ticker, row in bot_perf.iterrows():
                    print(f"  {ticker:<10}: {row['sum']:>8.2%}")
            
            print(f"\nCURRENT LIVE RECOMMENDATIONS (LATEST REBALANCE):")
            print("-"*50)
            holdings = backtest_results.get('current_holdings', {})
            if holdings:
                sorted_h = sorted(holdings.items(), key=lambda x: x[1], reverse=True)
                for ticker, weight in sorted_h:
                    if weight > 0.001:
                        print(f"  {ticker:<10}: {weight:>8.2%}")
            
            print(f"\nFEATURE ALPHA LOADINGS (DIRECTIONAL PREDICTIVE POWER):")
            print("-"*50)
            loadings = pd.Series(backtest_results.get('feature_loadings', {}))
            if not loadings.empty:
                top_features = loadings.abs().sort_values(ascending=False).head(12)
                for feat, _ in top_features.items():
                    direction = "bullish" if loadings[feat] > 0 else "bearish"
                    print(f"  {feat:<25}: {loadings[feat]:>8.4f} ({direction})")
            print("="*50 + "\n")

        # Save predictions bundle for Phase 2 execution testing
        pred_tag = "wf-skip" if skip_retraining else "wf-retrain"
        pred_tag = f"{pred_tag}_{model_tag}"
        pred_timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("saved_models", exist_ok=True)
        predictions = backtest_results.get("predictions") if isinstance(backtest_results, dict) else None
        if predictions:
            asset_cols = [
                c for c in clean_data.columns
                if c not in macro_ids and "_volume" not in c and not c.endswith(("_open", "_high", "_low", "_vwap"))
            ]
            pred_path = os.path.join("saved_models", f"predictions_{pred_timestamp}_{pred_tag}.pkl")
            with open(pred_path, "wb") as f:
                pickle.dump(
                    {
                        "metadata": {
                            "trained_at": pred_timestamp,
                            "train_start": config.START_DATE,
                            "train_end": config.TRAIN_END,
                            "holdout_start": config.HOLDOUT_START,
                            "holdout_end": config.HOLDOUT_END,
                            "top_n_default": 8,
                            "walkforward_retrain": not skip_retraining,
                            "source": "run.py",
                            "lightweight": lightweight_mode,
                            "ga_params": ga_params,
                        },
                        "predictions": predictions,
                        "asset_cols": asset_cols,
                        "clean_data": clean_data,
                    },
                    f,
                )
            print(f"  Saved predictions for Phase 2: {pred_path}")
        
        # 8. Save Results
        print("\n[Step 8/9] Saving results to results.pkl...")
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
        pbar.update(1)
        
        # 9. Show Dashboard
        print("\n[Step 9/9] Launching Interactive Symbolic Dashboard...")
        import dashboard
        pbar.update(1)
        dashboard.show_popup_dashboard()

if __name__ == "__main__":
    main()
