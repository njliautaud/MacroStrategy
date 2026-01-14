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
        optimizer = engine.RegimeAwareGeneticOptimizer(dev_prices, dev_features, dev_regimes)
        regime_models = optimizer.optimize()
        pbar.update(1)

        # Save trained model for execution testing (phase 2)
        os.makedirs("saved_models", exist_ok=True)
        model_timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join("saved_models", f"model_{model_timestamp}.pkl")
        model_bundle = {
            "metadata": {
                "trained_at": model_timestamp,
                "train_start": config.START_DATE,
                "train_end": config.TRAIN_END,
                "holdout_start": config.HOLDOUT_START,
                "holdout_end": config.HOLDOUT_END,
                "features_count": all_features.shape[1],
                "assets_count": len([c for c in clean_data.columns if "_volume" not in c]),
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
        )
        pbar.update(1)
        
        if "error" not in backtest_results:
            print(f"\nBacktest Performance (OOS):")
            print(f"  Annualized Return: {backtest_results['annualized_return']:.2%}")
            print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {backtest_results['max_drawdown']:.2%}")

        # Save predictions bundle for Phase 2 execution testing
        pred_tag = "wf-skip" if skip_retraining else "wf-retrain"
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
