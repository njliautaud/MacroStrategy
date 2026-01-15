import os
import pickle
import time

import pandas as pd
import numpy as np

import config
import data_manager
import engine


def build_data():
    dm = data_manager.DataManager()
    raw_data = dm.get_consolidated_data()
    raw_data = raw_data[~raw_data.index.duplicated(keep="first")]
    clean_data = dm.clean_data(raw_data)

    macro_ids = [sid for sub in config.ASSET_UNIVERSE["macro_indicators"].values() for sid in sub]
    available_macro_ids = [sid for sid in macro_ids if sid in clean_data.columns]
    macro_data = clean_data[available_macro_ids].dropna() if available_macro_ids else pd.DataFrame()

    tickers = [t for t in clean_data.columns if "_volume" not in t]
    asset_tickers = [t for t in tickers if t not in macro_ids]
    ce = engine.CausalEngine(asset_tickers)
    graph_macro = [m for m in macro_ids[:5] if m in clean_data.columns]
    dev_for_graph = clean_data.loc[:config.TRAIN_END]
    graph = ce.build_network(dev_for_graph[asset_tickers[:15] + graph_macro], max_lag=2)

    fe = engine.FeatureEngineer()
    all_features = fe.compute_all_features(clean_data, graph)

    return clean_data, all_features, macro_data, graph


def train_model(dev_prices, dev_features):
    regimes = pd.Series(0, index=dev_prices.index)
    optimizer = engine.RegimeAwareGeneticOptimizer(dev_prices, dev_features, regimes)
    return optimizer.optimize()


def generate_predictions(clean_data, feature_df, regime_models, top_n=8):
    macro_ids = [sid for sub in config.ASSET_UNIVERSE["macro_indicators"].values() for sid in sub]
    asset_cols = [
        c for c in clean_data.columns
        if c not in macro_ids and "_volume" not in c and not c.endswith(("_open", "_high", "_low", "_vwap"))
    ]
    returns_df = clean_data[asset_cols].pct_change().dropna()
    rebalance_dates = returns_df.resample("ME").last().index
    rebalance_dates = rebalance_dates[(rebalance_dates >= config.HOLDOUT_START) & (rebalance_dates <= config.HOLDOUT_END)]

    predictions = {}
    model_data = next(iter(regime_models.values()))
    func = engine.gp.compile(model_data["formula"], model_data["pset"])
    scaler = model_data.get("scaler")

    for dt in rebalance_dates:
        scores = {}
        for t in asset_cols:
            raw_vec = [
                feature_df[f"{t}_{f}"].asof(dt) if f"{t}_{f}" in feature_df.columns else 0
                for f in config.FEATURE_SET
            ]
            input_vec = raw_vec
            if scaler is not None:
                input_vec = scaler.transform([raw_vec])[0]
                input_vec = np.clip(input_vec, -5, 5)
            try:
                s = func(*input_vec)
                scores[t] = s if pd.notna(s) else 0.0
            except Exception:
                scores[t] = 0.0

        # Keep full scores; execution scripts can select top_n dynamically.
        predictions[dt] = scores

    return predictions, asset_cols


def main():
    print("Training model and generating holdout predictions...")
    clean_data, all_features, macro_data, graph = build_data()

    dev_prices = clean_data.loc[:config.TRAIN_END]
    dev_features = all_features.loc[:config.TRAIN_END]
    print(f"Training on {len(dev_prices)} history points with {all_features.shape[1]} features.")
    regime_models = train_model(dev_prices, dev_features)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("saved_models", exist_ok=True)

    model_path = os.path.join("saved_models", f"model_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "metadata": {
                    "trained_at": timestamp,
                    "train_start": config.START_DATE,
                    "train_end": config.TRAIN_END,
                    "holdout_start": config.HOLDOUT_START,
                    "holdout_end": config.HOLDOUT_END,
                    "features_count": all_features.shape[1],
                    "assets_count": len([c for c in clean_data.columns if "_volume" not in c]),
                },
                "regime_models": regime_models,
            },
            f,
        )

    predictions, asset_cols = generate_predictions(clean_data, all_features, regime_models)
    pred_path = os.path.join("saved_models", f"predictions_{timestamp}.pkl")
    with open(pred_path, "wb") as f:
        pickle.dump(
            {
                "metadata": {
                    "trained_at": timestamp,
                    "model_path": model_path,
                    "train_start": config.START_DATE,
                    "train_end": config.TRAIN_END,
                    "holdout_start": config.HOLDOUT_START,
                    "holdout_end": config.HOLDOUT_END,
                    "top_n_default": 8,
                },
                "predictions": predictions,
                "asset_cols": asset_cols,
                "clean_data": clean_data,
            },
            f,
        )

    print(f"Saved model: {model_path}")
    print(f"Saved predictions: {pred_path}")
    print("Phase 1 complete. Run test_execution.py to compare execution methods.")


if __name__ == "__main__":
    main()
