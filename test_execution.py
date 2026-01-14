import glob
import os
import pickle
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import engine
import config


def select_prediction_file():
    files = sorted(glob.glob(os.path.join("saved_models", "predictions_*.pkl")), reverse=True)
    if not files:
        print("No prediction bundles found in saved_models/. Run train_and_predict.py first.")
        return None

    models = []
    for path in files:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            meta = data.get("metadata", {})
            # Determine retrain status for display
            wf_retrain = meta.get("walkforward_retrain")
            retrain_status = "Retrained" if wf_retrain is True else ("Static" if wf_retrain is False else "Unknown")
            
            models.append(
                {
                    "path": path,
                    "trained_at": meta.get("trained_at", "unknown"),
                    "train_end": meta.get("train_end", ""),
                    "holdout_end": meta.get("holdout_end", ""),
                    "retrain_status": retrain_status,
                }
            )
        except Exception:
            continue

    models.sort(key=lambda m: m["trained_at"], reverse=True)

    root = tk.Tk()
    root.title("Select Prediction Model")
    root.geometry("800x300")

    label = tk.Label(root, text="Select a prediction bundle (newest first):")
    label.pack(pady=8)

    listbox = tk.Listbox(root, width=120, height=10)
    for idx, m in enumerate(models):
        display_text = f"[{m['retrain_status']}] {m['trained_at']} | train_end={m['train_end']} | holdout_end={m['holdout_end']} | {os.path.basename(m['path'])}"
        listbox.insert(idx, display_text)
    listbox.select_set(0)
    listbox.pack(padx=10, pady=5)

    selected = {"path": None}

    def on_select():
        sel = listbox.curselection()
        if sel:
            selected["path"] = models[sel[0]]["path"]
        root.destroy()

    btn = tk.Button(root, text="Use Selected", command=on_select)
    btn.pack(pady=8)

    root.mainloop()
    return selected["path"]


def compute_metrics(returns):
    if returns.empty:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / (ann_vol + 1e-9)
    equity = (1 + returns).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}


def scale_weights(raw_weights, hist_subset, method):
    target_vol = method.get("target_vol", 0.15)
    vol_mode = method.get("vol_mode", "std")

    if vol_mode == "downside":
        port_rets = hist_subset.dot(raw_weights)
        downside = port_rets[port_rets < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) else port_rets.std() * np.sqrt(252)
        port_vol = downside_vol
    else:
        port_vol = np.sqrt(np.dot(np.dot(raw_weights, hist_subset.cov() * 252), raw_weights))

    exposure_scale = target_vol / (port_vol + 1e-9)
    exposure_scale = min(exposure_scale, 1.5)
    return raw_weights * exposure_scale


def apply_execution_method(pred_bundle, method):
    clean_data = pred_bundle["clean_data"]
    predictions = pred_bundle["predictions"]
    asset_cols = pred_bundle["asset_cols"]

    returns_df = clean_data[asset_cols].pct_change().dropna()
    rebalance_dates = sorted(predictions.keys())
    if len(rebalance_dates) < 2:
        return {"error": "Insufficient rebalance dates."}

    top_n = method.get("top_n", pred_bundle.get("metadata", {}).get("top_n_default", 8))
    stop_loss_mult = method.get("stop_loss_mult", 2.0)
    take_profit = method.get("take_profit")
    trailing_start = method.get("trailing_start")
    trailing_gap = method.get("trailing_gap")

    portfolio_returns = []

    for i in range(len(rebalance_dates) - 1):
        start = rebalance_dates[i]
        end = rebalance_dates[i + 1]
        scores = predictions[start]

        selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
        if not selected:
            continue

        hist_subset = returns_df.loc[:start].tail(252)[selected]
        if hist_subset.empty:
            continue

        raw_weights = engine.PortfolioOptimizer.get_hrp_weights(hist_subset)
        final_weights = scale_weights(raw_weights, hist_subset, method)

        mask = (returns_df.index > start) & (returns_df.index <= end)
        period_returns = returns_df.loc[mask, selected]
        if period_returns.empty:
            continue

        asset_vols = hist_subset.std()
        cum_ret = {t: 0.0 for t in selected}
        peak_ret = {t: 0.0 for t in selected}
        exited = {t: False for t in selected}

        daily_port_rets = []
        for _, row in period_returns.iterrows():
            active_weights = final_weights.copy()
            for t in selected:
                if exited[t]:
                    active_weights[t] = 0.0
                    continue

                # Update cumulative and peak returns
                cum_ret[t] = (1 + cum_ret[t]) * (1 + row[t]) - 1
                peak_ret[t] = max(peak_ret[t], cum_ret[t])

                # Baseline stop-loss
                if row[t] < -(asset_vols[t] * stop_loss_mult):
                    exited[t] = True
                    active_weights[t] = 0.0
                    continue

                # Take-profit exit
                if take_profit is not None and cum_ret[t] >= take_profit:
                    exited[t] = True
                    active_weights[t] = 0.0
                    continue

                # Trailing stop exit
                if trailing_start is not None and trailing_gap is not None:
                    if peak_ret[t] >= trailing_start and cum_ret[t] <= (peak_ret[t] - trailing_gap):
                        exited[t] = True
                        active_weights[t] = 0.0
                        continue

            daily_port_rets.append((row * active_weights).sum())

        portfolio_returns.append(pd.Series(daily_port_rets, index=period_returns.index))

    if not portfolio_returns:
        return {"error": "No returns generated."}

    all_returns = pd.concat(portfolio_returns).sort_index()
    metrics = compute_metrics(all_returns)
    equity = (1 + all_returns).cumprod()
    return {"returns": all_returns, "equity": equity, "metrics": metrics}


def show_comparison(results):
    fig = Figure(figsize=(16, 10), dpi=100)
    ax_curve = fig.add_subplot(2, 1, 1)
    ax_table = fig.add_subplot(2, 1, 2)
    ax_table.axis("off")

    # Equity curves
    for name, res in results.items():
        if "equity" in res:
            ax_curve.plot(res["equity"], label=name)
    ax_curve.set_title("Execution Method Equity Curves (Holdout)")
    ax_curve.grid(alpha=0.2)
    ax_curve.legend(fontsize=8, ncol=2)

    # Metrics table
    lines = ["METHOD | ANN_RET | ANN_VOL | SHARPE | MAX_DD"]
    lines.append("-" * 60)
    for name, res in results.items():
        if "metrics" not in res:
            lines.append(f"{name} | ERROR")
            continue
        m = res["metrics"]
        lines.append(
            f"{name:<18} | {m['ann_return']:.2%} | {m['ann_vol']:.2%} | {m['sharpe']:.2f} | {m['max_dd']:.2%}"
        )
    ax_table.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace")

    root = tk.Tk()
    root.title("Execution Method Comparison")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.draw()

    root.mainloop()


def main():
    pred_path = select_prediction_file()
    if not pred_path:
        return

    with open(pred_path, "rb") as f:
        pred_bundle = pickle.load(f)

    # Define execution methods to compare (edit here for quick iteration)
    execution_methods = [
        {"name": "Baseline_StdVol_15", "vol_mode": "std", "target_vol": 0.15, "stop_loss_mult": 2.0},
        {"name": "DownsideVol_15", "vol_mode": "downside", "target_vol": 0.15, "stop_loss_mult": 2.0},
        {"name": "TakeProfit_10", "vol_mode": "std", "target_vol": 0.15, "stop_loss_mult": 2.0, "take_profit": 0.10},
        {"name": "Trailing_8_5", "vol_mode": "std", "target_vol": 0.15, "stop_loss_mult": 2.0, "trailing_start": 0.08, "trailing_gap": 0.05},
    ]

    results = {}
    for method in execution_methods:
        name = method["name"]
        res = apply_execution_method(pred_bundle, method)
        results[name] = res

    show_comparison(results)


if __name__ == "__main__":
    main()
