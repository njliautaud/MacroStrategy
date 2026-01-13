import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import numpy as np
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

def make_fullscreen(event, fig, ax):
    # Create a new window for the fullscreen plot
    top = tk.Toplevel(fig.canvas.manager.window)
    top.title(ax.get_title())
    
    full_fig = Figure(figsize=(12, 8), dpi=100)
    full_ax = full_fig.add_subplot(111)
    
    # Copy content from the original axis to the new axis
    for line in ax.get_lines():
        full_ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), 
                    linestyle=line.get_linestyle(), label=line.get_label(), linewidth=line.get_linewidth())
    
    full_ax.set_title(ax.get_title(), fontsize=16)
    full_ax.set_xlabel(ax.get_xlabel(), fontsize=12)
    full_ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    full_ax.legend(fontsize=10)
    full_ax.grid(True, alpha=0.2)
    
    canvas = FigureCanvasTkAgg(full_fig, master=top)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    toolbar = NavigationToolbar2Tk(canvas, top)
    toolbar.update()
    canvas_widget.focus_set()
    canvas.draw()

def load_results():
    if os.path.exists("results.pkl"):
        with open("results.pkl", "rb") as f:
            return pickle.load(f)
    return None

def show_popup_dashboard():
    results = load_results()
    if not results:
        print("No results found. Please run run.py first.")
        return

    backtest = results['backtest']
    
    plt.style.use('dark_background')

    # Create a scrollable Tkinter window to host the figure
    root = tk.Tk()
    root.title('Macro-Structural Engine: Symbolic & Regime-Aware (Scrollable)')
    root.geometry("1600x900")

    container = ttk.Frame(root)
    container.pack(fill="both", expand=True)

    canvas = tk.Canvas(container, background="black", highlightthickness=0)
    vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)

    vscroll.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = ttk.Frame(canvas)
    inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        # Keep inner frame width matched to canvas width
        canvas.itemconfig(inner_id, width=event.width)

    canvas.bind("<Configure>", _on_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    root.bind_all("<MouseWheel>", _on_mousewheel)

    # Larger figure for scrollable layout
    fig = plt.figure(figsize=(26, 34), dpi=90)
    
    # 1. Walk-Forward Equity Curve (Executed vs Shadow vs Model-Only)
    ax1 = plt.subplot2grid((9, 4), (0, 0), colspan=3, rowspan=2)
    
    rets = backtest['cumulative_returns']
    shadow = backtest.get('shadow_returns_no_vol', rets)
    model_only = backtest.get('model_only_returns', rets)
    
    ax1.plot(rets, color='#00ff00', label='Executed (Vol Targeted + Exits)', linewidth=3)
    ax1.plot(shadow, color='#00ffff', label='Shadow (HRP No-Vol)', linewidth=1.5, alpha=0.6)
    ax1.plot(model_only, color='#ffcc00', label='Model-Only (Equal Weight)', linewidth=1.5, alpha=0.5)
    
    if 'benchmark_returns' in backtest:
        ax1.plot(backtest['benchmark_returns'], color='#888888', linestyle='--', label='Control (SPY)', alpha=0.7)
    
    ax1.set_title('Engine Audit: Execution Impact vs. Raw Model Alpha', fontsize=14, pad=15)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.1)
    fig.canvas.mpl_connect('button_press_event', lambda event: make_fullscreen(event, fig, ax1) if event.inaxes == ax1 else None)

    # 2. Performance Stats Card (With Dual-Engine Analytics)
    ax2 = plt.subplot2grid((9, 4), (0, 3), rowspan=2)
    ax2.axis('off')
    
    p_val = backtest.get('p_value', 1.0)
    sig_color = '#00ff00' if p_val < 0.05 else '#ff4444'
    b_stats = backtest.get('bench_stats', {})
    
    stats_text = (
        f"DUAL-ENGINE ANALYTICS\n"
        f"-----------------------\n"
        f"Metric      Engine    SPY\n"
        f"Ann Ret:    {backtest['annualized_return']:>6.1%}  {b_stats.get('return', 0):>6.1%}\n"
        f"Ann Vol:    {backtest['annualized_vol']:>6.1%}  {b_stats.get('vol', 0):>6.1%}\n"
        f"Sharpe:     {backtest['sharpe_ratio']:>6.2f}  {b_stats.get('return', 0)/(b_stats.get('vol', 1)+1e-9):>6.2f}\n\n"
        f"ENGINE QUALITY AUDIT\n"
        f"-----------------------\n"
        f"Vol Impact: {backtest.get('vol_targeting_impact', 0):>+6.1%} (Risk Mgmt)\n"
        f"Model Alpha:{backtest.get('selection_alpha', 0):>+6.1%} (Selection)\n\n"
        f"VALIDATION (Holdout)\n"
        f"--------------------\n"
        f"T-Stat:     {backtest.get('t_stat', 0):.2f}\n"
        f"P-Value:    {p_val:.4f}\n"
        f"Placebo SR: {backtest.get('placebo_sharpe', 0.0):.2f}\n\n"
        f"GA PARAMS (DUAL)\n"
        f"--------------------\n"
        f"PRED POP: 1000 | GEN: 50\n"
        f"EXEC POP: 300  | GEN: 20"
    )
    ax2.text(0.0, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center', color='#ffffff', fontweight='bold')
    ax2.text(0.0, 0.05, "STRATEGY: " + ("SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"), color=sig_color, fontsize=11, family='monospace', fontweight='bold')

    # ... (Keep regimes and flows)

    # 3. Structural Pressure Heatmap (Rolling Cross-Asset Pressure)
    ax3 = plt.subplot2grid((9, 4), (2, 0), colspan=4)
    if 'backtest' in results and 'weights_history' in backtest:
        # Instead of regimes, show rolling mean of group flows to see rotation patterns
        if 'market_flows' in results and 'group_flows' in results['market_flows']:
            flows = results['market_flows']['group_flows']
            wf_start = rets.index[0]
            # Aggregate to weekly/monthly for pattern visibility
            rotation_data = flows.loc[wf_start:].resample('W').mean().T
            sns.heatmap(rotation_data, ax=ax3, cmap='RdYlGn', center=0, cbar=False, xticklabels=False)
            ax3.set_title('Cross-Asset Flow Rotation (Patterns of Pressure)', fontsize=12)
            ax3.set_ylabel('Groups', fontsize=9)
            ax3.tick_params(axis='y', labelsize=7)
        else:
            ax3.text(0.5, 0.5, 'Flow Data Missing', ha='center')
    fig.canvas.mpl_connect('button_press_event', lambda event: make_fullscreen(event, fig, ax3) if event.inaxes == ax3 else None)

    # 4. Global Flow Pressure by Group (equity sectors + ETF groups + commodities/proxies)
    ax4 = plt.subplot2grid((9, 4), (3, 0), colspan=4)
    if 'market_flows' in results and 'group_flows' in results['market_flows']:
        flows = results['market_flows']['group_flows']
        wf_start = rets.index[0]
        flows = flows.loc[wf_start:]
        # Plot top groups by variance, but keep names meaningful and stable
        top_groups = flows.var().sort_values(ascending=False).head(12).index
        for grp in top_groups:
            ax4.plot(flows[grp], label=grp, alpha=0.85, linewidth=1.3)
        ax4.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax4.set_title('Structural Flow Pressure by Group (eq_*, etf_*, cmd_*, proxy_*)', fontsize=12)
        ax4.legend(loc='upper left', fontsize=8, ncol=4)
        ax4.grid(alpha=0.1)

    # 5. Dynamic Sector Positioning (Heatmap)
    ax5 = plt.subplot2grid((9, 4), (4, 0), colspan=2, rowspan=2)
    weights_df = backtest['weights_history'].T
    sns.heatmap(weights_df, ax=ax5, cmap='magma', cbar_kws={'label': 'Weight'}, xticklabels=False)
    ax5.set_title('Dynamic Sector Positioning', fontsize=12)
    ax5.set_ylabel('Sectors', fontsize=10)

    # 6. Directional Alpha Contribution (Centered at 0)
    ax6 = plt.subplot2grid((9, 4), (4, 2), colspan=2, rowspan=2)
    loadings = pd.Series(backtest.get('feature_loadings', {}))
    loadings = loadings.sort_values(ascending=True)
    
    # Color coding: Green for positive predictors, Red for negative predictors
    colors = ['#00ff00' if x > 0 else '#ff4444' for x in loadings.values]
    ax6.barh(loadings.index, loadings.values, color=colors, alpha=0.8)
    ax6.axvline(0, color='white', linewidth=1, alpha=0.5)
    ax6.set_title('Feature Alpha Loading (Directional Influence)', fontsize=12)
    ax6.tick_params(axis='y', labelsize=8)
    ax6.grid(alpha=0.1, axis='x')

    # ... (Keep formula card)

    # ... (Keep formula card)

    # 7. Evolved Formula Card & Current Recommendations
    ax7 = plt.subplot2grid((9, 4), (6, 0), colspan=4)
    ax7.axis('off')
    
    # Left side: Formulas
    formula_text = "EVOLVED REGIME FORMULAS\n" + "="*25 + "\n"
    for r, model_data in results['regime_models'].items():
        prog = str(model_data["formula"])
        if len(prog) > 80: prog = prog[:77] + "..."
        formula_text += f"REGIME {r}: {prog}\n"
    ax7.text(0.01, 0.5, formula_text, fontsize=9, family='monospace', verticalalignment='center', color='#00ffff')
    
    # Right side: Current Live Snapshot
    current_h = backtest.get('current_holdings', {})
    holdings_text = "CURRENT LIVE RECOMMENDATIONS\n" + "="*28 + "\n"
    if current_h:
        # Sort by weight
        sorted_h = sorted(current_h.items(), key=lambda x: x[1], reverse=True)
        for ticker, weight in sorted_h:
            holdings_text += f"{ticker:<10} | {weight:>6.1%}\n"
    else:
        holdings_text += "No active recommendations."
    ax7.text(0.65, 0.5, holdings_text, fontsize=10, family='monospace', verticalalignment='center', color='#ffcc00', fontweight='bold')

    # 8. Asset Alpha Contributors
    ax8 = plt.subplot2grid((9, 4), (7, 0), colspan=2, rowspan=2)
    top_assets = backtest['asset_stats'].sort_values('sum', ascending=False).head(15)
    # Color coding: Green for positive, Red for negative
    colors = ['#00ff00' if x > 0 else '#ff4444' for x in top_assets['sum']]
    ax8.barh(top_assets.index, top_assets['sum'], color=colors, alpha=0.7)
    ax8.set_title('Top 15 Alpha-Positive Tickers', fontsize=12)
    ax8.invert_yaxis()
    ax8.tick_params(axis='y', labelsize=9)
    ax8.grid(alpha=0.1, axis='x')

    # 9. Inter-Market Dependency Monitor (Commodity/Energy/Metals -> Equities focus)
    ax9 = plt.subplot2grid((9, 4), (7, 2), colspan=2, rowspan=2)
    corr = backtest.get('correlation_comm_eq')
    if corr is not None and not corr.empty:
        sample_size = min(len(corr), 28)
        sns.heatmap(corr.iloc[:sample_size, :sample_size], ax=ax9, cmap='RdYlGn', center=0, annot=False, cbar=False)
        ax9.set_title('Inter-Market Dependency (Commodities + Proxies ↔ Equities)', fontsize=12)
        ax9.tick_params(axis='both', labelsize=6)
    elif 'correlation_matrix' in backtest:
        corr2 = backtest['correlation_matrix']
        sample_size = min(len(corr2), 28)
        sns.heatmap(corr2.iloc[:sample_size, :sample_size], ax=ax9, cmap='RdYlGn', center=0, annot=False, cbar=False)
        ax9.set_title('Inter-Market Dependency (Full Matrix)', fontsize=12)
        ax9.tick_params(axis='both', labelsize=6)
    else:
        ax9.text(0.5, 0.5, 'Matrix Data Missing', ha='center')

    plt.tight_layout()

    # Render figure into Tkinter and pack
    fig_canvas = FigureCanvasTkAgg(fig, master=inner)
    fig_canvas.draw()
    fig_widget = fig_canvas.get_tk_widget()
    fig_widget.pack(fill="both", expand=True)

    root.mainloop()

if __name__ == '__main__':
    show_popup_dashboard()
