import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import networkx as nx
import numpy as np
import seaborn as sns

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
    fig = plt.figure(figsize=(24, 18), dpi=100)
    fig.canvas.manager.set_window_title('Macro-Structural Engine: Symbolic & Regime-Aware')
    
    # 1. Walk-Forward Equity Curve
    ax1 = plt.subplot2grid((6, 4), (0, 0), colspan=3)
    ax1.plot(backtest['cumulative_returns'], color='#00ff00', label='Symbolic Regime Engine', linewidth=2)
    ax1.set_title('Out-of-Sample Walk-Forward Equity Curve')
    ax1.legend()
    ax1.grid(alpha=0.1)

    # 2. Performance Stats
    ax2 = plt.subplot2grid((6, 4), (0, 3))
    ax2.axis('off')
    stats_text = (
        f"SYMBOLIC ANALYTICS\n"
        f"-------------------\n"
        f"Ann. Return: {backtest['annualized_return']:.2%}\n"
        f"Sharpe:      {backtest['sharpe_ratio']:.2f}\n"
        f"Max DD:      {backtest['max_drawdown']:.2%}\n"
        f"Hit Rate:    {backtest['trade_analytics']['hit_rate']:.2%}\n"
    )
    ax2.text(0.05, 0.5, stats_text, fontsize=12, family='monospace', verticalalignment='center', color='#00ff00')

    # 3. Market Regimes (HMM)
    ax3 = plt.subplot2grid((6, 4), (1, 0), colspan=4)
    if 'regimes' in backtest:
        r = backtest['regimes']
        ax3.fill_between(r.index, 0, 1, where=(r==0), color='green', alpha=0.3, label='Regime 0 (Low Vol)')
        ax3.fill_between(r.index, 0, 1, where=(r==1), color='yellow', alpha=0.3, label='Regime 1 (Transition)')
        ax3.fill_between(r.index, 0, 1, where=(r==2), color='red', alpha=0.3, label='Regime 2 (High Vol)')
        ax3.set_title('Detected Market Regimes (Hidden Markov Model)')
        ax3.get_yaxis().set_visible(False)
        ax3.legend(loc='upper left', ncol=3, fontsize=8)

    # 4. Global Sector Net Flow Pressure
    ax4 = plt.subplot2grid((6, 4), (2, 0), colspan=4)
    if 'market_flows' in results:
        flows = results['market_flows']['sector_flows']
        wf_start = backtest['cumulative_returns'].index[0]
        flows = flows.loc[wf_start:]
        top_flow_sectors = flows.var().sort_values(ascending=False).head(6).index
        for sector in top_flow_sectors:
            ax4.plot(flows[sector], label=sector, alpha=0.7, linewidth=1.5)
        ax4.axhline(0, color='white', linestyle='--', alpha=0.5)
        ax4.set_title('Sector-Specific Net Flow Pressure (Aligned with WF)')
        ax4.legend(loc='upper left', fontsize=8, ncol=3)
        ax4.grid(alpha=0.1)

    # 5. Sector Exposure evolution
    ax5 = plt.subplot2grid((6, 4), (3, 0), colspan=2)
    weights_df = backtest['weights_history'].T
    top_sectors = weights_df.sum(axis=1).sort_values(ascending=False).head(8).index
    sns.heatmap(weights_df.loc[top_sectors], ax=ax5, cmap='magma', cbar_kws={'label': 'Weight'}, xticklabels=False)
    ax5.set_title('Dynamic Sector Positioning')

    # 6. Evolved Formula (Sample from first regime)
    ax6 = plt.subplot2grid((6, 4), (3, 2), colspan=2)
    ax6.axis('off')
    formula_text = "EVOLVED SYMBOLIC FORMULAS\n--------------------------\n"
    for r, model_data in results['regime_models'].items():
        # Truncate long formulas
        prog = str(model_data["formula"])
        if len(prog) > 80: prog = prog[:77] + "..."
        formula_text += f"Regime {r}: {prog}\n"
    ax6.text(0.05, 0.5, formula_text, fontsize=10, family='monospace', verticalalignment='center', color='#00ffff')

    # 7. Asset Alpha Contributors
    ax7 = plt.subplot2grid((6, 4), (4, 0), colspan=2)
    top_assets = backtest['asset_stats'].sort_values('sum', ascending=False).head(12)
    ax7.barh(top_assets.index, top_assets['sum'], color='#00ff00', alpha=0.6)
    ax7.set_title('Top 12 Out-of-Sample Assets')
    ax7.invert_yaxis()

    # 8. Causal Map
    ax8 = plt.subplot2grid((6, 4), (5, 0), colspan=4)
    G = results['graph']
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax8, node_size=80, node_color='#1f77b4', alpha=0.4)
    nx.draw_networkx_edges(G, pos, ax=ax8, edge_color='#555', alpha=0.2, width=0.3)
    nx.draw_networkx_labels(G, pos, ax=ax8, font_size=5, font_color='#fff')
    ax8.set_title('Causal Influence Propagation Network')
    ax8.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show_popup_dashboard()
