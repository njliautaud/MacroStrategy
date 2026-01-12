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
    fig = plt.figure(figsize=(22, 14))
    fig.canvas.manager.set_window_title('Macro-Structural Alpha Engine - Inflow/Outflow & Risk Analytics')
    
    # 1. Walk-Forward Equity Curve
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=3)
    ax1.plot(backtest['cumulative_returns'], color='#00ff00', label='Structural Alpha Engine', linewidth=2)
    if 'benchmark_returns' in backtest and backtest['benchmark_returns'] is not None:
        ax1.plot(backtest['benchmark_returns'], color='#888888', linestyle='--', label='Control (SPY)', alpha=0.7)
    ax1.set_title('Out-of-Sample Walk-Forward Equity Curve')
    ax1.legend()
    ax1.grid(alpha=0.1)

    # 2. Risk/Return Analytics
    ax2 = plt.subplot2grid((4, 4), (0, 3))
    ax2.axis('off')
    stats_text = (
        f"POSITIONING ANALYTICS\n"
        f"----------------------\n"
        f"Ann. Return: {backtest['annualized_return']:.2%}\n"
        f"Sharpe:      {backtest['sharpe_ratio']:.2f}\n"
        f"Sortino:     {backtest['sortino_ratio']:.2f}\n"
        f"Calmar:      {backtest['calmar_ratio']:.2f}\n"
        f"Max DD:      {backtest['max_drawdown']:.2%}\n"
        f"Win Rate:    {backtest['trade_analytics']['hit_rate']:.2%}\n"
        f"Total Trades: {backtest['trade_analytics']['total_trades']}\n"
    )
    ax2.text(0.05, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center', color='#00ff00')

    # 3. Sector Exposure Heatmap
    ax3 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
    weights_df = backtest['weights_history'].T
    top_sectors = weights_df.sum(axis=1).sort_values(ascending=False).head(10).index
    sns.heatmap(weights_df.loc[top_sectors], ax=ax3, cmap='viridis', cbar_kws={'label': 'Weight'}, xticklabels=False)
    ax3.set_title('Sector Exposure Evolution (Walk-Forward)')

    # 4. GA Feature Importance (Flow & Risk Focused)
    ax4 = plt.subplot2grid((4, 4), (1, 2), colspan=2)
    if 'feature_weights' in results:
        w = results['feature_weights']
        sorted_w = dict(sorted(w.items(), key=lambda item: abs(item[1]), reverse=True))
        colors = ['#00ff00' if v > 0 else '#ff4444' for v in sorted_w.values()]
        ax4.barh(list(sorted_w.keys()), list(sorted_w.values()), color=colors)
        ax4.set_title('GA Optimized Alpha Formula (Inflows vs Risk)')
        ax4.tick_params(axis='y', labelsize=8)

    # 5. Top Alpha Contributors (Individual Assets)
    ax5 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
    top_assets = backtest['asset_stats'].sort_values('sum', ascending=False).head(15)
    ax5.barh(top_assets.index, top_assets['sum'], color='#00ff00', alpha=0.6)
    ax5.set_title('Top 15 Alpha Contributors (Total Return Sum)')
    ax5.invert_yaxis()

    # 6. Sector Profitability (Flow Capture)
    ax6 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
    sector_stats = backtest['sector_stats'].sort_values('sum', ascending=False)
    ax6.barh(sector_stats.index, sector_stats['sum'], color=plt.cm.magma(np.linspace(0.3, 0.8, len(sector_stats))))
    ax6.set_title('Sector Profitability Leaderboard')
    ax6.invert_yaxis()

    # 7. Network Graph
    ax7 = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    G = results['graph']
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax7, node_size=120, node_color='#1f77b4', alpha=0.4)
    nx.draw_networkx_edges(G, pos, ax=ax7, edge_color='#555', alpha=0.2, width=0.4)
    nx.draw_networkx_labels(G, pos, ax=ax7, font_size=6, font_color='#fff')
    ax7.set_title('Causal Influence Propagation Map', fontsize=10)
    ax7.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show_popup_dashboard()
