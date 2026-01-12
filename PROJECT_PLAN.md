# Macro-Structural Alpha Engine - Project Plan

## Overall Project Goal
Build a system that learns its own alpha by capturing structural pressures and flows in markets, providing actionable probability-weighted insights for medium-term trading.

## Phase 1: Foundation (Completed)
- [x] Defined Asset Universe in `config.py`
- [x] Implemented Data Ingestion Layer (`data_manager.py`)
- [x] Established caching mechanism for FMP and FRED data

## Phase 2: Feature Engineering & Causal Discovery (Completed)
- [x] Implemented structural, flow, and catalyst features
- [x] Developed Causal Engine using Granger Causality
- [x] Built hierarchical mapping logic

## Phase 3: Hybrid Modeling & Scoring (Completed)
- [x] Integrated LSTM-based Deep Learning for structural pressure (optional torch)
- [x] Implemented Genetic Algorithm (DEAP) for optimal feature weight discovery
- [x] Developed comprehensive feature set (Momentum, Volatility, Sector/Market Relative, Causal Influence)
- [x] Developed scoring aggregation using GA-optimized decision weights

## Phase 4: Validation & Visualization (Completed)
- [x] Built Walk-Forward Backtesting framework with real return calculations
- [x] Developed interactive Dash/Plotly dashboard with performance metrics and equity curves
- [x] Integrated 'stable' FMP endpoints for unrestricted data access
- [x] Created unified orchestration script (`run.py`) and results persistence
- [x] Implemented detailed Sector and Asset-level profitability tracking
- [x] Added trade-level analytics (Hit Rate, Avg Win/Loss)

## Phase 5: Refinement & Expansion (Planned)
- [ ] Optimize GA fitness functions for better backtest alignment
- [ ] Add GNN (Graph Neural Network) layer for deeper propagation analysis
- [ ] Integrate streaming data for real-time flow monitoring
- [ ] Scenario simulation for macro shocks (e.g., "What if rates rise 50bps?")
