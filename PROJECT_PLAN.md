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
- [x] Developed native Matplotlib popup dashboard with performance metrics and equity curves
- [x] Integrated 'stable' FMP endpoints for unrestricted data access
- [x] Created unified orchestration script (`run.py`) and results persistence
- [x] Implemented detailed Sector and Asset-level profitability tracking
- [x] Added trade-level analytics (Hit Rate, Avg Win/Loss)

## Phase 5: Refinement & Expansion (Completed)
- [x] Expanded asset universe with additional industry ETFs and commodity proxies (macro-first)
- [x] Added explicit second-order cost-input baskets (commodities → sector cost headwinds/tailwinds)
- [x] Implemented Advanced Symbolic Regression for alpha formulas
- [x] Integrated `eaMuPlusLambda` for robust mathematical formula evolution
- [x] Implemented Dual-Engine Architecture (Prediction vs Execution)
- [x] Developed learned Execution GA for optimal exit management
- [x] Added detailed model performance analytics (IC, Selection Alpha, Vol Targeting Impact)
- [x] Added Alpha metrics (Annualized Alpha, Information Ratio) and Statistical Validation (Permutation, Placebo)

## Phase 6: Execution & Advanced UI (In Progress)
- [x] Implemented Scrollable Matplotlib Dashboard for dense data visualization
- [ ] Implement dynamic volatility targeting in backtester
- [ ] Add adaptive trailing stop-losses based on regime-aware volatility
- [ ] Implement catalyst-driven exit logic
- [ ] Add zoom/scroll interactive windows for detailed graph analysis
- [ ] Scenario simulation for macro shocks (e.g., "What if rates rise 50bps?")
