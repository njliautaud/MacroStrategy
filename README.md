# Macro-Structural Alpha Engine

## Overview
The Macro-Structural Alpha Engine is a data-driven system designed to capture flows, pressures, and catalysts across various asset classes (equities, sectors, commodities, ETFs). It generates probability-weighted structural insights for medium-term investment decisions by modeling the complex, multi-level relationships between assets.

## Key Principles
- **Multi-Level Relationship Tracking**: Assets are analyzed within their sector, macro, and commodity contexts.
- **Causal Effect Capture**: Identifies first- and second-order causal relationships using numeric data.
- **Purely Numerical**: Avoids NLP and news bias, focusing strictly on price, volume, and macro indicators.
- **Temporal Smoothing**: Uses lagged features and smoothed data to detect trends over days, weeks, and months.

## Project Structure
- `config.py`: Central configuration, asset universe definition, and API keys.
- `data_manager.py`: Handles data ingestion from FMP Pro and FRED, caching, and cleaning.
- `engine.py`: The core engine containing feature engineering, causal network construction, hybrid DL+GA modeling, and scoring logic.
- `dashboard.py`: Native popup dashboard using Matplotlib (click-to-expand workflow).
- `run.py`: The main execution script to run the entire pipeline.

## Performance & Validation
The engine includes a **Walk-Forward Expanding Backtest** system with advanced validation:
1.  **Run Analysis**: `python run.py` fetches data, evolves math formulas via Symbolic Regression, and simulates a walk-forward strategy.
2.  **Advanced GA Search**: Uses a large population (1000) and deep trees (6 branches) to discover non-linear alpha formulas using primitives like `sin`, `cos`, `log`, `exp`, `tanh`.
3.  **Statistical Validation**:
    -   **Permutation Test**: Shuffles excess returns to calculate an empirical p-value vs luck.
    -   **Placebo Test**: Shifts features forward to confirm signal decay (anti-overfit check).
    -   **Annualized Alpha**: Tracks pure performance above the benchmark (SPY).
4.  **View Results**: The **Scrollable Native Dashboard** shows:
    -   **Equity Curve vs Luck Thresholds**: Cumulative performance with 1/3-sigma bands.
    -   **Market Regimes**: HMM-based classification (Low/High Vol).
    -   **Feature Alpha Loading**: Directional contribution (Positive/Negative) of each feature to out-of-sample alpha.
    -   **Inter-Market Dependency**: Sorted correlation matrix focusing on Commodity-Equity linkages.

## Universe Expansion & 2nd Order Effects
This project emphasizes **commodities and cross-asset linkages**:
- Commodity futures + liquid proxy ETFs are included with **commodity flow proxies**.
- Second-order effects are modeled via **cost-input baskets** (e.g., how metals/energy prices pressure industrial/tech margins).
- **Symbolic Formulas** are evolved for each market regime independently.

## Current Phase: Implementation & Integration
The core engine and data layers are implemented. Future updates will focus on refining the hybrid GA/DL model parameters and expanding the asset universe.
