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
- `dashboard.py`: Interactive visualization dashboard using Dash and Plotly.
- `run.py`: The main execution script to run the entire pipeline.

## Performance & Visualization
The engine now includes a **Walk-Forward Backtest** system and an **Interactive Dashboard**:
1.  **Run Analysis**: `python run.py` will fetch the latest data, compute causal relationships, and simulate a medium-term strategy.
2.  **View Results**: `python dashboard.py` launches a web server (typically at `http://127.0.0.1:8050`) where you can see:
    - **Equity Curve**: The cumulative performance of the structural alpha strategy.
    - **Performance Metrics**: Annualized Return, Sharpe Ratio, and Max Drawdown.
    - **Causal Network Graph**: Visualizing how pressure propagates between assets and macro factors.
    - **Scoring Leaderboard**: The top-ranked assets by current structural and flow confirmation.

## Current Phase: Implementation & Integration
The core engine and data layers are implemented. Future updates will focus on refining the hybrid GA/DL model parameters and expanding the asset universe.
