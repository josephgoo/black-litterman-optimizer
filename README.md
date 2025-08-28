# Black–Litterman Bayesian Portfolio Optimizer

A practical Python project that implements the Black–Litterman (BL) model for portfolio construction, alongside classic mean–variance (MV) optimization. It provides utilities to fetch market data, compute CAPM equilibrium returns, specify investor views, blend them using BL, and compare efficient frontiers and optimal portfolios.

## Features
- Data loading from Yahoo Finance (yfinance)
- Return and covariance estimation (annualized)
- Market-implied equilibrium returns π via CAPM 
- View construction (absolute and relative) and uncertainty Ω
- Black–Litterman posterior returns/covariance
- Efficient frontier generation and visualization (MV vs BL)
- Tangent (max Sharpe) portfolio with robust handling when no risk-free asset exists
- Security Market Line (SML) plot and comparisons
 - Rolling backtests (monthly rebalance) for MV vs BL
 - Transaction costs and turnover tracking, including cost-sensitivity sweeps
 - Benchmark comparison vs S&P 500 (SPY)

## Tech Stack
- Python 3.10+
- pandas, numpy, matplotlib
- yfinance (data)
- cvxpy (optional if adding box/budget constraints)

## Quickstart
Create environment and install dependencies

```bash
# Option A: conda (recommended)
conda env create -f environment.yml
conda activate black-litterman-optimizer

# Option B: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Notebooks
- `notebooks/Black-Litterman1.ipynb`: A guided walkthrough of BL vs MV. Run it to reproduce the charts.
- `notebooks/Black-Litterman2.ipynb`: Rolling, monthly-rebalanced backtest comparing MV vs BL with a relative view.
	Includes turnover and transaction costs, cumulative return charts, performance tables,
	cost-sensitivity sweeps (varying bps), and a comparison vs S&P 500 (SPY).

### Reproducing the notebook results (defaults)
- Tickers: AAPL, MSFT, GOOGL, AMZN
- Period: from 2018-01-01
- Risk-free: 0.0 
- τ (tau): 0.05
- View: AAPL expected to outperform MSFT by 2% annually (relative view)

Run the notebook cells top to bottom. It fetches data, estimates μ and Σ, computes π from betas,
constructs the BL posterior, plots MV vs BL frontiers, and compares optimal portfolios.

For Part 2 (Backtest), the notebook:
- Uses a rolling 3-year lookback window with last-trading-day monthly rebalancing.
- Computes MV and BL weights each month (tangent portfolio if a risk-free is provided, otherwise a midpoint target-return portfolio).
- Tracks turnover and applies transaction costs on rebalance days (round-trip bps).
- Plots cumulative returns, reports CAGR/vol/Sharpe/max drawdown, and shows last-period weights.
- Adds a cost-sensitivity sweep and a benchmark comparison against SPY.

## Tests
Run unit tests:

```bash
pytest -q
```

## Findings (from the sample analysis)
- Midpoint comparison (BL frontier):
	- We compare MV and BL portfolios at the same target return (the BL midpoint). Expected returns match by construction; risk and allocations differ.
	- MV concentrates (and may short) historically strong assets (e.g., MSFT). BL diversifies more, reflecting the AAPL>MSFT view.
- SML insights: MSFT’s historical profile helps explain MV’s tilt without views.

Part 2 – Backtesting highlights:
- BL achieved higher Sharpe with lower drawdowns than MV once realistic transaction costs were applied.
- MV exhibited much higher turnover, leading to larger cost drag; BL’s blended prior/views stabilized allocations and reduced trading.
- Against SPY, BL provided competitive risk-adjusted returns with lower drawdowns in the sample window.

Results vary with tickers, dates, τ, and views.

## Suggested next steps
- Estimation robustness: Ledoit–Wolf/OAS shrinkage Σ; factor-model covariances; compare frontiers.
- Sensitivity: τ and view-confidence sweeps with Sharpe/weight heatmaps; add weekly/quarterly rebalance variants.
- Constraints and costs: Long-only/leverage caps; explicit transaction-cost-aware optimization (e.g., cvxpy).
- Attribution: Risk contributions, factor exposures, and view attribution.
- Benchmarking: Explore broader universes and alternative benchmarks (QQQ, equal-weight). 

## License
MIT License

Copyright (c) 2025 Joseph Goo Wei Zhen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
