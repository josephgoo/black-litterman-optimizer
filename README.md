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

## Tech Stack
- Python 3.10+
- pandas, numpy, matplotlib
- yfinance (data)
- cvxpy (optional if you later add box/budget constraints)

## Quickstart
1) Create environment and install dependencies

```bash
# Option A: conda (recommended)
conda env create -f environment.yml
conda activate black-litterman-optimizer

# Option B: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Example usage (script outline):

```python
import pandas as pd
from src import data_loader as dl
from src import implied_returns as ir
from src import views as vw
from src import optimizer as opt
from src import plots

# 1) Fetch prices
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
prices = dl.fetch_prices(tickers, start="2018-01-01").prices

# 2) Estimate returns & covariance (annualized)
mu_hist, Sigma_hist = dl.returns_and_covariance(prices, method="log")

# 3) Market caps -> weights and equilibrium returns
market_caps = {t: 1.0 for t in tickers}  # placeholder; replace with real caps
w_mkt = ir.market_caps_to_weights(market_caps)
delta = 2.5  # risk aversion (example)
pi = ir.compute_pi(Sigma_hist, w_mkt, delta)

# 4) Create views (example: AAPL expected to outperform MSFT by 2% annually)
P_row, q = vw.relative_view(tickers, long="AAPL", short="MSFT", magnitude=0.02)
P, Q = vw.build_PQ([(P_row, q)])

# 5) BL posterior
bl = opt.black_litterman_posterior(Sigma_hist, pi, P, Q, tau=0.05)

# 6) Compare frontiers and plot
curves = opt.compare_mv_vs_bl(mu_hist, Sigma_hist, bl.mu_bl, bl.Sigma_bl)
fig, ax = plots.plot_frontiers(*curves.values())
```

## Notebooks
- `notebooks/Black-Litterman.ipynb`: A guided walkthrough of BL vs MV. Run it to reproduce the charts.

### Reproducing the notebook results (defaults)
- Tickers: AAPL, MSFT, GOOGL, AMZN
- Period: from 2018-01-01
- Risk-free: 0.0 
- τ (tau): 0.05
- View: AAPL expected to outperform MSFT by 2% annually (relative view)

Run the notebook cells top to bottom. It fetches data, estimates μ and Σ, computes π from betas, constructs the BL posterior, plots MV vs BL frontiers, and compares optimal portfolios.

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

Results vary with tickers, dates, τ, and views.

## Suggested next steps
- Estimation robustness: Ledoit–Wolf/OAS shrinkage Σ; factor-model covariances; compare frontiers.
- Sensitivity: τ and view-confidence sweeps with Sharpe/weight heatmaps.
- Constraints and costs: Long-only/leverage caps; turnover and transaction costs.
- Backtesting: Walk-forward with monthly rebalance and realistic costs.
- Attribution: Risk contributions, factor exposures, and view attribution.

## License
See `LICENSE`.
