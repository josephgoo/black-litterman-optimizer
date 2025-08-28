"""
evaluator.py
------------

Backtesting utilities and performance metrics for portfolio strategies.

Functionality:
- portfolio_returns: Compute time series returns for fixed weights.
- performance_stats: Report annualized return, volatility, Sharpe, and max drawdown.

Intended Use:
Evaluate MV or BL-derived portfolios on historical return series.

Author: Joseph Goo Wei Zhen
Date: 2025-08-08
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_returns(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
	"""Compute portfolio returns time series for given weights."""
	returns = returns[weights.index]
	return returns.dot(weights)


def performance_stats(
	port_rets: pd.Series,
	periods_per_year: int = 252,
	risk_free: float = 0.0,
) -> dict:
	"""Compute core performance statistics for a return series.

	Returns a dict with the following keys:
	- cagr (float): Annualized arithmetic return, computed as mean(port_rets) * periods_per_year.
	  Units are in return per year (e.g., 0.08 for 8%).
	- vol (float): Annualized volatility (standard deviation), computed as std(port_rets, ddof=1)
	  times sqrt(periods_per_year). Units are in return per year.
	- sharpe (float): Sharpe ratio using the provided risk_free (per-year) as the benchmark,
	  defined as (cagr - risk_free) / vol. A small epsilon is added to the denominator to
	  avoid division by zero.
	- max_dd (float): Maximum drawdown of the cumulative wealth curve built from port_rets,
	  defined as min(cum / cum.cummax() - 1). This is a non-positive number (0 means no drawdown,
	  -0.2 means a 20% peak-to-trough loss).
	"""
	if port_rets.empty:
		return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
	ann_ret = port_rets.mean() * periods_per_year
	ann_vol = port_rets.std(ddof=1) * np.sqrt(periods_per_year)
	sharpe = (ann_ret - risk_free) / (ann_vol + 1e-12)
	cum = (1 + port_rets).cumprod()
	peak = cum.cummax()
	dd = (cum / peak - 1).min()
	return {"cagr": float(ann_ret), "vol": float(ann_vol), "sharpe": float(sharpe), "max_dd": float(dd)}

