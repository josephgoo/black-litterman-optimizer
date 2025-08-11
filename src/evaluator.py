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
	"""Basic stats: CAGR, vol, Sharpe, max drawdown."""
	if port_rets.empty:
		return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
	ann_ret = port_rets.mean() * periods_per_year
	ann_vol = port_rets.std(ddof=1) * np.sqrt(periods_per_year)
	sharpe = (ann_ret - risk_free) / (ann_vol + 1e-12)
	cum = (1 + port_rets).cumprod()
	peak = cum.cummax()
	dd = (cum / peak - 1).min()
	return {"cagr": float(ann_ret), "vol": float(ann_vol), "sharpe": float(sharpe), "max_dd": float(dd)}

