"""
data_loader.py
--------------

This module handles data acquisition and preprocessing for the
Black–Litterman Bayesian Portfolio Optimizer project.

Functionality:
- Download historical asset prices from Yahoo Finance via the yfinance API.
- Align and clean price data (sort index, drop missing values).
- Compute periodic returns (simple or log) and annualized mean returns.
- Estimate annualized covariance matrices with optional ridge shrinkage.
- Provide a convenience helper to return (mu, Sigma) from price data.

Intended Use:
This script is designed as the first stage in the optimizer pipeline,
supplying processed market data to:
	1. implied_returns.py (for CAPM-based equilibrium returns, π)
	2. views.py (for subjective views construction)
	3. optimizer.py (for MV/BL portfolio construction and frontiers)
	4. evaluator.py (for portfolio performance stats)
	5. plots.py (for MV vs BL frontier visualization)

Notes:
- Internet access is required for live downloads; for offline workflows,
  read CSVs into a wide price DataFrame and call compute_returns/returns_and_covariance.
- TRADING_DAYS is set to 252 for annualization.

Author: Joseph Goo Wei Zhen
Date: 2025-08-08
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
	import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - yfinance may not be installed for tests
	yf = None  # type: ignore


TRADING_DAYS = 252


@dataclass
class PriceData:
	tickers: list[str]
	prices: pd.DataFrame  # index: DatetimeIndex, columns: tickers


def fetch_prices(
	tickers: Iterable[str],
	start: Optional[str] = None,
	end: Optional[str] = None,
	interval: str = "1d",
	auto_adjust: bool = True,
) -> PriceData:
	"""Download adjusted close prices from Yahoo Finance.

	Parameters
	- tickers: list of ticker strings
	- start, end: date strings (e.g., '2015-01-01')
	- interval: '1d', '1wk', etc.
	- auto_adjust: adjust for dividends/splits

	Returns: PriceData with prices in wide format.
	"""
	tickers = list(tickers)
	if yf is None:
		raise ImportError(
			"yfinance is not available. Install it or provide prices from file."
		)
	data = yf.download(
		tickers=tickers,
		start=start,
		end=end,
		interval=interval,
		auto_adjust=auto_adjust,
		progress=False,
		group_by="ticker",
	)

	# yfinance returns different shapes depending on single vs multi tickers
	if len(tickers) == 1:
		# Single ticker: DataFrame with columns like ['Open','High',...]
		prices = data[["Close"]].rename(columns={"Close": tickers[0]})
	else:
		# Multi ticker: Column MultiIndex (Ticker, Field)
		closes = {t: data[(t, "Close")] for t in tickers}
		prices = pd.DataFrame(closes)

	prices.index = pd.to_datetime(prices.index)
	prices = prices.sort_index()
	return PriceData(tickers=tickers, prices=prices)


def align_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Drop rows with any NA and ensure increasing date index."""
	out = df.copy()
	out.index = pd.to_datetime(out.index)
	out = out.sort_index()
	out = out.dropna(how="any")
	return out


def compute_returns(
	prices: pd.DataFrame,
	method: str = "log",
	dropna: bool = True,
) -> pd.DataFrame:
	"""Compute period returns from prices.

	method: 'log' or 'simple'
	Returns a DataFrame aligned to price index (first row will be NA if dropna=False).
	"""
	prices = align_data(prices)
	if method == "log":
		rets = np.log(prices / prices.shift(1))
	elif method == "simple":
		rets = prices.pct_change()
	else:
		raise ValueError("method must be 'log' or 'simple'")
	return rets.dropna(how="any") if dropna else rets


def annualize_returns(returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> pd.Series:
	"""Annualize mean returns from periodic returns (log-add or compounding simple)."""
	if returns.empty:
		return pd.Series(dtype=float)
	if (returns < -1).any().any():
		raise ValueError("Simple returns less than -100% detected; check inputs.")
	# Use arithmetic mean of periodic simple returns as an approximation
	mu = returns.mean() * periods_per_year
	return mu


def estimate_covariance(
	returns: pd.DataFrame,
	periods_per_year: int = TRADING_DAYS,
	shrinkage: Optional[float] = None,
) -> pd.DataFrame:
	"""Estimate annualized covariance matrix; optional ridge shrinkage toward diagonal.

	shrinkage: float in [0,1]; 0=no shrink, 1=diagonal only.
	"""
	if returns.empty:
		raise ValueError("returns is empty")
	Sigma = returns.cov() * periods_per_year
	if shrinkage is not None:
		if not (0.0 <= shrinkage <= 1.0):
			raise ValueError("shrinkage must be in [0,1]")
		diag = np.diag(np.diag(Sigma.values))
		Sigma = (1 - shrinkage) * Sigma + shrinkage * pd.DataFrame(diag, index=Sigma.index, columns=Sigma.columns)
	return Sigma


def returns_and_covariance(
	prices: pd.DataFrame,
	method: str = "log",
	periods_per_year: int = TRADING_DAYS,
	shrinkage: Optional[float] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
	"""Convenience: compute annualized mean returns and covariance from prices."""
	rets = compute_returns(prices, method=method, dropna=True)
	mu = annualize_returns(rets, periods_per_year=periods_per_year)
	Sigma = estimate_covariance(rets, periods_per_year=periods_per_year, shrinkage=shrinkage)
	return mu, Sigma

