"""
market_portfolio.py
--------------------

Utilities for constructing and analyzing the market portfolio:
- Fetch market capitalizations (yfinance)
- Convert market caps to weights
- Compute market return and variance from μ, Σ, and weights
- Compute CAPM betas from Σ and market weights

Author: Joseph Goo Wei Zhen
Date: 2025-08-09
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore


def fetch_market_caps(tickers: List[str]) -> Dict[str, float]:
	"""Fetch current market capitalizations for a list of tickers using yfinance.

	Returns dict mapping ticker to market cap (USD).
	"""
	market_caps: Dict[str, float] = {}
	for t in tickers:
		info = yf.Ticker(t).info
		cap = info.get("marketCap", None)
		if cap is not None:
			market_caps[t] = cap
		else:
			print(f"Warning: No market cap found for {t}")
	return market_caps


def market_caps_to_weights(market_caps: Dict[str, float]) -> pd.Series:
	"""Normalize market caps to portfolio weights (sum to 1)."""
	s = pd.Series(market_caps, dtype=float)
	total = s.sum()
	if total <= 0:
		raise ValueError("Sum of market caps must be positive")
	return s / total


def market_return_and_variance(
	mu: pd.Series,
	Sigma: pd.DataFrame,
	w_mkt: pd.Series,
) -> Tuple[float, float]:
	"""Compute market portfolio expected return and variance.

	Returns (mkt_return, mkt_variance), both annualized to match μ and Σ.
	"""
	mu = mu.loc[w_mkt.index]
	Sigma = Sigma.loc[w_mkt.index, w_mkt.index]
	mkt_return = float(mu.dot(w_mkt))
	mkt_variance = float(w_mkt.values.T.dot(Sigma.values).dot(w_mkt.values))
	return mkt_return, mkt_variance
