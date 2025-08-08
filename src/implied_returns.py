"""
implied_returns.py
-------------------

Equilibrium returns (π) and helper utilities for the Black–Litterman model.

Key functions:
- market_implied_risk_aversion: estimate risk aversion delta from market.
- market_caps_to_weights: convert market caps to market portfolio weights.
- compute_pi: compute equilibrium returns π = δ · Σ · w_mkt.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def market_caps_to_weights(market_caps: Dict[str, float]) -> pd.Series:
	"""Normalize market caps to weights (sums to 1)."""
	s = pd.Series(market_caps, dtype=float)
	total = s.sum()
	if total <= 0:
		raise ValueError("Sum of market caps must be positive")
	return s / total


def market_implied_risk_aversion(
	market_return: float,
	market_variance: float,
) -> float:
	"""Estimate risk aversion δ from CAPM: E[R_m] - r_f = δ · Var(R_m).

	Here we expect inputs to be annualized excess market return and variance.
	"""
	if market_variance <= 0:
		raise ValueError("market_variance must be positive")
	return market_return / market_variance


def compute_pi(
	Sigma: pd.DataFrame,
	w_mkt: pd.Series,
	risk_aversion: float,
) -> pd.Series:
	"""Compute equilibrium returns π = δ · Σ · w_mkt.

	Ensures index alignment between Sigma and weights.
	"""
	# Align indices
	Sigma = Sigma.loc[w_mkt.index, w_mkt.index]
	pi = risk_aversion * Sigma.values.dot(w_mkt.values)
	return pd.Series(pi, index=w_mkt.index)

