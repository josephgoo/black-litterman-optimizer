"""
implied_returns.py
------------------

Equilibrium-return utilities for Black–Litterman.

Exports (kept minimal by design):
- asset_betas: CAPM betas vs the market from Σ and w_mkt
- compute_pi: CAPM prior via betas: π_i = β_i · (E[R_m] − r_f) (excess by default)

Author: Joseph Goo Wei Zhen
Date: 2025-08-08
"""

from __future__ import annotations

import pandas as pd


def asset_betas(
	Sigma: pd.DataFrame,
	w_mkt: pd.Series,
) -> pd.Series:
	"""Compute CAPM betas β_i = Cov(i, mkt) / Var(mkt) from covariance Σ and market weights w_mkt.

	Returns a Series aligned to asset tickers.
	"""
	Sigma = Sigma.loc[w_mkt.index, w_mkt.index]
	cov_i_m = Sigma.values.dot(w_mkt.values)  # shape (n,)
	var_m = float(w_mkt.values.T.dot(Sigma.values).dot(w_mkt.values))
	if var_m <= 0:
		raise ValueError("Market variance must be positive to compute betas")
	betas = cov_i_m / var_m
	return pd.Series(betas, index=w_mkt.index)


def compute_pi(
	Sigma: pd.DataFrame,
	w_mkt: pd.Series,
	market_excess_return: float,
) -> pd.Series:
	"""Compute CAPM prior via betas.

	π_i = β_i · (E[R_m] − r_f) for excess returns.
	"""
	betas = asset_betas(Sigma, w_mkt)
	pi_excess = betas * market_excess_return
	return pi_excess

