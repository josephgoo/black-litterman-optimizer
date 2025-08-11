"""
optimizer.py
------------

Core portfolio math: Black–Litterman posterior and mean–variance optimization.

Functionality:
- black_litterman_posterior: Compute posterior mean/covariance blending π with views.
- efficient_frontier: Generate risk/return curve and weights along the frontier.
- max_sharpe_portfolio: Compute the tangency (max Sharpe) portfolio.
- compare_mv_vs_bl: Build both MV (historic mu, Σ) and BL (posterior) frontiers.

Intended Use:
- Use with data and views to compare traditional MV vs BL-implied allocations and
	highlight the optimal (max Sharpe) portfolios.

Author: Joseph Goo Wei Zhen
Date: 2025-08-08
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BLResult:
	mu_bl: pd.Series            # Posterior mean E[r|Q]
	Sigma_bl: pd.DataFrame      # Posterior covariance Cov(r|Q)


def black_litterman_posterior(
	Sigma: pd.DataFrame,
	pi: pd.Series,
	P: pd.DataFrame,
	Q: pd.Series,
	tau: float,
	Omega: Optional[pd.DataFrame] = None,
) -> BLResult:
	"""Compute BL posterior moments (mean and covariance).

	If Omega is None, use Omega = diag(P Σ P^T) * tau.
	"""
	assets = Sigma.index
	Sigma = Sigma.loc[assets, assets]
	pi = pi.loc[assets]

    # No views: posterior mean = prior mean; posterior covariance = Σ + tau * Σ
	if P is None or P.empty:
		return BLResult(mu_bl=pi, Sigma_bl=Sigma * (1 + tau))

	if Omega is None:
		M = P.values @ Sigma.loc[P.columns, P.columns].values @ P.values.T
		Omega = pd.DataFrame(np.diag(np.diag(M) * tau), index=P.index, columns=P.index)
		
	# Compute posterior
	inv_tauSigma = np.linalg.inv((tau * Sigma).values)
	inv_Omega = np.linalg.inv(Omega.values)
	PT = P.values.T
	mu_part = inv_tauSigma @ pi.values + PT @ inv_Omega @ Q.values
	M = inv_tauSigma + PT @ inv_Omega @ P.values
	mu_bl = pd.Series(np.linalg.solve(M, mu_part), index=assets)
	Sigma_bl = Sigma + pd.DataFrame(np.linalg.inv(M), index=assets, columns=assets)
	return BLResult(mu_bl=mu_bl, Sigma_bl=Sigma_bl)


def _quad_solve_weights(
	mu: pd.Series,
	Sigma: pd.DataFrame,
	target_return: float,
	risk_free: float | None = 0.0,
) -> pd.Series:
	"""
	Closed-form portfolio frontier with equality constraints (no bounds by default).

	Solves weights for a given target return (minimum variance for that return).
	No box constraints here; can extend to cvxpy.
	"""
	ones = np.ones(len(mu))
	Sigma_inv = pd.DataFrame(np.linalg.pinv(Sigma.values), index=Sigma.index, columns=Sigma.columns)

	# No risk-free asset
	if risk_free is None:
		# Compute A, B, C, D
		A = float(ones.T @ Sigma_inv.values @ mu.values)
		B = float(mu.values.T @ Sigma_inv.values @ mu.values)
		C = float(ones.T @ Sigma_inv.values @ ones)
		D = B * C - A * A
		if abs(D) < 1e-16:
			raise ValueError("D is zero; cannot compute optimal weights for no risk-free asset.")
		# Compute g and h
		g = (B * (Sigma_inv @ ones) - A * (Sigma_inv @ mu.values)) / D
		h = (C * (Sigma_inv @ mu.values) - A * (Sigma_inv @ ones)) / D
		# Optimal weights
		w = g + h * target_return
		return pd.Series(w, index=mu.index)

	# With risk-free asset available
	excess = mu - risk_free
	K = float(excess.values.T @ Sigma_inv.values @ excess.values)
	if K == 0:
		raise ValueError("K is zero; cannot scale weights for target return.")
	scale = (target_return - risk_free) / K
	w = scale * (Sigma_inv.values @ excess.values)
	return pd.Series(w, index=mu.index)


def efficient_frontier(
	mu: pd.Series,
	Sigma: pd.DataFrame,
	n: int = 50,
	risk_free: float | None = 0.0,
) -> Tuple[pd.DataFrame, List[pd.Series]]:
	"""Generate efficient frontier points: risk (stdev), return, and weights.

	Returns a DataFrame with columns ['risk','return'] and list of weight Series.
	"""
	rets = np.linspace(mu.min(), mu.max(), n)
	pts = []
	weights = []
	for r in rets:
		w = _quad_solve_weights(mu, Sigma, target_return=r, risk_free=risk_free)
		port_ret = float(w @ mu)
		port_var = float(w.values.T @ Sigma.values @ w.values)
		pts.append((np.sqrt(port_var), port_ret))
		weights.append(w)
	df = pd.DataFrame(pts, columns=["risk", "return"])  # type: ignore
	return df, weights


def compare_mv_vs_bl(
	mu_hist: pd.Series,
	Sigma_hist: pd.DataFrame,
	mu_bl: pd.Series,
	Sigma_bl: pd.DataFrame,
	n: int = 50,
	risk_free: float | None = 0.0,
) -> Dict[str, Tuple[pd.DataFrame, List[pd.Series]]]:
	"""Build efficient frontiers for historical MV and BL and return both."""
	mv = efficient_frontier(mu_hist, Sigma_hist, n=n, risk_free=risk_free)
	bl = efficient_frontier(mu_bl, Sigma_bl, n=n, risk_free=risk_free)
	return {"mv": mv, "bl": bl}


def max_sharpe_portfolio(
	mu: pd.Series,
	Sigma: pd.DataFrame,
	risk_free: float | None = 0.0,
) -> Tuple[pd.Series, float, float, float]:
	"""Compute the maximum Sharpe ratio portfolio weights and its metrics.

	Returns: (weights, expected_return, risk, sharpe)
	"""
	# No risk-free asset: max Sharpe portfolio is undefined
	if risk_free is None:
		msg = "Tangent (max Sharpe) portfolio is undefined without a risk-free asset."
		print(msg)
		return None, None, None, msg

	# With risk-free asset available, use tangency portfolio (max Sharpe)
	Sigma_inv = pd.DataFrame(np.linalg.pinv(Sigma.values), index=Sigma.index, columns=Sigma.columns)
	ones = np.ones(len(mu))
	rf = float(risk_free)
	A = float(ones.T @ Sigma_inv.values @ mu.values)
	C = float(ones.T @ Sigma_inv.values @ ones)
	if np.isclose(rf, A / C, atol=1e-10):
		raise ValueError("No tangent portfolio exists if risk-free rate equals A/C.")
	excess = mu.values - rf * ones
	denom = float(ones.T @ Sigma_inv.values @ excess)
	if np.isclose(denom, 0.0, atol=1e-12):
		raise ValueError("Denominator is zero; cannot compute tangent portfolio.")
	w = Sigma_inv.values @ excess / denom
	w = pd.Series(w, index=mu.index)
	exp_ret = float(w @ mu)
	risk = float(np.sqrt(w.values.T @ Sigma.values @ w.values))
	sharpe = (exp_ret - rf) / (risk + 1e-12)
	return w, exp_ret, risk, sharpe

