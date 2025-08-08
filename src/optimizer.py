"""
optimizer.py
------------

Core Black–Litterman posterior returns and mean–variance optimization.

Key functions:
- black_litterman_posterior: compute posterior expected returns and covariance.
- mean_variance_opt: solve min variance for target return or max Sharpe.
- efficient_frontier: generate (risk, return, weights) along frontier.
- compare_mv_vs_bl: produce both MV (using historical mu) and BL frontiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BLResult:
	mu_bl: pd.Series
	Sigma_bl: pd.DataFrame


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

	if P is None or P.empty:
		return BLResult(mu_bl=pi, Sigma_bl=Sigma * (1 + tau))

	if Omega is None:
		M = P.values @ Sigma.loc[P.columns, P.columns].values @ P.values.T
		Omega = pd.DataFrame(np.diag(np.diag(M) * tau), index=P.index, columns=P.index)

	# Compute posterior
	inv_tauSigma = np.linalg.inv((tau * Sigma).values)
	PT = P.values.T
	middle = np.linalg.inv(P.values @ (tau * Sigma).values @ PT + Omega.values)
	mu_part = inv_tauSigma @ pi.values + PT @ middle @ Q.values
	Sigma_bl_inv = inv_tauSigma + PT @ middle @ P.values
	mu_bl = pd.Series(np.linalg.solve(Sigma_bl_inv, mu_part), index=assets)
	Sigma_bl = pd.DataFrame(np.linalg.inv(Sigma_bl_inv), index=assets, columns=assets)
	return BLResult(mu_bl=mu_bl, Sigma_bl=Sigma_bl)


def _quad_solve_weights(
	mu: pd.Series,
	Sigma: pd.DataFrame,
	target_return: Optional[float] = None,
	risk_free: float = 0.0,
	allow_short: bool = False,
) -> pd.Series:
	"""Closed-form efficient frontier with equality constraints (no bounds by default).

	Solves weights for either target return (minimum variance for that return) or
	max Sharpe (if target_return is None). No box constraints here; can extend to cvxpy.
	"""
	ones = np.ones(len(mu))
	Sigma_inv = pd.DataFrame(np.linalg.pinv(Sigma.values), index=Sigma.index, columns=Sigma.columns)
	A = float(ones.T @ Sigma_inv.values @ ones)
	B = float(ones.T @ Sigma_inv.values @ mu.values)
	C = float(mu.values.T @ Sigma_inv.values @ mu.values)
	D = A * C - B * B

	if target_return is None:
		# Max Sharpe: w ∝ Σ^{-1} (μ - r_f 1)
		excess = mu - risk_free
		w_unnorm = Sigma_inv.values @ excess.values
		w = w_unnorm / np.sum(w_unnorm)
		return pd.Series(w, index=mu.index)

	# Minimum variance for target return
	lam = (A * target_return - B) / D
	gamma = (C - B * target_return) / D
	w = lam * (Sigma_inv.values @ mu.values) + gamma * (Sigma_inv.values @ ones)
	return pd.Series(w, index=mu.index)


def efficient_frontier(
	mu: pd.Series,
	Sigma: pd.DataFrame,
	n: int = 50,
	risk_free: float = 0.0,
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
	risk_free: float = 0.0,
) -> Dict[str, Tuple[pd.DataFrame, List[pd.Series]]]:
	"""Build efficient frontiers for historical MV and BL and return both."""
	mv = efficient_frontier(mu_hist, Sigma_hist, n=n, risk_free=risk_free)
	bl = efficient_frontier(mu_bl, Sigma_bl, n=n, risk_free=risk_free)
	return {"mv": mv, "bl": bl}


def max_sharpe_portfolio(
	mu: pd.Series,
	Sigma: pd.DataFrame,
	risk_free: float = 0.0,
) -> Tuple[pd.Series, float, float, float]:
	"""Compute the maximum Sharpe ratio portfolio weights and its metrics.

	Returns: (weights, expected_return, risk, sharpe)
	"""
	w = _quad_solve_weights(mu, Sigma, target_return=None, risk_free=risk_free)
	exp_ret = float(w @ mu)
	risk = float(np.sqrt(w.values.T @ Sigma.values @ w.values))
	sharpe = (exp_ret - risk_free) / (risk + 1e-12)
	return w, exp_ret, risk, sharpe

