"""
views.py
--------

This module specifies subjective investor views and their uncertainties for
the Black–Litterman model.

Functionality:
- absolute_view: Set an absolute expected return view for a specific asset.
- relative_view: Set a relative view (asset A minus asset B equals magnitude).
- build_PQ: Stack a list of views into matrices P (loadings) and Q (view returns).
- tau_omega: Heuristic construction of view uncertainty matrix Ω via diag(PΣPᵀ)·τ.

Intended Use:
Construct (P, Q, Ω) inputs for optimizer.black_litterman_posterior.

Author: Joseph Goo Wei Zhen
Date: 2025-08-08
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def relative_view(tickers: List[str], long: str, short: str, magnitude: float) -> Tuple[pd.Series, float]:
	"""Create a relative view: return(long) - return(short) = magnitude.

	Returns a row vector p (as Series indexed by tickers) and q scalar.
	"""
	p = pd.Series(0.0, index=tickers)
	p[long] = 1.0
	p[short] = -1.0
	return p, magnitude


def absolute_view(tickers: List[str], asset: str, expected_return: float) -> Tuple[pd.Series, float]:
	"""Create an absolute view on a single asset expected return."""
	p = pd.Series(0.0, index=tickers)
	p[asset] = 1.0
	return p, expected_return


def build_PQ(views: List[Tuple[pd.Series, float]]) -> Tuple[pd.DataFrame, pd.Series]:
	"""Stack list of (p, q) into matrix P and vector Q."""
	if not views:
		# Return empty structures with zero rows
		return pd.DataFrame([]), pd.Series(dtype=float)
	P = pd.DataFrame([p.values for p, _ in views], columns=views[0][0].index)
	Q = pd.Series([q for _, q in views])
	return P, Q


def tau_omega(Sigma: pd.DataFrame, P: pd.DataFrame, tau: float = 0.05) -> pd.DataFrame:
	"""Default view uncertainty Ω = diag(P Σ P^T) · τ.

	This follows a common heuristic used in practice.
	"""
	if P.empty:
		return pd.DataFrame([])
	M = P.values @ Sigma.loc[P.columns, P.columns].values @ P.values.T
	return pd.DataFrame(np.diag(np.diag(M) * tau), index=P.index, columns=P.index)

