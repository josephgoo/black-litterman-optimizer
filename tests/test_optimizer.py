"""
test_optimizer.py
-----------------

Basic unit tests for Black–Litterman posterior and efficient frontier generation.

Author: Joseph Goo Wei Zhen
Date: 2025-08-08
"""

import numpy as np
import pandas as pd

from src import optimizer as opt


def _toy_data(n=4):
	idx = [f"A{i}" for i in range(n)]
	mu = pd.Series(np.linspace(0.05, 0.15, n), index=idx)
	Sigma = pd.DataFrame(0.02, index=idx, columns=idx)
	np.fill_diagonal(Sigma.values, 0.04)
	return mu, Sigma


def test_bl_shapes():
	mu, Sigma = _toy_data()
	pi = mu.copy()
	P = pd.DataFrame([[1, -1, 0, 0]], columns=mu.index)
	Q = pd.Series([0.02])
	res = opt.black_litterman_posterior(Sigma, pi, P, Q, tau=0.05)
	assert set(res.mu_bl.index) == set(mu.index)
	assert res.Sigma_bl.shape == Sigma.shape


def test_frontier_generation():
	mu, Sigma = _toy_data()
	df, w_list = opt.efficient_frontier(mu, Sigma, n=10)
	assert len(df) == 10
	assert len(w_list) == 10
