"""
plots.py
--------

Visualization utilities for efficient frontier comparison.

Functionality:
- plot_frontiers: Plot historical MV and BL frontiers and mark max Sharpe points.

Intended Use:
Display how BL views shift the efficient frontier and the location of the
optimal portfolios relative to traditional MV.

Author: Joseph Goo Wei Zhen
Date: 2025-08-08
"""

from __future__ import annotations

from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd


def plot_frontiers(
	mv: Tuple[pd.DataFrame, List[pd.Series]],
	bl: Tuple[pd.DataFrame, List[pd.Series]],
	title: str = "Efficient Frontier: MV vs BL",
):
	mv_df, _ = mv
	bl_df, _ = bl
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.plot(mv_df["risk"], mv_df["return"], label="Mean-Variance", color="tab:blue")
	ax.plot(bl_df["risk"], bl_df["return"], label="Black–Litterman", color="tab:orange")
	# Mark max Sharpe points (approx by slope vs RF at discrete points)
	def _mark_max(df, color, label):
		sharpe = df["return"] / df["risk"].replace(0, 1e-12)
		i = sharpe.idxmax()
		ax.scatter(df.loc[i, "risk"], df.loc[i, "return"], color=color, marker="o")
		ax.annotate(f"{label} max Sharpe", (df.loc[i, "risk"], df.loc[i, "return"]),
					textcoords="offset points", xytext=(5,5), fontsize=8, color=color)
	_mark_max(mv_df, "tab:blue", "MV")
	_mark_max(bl_df, "tab:orange", "BL")
	ax.set_xlabel("Risk (Std Dev)")
	ax.set_ylabel("Expected Return")
	ax.set_title(title)
	ax.grid(True, alpha=0.3)
	ax.legend()
	plt.tight_layout()
	return fig, ax

