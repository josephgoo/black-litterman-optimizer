"""
plots.py
--------

Visualization utilities: plot efficient frontier for MV and BL.
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
	ax.set_xlabel("Risk (Std Dev)")
	ax.set_ylabel("Expected Return")
	ax.set_title(title)
	ax.grid(True, alpha=0.3)
	ax.legend()
	plt.tight_layout()
	return fig, ax

