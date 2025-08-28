"""
backtest.py
-----------

Rolling-window backtests for MV and BL portfolio strategies.

What this provides:
- run_backtest: monthly rebalanced backtest using a rolling estimation window.
- Utilities to construct BL posterior each window and compute MV/BL tangent weights.

Assumptions:
- Annualization uses 252 trading days.
- Risk-free is a constant scalar (set to 0.0 by default). If set to None, tangent portfolio is undefined and will fall back to a midpoint target-return portfolio.
- Views are static across the backtest (example: AAPL outperforms MSFT by 2%).

Author: Joseph Goo Wei Zhen
Date: 2025-08-27
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import data_loader as dl
from . import implied_returns as ir
from . import market_portfolio as mp
from . import views as vw
from . import optimizer as opt
from . import evaluator as ev


TRADING_DAYS = 252


@dataclass
class BacktestResult:
    """Container for outputs of the rolling backtest.

    Attributes:
        mv_returns: Realized daily return series for the mean-variance (MV) strategy,
            indexed by trading date.
        bl_returns: Realized daily return series for the Black-Litterman (BL) strategy,
            indexed by trading date.
        mv_stats: Performance statistics for MV over the backtest horizon. Keys include
            at least 'cagr', 'vol', 'sharpe', and 'max_dd' (and may include others used
            by evaluator.performance_stats).
        bl_stats: Performance statistics for BL over the backtest horizon. Same schema
            as mv_stats.
        weights_mv: Mapping from each rebalance timestamp to the MV portfolio weights
            used for the subsequent holding period (Series aligned to the asset tickers).
        weights_bl: Mapping from each rebalance timestamp to the BL portfolio weights
            used for the subsequent holding period (Series aligned to the asset tickers).
        mv_turnover: Mapping from each rebalance timestamp to the MV turnover used for
            transaction cost calculation. First rebalance uses sum(|w|); subsequent
            rebalances use 0.5 * sum(|Δw|).
        bl_turnover: Mapping from each rebalance timestamp to the BL turnover used for
            transaction cost calculation, following the same convention as mv_turnover.
        mv_total_cost: Total cumulative transaction costs applied to MV returns over the
            entire backtest (expressed as a negative fraction of wealth).
        bl_total_cost: Total cumulative transaction costs applied to BL returns over the
            entire backtest (expressed as a negative fraction of wealth).
    """
    mv_returns: pd.Series
    bl_returns: pd.Series
    mv_stats: Dict[str, float]
    bl_stats: Dict[str, float]
    weights_mv: Dict[pd.Timestamp, pd.Series]
    weights_bl: Dict[pd.Timestamp, pd.Series]
    mv_turnover: Dict[pd.Timestamp, float]
    bl_turnover: Dict[pd.Timestamp, float]
    mv_total_cost: float
    bl_total_cost: float


def _rebalance_dates(index: pd.DatetimeIndex, start_idx: int, freq: str = "M") -> List[pd.Timestamp]:
    """Get rebalance dates as the last trading day of each period (e.g., month-end), after warmup.

    index: trading-day index
    start_idx: first usable index position after warmup
    freq: 'M' for month-end
    """
    # Last trading day per month
    month_last = index.to_series().groupby(index.to_period(freq)).last()
    cutoff_date = index[start_idx]
    rebal_dates = [ts for ts in month_last.values if ts >= cutoff_date]
    return rebal_dates


def _midpoint_target(mu: pd.Series) -> float:
    """A simple midpoint target return between min and max expected returns."""
    return float((mu.min() + mu.max()) / 2)


def _weights_for_policy(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    risk_free: Optional[float],
) -> pd.Series:
    """Choose a default policy for weights: tangent (max Sharpe) if defined; otherwise midpoint target.

    Returns a weight Series aligned to mu.index.
    """
    try:
        w, _, _, _ = opt.max_sharpe_portfolio(mu, Sigma, risk_free=risk_free)
        if w is not None:
            return w
    except Exception:
        pass
    # Fallback to midpoint target-return portfolio
    target = _midpoint_target(mu)
    return opt._quad_solve_weights(mu, Sigma, target_return=target, risk_free=risk_free)


def run_backtest(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    window_years: int = 3,
    tau: float = 0.05,
    risk_free: Optional[float] = 0.0,
    view_spec: Optional[Tuple[str, str, float]] = None,
    periods_per_year: int = TRADING_DAYS,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """Run a monthly rebalanced backtest comparing MV and BL strategies.

    Strategy per window:
    - Estimate μ, Σ from prior window_years of daily returns (annualized).
    - Build CAPM π via betas and market caps.
    - Apply the specified view (if provided) to get BL posterior.
    - Compute weights via tangent (max Sharpe) portfolio if feasible, else midpoint target.
    - Hold weights for the next month; compute realized daily returns.

    view_spec: (long_ticker, short_ticker, magnitude), e.g., ("AAPL", "MSFT", 0.02)
    cost_bps: round-trip transaction cost rate per $ traded, in basis points (e.g., 10 = 10 bps = 0.1%).

    Turnover convention:
    - First rebalance (from cash): turnover = sum(|w_new|)
    - Subsequent: turnover = 0.5 * sum(|w_new - w_prev|)  # avoids double counting matched trades
    """
    # Fetch prices and compute daily simple returns
    prices = dl.fetch_prices(tickers, start=start, end=end).prices
    rets = dl.compute_returns(prices, method="simple", dropna=True)

    # Warmup window length in trading days
    window_days = int(window_years * periods_per_year)
    if len(rets) <= window_days:
        raise ValueError("Not enough data for the specified window_years")

    # Static market caps -> weights (can be made time-varying if desired)
    # (CAVEAT: yfinance info is current; ideally use historical caps)
    mkt_caps = mp.fetch_market_caps(tickers)
    # Fallback to equal weights if market caps are missing/unavailable
    try:
        w_mkt = mp.market_caps_to_weights(mkt_caps)
    except Exception:
        w_mkt = pd.Series(1.0, index=tickers)
        w_mkt = w_mkt / w_mkt.sum()

    # Setup views (static across backtest)
    P = Q = None
    if view_spec is not None:
        long_t, short_t, mag = view_spec
        P_row, q = vw.relative_view(tickers, long=long_t, short=short_t, magnitude=mag)
        P, Q = vw.build_PQ([(P_row, q)])

    # Build list of rebalance dates
    idx = rets.index
    rebal_dates = _rebalance_dates(idx, start_idx=window_days, freq="M")
    if not rebal_dates:
        raise ValueError("No rebalance dates found; check data/index")

    mv_series: List[pd.Series] = []
    bl_series: List[pd.Series] = []
    weights_mv: Dict[pd.Timestamp, pd.Series] = {}
    weights_bl: Dict[pd.Timestamp, pd.Series] = {}
    mv_turnover: Dict[pd.Timestamp, float] = {}
    bl_turnover: Dict[pd.Timestamp, float] = {}

    cost_rate = cost_bps / 10000.0

    prev_mv = None
    prev_bl = None

    for i, t in enumerate(rebal_dates):
        # Estimation window ends at t-1
        hist = rets.loc[:t].tail(window_days)
        mu_h = dl.annualize_returns(hist, periods_per_year=periods_per_year)
        Sigma_h = dl.estimate_covariance(hist, periods_per_year=periods_per_year)

        # Market-implied π via betas
        mkt_ret, _ = mp.market_return_and_variance(mu_h, Sigma_h, w_mkt)
        pi = ir.compute_pi(Sigma_h, w_mkt, market_excess_return=mkt_ret - (risk_free or 0.0))

        # BL posterior
        if P is not None:
            bl_res = opt.black_litterman_posterior(Sigma_h, pi, P, Q, tau=tau)
            mu_bl, Sigma_bl = bl_res.mu_bl, bl_res.Sigma_bl
        else:
            mu_bl, Sigma_bl = mu_h, Sigma_h

        # Choose weights
        w_mv = _weights_for_policy(mu_h, Sigma_h, risk_free=risk_free)
        w_bl = _weights_for_policy(mu_bl, Sigma_bl, risk_free=risk_free)
        weights_mv[t] = w_mv
        weights_bl[t] = w_bl

        # Turnover calculations
        if prev_mv is None:
            mv_to = float(np.sum(np.abs(w_mv.values)))
        else:
            dv = (w_mv - prev_mv).reindex(w_mv.index).fillna(0.0)
            mv_to = float(0.5 * np.sum(np.abs(dv.values)))
        if prev_bl is None:
            bl_to = float(np.sum(np.abs(w_bl.values)))
        else:
            dv = (w_bl - prev_bl).reindex(w_bl.index).fillna(0.0)
            bl_to = float(0.5 * np.sum(np.abs(dv.values)))
        mv_turnover[t] = mv_to
        bl_turnover[t] = bl_to

        # Holding period: from t (inclusive) to next rebalance (exclusive)
        t_next = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else idx[-1]
        hold = rets.loc[(rets.index > t) & (rets.index <= t_next)]
        if not hold.empty:
            # Raw returns
            r_mv = hold.dot(w_mv)
            r_bl = hold.dot(w_bl)
            # Apply transaction cost on first day of holding period
            first_day = r_mv.index[0]
            r_mv.loc[first_day] += -cost_rate * mv_to
            r_bl.loc[first_day] += -cost_rate * bl_to
            mv_series.append(r_mv)
            bl_series.append(r_bl)

        prev_mv = w_mv
        prev_bl = w_bl

    mv_rets = pd.concat(mv_series).sort_index()
    bl_rets = pd.concat(bl_series).sort_index()

    mv_stats = ev.performance_stats(mv_rets, periods_per_year=periods_per_year, risk_free=(risk_free or 0.0))
    bl_stats = ev.performance_stats(bl_rets, periods_per_year=periods_per_year, risk_free=(risk_free or 0.0))

    return BacktestResult(
        mv_returns=mv_rets,
        bl_returns=bl_rets,
        mv_stats=mv_stats,
        bl_stats=bl_stats,
        weights_mv=weights_mv,
        weights_bl=weights_bl,
        mv_turnover=mv_turnover,
        bl_turnover=bl_turnover,
        mv_total_cost=float(-cost_rate * sum(mv_turnover.values())),
        bl_total_cost=float(-cost_rate * sum(bl_turnover.values())),
    )
