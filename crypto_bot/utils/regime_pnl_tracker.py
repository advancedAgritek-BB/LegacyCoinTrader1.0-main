import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .logger import LOG_DIR

LOG_FILE = LOG_DIR / "regime_pnl.csv"


def log_trade(regime: str, strategy: str, pnl: float) -> None:
    """Append realized PnL for ``strategy`` in ``regime`` to the CSV log."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "regime": regime,
        "strategy": strategy,
        "pnl": float(pnl),
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([record])
    header = not LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", header=header, index=False)


def _calc_metrics(pnls: pd.Series) -> Dict[str, float]:
    equity = pnls.cumsum()
    running_max = equity.cummax()
    drawdown = (1 - equity / running_max).max() if not equity.empty else 0.0
    sharpe = 0.0
    if len(pnls) > 1 and pnls.std() != 0:
        sharpe = pnls.mean() / pnls.std() * (len(pnls) ** 0.5)
    return {
        "pnl": float(pnls.sum()),
        "sharpe": float(sharpe),
        "drawdown": float(drawdown),
    }


def get_metrics(regime: str | None = None, path: str | Path = LOG_FILE) -> Dict[str, Dict[str, Any]]:
    """Return PnL metrics grouped by regime and strategy."""
    file = Path(path)
    if not file.exists():
        return {}
    df = pd.read_csv(file)
    if df.empty:
        return {}
    if regime:
        df = df[df["regime"] == regime]
    metrics: Dict[str, Dict[str, Any]] = {}
    for (reg, strat), group in df.groupby(["regime", "strategy"]):
        stats = _calc_metrics(group["pnl"])
        metrics.setdefault(reg, {})[strat] = stats
    return metrics


def compute_weights(regime: str, path: str | Path = LOG_FILE) -> Dict[str, float]:
    """Return normalized weights for strategies in ``regime`` based on recent performance."""
    data = get_metrics(regime, path)
    strategies = data.get(regime, {})
    if not strategies:
        return {}
    
    # Enhanced scoring with recency bias and volatility adjustment
    scores = {}
    for strategy, metrics in strategies.items():
        # Base score using Sharpe ratio
        base_score = metrics.get("sharpe", 0.0)
        
        # Apply recency bias - recent trades get higher weight
        recency_bias = _calculate_recency_bias(strategy, regime, path)
        
        # Apply volatility adjustment - higher volatility strategies get bonus in volatile regimes
        volatility_bonus = _calculate_volatility_bonus(strategy, regime, metrics)
        
        # Apply drawdown penalty
        drawdown_penalty = _calculate_drawdown_penalty(metrics)
        
        # Calculate final score
        final_score = base_score * recency_bias * (1 + volatility_bonus) * (1 - drawdown_penalty)
        scores[strategy] = max(0.0, final_score)
    
    # Normalize scores
    total = sum(scores.values())
    if not total:
        return {s: 1 / len(scores) for s in scores}
    
    return {s: sc / total for s, sc in scores.items()}


def _calculate_recency_bias(strategy: str, regime: str, path: str | Path) -> float:
    """Calculate recency bias based on recent trade activity."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            return 1.0
        
        # Filter for strategy and regime
        mask = (df["strategy"] == strategy) & (df["regime"] == regime)
        strategy_trades = df[mask]
        
        if strategy_trades.empty:
            return 0.8  # Penalty for no recent trades
        
        # Calculate days since last trade
        strategy_trades["timestamp"] = pd.to_datetime(strategy_trades["timestamp"])
        days_since_last = (pd.Timestamp.now() - strategy_trades["timestamp"].max()).days
        
        # Recency bias: more recent trades get higher weight
        if days_since_last <= 1:
            return 1.3  # Bonus for very recent trades
        elif days_since_last <= 3:
            return 1.2  # Bonus for recent trades
        elif days_since_last <= 7:
            return 1.1  # Small bonus for recent trades
        elif days_since_last <= 14:
            return 1.0  # Neutral for older trades
        else:
            return 0.9  # Penalty for old trades
            
    except Exception:
        return 1.0


def _calculate_volatility_bonus(strategy: str, regime: str, metrics: Dict[str, Any]) -> float:
    """Calculate volatility bonus for high-frequency strategies in volatile regimes."""
    # High-frequency strategies get bonus in volatile regimes
    high_freq_strategies = {
        "micro_scalp_bot", "sniper_bot", "flash_crash_bot", 
        "meme_wave_bot", "hft_engine", "bounce_scalper"
    }
    
    if strategy in high_freq_strategies and regime == "volatile":
        return 0.3  # 30% bonus for high-frequency strategies in volatile markets
    elif strategy in high_freq_strategies and regime in ["breakout", "trending"]:
        return 0.2  # 20% bonus for high-frequency strategies in trending markets
    elif strategy in high_freq_strategies:
        return 0.1  # 10% bonus for high-frequency strategies in other markets
    
    return 0.0


def _calculate_drawdown_penalty(metrics: Dict[str, Any]) -> float:
    """Calculate drawdown penalty to avoid strategies with high drawdowns."""
    drawdown = metrics.get("drawdown", 0.0)
    
    # Progressive drawdown penalty
    if drawdown <= 0.05:  # 5% or less
        return 0.0
    elif drawdown <= 0.10:  # 5-10%
        return 0.1
    elif drawdown <= 0.20:  # 10-20%
        return 0.2
    elif drawdown <= 0.30:  # 20-30%
        return 0.4
    else:  # 30%+
        return 0.6


def get_recent_win_rate(
    window: int = 20,
    path: str | Path = LOG_FILE,
    strategy: str | None = None,
) -> float:
    """Return the fraction of profitable trades.

    Parameters
    ----------
    window : int, optional
        Number of most recent trades to evaluate (default ``20``).
    path : str or Path, optional
        CSV log file location (defaults to :data:`LOG_FILE`).
    strategy : str, optional
        If given, filter trades for the specified strategy.

    Returns
    -------
    float
        Win rate as ``wins / total`` over the evaluated trades.
    """
    file = Path(path)
    if not file.exists():
        return 0.0
    df = pd.read_csv(file)
    if df.empty:
        return 0.0
    if strategy is not None and "strategy" in df.columns:
        df = df[df["strategy"] == strategy]

    recent = df.tail(window)
    if strategy is not None and "strategy" in recent.columns:
        recent = recent[recent["strategy"] == strategy]

    wins = (recent["pnl"] > 0).sum()
    total = len(recent)
    return float(wins / total) if total else 0.0
