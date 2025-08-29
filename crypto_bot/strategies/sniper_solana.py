from __future__ import annotations

"""Simplified Solana sniping strategy."""

from typing import Tuple, Optional

import pandas as pd
import ta

from crypto_bot.utils.pyth_utils import get_pyth_price

from crypto_bot.fund_manager import auto_convert_funds


class RugCheckAPI:
    """Placeholder API returning a rug risk score between 0 and 1."""

    @staticmethod
    def risk_score(token: str) -> float:
        return 0.0


async def on_trade_filled(
    wallet: str,
    token: str,
    profit_token: str,
    amount: float,
    *,
    dry_run: bool = True,
    slippage_bps: int = 50,
) -> dict:
    """Convert trade profits back to BTC using the fund manager helper."""

    return await auto_convert_funds(
        wallet,
        token,
        profit_token,
        amount,
        dry_run=dry_run,
        slippage_bps=slippage_bps,
    )


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Return a signal score and direction based on ATR jumps."""

    if df is None or df.empty:
        return 0.0, "none"
    
    # Add debug logging for paper trading
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Generating Solana signal for DataFrame with {len(df)} rows")

    params = config or {}
    atr_window = int(params.get("atr_window", 14))
    jump_mult = float(params.get("jump_mult", 4.0))
    rug_threshold = float(params.get("rug_threshold", 0.5))
    profit_target = float(params.get("profit_target_pct", 0.05))
    token = params.get("token", "")
    entry_price = params.get("entry_price")
    is_trading = bool(params.get("is_trading", True))
    conf_pct = float(params.get("conf_pct", 0.0))

    if not is_trading or conf_pct > 0.5:
        return 0.0, "none"

    if len(df) < atr_window + 1:
        return 0.0, "none"

    # Use live price from Pyth if a token is provided
    if token:
        price = get_pyth_price(f"Crypto.{token}/USD", config)
        try:
            df = df.copy()
            df.at[df.index[-1], "close"] = float(price)
        except Exception:
            pass

    atr = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=atr_window
    )
    if atr.empty or pd.isna(atr.iloc[-1]):
        logger.debug("ATR calculation failed - empty or NaN")
        return 0.0, "none"

    current_atr = atr.iloc[-1]
    price_change = df["close"].iloc[-1] - df["close"].iloc[-2]
    jump_threshold = current_atr * jump_mult
    
    logger.debug(f"ATR: {current_atr:.6f}, Price change: {price_change:.6f}, Jump threshold: {jump_threshold:.6f}")
    
    if abs(price_change) >= jump_threshold:
        direction = "long" if price_change > 0 else "short"
        logger.debug(f"Signal triggered: {direction} with score 1.0")
        if token and RugCheckAPI.risk_score(token) >= rug_threshold:
            logger.debug(f"Rug check failed for token {token}")
            return 0.0, "none"
        return 1.0, direction

    if entry_price is not None:
        if df["close"].iloc[-1] >= float(entry_price) * (1 + profit_target):
            logger.debug(f"Profit target reached: {df['close'].iloc[-1]:.6f} >= {float(entry_price) * (1 + profit_target):.6f}")
            return 1.0, "close"

    logger.debug("No signal generated - conditions not met")
    return 0.0, "none"


class regime_filter:
    """Match volatile regime on Solana."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "volatile"

