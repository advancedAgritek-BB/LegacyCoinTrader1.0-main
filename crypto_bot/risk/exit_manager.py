"""Position exit helpers with improved trailing stop and take profit logic."""

from typing import Tuple

import pandas as pd
import ta

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.volatility_filter import calc_atr
from pathlib import Path


# Use the main bot log for exit messages
logger = setup_logger(__name__, LOG_DIR / "bot.log")


def calculate_trailing_stop(
    price_series: pd.Series, trail_pct: float = 0.1
) -> float:
    """Return a trailing stop from the high of ``price_series``.

    Parameters
    ----------
    price_series : pd.Series
        Series of closing prices.
    trail_pct : float, optional
        Percentage to trail below the maximum price.

    Returns
    -------
    float
        Calculated trailing stop value.
    """
    highest = price_series.max()
    stop = highest * (1 - trail_pct)
    logger.info("Calculated trailing stop %.4f from high %.4f", stop, highest)
    return stop


def calculate_trailing_stop_short(
    price_series: pd.Series, trail_pct: float = 0.1
) -> float:
    """Return a trailing stop from the low of ``price_series`` for short positions.

    Parameters
    ----------
    price_series : pd.Series
        Series of closing prices.
    trail_pct : float, optional
        Percentage to trail above the minimum price.

    Returns
    -------
    float
        Calculated trailing stop value above the low.
    """
    lowest = price_series.min()
    stop = lowest * (1 + trail_pct)
    logger.info("Calculated short trailing stop %.4f from low %.4f", stop, lowest)
    return stop


def calculate_atr_trailing_stop(df: pd.DataFrame, atr_factor: float = 2.0) -> float:
    """Return an ATR based trailing stop.

    The stop is calculated as ``highest_price_since_entry - ATR * atr_factor``.
    ``df`` should contain the OHLC data from trade entry to the current bar.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing ``high``, ``low`` and ``close`` columns.
    atr_factor : float, optional
        Multiplier applied to the ATR value.

    Returns
    -------
    float
        Calculated trailing stop using ATR.
    """
    try:
        highest = df["close"].max()
        atr = calc_atr(df)
        
        # Fallback ATR calculation if the cached version fails
        if pd.isna(atr) or atr == 0:
            logger.warning("Cached ATR failed, using fallback calculation")
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
        
        if pd.isna(atr) or atr == 0:
            logger.warning("Fallback ATR also failed, using percentage-based stop")
            return highest * 0.95  # Fallback to 5% trailing stop
        
        stop = highest - atr * atr_factor
        logger.info(
            "Calculated ATR trailing stop %.4f using high %.4f and ATR %.4f",
            stop,
            highest,
            atr,
        )
        return stop
    except Exception as e:
        logger.error(f"Error calculating ATR trailing stop: {e}")
        # Fallback to percentage-based trailing stop
        highest = df["close"].max()
        return highest * 0.95


def calculate_atr_trailing_stop_short(df: pd.DataFrame, atr_factor: float = 2.0) -> float:
    """Return an ATR based trailing stop for short positions.

    The stop is calculated as ``lowest_price_since_entry + ATR * atr_factor``.
    ``df`` should contain the OHLC data from trade entry to the current bar.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing ``high``, ``low`` and ``close`` columns.
    atr_factor : float, optional
        Multiplier applied to the ATR value.

    Returns
    -------
    float
        Calculated trailing stop using ATR for short positions.
    """
    try:
        lowest = df["close"].min()
        atr = calc_atr(df)
        
        # Fallback ATR calculation if the cached version fails
        if pd.isna(atr) or atr == 0:
            logger.warning("Cached ATR failed, using fallback calculation")
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
        
        if pd.isna(atr) or atr == 0:
            logger.warning("Fallback ATR also failed, using percentage-based stop")
            return lowest * 1.05  # Fallback to 5% trailing stop
        
        stop = lowest + atr * atr_factor
        logger.info(
            "Calculated ATR short trailing stop %.4f using low %.4f and ATR %.4f",
            stop,
            lowest,
            atr,
        )
        return stop
    except Exception as e:
        logger.error(f"Error calculating ATR short trailing stop: {e}")
        # Fallback to percentage-based trailing stop
        lowest = df["close"].min()
        return lowest * 1.05


def momentum_healthy(df: pd.DataFrame) -> bool:
    """Check RSI, MACD and volume to gauge trend health.

    Parameters
    ----------
    df : pd.DataFrame
        Historical OHLCV data used to compute indicators.

    Returns
    -------
    bool
        ``True`` if the momentum indicators confirm strength.
    """
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    vol_avg = df['volume'].rolling(3).mean()
    # Ensure at least two non-null volume averages exist before comparing
    if vol_avg.dropna().shape[0] < 2:
        return False
    vol_rising = vol_avg.iloc[-1] > vol_avg.iloc[-2]

    latest = df.iloc[-1]
    # Verify momentum indicators have valid values
    if (
        pd.isna(latest.get('rsi'))
        or pd.isna(latest.get('macd'))
        or pd.isna(latest.get('macd_signal'))
    ):
        return False

    return bool(
        latest['rsi'] > 55
        and latest['macd'] > latest['macd_signal']
        and vol_rising
    )


def _assess_momentum_strength(df: pd.DataFrame) -> float:
    """Assess momentum strength on a scale of 0-1, less restrictive than momentum_healthy."""
    try:
        df = df.copy()
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        
        latest = df.iloc[-1]
        
        # Check if indicators have valid values
        if (pd.isna(latest.get('rsi')) or 
            pd.isna(latest.get('macd')) or 
            pd.isna(latest.get('macd_signal'))):
            return 0.5  # Neutral if data unavailable
        
        # Calculate momentum score components
        rsi_score = min(1.0, max(0.0, (latest['rsi'] - 30) / 40))  # 30-70 range
        macd_score = 1.0 if latest['macd'] > latest['macd_signal'] else 0.0
        
        # Volume trend (less strict)
        vol_avg = df['volume'].rolling(3).mean()
        if vol_avg.dropna().shape[0] >= 2:
            vol_score = 1.0 if vol_avg.iloc[-1] > vol_avg.iloc[-2] else 0.5
        else:
            vol_score = 0.5
        
        # Weighted average
        momentum_strength = (rsi_score * 0.4 + macd_score * 0.4 + vol_score * 0.2)
        
        return momentum_strength
        
    except Exception:
        return 0.5  # Neutral on error


def should_exit(
    df: pd.DataFrame,
    current_price: float,
    trailing_stop: float,
    config: dict,
    risk_manager=None,
    position_side: str = "buy",  # Add position side parameter
    entry_price: float = None,  # Add entry price for take profit calculations
) -> Tuple[bool, float]:
    """Determine whether to exit a position and update trailing stop.

    Parameters
    ----------
    df : pd.DataFrame
        Recent market data.
    current_price : float
        Latest traded price.
    trailing_stop : float
        Current trailing stop value.
    config : dict
        Strategy configuration.
    risk_manager : object, optional
        Risk manager instance for stop order handling.
    position_side : str
        Position side: "buy" for long, "sell" for short.
    entry_price : float, optional
        Entry price for take profit calculations.

    Returns
    -------
    Tuple[bool, float]
        Flag indicating whether to exit and the updated stop price.
    """
    exit_signal = False
    new_stop = trailing_stop
    
    # Check take profit first (if configured)
    if entry_price is not None:
        exit_cfg = config.get('exit_strategy', {})
        take_profit_pct = exit_cfg.get('take_profit_pct', 0.0)
        
        if take_profit_pct > 0:
            if position_side == "buy":  # Long position
                take_profit_price = entry_price * (1 + take_profit_pct)
                if current_price >= take_profit_price:
                    logger.info(
                        "Take profit hit at %.4f (target: %.4f) for long position",
                        current_price,
                        take_profit_price,
                    )
                    return True, new_stop
            else:  # Short position
                take_profit_price = entry_price * (1 - take_profit_pct)
                if current_price <= take_profit_price:
                    logger.info(
                        "Take profit hit at %.4f (target: %.4f) for short position",
                        current_price,
                        take_profit_price,
                    )
                    return True, new_stop
    
    # Check hard stop loss first (this should always trigger regardless of momentum)
    if entry_price is not None:
        # Get hard stop loss percentage from config
        hard_stop_pct = config.get('risk', {}).get('stop_loss_pct', 0.015)  # Default 1.5%
        
        if position_side == "buy":  # Long position
            hard_stop_price = entry_price * (1 - hard_stop_pct)
            if current_price <= hard_stop_price:
                logger.info(
                    "HARD STOP LOSS triggered at %.4f (target: %.4f) for long position",
                    current_price,
                    hard_stop_price,
                )
                return True, new_stop
        else:  # Short position
            hard_stop_price = entry_price * (1 + hard_stop_pct)
            if current_price >= hard_stop_price:
                logger.info(
                    "HARD STOP LOSS triggered at %.4f (target: %.4f) for short position",
                    current_price,
                    hard_stop_price,
                )
                return True, new_stop
    
    # Check if price hit trailing stop based on position side
    if position_side == "buy":  # Long position
        stop_hit = current_price < trailing_stop
    else:  # Short position
        stop_hit = current_price > trailing_stop
    
    if stop_hit and trailing_stop > 0:
        # For trailing stops, we can still apply momentum checks but make them less restrictive
        # This allows for some flexibility while maintaining risk control
        momentum_strength = _assess_momentum_strength(df)
        
        # Allow exit if momentum is weak or if we're in a significant loss
        pnl_pct = abs((current_price - entry_price) / entry_price) if entry_price else 0
        significant_loss = pnl_pct > 0.05  # 5% loss threshold
        
        if momentum_strength < 0.6 or significant_loss:  # More permissive momentum check
            logger.info(
                "Price %.4f hit trailing stop %.4f for %s position (momentum: %.2f, loss: %.2f%%)",
                current_price,
                trailing_stop,
                "long" if position_side == "buy" else "short",
                momentum_strength,
                pnl_pct * 100,
            )
            exit_signal = True
            if risk_manager and getattr(risk_manager, "stop_order", None):
                order = risk_manager.stop_order
                entry = order.get("entry_price")
                direction = order.get("direction")
                strategy = order.get("strategy", "")
                symbol = order.get("symbol", config.get("symbol", ""))
                confidence = order.get("confidence", 0.0)
                if entry is not None and direction:
                    pnl = (current_price - entry) * (
                        1 if direction == "buy" else -1
                    )
                    from crypto_bot.utils.pnl_logger import log_pnl

                    log_pnl(
                        strategy,
                        symbol,
                        entry,
                        current_price,
                        pnl,
                        confidence,
                        direction,
                    )
    else:
        if trailing_stop > 0:
            exit_cfg = config.get('exit_strategy', {})
            if 'trailing_stop_factor' in exit_cfg:
                if position_side == "buy":
                    trailed = calculate_atr_trailing_stop(
                        df,
                        exit_cfg['trailing_stop_factor'],
                    )
                else:  # Short position
                    trailed = calculate_atr_trailing_stop_short(
                        df,
                        exit_cfg['trailing_stop_factor'],
                    )
            else:
                if position_side == "buy":
                    trailed = calculate_trailing_stop(
                        df['close'],
                        exit_cfg['trailing_stop_pct'],
                    )
                else:  # Short position
                    trailed = calculate_trailing_stop_short(
                        df['close'],
                        exit_cfg['trailing_stop_pct'],
                    )
            
            # Update stop only if it's better (higher for long, lower for short)
            # But also allow moving the stop to protect against losses
            if position_side == "buy":
                # For long positions, allow moving stop up to protect profits
                # AND allow moving stop down to protect against larger losses
                if trailed > trailing_stop or (trailed < trailing_stop and current_price < entry_price):
                    new_stop = trailed
                    logger.info("Trailing stop moved to %.4f", new_stop)
            elif position_side == "sell":
                # For short positions, allow moving stop down to protect profits
                # AND allow moving stop up to protect against larger losses
                if trailed < trailing_stop or (trailed > trailing_stop and current_price > entry_price):
                    new_stop = trailed
                    logger.info("Trailing stop moved to %.4f", new_stop)
    
    return exit_signal, new_stop


def get_partial_exit_percent(pnl_pct: float) -> int:
    """Return percent of position to close based on profit.

    Parameters
    ----------
    pnl_pct : float
        Unrealized profit or loss percentage.

    Returns
    -------
    int
        Portion of the position to close expressed as a percentage.
    """
    if pnl_pct > 100:
        return 50
    if pnl_pct > 50:
        return 30
    if pnl_pct > 25:
        return 20
    return 0

# Enhanced Exit Manager for Aggressive Profit Maximization
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from crypto_bot.utils.logger import setup_logger, LOG_DIR
from crypto_bot.utils.telegram import TelegramNotifier

logger = setup_logger(__name__, LOG_DIR / "exit_manager.log")


@dataclass
class Position:
    """Enhanced position tracking with exit management."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    entry_time: datetime
    size: float
    strategy: str
    confidence: float
    regime: str
    
    # Exit management
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    partial_exits: List[Dict] = field(default_factory=list)
    
    # Performance tracking
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    current_pnl: float = 0.0
    max_pnl: float = 0.0
    max_drawdown: float = 0.0
    
    # Dynamic exit parameters
    exit_strategy: str = "aggressive"  # "aggressive", "conservative", "hybrid"
    trailing_stop_activated: bool = False
    partial_exit_count: int = 0


@dataclass
class ExitConfig:
    """Configuration for aggressive exit strategies."""
    # Stop loss settings
    initial_stop_loss_pct: float = 0.012  # 1.2% initial stop loss
    trailing_stop_pct: float = 0.015  # 1.5% trailing stop
    min_trailing_distance: float = 0.008  # Minimum 0.8% trailing distance
    
    # Take profit settings
    initial_take_profit_pct: float = 0.04  # 4% initial take profit
    partial_exit_levels: List[float] = field(default_factory=lambda: [0.02, 0.04, 0.06, 0.08])
    partial_exit_sizes: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    
    # Dynamic adjustments
    volatility_multiplier: float = 1.2  # Adjust exits based on volatility
    momentum_multiplier: float = 1.1  # Adjust exits based on momentum
    regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "volatile": 1.3,      # More aggressive in volatile markets
        "breakout": 1.2,      # More aggressive on breakouts
        "trending": 1.1,      # Slightly aggressive in trends
        "sideways": 0.9,      # Less aggressive in sideways markets
        "mean-reverting": 0.8  # Conservative in mean reversion
    })
    
    # Risk management
    max_position_size_pct: float = 0.15  # Maximum 15% per position
    max_daily_loss_pct: float = 0.05  # Maximum 5% daily loss
    correlation_limit: float = 0.7  # Maximum correlation between positions


class AggressiveExitManager:
    """
    Enhanced exit manager implementing aggressive profit-taking strategies
    with dynamic position sizing and regime-aware exits.
    """
    
    def __init__(self, config: ExitConfig, notifier: Optional[TelegramNotifier] = None):
        self.config = config
        self.notifier = notifier
        self.positions: Dict[str, Position] = {}
        self.exit_history: List[Dict] = []
        self.daily_pnl: float = 0.0
        self.last_reset_date: datetime = datetime.now().date()
        
        # Performance tracking
        self.total_exits: int = 0
        self.profitable_exits: int = 0
        self.avg_exit_time: timedelta = timedelta(0)
        self.max_drawdown: float = 0.0
        
        # Start monitoring loop
        asyncio.create_task(self._monitor_positions())
    
    async def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        strategy: str,
        confidence: float,
        regime: str,
        current_price: float
    ) -> Position:
        """Open a new position with aggressive exit management."""
        # Calculate dynamic exit parameters based on regime and volatility
        exit_params = self._calculate_dynamic_exits(entry_price, current_price, regime, confidence)
        
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(),
            size=size,
            strategy=strategy,
            confidence=confidence,
            regime=regime,
            stop_loss_price=exit_params["stop_loss"],
            take_profit_price=exit_params["take_profit"],
            trailing_stop_price=exit_params["trailing_stop"],
            exit_strategy=self._select_exit_strategy(regime, confidence)
        )
        
        self.positions[symbol] = position
        
        # Log position opening
        logger.info(f"Opened {side} position in {symbol} at {entry_price:.6f} "
                   f"with {strategy} strategy (confidence: {confidence:.2f})")
        
        if self.notifier:
            await self.notifier.send_message(
                f"🟢 Opened {side} position in {symbol}\n"
                f"Entry: {entry_price:.6f}\n"
                f"Strategy: {strategy}\n"
                f"Confidence: {confidence:.2f}\n"
                f"Regime: {regime}\n"
                f"Stop Loss: {exit_params['stop_loss']:.6f}\n"
                f"Take Profit: {exit_params['take_profit']:.6f}"
            )
        
        return position
    
    def _calculate_dynamic_exits(
        self, 
        entry_price: float, 
        current_price: float, 
        regime: str, 
        confidence: float
    ) -> Dict[str, float]:
        """Calculate dynamic exit levels based on regime, volatility, and confidence."""
        # Base multipliers
        regime_mult = self.config.regime_multipliers.get(regime, 1.0)
        confidence_mult = 0.8 + (confidence * 0.4)  # 0.8 to 1.2 range
        
        # Calculate base percentages
        base_sl_pct = self.config.initial_stop_loss_pct
        base_tp_pct = self.config.initial_take_profit_pct
        base_trailing_pct = self.config.trailing_stop_pct
        
        # Apply regime and confidence adjustments
        adjusted_sl_pct = base_sl_pct * regime_mult * confidence_mult
        adjusted_tp_pct = base_tp_pct * regime_mult * confidence_mult
        adjusted_trailing_pct = base_trailing_pct * regime_mult * confidence_mult
        
        # Calculate exit prices
        if regime in ["volatile", "breakout"]:
            # More aggressive exits in volatile/breakout regimes
            stop_loss = entry_price * (1 - adjusted_sl_pct * 0.8)  # Tighter stops
            take_profit = entry_price * (1 + adjusted_tp_pct * 1.2)  # Higher targets
            trailing_stop = entry_price * (1 - adjusted_trailing_pct * 0.8)
        else:
            # Standard exits for other regimes
            stop_loss = entry_price * (1 - adjusted_sl_pct)
            take_profit = entry_price * (1 + adjusted_tp_pct)
            trailing_stop = entry_price * (1 - adjusted_trailing_pct)
        
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop": trailing_stop
        }
    
    def _select_exit_strategy(self, regime: str, confidence: float) -> str:
        """Select exit strategy based on regime and confidence."""
        if regime in ["volatile", "breakout"] and confidence > 0.7:
            return "aggressive"
        elif regime in ["trending"] and confidence > 0.6:
            return "hybrid"
        else:
            return "conservative"
    
    async def update_position(
        self, 
        symbol: str, 
        current_price: float,
        volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Update position and check for exit conditions."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        old_pnl = position.current_pnl
        
        # Update position metrics
        position.current_pnl = self._calculate_pnl(position, current_price)
        position.highest_price = max(position.highest_price, current_price)
        position.lowest_price = min(position.lowest_price, current_price)
        
        # Check for exit conditions
        exit_signal = await self._check_exit_conditions(position, current_price, volume, volatility)
        
        if exit_signal:
            await self._execute_exit(position, exit_signal, current_price)
            return exit_signal
        
        # Update trailing stops
        self._update_trailing_stops(position, current_price)
        
        # Check for partial exits
        partial_exit = self._check_partial_exit(position, current_price)
        if partial_exit:
            await self._execute_partial_exit(position, partial_exit, current_price)
        
        return None
    
    def _calculate_pnl(self, position: Position, current_price: float) -> float:
        """Calculate current PnL for position."""
        if position.side == "long":
            return (current_price - position.entry_price) / position.entry_price
        else:  # short
            return (position.entry_price - current_price) / position.entry_price
    
    async def _check_exit_conditions(
        self, 
        position: Position, 
        current_price: float,
        volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Check if position should be exited."""
        # Stop loss check
        if self._should_stop_loss(position, current_price):
            return {
                "reason": "stop_loss",
                "price": current_price,
                "pnl": position.current_pnl,
                "exit_type": "full"
            }
        
        # Take profit check
        if self._should_take_profit(position, current_price):
            return {
                "reason": "take_profit",
                "price": current_price,
                "pnl": position.current_pnl,
                "exit_type": "full"
            }
        
        # Trailing stop check
        if self._should_trailing_stop(position, current_price):
            return {
                "reason": "trailing_stop",
                "price": current_price,
                "pnl": position.current_pnl,
                "exit_type": "full"
            }
        
        # Dynamic exit based on regime changes
        regime_exit = self._check_regime_exit(position, current_price, volume, volatility)
        if regime_exit:
            return regime_exit
        
        return None
    
    def _should_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss should be triggered."""
        if position.side == "long":
            return current_price <= position.stop_loss_price
        else:  # short
            return current_price >= position.stop_loss_price
    
    def _should_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if take profit should be triggered."""
        if position.side == "long":
            return current_price >= position.take_profit_price
        else:  # short
            return current_price <= position.take_profit_price
    
    def _should_trailing_stop(self, position: Position, current_price: float) -> bool:
        """Check if trailing stop should be triggered."""
        if not position.trailing_stop_activated or not position.trailing_stop_price:
            return False
        
        if position.side == "long":
            return current_price <= position.trailing_stop_price
        else:  # short
            return current_price >= position.trailing_stop_price
    
    def _check_regime_exit(
        self, 
        position: Position, 
        current_price: float,
        volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Check for regime-based exits."""
        # Exit if position has been open too long without progress
        time_open = datetime.now() - position.entry_time
        
        if position.regime == "volatile" and time_open > timedelta(hours=2):
            if abs(position.current_pnl) < 0.01:  # No significant progress
                return {
                    "reason": "regime_timeout",
                    "price": current_price,
                    "pnl": position.current_pnl,
                    "exit_type": "full"
                }
        
        # Exit if momentum has significantly reversed
        if position.regime == "trending" and time_open > timedelta(hours=4):
            if position.current_pnl < -0.02:  # 2% loss in trending market
                return {
                    "reason": "momentum_reversal",
                    "price": current_price,
                    "pnl": position.current_pnl,
                    "exit_type": "full"
                }
        
        return None
    
    def _update_trailing_stops(self, position: Position, current_price: float):
        """Update trailing stop levels."""
        if position.side == "long":
            # Update trailing stop for long positions
            if current_price > position.entry_price:
                new_trailing_stop = current_price * (1 - self.config.trailing_stop_pct)
                if not position.trailing_stop_price or new_trailing_stop > position.trailing_stop_price:
                    position.trailing_stop_price = new_trailing_stop
                    position.trailing_stop_activated = True
        else:
            # Update trailing stop for short positions
            if current_price < position.entry_price:
                new_trailing_stop = current_price * (1 + self.config.trailing_stop_pct)
                if not position.trailing_stop_price or new_trailing_stop < position.trailing_stop_price:
                    position.trailing_stop_price = new_trailing_stop
                    position.trailing_stop_activated = True
    
    def _check_partial_exit(self, position: Position, current_price: float) -> Optional[Dict[str, Any]]:
        """Check if partial exit should be executed."""
        if position.partial_exit_count >= len(self.config.partial_exit_levels):
            return None
        
        exit_level = self.config.partial_exit_levels[position.partial_exit_count]
        exit_size = self.config.partial_exit_sizes[position.partial_exit_count]
        
        if position.side == "long" and position.current_pnl >= exit_level:
            return {
                "reason": "partial_profit",
                "level": exit_level,
                "size": exit_size,
                "exit_type": "partial"
            }
        elif position.side == "short" and position.current_pnl >= exit_level:
            return {
                "reason": "partial_profit",
                "level": exit_level,
                "size": exit_size,
                "exit_type": "partial"
            }
        
        return None
    
    async def _execute_exit(self, position: Position, exit_signal: Dict[str, Any], exit_price: float):
        """Execute full position exit."""
        # Calculate final PnL
        final_pnl = self._calculate_pnl(position, exit_price)
        
        # Log exit
        logger.info(f"Exiting {position.side} position in {position.symbol} "
                   f"at {exit_price:.6f} (PnL: {final_pnl:.4f}) "
                   f"Reason: {exit_signal['reason']}")
        
        # Update statistics
        self._update_exit_statistics(position, final_pnl, exit_signal)
        
        # Send notification
        if self.notifier:
            await self.notifier.send_message(
                f"🔴 Exited {position.side} position in {position.symbol}\n"
                f"Exit Price: {exit_price:.6f}\n"
                f"PnL: {final_pnl:.4f}\n"
                f"Reason: {exit_signal['reason']}\n"
                f"Strategy: {position.strategy}\n"
                f"Regime: {position.regime}"
            )
        
        # Remove position
        del self.positions[position.symbol]
        
        # Record exit history
        self.exit_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "pnl": final_pnl,
            "strategy": position.strategy,
            "regime": position.regime,
            "reason": exit_signal["reason"],
            "exit_type": exit_signal["exit_type"],
            "time_open": (datetime.now() - position.entry_time).total_seconds() / 3600
        })
    
    async def _execute_partial_exit(self, position: Position, partial_exit: Dict[str, Any], exit_price: float):
        """Execute partial position exit."""
        exit_size = partial_exit["size"]
        exit_amount = position.size * exit_size
        
        # Update position size
        position.size -= exit_amount
        position.partial_exit_count += 1
        
        # Calculate partial PnL
        partial_pnl = self._calculate_pnl(position, exit_price) * exit_size
        
        # Log partial exit
        logger.info(f"Partial exit from {position.side} position in {position.symbol} "
                   f"at {exit_price:.6f} (Size: {exit_amount:.6f}, PnL: {partial_pnl:.4f})")
        
        # Send notification
        if self.notifier:
            await self.notifier.send_message(
                f"💰 Partial exit from {position.side} position in {position.symbol}\n"
                f"Exit Price: {exit_price:.6f}\n"
                f"Exit Size: {exit_amount:.6f}\n"
                f"Partial PnL: {partial_pnl:.4f}\n"
                f"Remaining Size: {position.size:.6f}"
            )
    
    def _update_exit_statistics(self, position: Position, final_pnl: float, exit_signal: Dict[str, Any]):
        """Update exit statistics."""
        self.total_exits += 1
        if final_pnl > 0:
            self.profitable_exits += 1
        
        # Update average exit time
        time_open = datetime.now() - position.entry_time
        if self.total_exits == 1:
            self.avg_exit_time = time_open
        else:
            self.avg_exit_time = (self.avg_exit_time * (self.total_exits - 1) + time_open) / self.total_exits
        
        # Update max drawdown
        if final_pnl < self.max_drawdown:
            self.max_drawdown = final_pnl
        
        # Update daily PnL
        self.daily_pnl += final_pnl
    
    async def _monitor_positions(self):
        """Background task to monitor all positions."""
        while True:
            try:
                # Reset daily PnL at midnight
                current_date = datetime.now().date()
                if current_date != self.last_reset_date:
                    self.daily_pnl = 0.0
                    self.last_reset_date = current_date
                
                # Check for daily loss limit
                if self.daily_pnl < -self.config.max_daily_loss_pct:
                    await self._emergency_exit_all("daily_loss_limit")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _emergency_exit_all(self, reason: str):
        """Emergency exit all positions."""
        logger.warning(f"Emergency exit all positions: {reason}")
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            await self._execute_exit(position, {
                "reason": reason,
                "price": position.entry_price,  # Use entry price as fallback
                "pnl": 0.0,
                "exit_type": "emergency"
            }, position.entry_price)
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of current positions and performance."""
        total_positions = len(self.positions)
        total_pnl = sum(pos.current_pnl for pos in self.positions.values())
        
        return {
            "total_positions": total_positions,
            "total_pnl": total_pnl,
            "daily_pnl": self.daily_pnl,
            "positions": [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "current_pnl": pos.current_pnl,
                    "strategy": pos.strategy,
                    "regime": pos.regime,
                    "time_open": (datetime.now() - pos.entry_time).total_seconds() / 3600
                }
                for pos in self.positions.values()
            ],
            "exit_statistics": {
                "total_exits": self.total_exits,
                "profitable_exits": self.profitable_exits,
                "win_rate": self.profitable_exits / self.total_exits if self.total_exits > 0 else 0.0,
                "avg_exit_time_hours": self.avg_exit_time.total_seconds() / 3600,
                "max_drawdown": self.max_drawdown
            }
        }
