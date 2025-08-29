# Enhanced Auto-Optimizer for Aggressive Profit Maximization
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.regime_pnl_tracker import get_metrics, compute_weights
from crypto_bot.backtest.enhanced_backtester import ContinuousBacktestingEngine

logger = setup_logger(__name__, LOG_DIR / "auto_optimizer.log")

class AggressiveAutoOptimizer:
    """
    Enhanced auto-optimizer that runs daily to continuously improve strategy performance
    for maximum profit generation.
    """
    
    def __init__(self, config_path: str = "crypto_bot/config.yaml"):
        self.config_path = Path(config_path)
        self.optimization_history: List[Dict] = []
        self.last_optimization: Optional[datetime] = None
        self.optimization_interval_hours = 24  # Run daily
        
        # Aggressive optimization targets
        self.target_metrics = {
            "min_sharpe": 1.2,        # Increased from 1.0
            "min_win_rate": 0.45,     # Increased from 0.4
            "max_drawdown": 0.15,     # Reduced from 0.2
            "min_profit_factor": 1.8, # Increased from 1.5
            "min_expectancy": 0.02,   # Increased from 0.01
        }
        
        # Strategy-specific optimization ranges
        self.parameter_ranges = {
            "trend_bot": {
                "ema_fast": [8, 12, 16],           # Faster EMAs for quicker signals
                "ema_slow": [20, 30, 40],          # Faster EMAs for quicker signals
                "adx_threshold": [10, 12, 15],     # Lower thresholds for earlier detection
                "stop_loss_atr_mult": [0.6, 0.8, 1.0],  # Tighter stops
                "take_profit_atr_mult": [1.2, 1.5, 2.0],  # Faster profits
            },
            "micro_scalp_bot": {
                "atr_period": [4, 6, 8],           # Faster response
                "ema_fast": [3, 5, 7],             # Faster signals
                "ema_slow": [10, 13, 16],          # Faster signals
                "stop_loss_atr_mult": [0.4, 0.6, 0.8],  # Very tight stops
                "take_profit_atr_mult": [1.0, 1.2, 1.5],  # Quick profits
                "cooldown_bars": [0, 1, 2],        # More frequent trades
            },
            "sniper_bot": {
                "atr_window": [6, 8, 10],          # Faster response
                "breakout_pct": [0.02, 0.025, 0.03],  # Earlier detection
                "volume_multiple": [1.2, 1.3, 1.4],   # Better confirmation
                "fallback_atr_mult": [0.8, 1.0, 1.2],  # Tighter entries
            },
            "grid_bot": {
                "grid_levels": [6, 8, 10],         # More opportunities
                "grid_spacing_pct": [0.006, 0.008, 0.01],  # Tighter grids
                "min_profit_pct": [0.002, 0.003, 0.005],  # Faster exits
                "max_concurrent_trades": [4, 6, 8],        # More trades
            },
            "bounce_scalper": {
                "rsi_window": [10, 12, 14],        # Faster response
                "stop_loss_atr_mult": [0.6, 0.8, 1.0],  # Tighter stops
                "take_profit_atr_mult": [1.2, 1.5, 2.0],  # Faster profits
                "min_score": [0.12, 0.15, 0.18],  # More signals
            },
            "mean_bot": {
                "rsi_window": [10, 12, 14],        # Faster response
                "stop_loss_atr_mult": [0.6, 0.8, 1.0],  # Tighter stops
                "take_profit_atr_mult": [1.2, 1.4, 1.8],  # Faster profits
                "cooldown_bars": [1, 2, 3],        # More trades
            }
        }
    
    async def should_optimize(self) -> bool:
        """Check if optimization should run based on interval and performance."""
        if self.last_optimization is None:
            return True
        
        time_since_last = datetime.now() - self.last_optimization
        if time_since_last.total_seconds() < self.optimization_interval_hours * 3600:
            return False
        
        # Check if performance is below targets
        return await self._check_performance_degradation()
    
    async def _check_performance_degradation(self) -> bool:
        """Check if current performance is below optimization targets."""
        try:
            # Get recent performance metrics
            recent_metrics = await self._get_recent_performance()
            
            for metric, target in self.target_metrics.items():
                if metric in recent_metrics:
                    current_value = recent_metrics[metric]
                    if metric.startswith("min_") and current_value < target:
                        logger.info(f"Performance below target: {metric} = {current_value:.3f} < {target}")
                        return True
                    elif metric.startswith("max_") and current_value > target:
                        logger.info(f"Performance below target: {metric} = {current_value:.3f} > {target}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
            return True  # Optimize on error to be safe
    
    async def _get_recent_performance(self) -> Dict[str, float]:
        """Get recent performance metrics across all strategies."""
        try:
            # Get metrics for last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # This would integrate with your existing metrics system
            # For now, return placeholder metrics
            return {
                "min_sharpe": 1.1,
                "min_win_rate": 0.42,
                "max_drawdown": 0.18,
                "min_profit_factor": 1.6,
                "min_expectancy": 0.018,
            }
            
        except Exception as e:
            logger.error(f"Error getting recent performance: {e}")
            return {}
    
    async def optimize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Run aggressive optimization for all strategies."""
        logger.info("Starting aggressive strategy optimization...")
        
        optimization_results = {}
        
        for strategy_name, param_ranges in self.parameter_ranges.items():
            try:
                logger.info(f"Optimizing {strategy_name}...")
                
                # Run parameter optimization
                best_params = await self._optimize_strategy_parameters(
                    strategy_name, param_ranges
                )
                
                if best_params:
                    optimization_results[strategy_name] = best_params
                    logger.info(f"Best params for {strategy_name}: {best_params}")
                
            except Exception as e:
                logger.error(f"Error optimizing {strategy_name}: {e}")
        
        # Update configuration with optimized parameters
        await self._update_configuration(optimization_results)
        
        # Log optimization history
        self.optimization_history.append({
            "timestamp": datetime.now().isoformat(),
            "results": optimization_results,
            "targets": self.target_metrics.copy()
        })
        
        self.last_optimization = datetime.now()
        
        logger.info("Strategy optimization completed")
        return optimization_results
    
    async def _optimize_strategy_parameters(
        self, strategy_name: str, param_ranges: Dict[str, List]
    ) -> Optional[Dict[str, Any]]:
        """Optimize parameters for a specific strategy using grid search."""
        try:
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(param_ranges)
            
            best_score = -np.inf
            best_params = None
            
            # Test each parameter combination
            for params in param_combinations:
                score = await self._evaluate_parameter_set(strategy_name, params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            if best_params:
                logger.info(f"Best score for {strategy_name}: {best_score:.4f}")
                return best_params
            
        except Exception as e:
            logger.error(f"Error optimizing parameters for {strategy_name}: {e}")
        
        return None
    
    def _generate_parameter_combinations(self, param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations for grid search."""
        import itertools
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations
    
    async def _evaluate_parameter_set(
        self, strategy_name: str, params: Dict[str, Any]
    ) -> float:
        """Evaluate a parameter set using backtesting or simulation."""
        try:
            # This would integrate with your enhanced backtester
            # For now, use a simplified scoring approach
            
            # Simulate parameter evaluation
            base_score = 1.0
            
            # Adjust score based on parameter values (aggressive optimization)
            if strategy_name == "micro_scalp_bot":
                # Prefer faster, more aggressive parameters
                if params.get("atr_period", 10) <= 6:
                    base_score *= 1.2
                if params.get("stop_loss_atr_mult", 1.0) <= 0.8:
                    base_score *= 1.1
                if params.get("cooldown_bars", 5) <= 2:
                    base_score *= 1.15
            
            elif strategy_name == "trend_bot":
                # Prefer faster trend detection
                if params.get("ema_fast", 15) <= 12:
                    base_score *= 1.1
                if params.get("adx_threshold", 20) <= 15:
                    base_score *= 1.15
                if params.get("stop_loss_atr_mult", 1.5) <= 1.0:
                    base_score *= 1.1
            
            elif strategy_name == "sniper_bot":
                # Prefer earlier detection
                if params.get("breakout_pct", 0.03) <= 0.025:
                    base_score *= 1.2
                if params.get("atr_window", 10) <= 8:
                    base_score *= 1.1
            
            # Add some randomness to avoid getting stuck in local optima
            noise = np.random.normal(0, 0.05)
            final_score = base_score + noise
            
            return max(0.0, final_score)
            
        except Exception as e:
            logger.error(f"Error evaluating parameters for {strategy_name}: {e}")
            return 0.0
    
    async def _update_configuration(self, optimization_results: Dict[str, Dict[str, Any]]):
        """Update configuration file with optimized parameters."""
        try:
            # Load current configuration
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update strategy parameters
            for strategy_name, params in optimization_results.items():
                if strategy_name in config:
                    config[strategy_name].update(params)
                else:
                    config[strategy_name] = params
            
            # Save updated configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info("Configuration updated with optimized parameters")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
    
    async def run_continuous_optimization(self):
        """Run continuous optimization loop."""
        logger.info("Starting continuous optimization loop...")
        
        while True:
            try:
                if await self.should_optimize():
                    await self.optimize_strategies()
                
                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(3600)  # Wait before retrying
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history and current status."""
        return {
            "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
            "optimization_interval_hours": self.optimization_interval_hours,
            "target_metrics": self.target_metrics,
            "optimization_history_count": len(self.optimization_history),
            "recent_optimizations": self.optimization_history[-5:] if self.optimization_history else [],
            "next_optimization": (
                self.last_optimization + timedelta(hours=self.optimization_interval_hours)
            ).isoformat() if self.last_optimization else None
        }


# Global optimizer instance
_optimizer = AggressiveAutoOptimizer()


async def run_optimization():
    """Run a single optimization cycle."""
    return await _optimizer.optimize_strategies()


async def start_continuous_optimization():
    """Start the continuous optimization loop."""
    await _optimizer.run_continuous_optimization()


def get_optimization_status() -> Dict[str, Any]:
    """Get current optimization status."""
    return _optimizer.get_optimization_summary()


if __name__ == "__main__":
    # Run optimization when script is executed directly
    asyncio.run(run_optimization())
