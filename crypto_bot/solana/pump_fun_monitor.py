"""
Pump.fun Liquidity Pool Launch Monitoring Service

This service provides intelligent detection and analysis of new token launches
on pump.fun with high probability of significant pumps. It integrates with
the existing pump detection infrastructure and provides real-time monitoring,
scoring, and execution recommendations.
"""

import asyncio
import logging
import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
import aiohttp
import numpy as np
import pandas as pd

from .pump_detector import PumpDetector, PoolAnalysis, PumpSignal
from .safety import is_safe
from .score import calculate_token_score
from .token_utils import get_token_metadata, validate_token_address
from .pyth_utils import get_pyth_price
from ..utils.telemetry import telemetry
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PumpFunLaunch:
    """Represents a detected pump.fun token launch."""
    
    # Core identifiers
    pool_address: str
    token_mint: str
    token_symbol: str
    token_name: str
    
    # Launch metadata
    launch_time: float
    initial_liquidity: float
    initial_price: float
    creator_wallet: str
    
    # Pool configuration
    fee_tier: str
    pool_type: str  # "standard", "concentrated", "stable"
    
    # Market data
    current_price: float
    current_liquidity: float
    volume_24h: float
    price_change_24h: float
    
    # Analysis scores
    pump_probability: float = 0.0
    risk_score: float = 0.0
    timing_score: float = 0.0
    composite_score: float = 0.0
    
    # Signals and alerts
    signals: List[PumpSignal] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    
    # Metadata
    last_updated: float = field(default_factory=time.time)
    is_active: bool = True


@dataclass
class LaunchFilter:
    """Configuration for filtering token launches."""
    
    # Liquidity thresholds
    min_initial_liquidity: float = 1000.0  # USD
    max_initial_liquidity: float = 1000000.0  # USD
    
    # Price thresholds
    min_initial_price: float = 0.000001  # USD
    max_initial_price: float = 1.0  # USD
    
    # Creator requirements
    require_verified_creator: bool = False
    min_creator_balance: float = 100.0  # SOL
    
    # Pool requirements
    allowed_fee_tiers: List[str] = field(default_factory=lambda: ["0.3%", "1%", "0.05%"])
    allowed_pool_types: List[str] = field(default_factory=lambda: ["standard"])
    
    # Score thresholds
    min_pump_probability: float = 0.6
    max_risk_score: float = 0.4
    min_composite_score: float = 0.7


class PumpFunMonitor:
    """
    Intelligent pump.fun liquidity pool launch monitoring service.
    
    Features:
    - Real-time pool creation detection
    - Multi-factor launch scoring
    - Social sentiment integration
    - Risk assessment and filtering
    - Execution opportunity identification
    - Historical performance tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor_config = config.get("pump_fun_monitor", {})
        
        # Core components
        self.pump_detector = PumpDetector(config)
        self.launch_filter = LaunchFilter(**self.monitor_config.get("launch_filter", {}))
        
        # Monitoring state
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Data sources
        self.websocket_url = "wss://api.pump.fun/v1/ws"
        self.rpc_url = "https://api.pump.fun/v1"
        self.api_key = self._get_env_or_config("PUMPFUN_API_KEY", "api_key")
        self.wallet_address = self._get_env_or_config("PUMPFUN_WALLET", "wallet_address")
        self.private_key = self._get_env_or_config("PUMPFUN_PRIVATE_KEY", "private_key")
        
        # Launch tracking
        self.active_launches: Dict[str, PumpFunLaunch] = {}
        self.launch_history: List[PumpFunLaunch] = []
        self.launch_signals: List[PumpSignal] = []
        
        # Performance tracking
        self.performance_metrics = {
            "total_launches_detected": 0,
            "high_probability_launches": 0,
            "successful_pumps": 0,
            "average_pump_return": 0.0,
            "rug_pulls_detected": 0
        }
        
        # Callbacks
        self.launch_callbacks: List[Callable[[PumpFunLaunch], None]] = []
        self.alert_callbacks: List[Callable[[str, Dict], None]] = []
        
        # Rate limiting
        self.last_api_call = 0
        self.api_call_interval = 0.1  # 100ms between calls
        
        # Caching
        self.token_metadata_cache: Dict[str, Dict] = {}
        self.metadata_cache_ttl = 300  # 5 minutes
        
    def _get_env_or_config(self, env_key: str, config_key: str) -> Optional[str]:
        """Get value from environment variable or config, preferring environment."""
        import os
        from dotenv import load_dotenv
        
        # Load .env file if it exists
        load_dotenv()
        
        # Try environment variable first
        env_value = os.getenv(env_key)
        if env_value:
            return env_value
        # Fall back to config
        return self.monitor_config.get(config_key)
        
    async def start(self):
        """Start the pump.fun monitoring service."""
        if self.monitoring:
            logger.warning("Pump.fun monitor already running")
            return
            
        try:
            logger.info("Starting pump.fun launch monitoring service...")
            
            # Initialize pump detector
            await self.pump_detector.start()
            
            # Start monitoring task
            self.monitoring = True
            self.monitor_task = asyncio.create_task(self._monitor_launches())
            
            # Start performance tracking
            asyncio.create_task(self._track_performance())
            
            logger.info("Pump.fun monitor started successfully")
            telemetry.inc("pump_fun_monitor.started")
            
        except Exception as exc:
            logger.error(f"Failed to start pump.fun monitor: {exc}")
            self.monitoring = False
            raise
            
    async def stop(self):
        """Stop the pump.fun monitoring service."""
        if not self.monitoring:
            return
            
        try:
            logger.info("Stopping pump.fun launch monitoring service...")
            
            self.monitoring = False
            
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
                    
            # Stop pump detector
            await self.pump_detector.stop()
            
            logger.info("Pump.fun monitor stopped successfully")
            telemetry.inc("pump_fun_monitor.stopped")
            
        except Exception as exc:
            logger.error(f"Error stopping pump.fun monitor: {exc}")
            
    async def _monitor_launches(self):
        """Main monitoring loop for pump.fun launches."""
        logger.info("Starting pump.fun launch monitoring loop...")
        
        while self.monitoring:
            try:
                # Check for new launches
                await self._check_new_launches()
                
                # Update existing launches
                await self._update_active_launches()
                
                # Analyze and score launches
                await self._analyze_launches()
                
                # Generate alerts
                await self._generate_alerts()
                
                # Wait for next cycle
                await asyncio.sleep(self.monitor_config.get("monitor_interval_seconds", 30))
                
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in launch monitoring loop: {exc}")
                await asyncio.sleep(5)
                
    async def _check_new_launches(self):
        """Check for new token launches on pump.fun."""
        try:
            # Get recent pool creations
            new_pools = await self._get_recent_pools()
            
            for pool in new_pools:
                if pool["pool_address"] not in self.active_launches:
                    await self._process_new_launch(pool)
                    
        except Exception as exc:
            logger.error(f"Error checking new launches: {exc}")
            
    async def _get_recent_pools(self) -> List[Dict]:
        """Get recent pool creations from pump.fun API."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                
                async with session.get(
                    f"{self.rpc_url}/pools/recent",
                    headers=headers,
                    params={"limit": 50, "timeframe": "1h"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("pools", [])
                    else:
                        logger.warning(f"Failed to get recent pools: {response.status}")
                        return []
                        
        except Exception as exc:
            logger.error(f"Error fetching recent pools: {exc}")
            return []
            
    async def _process_new_launch(self, pool_data: Dict):
        """Process a newly detected token launch."""
        try:
            pool_address = pool_data["pool_address"]
            token_mint = pool_data["token_mint"]
            
            logger.info(f"Processing new launch: {token_mint} at {pool_address}")
            
            # Get token metadata
            token_metadata = await self._get_token_metadata(token_mint)
            
            # Create launch object
            launch = PumpFunLaunch(
                pool_address=pool_address,
                token_mint=token_mint,
                token_symbol=token_metadata.get("symbol", "UNKNOWN"),
                token_name=token_metadata.get("name", "Unknown Token"),
                launch_time=time.time(),
                initial_liquidity=pool_data.get("initial_liquidity", 0.0),
                initial_price=pool_data.get("initial_price", 0.0),
                creator_wallet=pool_data.get("creator", ""),
                fee_tier=pool_data.get("fee_tier", "0.3%"),
                pool_type=pool_data.get("pool_type", "standard"),
                current_price=pool_data.get("initial_price", 0.0),
                current_liquidity=pool_data.get("initial_liquidity", 0.0),
                volume_24h=0.0,
                price_change_24h=0.0
            )
            
            # Add to active launches
            self.active_launches[pool_address] = launch
            self.launch_history.append(launch)
            
            # Update performance metrics
            self.performance_metrics["total_launches_detected"] += 1
            
            # Notify callbacks
            for callback in self.launch_callbacks:
                try:
                    callback(launch)
                except Exception as exc:
                    logger.error(f"Error in launch callback: {exc}")
                    
            logger.info(f"New launch processed: {launch.token_symbol} ({launch.token_mint})")
            
        except Exception as exc:
            logger.error(f"Error processing new launch: {exc}")
            
    async def _update_active_launches(self):
        """Update data for active launches."""
        try:
            for pool_address, launch in list(self.active_launches.items()):
                if not launch.is_active:
                    continue
                    
                # Check if launch is still recent (within 24 hours)
                if time.time() - launch.launch_time > 86400:
                    launch.is_active = False
                    continue
                    
                # Update market data
                await self._update_launch_data(launch)
                
        except Exception as exc:
            logger.error(f"Error updating active launches: {exc}")
            
    async def _update_launch_data(self, launch: PumpFunLaunch):
        """Update market data for a specific launch."""
        try:
            # Get current pool data
            pool_data = await self._get_pool_data(launch.pool_address)
            
            if pool_data:
                # Update price and liquidity
                launch.current_price = pool_data.get("current_price", launch.current_price)
                launch.current_liquidity = pool_data.get("current_liquidity", launch.current_liquidity)
                launch.volume_24h = pool_data.get("volume_24h", launch.volume_24h)
                
                # Calculate price change
                if launch.initial_price > 0:
                    launch.price_change_24h = (
                        (launch.current_price - launch.initial_price) / launch.initial_price * 100
                    )
                    
                launch.last_updated = time.time()
                
        except Exception as exc:
            logger.error(f"Error updating launch data for {launch.token_symbol}: {exc}")
            
    async def _analyze_launches(self):
        """Analyze and score active launches."""
        try:
            for launch in self.active_launches.values():
                if not launch.is_active:
                    continue
                    
                # Calculate pump probability
                pump_probability = await self._calculate_pump_probability(launch)
                launch.pump_probability = pump_probability
                
                # Calculate risk score
                risk_score = await self._calculate_risk_score(launch)
                launch.risk_score = risk_score
                
                # Calculate timing score
                timing_score = await self._calculate_timing_score(launch)
                launch.timing_score = timing_score
                
                # Calculate composite score
                composite_score = self._calculate_composite_score(launch)
                launch.composite_score = composite_score
                
                # Generate signals
                signals = await self._generate_launch_signals(launch)
                launch.signals = signals
                
                # Check if this is a high-probability launch
                if composite_score >= self.launch_filter.min_composite_score:
                    self.performance_metrics["high_probability_launches"] += 1
                    
        except Exception as exc:
            logger.error(f"Error analyzing launches: {exc}")
            
    async def _calculate_pump_probability(self, launch: PumpFunLaunch) -> float:
        """Calculate the probability of a significant pump."""
        try:
            # Use the existing pump detector
            pool_analysis = await self._create_pool_analysis(launch)
            return pool_analysis.pump_probability
            
        except Exception as exc:
            logger.error(f"Error calculating pump probability: {exc}")
            return 0.0
            
    async def _calculate_risk_score(self, launch: PumpFunLaunch) -> float:
        """Calculate the risk score for a launch."""
        try:
            risk_factors = []
            
            # Liquidity risk
            if launch.initial_liquidity < 10000:
                risk_factors.append(0.8)
            elif launch.initial_liquidity < 50000:
                risk_factors.append(0.4)
            else:
                risk_factors.append(0.1)
                
            # Creator risk
            if launch.creator_wallet:
                creator_balance = await self._get_wallet_balance(launch.creator_wallet)
                if creator_balance < 50:  # Less than 50 SOL
                    risk_factors.append(0.7)
                else:
                    risk_factors.append(0.2)
                    
            # Price risk
            if launch.initial_price < 0.00001:
                risk_factors.append(0.6)
            elif launch.initial_price > 0.1:
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)
                
            # Return average risk score
            return np.mean(risk_factors) if risk_factors else 0.5
            
        except Exception as exc:
            logger.error(f"Error calculating risk score: {exc}")
            return 0.5
            
    async def _calculate_timing_score(self, launch: PumpFunLaunch) -> float:
        """Calculate the optimal timing score for entry."""
        try:
            # Time since launch
            time_since_launch = time.time() - launch.launch_time
            
            # Optimal entry window: 5-30 minutes after launch
            if time_since_launch < 300:  # Less than 5 minutes
                timing_score = 0.3  # Too early
            elif time_since_launch < 1800:  # 5-30 minutes
                timing_score = 0.9  # Optimal window
            elif time_since_launch < 3600:  # 30-60 minutes
                timing_score = 0.6  # Still good
            else:
                timing_score = 0.2  # Too late
                
            return timing_score
            
        except Exception as exc:
            logger.error(f"Error calculating timing score: {exc}")
            return 0.5
            
    def _calculate_composite_score(self, launch: PumpFunLaunch) -> float:
        """Calculate the composite score for a launch."""
        try:
            # Weighted combination of scores
            weights = {
                "pump_probability": 0.4,
                "timing_score": 0.3,
                "risk_score": 0.3
            }
            
            # Risk score is inverted (lower is better)
            risk_adjusted = 1.0 - launch.risk_score
            
            composite_score = (
                launch.pump_probability * weights["pump_probability"] +
                launch.timing_score * weights["timing_score"] +
                risk_adjusted * weights["risk_score"]
            )
            
            return min(composite_score, 1.0)
            
        except Exception as exc:
            logger.error(f"Error calculating composite score: {exc}")
            return 0.0
            
    async def _generate_launch_signals(self, launch: PumpFunLaunch) -> List[PumpSignal]:
        """Generate trading signals for a launch."""
        try:
            signals = []
            
            # High probability pump signal
            if launch.composite_score >= 0.8:
                signals.append(PumpSignal(
                    signal_type="HIGH_PROBABILITY_PUMP",
                    strength=launch.composite_score,
                    confidence=0.9,
                    timestamp=time.time(),
                    details={
                        "token_symbol": launch.token_symbol,
                        "pool_address": launch.pool_address,
                        "composite_score": launch.composite_score
                    }
                ))
                
            # Early entry opportunity
            if launch.timing_score >= 0.8:
                signals.append(PumpSignal(
                    signal_type="EARLY_ENTRY_OPPORTUNITY",
                    strength=launch.timing_score,
                    confidence=0.8,
                    timestamp=time.time(),
                    details={
                        "token_symbol": launch.token_symbol,
                        "time_since_launch": time.time() - launch.launch_time
                    }
                ))
                
            # Risk alert
            if launch.risk_score >= 0.7:
                signals.append(PumpSignal(
                    signal_type="HIGH_RISK_ALERT",
                    strength=launch.risk_score,
                    confidence=0.9,
                    timestamp=time.time(),
                    details={
                        "token_symbol": launch.token_symbol,
                        "risk_factors": ["Low liquidity", "Suspicious creator", "Extreme price"]
                    }
                ))
                
            return signals
            
        except Exception as exc:
            logger.error(f"Error generating launch signals: {exc}")
            return []
            
    async def _create_pool_analysis(self, launch: PumpFunLaunch) -> PoolAnalysis:
        """Create a pool analysis object for the pump detector."""
        try:
            # This would integrate with your existing pump detector
            # For now, return a basic analysis
            return PoolAnalysis(
                pool_address=launch.pool_address,
                token_mint=launch.token_mint,
                initial_liquidity=launch.initial_liquidity,
                current_liquidity=launch.current_liquidity,
                liquidity_change_rate=0.0,
                liquidity_stability_score=0.5,
                transaction_velocity=0.0,
                unique_wallets=0,
                avg_transaction_size=0.0,
                whale_activity_score=0.0,
                price_momentum=0.0,
                volume_spike_factor=0.0,
                volume_consistency=0.0,
                price_stability=0.0,
                social_buzz_score=0.0,
                sentiment_score=0.0,
                influencer_mentions=0,
                rsi=50.0,
                bollinger_position=0.5,
                volume_profile_score=0.0,
                rug_risk_score=launch.risk_score,
                dev_activity_score=0.0,
                tokenomics_score=0.0,
                pump_probability=launch.pump_probability,
                timing_score=launch.timing_score,
                risk_adjusted_score=launch.composite_score
            )
            
        except Exception as exc:
            logger.error(f"Error creating pool analysis: {exc}")
            return PoolAnalysis(
                pool_address=launch.pool_address,
                token_mint=launch.token_mint,
                initial_liquidity=0.0,
                current_liquidity=0.0,
                liquidity_change_rate=0.0,
                liquidity_stability_score=0.0,
                transaction_velocity=0.0,
                unique_wallets=0,
                avg_transaction_size=0.0,
                whale_activity_score=0.0,
                price_momentum=0.0,
                volume_spike_factor=0.0,
                volume_consistency=0.0,
                price_stability=0.0,
                social_buzz_score=0.0,
                sentiment_score=0.0,
                influencer_mentions=0,
                rsi=50.0,
                bollinger_position=0.5,
                volume_profile_score=0.0,
                rug_risk_score=0.5,
                dev_activity_score=0.0,
                tokenomics_score=0.0
            )
            
    async def _generate_alerts(self):
        """Generate alerts for significant events."""
        try:
            for launch in self.active_launches.values():
                if not launch.is_active:
                    continue
                    
                # High probability pump alert
                if launch.composite_score >= 0.8 and not any(
                    s.signal_type == "HIGH_PROBABILITY_PUMP" for s in launch.signals
                ):
                    alert_msg = f"🚀 HIGH PROBABILITY PUMP: {launch.token_symbol} ({launch.token_mint})"
                    alert_data = {
                        "type": "HIGH_PROBABILITY_PUMP",
                        "launch": launch,
                        "timestamp": time.time()
                    }
                    
                    # Notify alert callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert_msg, alert_data)
                        except Exception as exc:
                            logger.error(f"Error in alert callback: {exc}")
                            
        except Exception as exc:
            logger.error(f"Error generating alerts: {exc}")
            
    async def _track_performance(self):
        """Track performance metrics for launches."""
        try:
            while self.monitoring:
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Wait for next update
                await asyncio.sleep(300)  # 5 minutes
                
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Error tracking performance: {exc}")
            
    async def _update_performance_metrics(self):
        """Update performance metrics based on launch outcomes."""
        try:
            # Calculate successful pumps (launches with >100% price increase)
            successful_pumps = sum(
                1 for launch in self.launch_history
                if launch.price_change_24h > 100
            )
            
            # Calculate average return
            returns = [
                launch.price_change_24h for launch in self.launch_history
                if launch.price_change_24h > 0
            ]
            
            avg_return = np.mean(returns) if returns else 0.0
            
            # Update metrics
            self.performance_metrics["successful_pumps"] = successful_pumps
            self.performance_metrics["average_pump_return"] = avg_return
            
        except Exception as exc:
            logger.error(f"Error updating performance metrics: {exc}")
            
    async def _get_token_metadata(self, token_mint: str) -> Dict:
        """Get token metadata with caching."""
        try:
            # Check cache
            if token_mint in self.token_metadata_cache:
                cached = self.token_metadata_cache[token_mint]
                if time.time() - cached["timestamp"] < self.metadata_cache_ttl:
                    return cached["data"]
                    
            # Fetch from API
            metadata = await get_token_metadata(token_mint)
            
            # Cache the result
            self.token_metadata_cache[token_mint] = {
                "data": metadata,
                "timestamp": time.time()
            }
            
            return metadata
            
        except Exception as exc:
            logger.error(f"Error getting token metadata: {exc}")
            return {"symbol": "UNKNOWN", "name": "Unknown Token"}
            
    async def _get_pool_data(self, pool_address: str) -> Optional[Dict]:
        """Get current pool data from pump.fun API."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                
                async with session.get(
                    f"{self.rpc_url}/pools/{pool_address}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return None
                        
        except Exception as exc:
            logger.error(f"Error getting pool data: {exc}")
            return None
            
    async def _get_wallet_balance(self, wallet_address: str) -> float:
        """Get SOL balance for a wallet address."""
        try:
            # This would integrate with your existing Solana RPC connection
            # For now, return a default value
            return 100.0
            
        except Exception as exc:
            logger.error(f"Error getting wallet balance: {exc}")
            return 0.0
            
    async def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.api_call_interval:
            await asyncio.sleep(self.api_call_interval - time_since_last_call)
            
        self.last_api_call = time.time()
        
    def add_launch_callback(self, callback: Callable[[PumpFunLaunch], None]):
        """Add a callback for new launch notifications."""
        self.launch_callbacks.append(callback)
        
    def add_alert_callback(self, callback: Callable[[str, Dict], None]):
        """Add a callback for alert notifications."""
        self.alert_callbacks.append(callback)
        
    def get_active_launches(self) -> List[PumpFunLaunch]:
        """Get list of active launches."""
        return list(self.active_launches.values())
        
    def get_high_probability_launches(self) -> List[PumpFunLaunch]:
        """Get list of high-probability launches."""
        return [
            launch for launch in self.active_launches.values()
            if launch.composite_score >= self.launch_filter.min_composite_score
        ]
        
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
        
    def get_launch_history(self) -> List[PumpFunLaunch]:
        """Get launch history."""
        return self.launch_history.copy()


# Factory function
def create_pump_fun_monitor(config: Dict[str, Any]) -> PumpFunMonitor:
    """Create and configure a pump.fun monitor instance."""
    return PumpFunMonitor(config)
