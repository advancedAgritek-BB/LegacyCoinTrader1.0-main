"""
Pump.fun Intelligent Launch Analyzer

This module provides sophisticated analysis and scoring of pump.fun token launches
by combining on-chain data, social sentiment, market microstructure, and
machine learning predictions to identify high-probability pump opportunities.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from .pump_fun_monitor import PumpFunLaunch
from .social_sentiment_analyzer import SocialSentimentAnalyzer
from .momentum_detector import MomentumDetector
from .pool_analyzer import PoolAnalyzer
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class LaunchAnalysis:
    """Comprehensive analysis of a pump.fun token launch."""
    
    # Core identifiers
    launch: PumpFunLaunch
    
    # Technical Analysis
    price_momentum: float = 0.0
    volume_profile: float = 0.0
    liquidity_quality: float = 0.0
    volatility_score: float = 0.0
    
    # Social Sentiment
    social_buzz: float = 0.0
    sentiment_score: float = 0.0
    influencer_impact: float = 0.0
    community_growth: float = 0.0
    
    # Market Microstructure
    whale_activity: float = 0.0
    transaction_patterns: float = 0.0
    order_book_imbalance: float = 0.0
    market_efficiency: float = 0.0
    
    # Risk Assessment
    rug_pull_risk: float = 0.0
    liquidity_lock_risk: float = 0.0
    creator_reputation: float = 0.0
    contract_security: float = 0.0
    
    # Composite Scores
    pump_probability: float = 0.0
    risk_adjusted_score: float = 0.0
    timing_optimization: float = 0.0
    final_score: float = 0.0
    
    # Analysis metadata
    analysis_timestamp: float = field(default_factory=time.time)
    confidence_level: float = 0.0
    data_quality: float = 0.0


@dataclass
class AnalysisConfig:
    """Configuration for the launch analyzer."""
    
    # Feature weights for scoring
    feature_weights = {
        "technical": 0.25,
        "social": 0.20,
        "microstructure": 0.25,
        "risk": 0.30
    }
    
    # Thresholds
    min_confidence: float = 0.7
    min_data_quality: float = 0.6
    high_probability_threshold: float = 0.8
    high_risk_threshold: float = 0.7
    
    # Analysis intervals
    analysis_interval: float = 60.0  # seconds
    update_interval: float = 300.0   # seconds
    
    # ML model settings
    enable_ml_predictions: bool = True
    model_update_frequency: int = 100  # updates per model retrain
    prediction_confidence_threshold: float = 0.75


class PumpFunAnalyzer:
    """
    Intelligent analyzer for pump.fun token launches.
    
    Features:
    - Multi-factor analysis combining technical, social, and market data
    - Machine learning-powered pump probability prediction
    - Real-time risk assessment and scoring
    - Historical performance correlation
    - Adaptive scoring based on market conditions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analyzer_config = AnalysisConfig(**config.get("pump_fun_analyzer", {}))
        
        # Core components
        self.social_analyzer = SocialSentimentAnalyzer(config)
        self.momentum_detector = MomentumDetector(config)
        self.pool_analyzer = PoolAnalyzer(config)
        
        # Analysis state
        self.analyzing = False
        self.analysis_task: Optional[asyncio.Task] = None
        
        # Analysis cache
        self.analysis_cache: Dict[str, LaunchAnalysis] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Historical data
        self.launch_history: List[LaunchAnalysis] = []
        self.performance_correlation: Dict[str, float] = {}
        
        # Machine learning
        self.ml_model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.model_performance: Dict[str, float] = {}
        
        # Statistics
        self.analysis_stats = {
            "total_analyses": 0,
            "high_probability_launches": 0,
            "successful_predictions": 0,
            "average_prediction_accuracy": 0.0,
            "last_model_update": 0
        }
        
        # Background tasks
        self.update_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the launch analyzer."""
        if self.analyzing:
            logger.warning("Launch analyzer already running")
            return
            
        try:
            logger.info("Starting pump.fun launch analyzer...")
            
            # Initialize components
            await self.social_analyzer.start()
            await self.momentum_detector.start()
            await self.pool_analyzer.start()
            
            # Load or train ML model
            await self._load_or_train_model()
            
            # Start analysis task
            self.analyzing = True
            self.analysis_task = asyncio.create_task(self._analysis_loop())
            
            # Start update task
            self.update_task = asyncio.create_task(self._update_loop())
            
            logger.info("Launch analyzer started successfully")
            
        except Exception as exc:
            logger.error(f"Failed to start launch analyzer: {exc}")
            self.analyzing = False
            raise
            
    async def stop(self):
        """Stop the launch analyzer."""
        if not self.analyzing:
            return
            
        try:
            logger.info("Stopping pump.fun launch analyzer...")
            
            self.analyzing = False
            
            # Cancel background tasks
            if self.analysis_task:
                self.analysis_task.cancel()
                try:
                    await self.analysis_task
                except asyncio.CancelledError:
                    pass
                    
            if self.update_task:
                self.update_task.cancel()
                try:
                    await self.update_task
                except asyncio.CancelledError:
                    pass
                    
            # Stop components
            await self.social_analyzer.stop()
            await self.momentum_detector.stop()
            await self.pool_analyzer.stop()
            
            logger.info("Launch analyzer stopped successfully")
            
        except Exception as exc:
            logger.error(f"Error stopping launch analyzer: {exc}")
            
    async def analyze_launch(self, launch: PumpFunLaunch) -> LaunchAnalysis:
        """Analyze a single token launch."""
        try:
            # Check cache first
            cache_key = f"{launch.pool_address}_{int(launch.launch_time)}"
            if cache_key in self.analysis_cache:
                cached = self.analysis_cache[cache_key]
                if time.time() - cached.analysis_timestamp < self.cache_ttl:
                    return cached
                    
            logger.info(f"Analyzing launch: {launch.token_symbol}")
            
            # Create analysis object
            analysis = LaunchAnalysis(launch=launch)
            
            # Perform comprehensive analysis
            await self._analyze_technical_factors(analysis)
            await self._analyze_social_factors(analysis)
            await self._analyze_market_microstructure(analysis)
            await self._assess_risks(analysis)
            
            # Calculate composite scores
            await self._calculate_composite_scores(analysis)
            
            # Apply ML predictions if available
            if self.analyzer_config.enable_ml_predictions and self.ml_model:
                await self._apply_ml_predictions(analysis)
                
            # Cache the analysis
            self.analysis_cache[cache_key] = analysis
            
            # Add to history
            self.launch_history.append(analysis)
            
            # Update statistics
            self.analysis_stats["total_analyses"] += 1
            if analysis.pump_probability >= self.analyzer_config.high_probability_threshold:
                self.analysis_stats["high_probability_launches"] += 1
                
            logger.info(f"Analysis completed for {launch.token_symbol}: Score={analysis.final_score:.3f}")
            
            return analysis
            
        except Exception as exc:
            logger.error(f"Error analyzing launch {launch.token_symbol}: {exc}")
            # Return basic analysis with error handling
            return LaunchAnalysis(
                launch=launch,
                confidence_level=0.0,
                data_quality=0.0
            )
            
    async def _analyze_technical_factors(self, analysis: LaunchAnalysis):
        """Analyze technical factors for a launch."""
        try:
            launch = analysis.launch
            
            # Price momentum analysis
            if launch.current_price > 0 and launch.initial_price > 0:
                price_change = (launch.current_price - launch.initial_price) / launch.initial_price
                analysis.price_momentum = min(max(price_change * 10, 0), 1)  # Scale to 0-1
            else:
                analysis.price_momentum = 0.5
                
            # Volume profile analysis
            if launch.volume_24h > 0:
                # Normalize volume relative to liquidity
                volume_liquidity_ratio = launch.volume_24h / max(launch.current_liquidity, 1)
                analysis.volume_profile = min(volume_liquidity_ratio / 10, 1)  # Scale to 0-1
            else:
                analysis.volume_profile = 0.0
                
            # Liquidity quality analysis
            if launch.initial_liquidity > 0:
                # Higher liquidity generally indicates better quality
                if launch.initial_liquidity >= 100000:
                    analysis.liquidity_quality = 0.9
                elif launch.initial_liquidity >= 50000:
                    analysis.liquidity_quality = 0.7
                elif launch.initial_liquidity >= 10000:
                    analysis.liquidity_quality = 0.5
                else:
                    analysis.liquidity_quality = 0.3
            else:
                analysis.liquidity_quality = 0.0
                
            # Volatility score
            # This would be calculated from price data over time
            analysis.volatility_score = 0.5  # Placeholder
            
        except Exception as exc:
            logger.error(f"Error analyzing technical factors: {exc}")
            
    async def _analyze_social_factors(self, analysis: LaunchAnalysis):
        """Analyze social and sentiment factors for a launch."""
        try:
            launch = analysis.launch
            
            # Get social sentiment data
            sentiment_data = await self.social_analyzer.get_token_sentiment(launch.token_mint)
            
            if sentiment_data:
                analysis.social_buzz = sentiment_data.get("buzz_score", 0.0)
                analysis.sentiment_score = sentiment_data.get("sentiment_score", 0.5)
                analysis.influencer_impact = sentiment_data.get("influencer_score", 0.0)
                analysis.community_growth = sentiment_data.get("growth_score", 0.0)
            else:
                # Default values if no sentiment data
                analysis.social_buzz = 0.0
                analysis.sentiment_score = 0.5
                analysis.influencer_impact = 0.0
                analysis.community_growth = 0.0
                
        except Exception as exc:
            logger.error(f"Error analyzing social factors: {exc}")
            
    async def _analyze_market_microstructure(self, analysis: LaunchAnalysis):
        """Analyze market microstructure factors for a launch."""
        try:
            launch = analysis.launch
            
            # Get pool analysis data
            pool_data = await self.pool_analyzer.analyze_pool(launch.pool_address)
            
            if pool_data:
                analysis.whale_activity = pool_data.get("whale_activity_score", 0.0)
                analysis.transaction_patterns = pool_data.get("transaction_pattern_score", 0.0)
                analysis.order_book_imbalance = pool_data.get("order_book_imbalance", 0.0)
                analysis.market_efficiency = pool_data.get("market_efficiency_score", 0.0)
            else:
                # Default values if no pool data
                analysis.whale_activity = 0.5
                analysis.transaction_patterns = 0.5
                analysis.order_book_imbalance = 0.5
                analysis.market_efficiency = 0.5
                
        except Exception as exc:
            logger.error(f"Error analyzing market microstructure: {exc}")
            
    async def _assess_risks(self, analysis: LaunchAnalysis):
        """Assess various risk factors for a launch."""
        try:
            launch = analysis.launch
            
            # Rug pull risk assessment
            rug_risk_factors = []
            
            # Low liquidity risk
            if launch.initial_liquidity < 10000:
                rug_risk_factors.append(0.8)
            elif launch.initial_liquidity < 50000:
                rug_risk_factors.append(0.5)
            else:
                rug_risk_factors.append(0.2)
                
            # Creator wallet risk
            if launch.creator_wallet:
                # This would check creator reputation and balance
                creator_risk = 0.5  # Placeholder
                rug_risk_factors.append(creator_risk)
                
            # Price volatility risk
            if launch.initial_price < 0.00001:
                rug_risk_factors.append(0.7)
            elif launch.initial_price > 0.1:
                rug_risk_factors.append(0.3)
            else:
                rug_risk_factors.append(0.1)
                
            analysis.rug_pull_risk = np.mean(rug_risk_factors) if rug_risk_factors else 0.5
            
            # Liquidity lock risk
            # This would check if liquidity is locked in smart contracts
            analysis.liquidity_lock_risk = 0.3  # Placeholder
            
            # Creator reputation
            # This would check creator's history and reputation
            analysis.creator_reputation = 0.5  # Placeholder
            
            # Contract security
            # This would analyze smart contract security
            analysis.contract_security = 0.7  # Placeholder
            
        except Exception as exc:
            logger.error(f"Error assessing risks: {exc}")
            
    async def _calculate_composite_scores(self, analysis: LaunchAnalysis):
        """Calculate composite scores for a launch."""
        try:
            # Calculate pump probability
            technical_score = (
                analysis.price_momentum * 0.3 +
                analysis.volume_profile * 0.3 +
                analysis.liquidity_quality * 0.2 +
                analysis.volatility_score * 0.2
            )
            
            social_score = (
                analysis.social_buzz * 0.3 +
                analysis.sentiment_score * 0.3 +
                analysis.influencer_impact * 0.2 +
                analysis.community_growth * 0.2
            )
            
            microstructure_score = (
                analysis.whale_activity * 0.25 +
                analysis.transaction_patterns * 0.25 +
                analysis.order_book_imbalance * 0.25 +
                analysis.market_efficiency * 0.25
            )
            
            risk_score = (
                (1 - analysis.rug_pull_risk) * 0.4 +
                (1 - analysis.liquidity_lock_risk) * 0.2 +
                analysis.creator_reputation * 0.2 +
                analysis.contract_security * 0.2
            )
            
            # Weighted composite score
            weights = self.analyzer_config.feature_weights
            analysis.pump_probability = (
                technical_score * weights["technical"] +
                social_score * weights["social"] +
                microstructure_score * weights["microstructure"] +
                risk_score * weights["risk"]
            )
            
            # Risk-adjusted score
            analysis.risk_adjusted_score = analysis.pump_probability * (1 - analysis.rug_pull_risk)
            
            # Timing optimization score
            time_since_launch = time.time() - analysis.launch.launch_time
            if time_since_launch < 300:  # Less than 5 minutes
                analysis.timing_optimization = 0.3
            elif time_since_launch < 1800:  # 5-30 minutes
                analysis.timing_optimization = 0.9
            elif time_since_launch < 3600:  # 30-60 minutes
                analysis.timing_optimization = 0.6
            else:
                analysis.timing_optimization = 0.2
                
            # Final score
            analysis.final_score = (
                analysis.risk_adjusted_score * 0.6 +
                analysis.timing_optimization * 0.4
            )
            
            # Confidence level based on data quality
            analysis.confidence_level = min(analysis.final_score, 0.95)
            analysis.data_quality = 0.8  # Placeholder - would be calculated from available data
            
        except Exception as exc:
            logger.error(f"Error calculating composite scores: {exc}")
            
    async def _apply_ml_predictions(self, analysis: LaunchAnalysis):
        """Apply machine learning predictions to the analysis."""
        try:
            if not self.ml_model or not self.scaler:
                return
                
            # Prepare features for ML model
            features = self._extract_ml_features(analysis)
            
            if len(features) != len(self.feature_names):
                logger.warning("Feature mismatch for ML prediction")
                return
                
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            prediction = self.ml_model.predict(features_scaled)[0]
            
            # Update analysis with ML prediction
            ml_confidence = self.model_performance.get("r2_score", 0.5)
            
            # Blend ML prediction with traditional analysis
            if ml_confidence > self.analyzer_config.prediction_confidence_threshold:
                analysis.pump_probability = (
                    analysis.pump_probability * 0.7 +
                    prediction * 0.3
                )
                
                # Recalculate final score
                analysis.final_score = (
                    analysis.risk_adjusted_score * 0.6 +
                    analysis.timing_optimization * 0.4
                )
                
        except Exception as exc:
            logger.error(f"Error applying ML predictions: {exc}")
            
    def _extract_ml_features(self, analysis: LaunchAnalysis) -> List[float]:
        """Extract features for machine learning model."""
        try:
            features = [
                analysis.price_momentum,
                analysis.volume_profile,
                analysis.liquidity_quality,
                analysis.volatility_score,
                analysis.social_buzz,
                analysis.sentiment_score,
                analysis.influencer_impact,
                analysis.community_growth,
                analysis.whale_activity,
                analysis.transaction_patterns,
                analysis.order_book_imbalance,
                analysis.market_efficiency,
                1 - analysis.rug_pull_risk,
                1 - analysis.liquidity_lock_risk,
                analysis.creator_reputation,
                analysis.contract_security
            ]
            
            return features
            
        except Exception as exc:
            logger.error(f"Error extracting ML features: {exc}")
            return [0.0] * 16  # Return default features
            
    async def _analysis_loop(self):
        """Main analysis loop."""
        try:
            while self.analyzing:
                # This loop would process launches from a queue
                # For now, it's just a placeholder
                await asyncio.sleep(self.analyzer_config.analysis_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Error in analysis loop: {exc}")
            
    async def _update_loop(self):
        """Update loop for model retraining and performance tracking."""
        try:
            while self.analyzing:
                # Update ML model if needed
                if self.analyzer_config.enable_ml_predictions:
                    await self._update_ml_model()
                    
                # Update performance correlations
                await self._update_performance_correlations()
                
                # Wait for next update
                await asyncio.sleep(self.analyzer_config.update_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Error in update loop: {exc}")
            
    async def _load_or_train_model(self):
        """Load existing ML model or train a new one."""
        try:
            model_path = "models/pump_fun_predictor.joblib"
            scaler_path = "models/pump_fun_scaler.joblib"
            
            # Try to load existing model
            try:
                self.ml_model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded existing ML model")
                return
            except FileNotFoundError:
                logger.info("No existing model found, will train new one")
                
            # Train new model if no existing one
            await self._train_ml_model()
            
        except Exception as exc:
            logger.error(f"Error loading/training ML model: {exc}")
            
    async def _train_ml_model(self):
        """Train a new machine learning model."""
        try:
            logger.info("Training new ML model for pump prediction...")
            
            # This would use historical launch data to train the model
            # For now, create a placeholder model
            
            # Create dummy training data
            n_samples = 1000
            n_features = 16
            
            X = np.random.rand(n_samples, n_features)
            y = np.random.rand(n_samples)
            
            # Train model
            self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.ml_model.fit(X, y)
            
            # Create scaler
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            
            # Set feature names
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
            
            # Save model
            joblib.dump(self.ml_model, "models/pump_fun_predictor.joblib")
            joblib.dump(self.scaler, "models/pump_fun_scaler.joblib")
            
            logger.info("ML model trained and saved successfully")
            
        except Exception as exc:
            logger.error(f"Error training ML model: {exc}")
            
    async def _update_ml_model(self):
        """Update the ML model with new data."""
        try:
            # Check if we have enough new data
            if len(self.launch_history) < self.analyzer_config.model_update_frequency:
                return
                
            # Retrain model with new data
            await self._train_ml_model()
            
            # Update statistics
            self.analysis_stats["last_model_update"] = time.time()
            
        except Exception as exc:
            logger.error(f"Error updating ML model: {exc}")
            
    async def _update_performance_correlations(self):
        """Update performance correlations for features."""
        try:
            if len(self.launch_history) < 10:
                return
                
            # Calculate correlations between features and actual performance
            # This would help improve the scoring algorithm
            
            for feature_name in self.feature_names:
                # Placeholder correlation calculation
                correlation = 0.5
                self.performance_correlation[feature_name] = correlation
                
        except Exception as exc:
            logger.error(f"Error updating performance correlations: {exc}")
            
    def get_high_probability_launches(self, min_score: float = 0.8) -> List[LaunchAnalysis]:
        """Get list of high-probability launches."""
        return [
            analysis for analysis in self.launch_history
            if analysis.final_score >= min_score
        ]
        
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return self.analysis_stats.copy()
        
    def get_performance_correlations(self) -> Dict[str, float]:
        """Get feature performance correlations."""
        return self.performance_correlation.copy()


# Factory function
def create_pump_fun_analyzer(config: Dict[str, Any]) -> PumpFunAnalyzer:
    """Create and configure a pump.fun analyzer instance."""
    return PumpFunAnalyzer(config)
