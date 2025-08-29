"""
Pump.fun Service Orchestrator

This module orchestrates all pump.fun monitoring services including the main monitor,
WebSocket monitor, analyzer, and execution engine. It provides a unified interface
for monitoring, analyzing, and executing on pump.fun token launches.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, deque
import yaml
from pathlib import Path

from .pump_fun_monitor import PumpFunMonitor, PumpFunLaunch, create_pump_fun_monitor
from .pump_fun_websocket import PumpFunWebSocketMonitor, create_pump_fun_websocket_monitor
from .pump_fun_analyzer import PumpFunAnalyzer, LaunchAnalysis, create_pump_fun_analyzer
from .pump_sniper_orchestrator import PumpSniperOrchestrator, SniperDecision
from ..utils.logger import setup_logger
from ..utils.telemetry import telemetry

logger = setup_logger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the pump.fun orchestrator."""
    
    # Service enablement
    enable_main_monitor: bool = True
    enable_websocket_monitor: bool = True
    enable_analyzer: bool = True
    enable_execution: bool = True
    
    # Monitoring settings
    monitor_interval: float = 30.0  # seconds
    analysis_interval: float = 60.0  # seconds
    execution_interval: float = 15.0  # seconds
    
    # Thresholds
    min_score_for_execution: float = 0.8
    max_risk_for_execution: float = 0.3
    min_liquidity_for_execution: float = 10000.0  # USD
    
    # Notification settings
    enable_telegram_notifications: bool = True
    enable_discord_notifications: bool = False
    enable_email_notifications: bool = False
    
    # Performance tracking
    track_performance: bool = True
    performance_update_interval: float = 300.0  # seconds


@dataclass
class OrchestratorState:
    """Current state of the pump.fun orchestrator."""
    
    # Service status
    main_monitor_running: bool = False
    websocket_monitor_running: bool = False
    analyzer_running: bool = False
    execution_running: bool = False
    
    # Current launches
    active_launches: int = 0
    high_probability_launches: int = 0
    launches_analyzed: int = 0
    executions_attempted: int = 0
    
    # Performance metrics
    successful_pumps: int = 0
    average_return: float = 0.0
    total_volume_traded: float = 0.0
    
    # Last update
    last_update: float = field(default_factory=time.time)


class PumpFunOrchestrator:
    """
    Main orchestrator for pump.fun monitoring and execution services.
    
    Features:
    - Coordinates all monitoring services
    - Manages launch analysis pipeline
    - Handles execution decisions
    - Provides unified monitoring interface
    - Performance tracking and reporting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orchestrator_config = OrchestratorConfig(**config.get("pump_fun_orchestrator", {}))
        
        # Core services
        self.main_monitor: Optional[PumpFunMonitor] = None
        self.websocket_monitor: Optional[PumpFunWebSocketMonitor] = None
        self.analyzer: Optional[PumpFunAnalyzer] = None
        self.executor: Optional[PumpSniperOrchestrator] = None
        
        # Orchestrator state
        self.running = False
        self.orchestrator_task: Optional[asyncio.Task] = None
        self.state = OrchestratorState()
        
        # Launch management
        self.launch_queue: deque = deque(maxlen=1000)
        self.analysis_queue: deque = deque(maxlen=1000)
        self.execution_queue: deque = deque(maxlen=100)
        
        # Callbacks and notifications
        self.launch_callbacks: List[Callable[[PumpFunLaunch], None]] = []
        self.analysis_callbacks: List[Callable[[LaunchAnalysis], None]] = []
        self.execution_callbacks: List[Callable[[SniperDecision], None]] = []
        self.alert_callbacks: List[Callable[[str, Dict], None]] = []
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.performance_task: Optional[asyncio.Task] = None
        
        # Configuration validation
        self._validate_config()
        
    def _validate_config(self):
        """Validate orchestrator configuration."""
        try:
            # Check required services
            if not any([
                self.orchestrator_config.enable_main_monitor,
                self.orchestrator_config.enable_websocket_monitor
            ]):
                raise ValueError("At least one monitoring service must be enabled")
                
            # Validate thresholds
            if self.orchestrator_config.min_score_for_execution <= 0 or self.orchestrator_config.min_score_for_execution > 1:
                raise ValueError("min_score_for_execution must be between 0 and 1")
                
            if self.orchestrator_config.max_risk_for_execution < 0 or self.orchestrator_config.max_risk_for_execution > 1:
                raise ValueError("max_risk_for_execution must be between 0 and 1")
                
            logger.info("Orchestrator configuration validated successfully")
            
        except Exception as exc:
            logger.error(f"Configuration validation failed: {exc}")
            raise
            
    async def start(self):
        """Start the pump.fun orchestrator."""
        if self.running:
            logger.warning("Pump.fun orchestrator already running")
            return
            
        try:
            logger.info("Starting pump.fun service orchestrator...")
            
            # Initialize services based on configuration
            await self._initialize_services()
            
            # Start enabled services
            await self._start_services()
            
            # Start orchestrator task
            self.running = True
            self.orchestrator_task = asyncio.create_task(self._orchestrator_loop())
            
            # Start performance tracking
            if self.orchestrator_config.track_performance:
                self.performance_task = asyncio.create_task(self._performance_tracking_loop())
                
            logger.info("Pump.fun orchestrator started successfully")
            telemetry.inc("pump_fun_orchestrator.started")
            
        except Exception as exc:
            logger.error(f"Failed to start pump.fun orchestrator: {exc}")
            self.running = False
            raise
            
    async def stop(self):
        """Stop the pump.fun orchestrator."""
        if not self.running:
            return
            
        try:
            logger.info("Stopping pump.fun service orchestrator...")
            
            self.running = False
            
            # Cancel background tasks
            if self.orchestrator_task:
                self.orchestrator_task.cancel()
                try:
                    await self.orchestrator_task
                except asyncio.CancelledError:
                    pass
                    
            if self.performance_task:
                self.performance_task.cancel()
                try:
                    await self.performance_task
                except asyncio.CancelledError:
                    pass
                    
            # Stop all services
            await self._stop_services()
            
            logger.info("Pump.fun orchestrator stopped successfully")
            telemetry.inc("pump_fun_orchestrator.stopped")
            
        except Exception as exc:
            logger.error(f"Error stopping pump.fun orchestrator: {exc}")
            
    async def _initialize_services(self):
        """Initialize all pump.fun services."""
        try:
            logger.info("Initializing pump.fun services...")
            
            # Initialize main monitor
            if self.orchestrator_config.enable_main_monitor:
                self.main_monitor = create_pump_fun_monitor(self.config)
                logger.info("Main monitor initialized")
                
            # Initialize WebSocket monitor
            if self.orchestrator_config.enable_websocket_monitor and self.main_monitor:
                self.websocket_monitor = create_pump_fun_websocket_monitor(self.config, self.main_monitor)
                logger.info("WebSocket monitor initialized")
                
            # Initialize analyzer
            if self.orchestrator_config.enable_analyzer:
                self.analyzer = create_pump_fun_analyzer(self.config)
                logger.info("Analyzer initialized")
                
            # Initialize executor
            if self.orchestrator_config.enable_execution:
                # This would integrate with your existing pump sniper
                # For now, create a placeholder
                logger.info("Execution service initialized (placeholder)")
                
            # Set up callbacks
            self._setup_service_callbacks()
            
            logger.info("All services initialized successfully")
            
        except Exception as exc:
            logger.error(f"Error initializing services: {exc}")
            raise
            
    async def _start_services(self):
        """Start all enabled services."""
        try:
            logger.info("Starting pump.fun services...")
            
            # Start main monitor
            if self.main_monitor:
                await self.main_monitor.start()
                self.state.main_monitor_running = True
                logger.info("Main monitor started")
                
            # Start WebSocket monitor
            if self.websocket_monitor:
                await self.websocket_monitor.start()
                self.state.websocket_monitor_running = True
                logger.info("WebSocket monitor started")
                
            # Start analyzer
            if self.analyzer:
                await self.analyzer.start()
                self.state.analyzer_running = True
                logger.info("Analyzer started")
                
            # Start executor
            if self.executor:
                # await self.executor.start()
                self.state.execution_running = True
                logger.info("Execution service started")
                
            logger.info("All services started successfully")
            
        except Exception as exc:
            logger.error(f"Error starting services: {exc}")
            raise
            
    async def _stop_services(self):
        """Stop all running services."""
        try:
            logger.info("Stopping pump.fun services...")
            
            # Stop main monitor
            if self.main_monitor and self.state.main_monitor_running:
                await self.main_monitor.stop()
                self.state.main_monitor_running = False
                
            # Stop WebSocket monitor
            if self.websocket_monitor and self.state.websocket_monitor_running:
                await self.websocket_monitor.stop()
                self.state.websocket_monitor_running = False
                
            # Stop analyzer
            if self.analyzer and self.state.analyzer_running:
                await self.analyzer.stop()
                self.state.analyzer_running = False
                
            # Stop executor
            if self.executor and self.state.execution_running:
                # await self.executor.stop()
                self.state.execution_running = False
                
            logger.info("All services stopped successfully")
            
        except Exception as exc:
            logger.error(f"Error stopping services: {exc}")
            
    def _setup_service_callbacks(self):
        """Set up callbacks for all services."""
        try:
            # Main monitor callbacks
            if self.main_monitor:
                self.main_monitor.add_launch_callback(self._handle_new_launch)
                self.main_monitor.add_alert_callback(self._handle_alert)
                
            # WebSocket monitor callbacks
            if self.websocket_monitor:
                self.websocket_monitor.add_message_callback(self._handle_websocket_message)
                self.websocket_monitor.add_connection_callback(self._handle_connection_change)
                
            # Analyzer callbacks (if direct integration)
            # if self.analyzer:
            #     # Set up analyzer callbacks
            #     pass
                
            logger.info("Service callbacks configured successfully")
            
        except Exception as exc:
            logger.error(f"Error setting up service callbacks: {exc}")
            
    async def _orchestrator_loop(self):
        """Main orchestrator loop."""
        try:
            logger.info("Starting pump.fun orchestrator loop...")
            
            while self.running:
                try:
                    # Process launch queue
                    await self._process_launch_queue()
                    
                    # Process analysis queue
                    await self._process_analysis_queue()
                    
                    # Process execution queue
                    await self._process_execution_queue()
                    
                    # Update orchestrator state
                    await self._update_orchestrator_state()
                    
                    # Wait for next cycle
                    await asyncio.sleep(self.orchestrator_config.monitor_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error(f"Error in orchestrator loop: {exc}")
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Error in orchestrator loop: {exc}")
            
    async def _process_launch_queue(self):
        """Process new launches from the queue."""
        try:
            while self.launch_queue and self.running:
                launch = self.launch_queue.popleft()
                
                # Add to analysis queue
                self.analysis_queue.append(launch)
                
                # Update state
                self.state.active_launches += 1
                
                # Notify callbacks
                for callback in self.launch_callbacks:
                    try:
                        callback(launch)
                    except Exception as exc:
                        logger.error(f"Error in launch callback: {exc}")
                        
        except Exception as exc:
            logger.error(f"Error processing launch queue: {exc}")
            
    async def _process_analysis_queue(self):
        """Process launches in the analysis queue."""
        try:
            while self.analysis_queue and self.running and self.analyzer:
                launch = self.analysis_queue.popleft()
                
                # Analyze the launch
                analysis = await self.analyzer.analyze_launch(launch)
                
                # Update state
                self.state.launches_analyzed += 1
                
                # Check if it's a high-probability launch
                if analysis.final_score >= self.orchestrator_config.min_score_for_execution:
                    self.state.high_probability_launches += 1
                    
                    # Add to execution queue
                    self.execution_queue.append(analysis)
                    
                # Notify callbacks
                for callback in self.analysis_callbacks:
                    try:
                        callback(analysis)
                    except Exception as exc:
                        logger.error(f"Error in analysis callback: {exc}")
                        
        except Exception as exc:
            logger.error(f"Error processing analysis queue: {exc}")
            
    async def _process_execution_queue(self):
        """Process launches in the execution queue."""
        try:
            while self.execution_queue and self.running:
                analysis = self.execution_queue.popleft()
                
                # Check execution criteria
                if self._should_execute(analysis):
                    # Attempt execution
                    await self._execute_launch(analysis)
                    
                    # Update state
                    self.state.executions_attempted += 1
                    
        except Exception as exc:
            logger.error(f"Error processing execution queue: {exc}")
            
    def _should_execute(self, analysis: LaunchAnalysis) -> bool:
        """Determine if a launch should be executed on."""
        try:
            # Check score threshold
            if analysis.final_score < self.orchestrator_config.min_score_for_execution:
                return False
                
            # Check risk threshold
            if analysis.rug_pull_risk > self.orchestrator_config.max_risk_for_execution:
                return False
                
            # Check liquidity threshold
            if analysis.launch.current_liquidity < self.orchestrator_config.min_liquidity_for_execution:
                return False
                
            # Check timing
            if analysis.timing_optimization < 0.6:
                return False
                
            return True
            
        except Exception as exc:
            logger.error(f"Error checking execution criteria: {exc}")
            return False
            
    async def _execute_launch(self, analysis: LaunchAnalysis):
        """Execute on a high-probability launch."""
        try:
            logger.info(f"Executing on launch: {analysis.launch.token_symbol} (Score: {analysis.final_score:.3f})")
            
            # This would integrate with your existing execution system
            # For now, just log the execution decision
            
            # Create execution decision
            decision = SniperDecision(
                token_mint=analysis.launch.token_mint,
                pool_address=analysis.launch.pool_address,
                action="BUY",
                confidence=analysis.final_score,
                risk_score=analysis.rug_pull_risk,
                timestamp=time.time(),
                details={
                    "analysis_score": analysis.final_score,
                    "pump_probability": analysis.pump_probability,
                    "timing_optimization": analysis.timing_optimization
                }
            )
            
            # Notify execution callbacks
            for callback in self.execution_callbacks:
                try:
                    callback(decision)
                except Exception as exc:
                    logger.error(f"Error in execution callback: {exc}")
                    
            logger.info(f"Execution decision made for {analysis.launch.token_symbol}")
            
        except Exception as exc:
            logger.error(f"Error executing launch: {exc}")
            
    async def _update_orchestrator_state(self):
        """Update the orchestrator state."""
        try:
            # Update launch counts
            if self.main_monitor:
                active_launches = self.main_monitor.get_active_launches()
                self.state.active_launches = len(active_launches)
                
                high_prob_launches = self.main_monitor.get_high_probability_launches()
                self.state.high_probability_launches = len(high_prob_launches)
                
            # Update analysis count
            if self.analyzer:
                analysis_stats = self.analyzer.get_analysis_stats()
                self.state.launches_analyzed = analysis_stats.get("total_analyses", 0)
                
            # Update timestamp
            self.state.last_update = time.time()
            
        except Exception as exc:
            logger.error(f"Error updating orchestrator state: {exc}")
            
    async def _performance_tracking_loop(self):
        """Track performance metrics."""
        try:
            while self.running:
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Wait for next update
                await asyncio.sleep(self.orchestrator_config.performance_update_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Error in performance tracking loop: {exc}")
            
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            if self.main_monitor:
                metrics = self.main_monitor.get_performance_metrics()
                
                self.state.successful_pumps = metrics.get("successful_pumps", 0)
                self.state.average_return = metrics.get("average_pump_return", 0.0)
                
            # Store performance history
            performance_record = {
                "timestamp": time.time(),
                "active_launches": self.state.active_launches,
                "high_probability_launches": self.state.high_probability_launches,
                "successful_pumps": self.state.successful_pumps,
                "average_return": self.state.average_return
            }
            
            self.performance_history.append(performance_record)
            
            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as exc:
            logger.error(f"Error updating performance metrics: {exc}")
            
    def _handle_new_launch(self, launch: PumpFunLaunch):
        """Handle new launch from monitor service."""
        try:
            # Add to launch queue
            self.launch_queue.append(launch)
            
            logger.debug(f"New launch queued: {launch.token_symbol}")
            
        except Exception as exc:
            logger.error(f"Error handling new launch: {exc}")
            
    def _handle_alert(self, alert_msg: str, alert_data: Dict):
        """Handle alert from monitor service."""
        try:
            # Notify alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_msg, alert_data)
                except Exception as exc:
                    logger.error(f"Error in alert callback: {exc}")
                    
        except Exception as exc:
            logger.error(f"Error handling alert: {exc}")
            
    def _handle_websocket_message(self, message):
        """Handle WebSocket message from monitor service."""
        try:
            # This would process WebSocket messages
            # For now, just log
            logger.debug(f"WebSocket message received: {message.message_type}")
            
        except Exception as exc:
            logger.error(f"Error handling WebSocket message: {exc}")
            
    def _handle_connection_change(self, connected: bool):
        """Handle WebSocket connection state change."""
        try:
            if connected:
                logger.info("WebSocket connection established")
            else:
                logger.warning("WebSocket connection lost")
                
        except Exception as exc:
            logger.error(f"Error handling connection change: {exc}")
            
    # Public interface methods
    def add_launch_callback(self, callback: Callable[[PumpFunLaunch], None]):
        """Add a callback for new launch notifications."""
        self.launch_callbacks.append(callback)
        
    def add_analysis_callback(self, callback: Callable[[LaunchAnalysis], None]):
        """Add a callback for analysis notifications."""
        self.analysis_callbacks.append(callback)
        
    def add_execution_callback(self, callback: Callable[[SniperDecision], None]):
        """Add a callback for execution notifications."""
        self.execution_callbacks.append(callback)
        
    def add_alert_callback(self, callback: Callable[[str, Dict], None]):
        """Add a callback for alert notifications."""
        self.alert_callbacks.append(callback)
        
    def get_orchestrator_state(self) -> OrchestratorState:
        """Get current orchestrator state."""
        return self.state
        
    def get_performance_history(self) -> List[Dict]:
        """Get performance history."""
        return self.performance_history.copy()
        
    def get_active_launches(self) -> List[PumpFunLaunch]:
        """Get list of active launches."""
        if self.main_monitor:
            return self.main_monitor.get_active_launches()
        return []
        
    def get_high_probability_launches(self) -> List[PumpFunLaunch]:
        """Get list of high-probability launches."""
        if self.main_monitor:
            return self.main_monitor.get_high_probability_launches()
        return []
        
    def is_healthy(self) -> bool:
        """Check if the orchestrator is healthy."""
        if not self.running:
            return False
            
        # Check service health
        if self.main_monitor and not self.state.main_monitor_running:
            return False
            
        if self.websocket_monitor and not self.state.websocket_monitor_running:
            return False
            
        if self.analyzer and not self.state.analyzer_running:
            return False
            
        return True


# Factory function
def create_pump_fun_orchestrator(config: Dict[str, Any]) -> PumpFunOrchestrator:
    """Create and configure a pump.fun orchestrator instance."""
    return PumpFunOrchestrator(config)
