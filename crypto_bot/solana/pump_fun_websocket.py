"""
Pump.fun WebSocket Real-time Monitoring Service

This service provides ultra-low latency monitoring of pump.fun liquidity pool
creations using WebSocket connections for real-time launch detection.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import websockets
import aiohttp

from .pump_fun_monitor import PumpFunLaunch, PumpFunMonitor
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class WebSocketMessage:
    """Represents a WebSocket message from pump.fun."""
    
    message_type: str
    timestamp: float
    data: Dict[str, Any]
    raw_message: str


class PumpFunWebSocketMonitor:
    """
    WebSocket-based real-time monitor for pump.fun launches.
    
    Features:
    - Ultra-low latency pool creation detection
    - Automatic reconnection and error handling
    - Message filtering and processing
    - Integration with main monitor service
    """
    
    def __init__(self, config: Dict[str, Any], main_monitor: PumpFunMonitor):
        self.config = config
        self.ws_config = config.get("pump_fun_websocket", {})
        self.main_monitor = main_monitor
        
        # WebSocket configuration
        self.websocket_url = self.ws_config.get("websocket_url", "wss://api.pump.fun/v1/ws")
        self.api_key = self._get_env_or_config("PUMPFUN_API_KEY", "api_key")
        self.reconnect_interval = self.ws_config.get("reconnect_interval", 5)
        self.max_reconnect_attempts = self.ws_config.get("max_reconnect_attempts", 10)
        self.heartbeat_interval = self.ws_config.get("heartbeat_interval", 30)
        
        # Connection state
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.monitoring = False
        self.reconnect_attempts = 0
        
        # Message processing
        self.message_handlers: Dict[str, Callable] = {
            "pool_created": self._handle_pool_created,
            "pool_updated": self._handle_pool_updated,
            "token_launched": self._handle_token_launched,
            "liquidity_added": self._handle_liquidity_added,
            "price_update": self._handle_price_update,
            "volume_update": self._handle_volume_update
        }
        
        # Statistics
        self.stats = {
            "messages_received": 0,
            "pools_detected": 0,
            "launches_detected": 0,
            "connection_errors": 0,
            "reconnections": 0,
            "last_message_time": 0
        }
        
        # Callbacks
        self.message_callbacks: List[Callable[[WebSocketMessage], None]] = []
        self.connection_callbacks: List[Callable[[bool], None]] = []
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
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
        return self.ws_config.get(config_key)
        
    async def start(self):
        """Start the WebSocket monitoring service."""
        if self.monitoring:
            logger.warning("WebSocket monitor already running")
            return
            
        try:
            logger.info("Starting pump.fun WebSocket monitoring service...")
            
            self.monitoring = True
            
            # Start monitoring task
            self.monitor_task = asyncio.create_task(self._monitor_websocket())
            
            # Start heartbeat task
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info("WebSocket monitor started successfully")
            
        except Exception as exc:
            logger.error(f"Failed to start WebSocket monitor: {exc}")
            self.monitoring = False
            raise
            
    async def stop(self):
        """Stop the WebSocket monitoring service."""
        if not self.monitoring:
            return
            
        try:
            logger.info("Stopping pump.fun WebSocket monitoring service...")
            
            self.monitoring = False
            
            # Cancel background tasks
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
                    
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
                    
            # Close WebSocket connection
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
                
            self.connected = False
            
            logger.info("WebSocket monitor stopped successfully")
            
        except Exception as exc:
            logger.error(f"Error stopping WebSocket monitor: {exc}")
            
    async def _monitor_websocket(self):
        """Main WebSocket monitoring loop."""
        logger.info("Starting WebSocket monitoring loop...")
        
        while self.monitoring:
            try:
                # Connect to WebSocket
                await self._connect_websocket()
                
                # Monitor for messages
                await self._monitor_messages()
                
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in WebSocket monitoring loop: {exc}")
                self.stats["connection_errors"] += 1
                
                # Wait before reconnecting
                await asyncio.sleep(self.reconnect_interval)
                
    async def _connect_websocket(self):
        """Establish WebSocket connection to pump.fun."""
        try:
            logger.info(f"Connecting to pump.fun WebSocket: {self.websocket_url}")
            
            # Prepare connection headers
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.websocket_url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.connected = True
            self.reconnect_attempts = 0
            
            # Send subscription message
            await self._subscribe_to_channels()
            
            # Notify connection callbacks
            for callback in self.connection_callbacks:
                try:
                    callback(True)
                except Exception as exc:
                    logger.error(f"Error in connection callback: {exc}")
                    
            logger.info("Successfully connected to pump.fun WebSocket")
            
        except Exception as exc:
            logger.error(f"Failed to connect to WebSocket: {exc}")
            self.connected = False
            raise
            
    async def _subscribe_to_channels(self):
        """Subscribe to relevant WebSocket channels."""
        try:
            if not self.websocket:
                return
                
            # Subscribe to pool creation events
            subscribe_message = {
                "action": "subscribe",
                "channels": [
                    "pool_creations",
                    "pool_updates", 
                    "token_launches",
                    "liquidity_events",
                    "price_feeds"
                ]
            }
            
            if self.api_key:
                subscribe_message["api_key"] = self.api_key
                
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info("Subscribed to pump.fun WebSocket channels")
            
        except Exception as exc:
            logger.error(f"Error subscribing to channels: {exc}")
            
    async def _monitor_messages(self):
        """Monitor WebSocket for incoming messages."""
        try:
            if not self.websocket:
                return
                
            async for message in self.websocket:
                if not self.monitoring:
                    break
                    
                try:
                    # Parse message
                    parsed_message = await self._parse_message(message)
                    if parsed_message:
                        await self._process_message(parsed_message)
                        
                except Exception as exc:
                    logger.error(f"Error processing WebSocket message: {exc}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.connected = False
        except Exception as exc:
            logger.error(f"Error monitoring WebSocket messages: {exc}")
            self.connected = False
            
    async def _parse_message(self, raw_message: str) -> Optional[WebSocketMessage]:
        """Parse raw WebSocket message."""
        try:
            # Parse JSON
            data = json.loads(raw_message)
            
            # Extract message type
            message_type = data.get("type", "unknown")
            timestamp = data.get("timestamp", time.time())
            
            # Create message object
            message = WebSocketMessage(
                message_type=message_type,
                timestamp=timestamp,
                data=data,
                raw_message=raw_message
            )
            
            # Update statistics
            self.stats["messages_received"] += 1
            self.stats["last_message_time"] = time.time()
            
            return message
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse WebSocket message: {raw_message}")
            return None
        except Exception as exc:
            logger.error(f"Error parsing WebSocket message: {exc}")
            return None
            
    async def _process_message(self, message: WebSocketMessage):
        """Process a parsed WebSocket message."""
        try:
            # Notify message callbacks
            for callback in self.message_callbacks:
                try:
                    callback(message)
                except Exception as exc:
                    logger.error(f"Error in message callback: {exc}")
                    
            # Handle message based on type
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                logger.debug(f"No handler for message type: {message.message_type}")
                
        except Exception as exc:
            logger.error(f"Error processing message: {exc}")
            
    async def _handle_pool_created(self, message: WebSocketMessage):
        """Handle pool creation event."""
        try:
            pool_data = message.data.get("pool", {})
            
            logger.info(f"Pool created detected: {pool_data.get('pool_address')}")
            
            # Create launch object
            launch = PumpFunLaunch(
                pool_address=pool_data.get("pool_address", ""),
                token_mint=pool_data.get("token_mint", ""),
                token_symbol=pool_data.get("token_symbol", "UNKNOWN"),
                token_name=pool_data.get("token_name", "Unknown Token"),
                launch_time=message.timestamp,
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
            
            # Add to main monitor
            self.main_monitor.active_launches[launch.pool_address] = launch
            self.main_monitor.launch_history.append(launch)
            
            # Update statistics
            self.stats["pools_detected"] += 1
            self.stats["launches_detected"] += 1
            
            logger.info(f"New launch processed via WebSocket: {launch.token_symbol}")
            
        except Exception as exc:
            logger.error(f"Error handling pool created event: {exc}")
            
    async def _handle_pool_updated(self, message: WebSocketMessage):
        """Handle pool update event."""
        try:
            pool_data = message.data.get("pool", {})
            pool_address = pool_data.get("pool_address")
            
            if pool_address in self.main_monitor.active_launches:
                launch = self.main_monitor.active_launches[pool_address]
                
                # Update launch data
                launch.current_price = pool_data.get("current_price", launch.current_price)
                launch.current_liquidity = pool_data.get("current_liquidity", launch.current_liquidity)
                launch.volume_24h = pool_data.get("volume_24h", launch.volume_24h)
                launch.last_updated = time.time()
                
                # Calculate price change
                if launch.initial_price > 0:
                    launch.price_change_24h = (
                        (launch.current_price - launch.initial_price) / launch.initial_price * 100
                    )
                    
        except Exception as exc:
            logger.error(f"Error handling pool update event: {exc}")
            
    async def _handle_token_launched(self, message: WebSocketMessage):
        """Handle token launch event."""
        try:
            token_data = message.data.get("token", {})
            
            logger.info(f"Token launch detected: {token_data.get('symbol')}")
            
            # This event might provide additional context for existing pools
            # or indicate a new launch mechanism
            
        except Exception as exc:
            logger.error(f"Error handling token launch event: {exc}")
            
    async def _handle_liquidity_added(self, message: WebSocketMessage):
        """Handle liquidity addition event."""
        try:
            liquidity_data = message.data.get("liquidity", {})
            pool_address = liquidity_data.get("pool_address")
            
            if pool_address in self.main_monitor.active_launches:
                launch = self.main_monitor.active_launches[pool_address]
                
                # Update liquidity
                new_liquidity = liquidity_data.get("amount", 0.0)
                launch.current_liquidity += new_liquidity
                launch.last_updated = time.time()
                
                logger.debug(f"Liquidity updated for {launch.token_symbol}: +{new_liquidity}")
                
        except Exception as exc:
            logger.error(f"Error handling liquidity added event: {exc}")
            
    async def _handle_price_update(self, message: WebSocketMessage):
        """Handle price update event."""
        try:
            price_data = message.data.get("price", {})
            pool_address = price_data.get("pool_address")
            
            if pool_address in self.main_monitor.active_launches:
                launch = self.main_monitor.active_launches[pool_address]
                
                # Update price
                new_price = price_data.get("price", launch.current_price)
                launch.current_price = new_price
                launch.last_updated = time.time()
                
                # Calculate price change
                if launch.initial_price > 0:
                    launch.price_change_24h = (
                        (new_price - launch.initial_price) / launch.initial_price * 100
                    )
                    
        except Exception as exc:
            logger.error(f"Error handling price update event: {exc}")
            
    async def _handle_volume_update(self, message: WebSocketMessage):
        """Handle volume update event."""
        try:
            volume_data = message.data.get("volume", {})
            pool_address = volume_data.get("pool_address")
            
            if pool_address in self.main_monitor.active_launches:
                launch = self.main_monitor.active_launches[pool_address]
                
                # Update volume
                launch.volume_24h = volume_data.get("volume_24h", launch.volume_24h)
                launch.last_updated = time.time()
                
        except Exception as exc:
            logger.error(f"Error handling volume update event: {exc}")
            
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages to keep connection alive."""
        try:
            while self.monitoring:
                if self.connected and self.websocket:
                    try:
                        # Send heartbeat
                        heartbeat_message = {
                            "action": "ping",
                            "timestamp": time.time()
                        }
                        
                        await self.websocket.send(json.dumps(heartbeat_message))
                        logger.debug("Heartbeat sent")
                        
                    except Exception as exc:
                        logger.warning(f"Failed to send heartbeat: {exc}")
                        self.connected = False
                        
                # Wait for next heartbeat
                await asyncio.sleep(self.heartbeat_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Error in heartbeat loop: {exc}")
            
    async def _reconnect(self):
        """Attempt to reconnect to WebSocket."""
        try:
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.error("Max reconnection attempts reached")
                return False
                
            logger.info(f"Attempting to reconnect (attempt {self.reconnect_attempts + 1})")
            
            # Close existing connection
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
                
            self.connected = False
            
            # Wait before reconnecting
            await asyncio.sleep(self.reconnect_interval * (self.reconnect_attempts + 1))
            
            # Attempt reconnection
            await self._connect_websocket()
            
            self.reconnect_attempts += 1
            self.stats["reconnections"] += 1
            
            return True
            
        except Exception as exc:
            logger.error(f"Reconnection failed: {exc}")
            self.reconnect_attempts += 1
            return False
            
    def add_message_callback(self, callback: Callable[[WebSocketMessage], None]):
        """Add a callback for WebSocket message notifications."""
        self.message_callbacks.append(callback)
        
    def add_connection_callback(self, callback: Callable[[bool], None]):
        """Add a callback for connection state changes."""
        self.connection_callbacks.append(callback)
        
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and statistics."""
        return {
            "connected": self.connected,
            "monitoring": self.monitoring,
            "reconnect_attempts": self.reconnect_attempts,
            "statistics": self.stats.copy()
        }
        
    def is_healthy(self) -> bool:
        """Check if the WebSocket monitor is healthy."""
        if not self.monitoring:
            return False
            
        # Check if we've received messages recently
        if time.time() - self.stats["last_message_time"] > 300:  # 5 minutes
            return False
            
        return self.connected


# Factory function
def create_pump_fun_websocket_monitor(config: Dict[str, Any], main_monitor: PumpFunMonitor) -> PumpFunWebSocketMonitor:
    """Create and configure a pump.fun WebSocket monitor instance."""
    return PumpFunWebSocketMonitor(config, main_monitor)
