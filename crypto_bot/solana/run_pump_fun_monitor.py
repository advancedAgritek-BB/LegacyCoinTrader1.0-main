#!/usr/bin/env python3
"""
Pump.fun Monitoring Service CLI

This script provides a command-line interface to run the pump.fun monitoring service
with various options for monitoring, analysis, and execution.
"""

import asyncio
import argparse
import logging
import signal
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Add the parent directory to the path to import crypto_bot modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from crypto_bot.solana.pump_fun_orchestrator import create_pump_fun_orchestrator
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__)


class PumpFunMonitorCLI:
    """Command-line interface for the pump.fun monitoring service."""
    
    def __init__(self):
        self.orchestrator = None
        self.running = False
        
    async def start(self, config_path: str, config_overrides: Dict[str, Any]):
        """Start the pump.fun monitoring service."""
        try:
            # Load configuration
            config = self._load_config(config_path, config_overrides)
            
            # Create orchestrator
            self.orchestrator = create_pump_fun_orchestrator(config)
            
            # Set up callbacks
            self._setup_callbacks()
            
            # Start the service
            await self.orchestrator.start()
            
            self.running = True
            logger.info("Pump.fun monitoring service started successfully")
            
            # Keep the service running
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as exc:
            logger.error(f"Error starting service: {exc}")
            raise
        finally:
            await self._cleanup()
            
    async def stop(self):
        """Stop the pump.fun monitoring service."""
        if self.orchestrator and self.running:
            try:
                await self.orchestrator.stop()
                logger.info("Pump.fun monitoring service stopped")
            except Exception as exc:
                logger.error(f"Error stopping service: {exc}")
                
        self.running = False
        
    async def _cleanup(self):
        """Clean up resources."""
        try:
            await self.stop()
        except Exception as exc:
            logger.error(f"Error during cleanup: {exc}")
            
    def _load_config(self, config_path: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from file with optional overrides."""
        try:
            # Load base config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Apply overrides
            if overrides:
                self._merge_config(config, overrides)
                
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except Exception as exc:
            logger.error(f"Error loading configuration: {exc}")
            raise
            
    def _merge_config(self, config: Dict[str, Any], overrides: Dict[str, Any]):
        """Merge configuration overrides into base config."""
        for key, value in overrides.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                self._merge_config(config[key], value)
            else:
                config[key] = value
                
    def _setup_callbacks(self):
        """Set up callbacks for the orchestrator."""
        if not self.orchestrator:
            return
            
        # Launch callback
        self.orchestrator.add_launch_callback(self._on_new_launch)
        
        # Analysis callback
        self.orchestrator.add_analysis_callback(self._on_analysis_complete)
        
        # Execution callback
        self.orchestrator.add_execution_callback(self._on_execution_decision)
        
        # Alert callback
        self.orchestrator.add_alert_callback(self._on_alert)
        
    def _on_new_launch(self, launch):
        """Handle new launch notification."""
        logger.info(f"🚀 NEW LAUNCH: {launch.token_symbol} ({launch.token_mint})")
        logger.info(f"   Pool: {launch.pool_address}")
        logger.info(f"   Initial Liquidity: ${launch.initial_liquidity:,.2f}")
        logger.info(f"   Initial Price: ${launch.initial_price:.8f}")
        logger.info(f"   Creator: {launch.creator_wallet}")
        
    def _on_analysis_complete(self, analysis):
        """Handle analysis completion notification."""
        launch = analysis.launch
        logger.info(f"📊 ANALYSIS: {launch.token_symbol}")
        logger.info(f"   Final Score: {analysis.final_score:.3f}")
        logger.info(f"   Pump Probability: {analysis.pump_probability:.3f}")
        logger.info(f"   Risk Score: {analysis.rug_pull_risk:.3f}")
        logger.info(f"   Timing Score: {analysis.timing_optimization:.3f}")
        
        if analysis.final_score >= 0.8:
            logger.info(f"   ⭐ HIGH PROBABILITY LAUNCH DETECTED!")
            
    def _on_execution_decision(self, decision):
        """Handle execution decision notification."""
        logger.info(f"⚡ EXECUTION: {decision.token_mint}")
        logger.info(f"   Action: {decision.action}")
        logger.info(f"   Confidence: {decision.confidence:.3f}")
        logger.info(f"   Risk Score: {decision.risk_score:.3f}")
        
    def _on_alert(self, alert_msg: str, alert_data: Dict):
        """Handle alert notification."""
        logger.warning(f"🚨 ALERT: {alert_msg}")
        if alert_data:
            logger.warning(f"   Details: {alert_data}")


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Pump.fun Liquidity Pool Launch Monitoring Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python run_pump_fun_monitor.py
  
  # Run with custom config file
  python run_pump_fun_monitor.py -c config/custom_pump_fun_config.yaml
  
  # Run with specific overrides
  python run_pump_fun_monitor.py --override pump_fun_orchestrator.enable_execution false
  
  # Run in test mode
  python run_pump_fun_monitor.py --override development.test_mode true
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        default="config/pump_fun_config.yaml",
        help="Path to configuration file (default: config/pump_fun_config.yaml)"
    )
    
    parser.add_argument(
        "--override",
        action="append",
        help="Override configuration values (format: key.subkey=value)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without starting the service"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Parse overrides
    config_overrides = {}
    if args.override:
        for override in args.override:
            try:
                key, value = override.split("=", 1)
                keys = key.split(".")
                
                # Convert value to appropriate type
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "").isdigit():
                    value = float(value)
                    
                # Build nested dictionary
                current = config_overrides
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                
            except ValueError:
                logger.error(f"Invalid override format: {override}")
                sys.exit(1)
                
    # Check if config file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.error("Please create a configuration file or specify a valid path with -c")
        sys.exit(1)
        
    # Validate configuration
    try:
        with open(args.config, 'r') as f:
            yaml.safe_load(f)
        logger.info("Configuration file is valid")
    except Exception as exc:
        logger.error(f"Invalid configuration file: {exc}")
        sys.exit(1)
        
    if args.dry_run:
        logger.info("Configuration validation completed successfully")
        sys.exit(0)
        
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run CLI
    cli = PumpFunMonitorCLI()
    
    try:
        # Run the service
        asyncio.run(cli.start(args.config, config_overrides))
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as exc:
        logger.error(f"Service failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
