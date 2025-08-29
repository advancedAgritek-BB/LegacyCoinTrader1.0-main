# 🚀 Pump.fun Liquidity Pool Launch Monitoring Service

A comprehensive, intelligent monitoring service for detecting and analyzing new token launches on pump.fun with high probability of significant pumps. This service combines real-time monitoring, multi-factor analysis, machine learning predictions, and automated execution capabilities.

## ✨ Features

### 🔍 **Intelligent Launch Detection**
- **Real-time monitoring** of pump.fun liquidity pool creations
- **WebSocket integration** for ultra-low latency detection
- **Multi-source data aggregation** from on-chain and social sources
- **Automatic filtering** based on configurable criteria

### 🧠 **Advanced Analysis Engine**
- **Multi-factor scoring** combining technical, social, and market data
- **Machine learning predictions** for pump probability
- **Risk assessment** including rug pull detection
- **Timing optimization** for optimal entry points

### ⚡ **Execution Integration**
- **Automated execution** on high-probability launches
- **Risk management** with configurable thresholds
- **Integration** with existing pump sniper infrastructure
- **Performance tracking** and optimization

### 📊 **Comprehensive Monitoring**
- **Real-time alerts** via Telegram, Discord, and email
- **Performance metrics** and historical analysis
- **Dashboard interface** for monitoring and control
- **Extensive logging** and debugging capabilities

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Pump.fun Orchestrator                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Monitor   │  │  WebSocket  │  │  Analyzer   │        │
│  │   Service   │  │   Monitor   │  │   Engine    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Social    │  │  Momentum   │  │    Pool     │        │
│  │ Sentiment   │  │  Detector   │  │  Analyzer   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Risk      │  │ Execution   │  │ Performance │        │
│  │  Manager    │  │   Engine    │  │   Tracker   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone the repository
git clone <your-repo-url>
cd LegacyCoinTrader1.0-main

# Install dependencies
pip install -r requirements.txt

# Install additional ML dependencies
pip install scikit-learn joblib websockets
```

### 2. **Configuration**

```bash
# Copy and edit the configuration file
cp config/pump_fun_config.yaml config/my_pump_fun_config.yaml

# Edit the configuration with your API keys and preferences
nano config/my_pump_fun_config.yaml
```

**Key configuration sections:**
- **API Keys**: Add your pump.fun API key
- **Thresholds**: Adjust scoring and risk thresholds
- **Notifications**: Configure Telegram/Discord webhooks
- **Execution**: Set risk management parameters

### 3. **Running the Service**

```bash
# Basic run
python crypto_bot/solana/run_pump_fun_monitor.py

# With custom config
python crypto_bot/solana/run_pump_fun_monitor.py -c config/my_pump_fun_config.yaml

# With overrides
python crypto_bot/solana/run_pump_fun_monitor.py --override pump_fun_orchestrator.enable_execution false

# Test mode (validate config only)
python crypto_bot/solana/run_pump_fun_monitor.py --dry-run
```

## 📋 Configuration Guide

### **Main Service Configuration**

```yaml
pump_fun_orchestrator:
  # Enable/disable services
  enable_main_monitor: true
  enable_websocket_monitor: true
  enable_analyzer: true
  enable_execution: true
  
  # Execution thresholds
  min_score_for_execution: 0.8
  max_risk_for_execution: 0.3
  min_liquidity_for_execution: 10000.0
```

### **Launch Filtering**

```yaml
pump_fun_monitor:
  launch_filter:
    min_initial_liquidity: 1000.0      # Minimum liquidity in USD
    max_initial_liquidity: 1000000.0   # Maximum liquidity in USD
    min_initial_price: 0.000001        # Minimum token price
    max_initial_price: 1.0             # Maximum token price
    require_verified_creator: false    # Require verified creator
    min_creator_balance: 100.0         # Minimum creator SOL balance
```

### **Analysis Weights**

```yaml
pump_fun_analyzer:
  feature_weights:
    technical: 0.25        # Technical analysis weight
    social: 0.20           # Social sentiment weight
    microstructure: 0.25   # Market microstructure weight
    risk: 0.30             # Risk assessment weight
```

### **Risk Management**

```yaml
risk_manager:
  max_position_size: 0.1           # Max 10% of portfolio per trade
  max_risk_per_trade: 0.02        # Max 2% risk per trade
  enable_stop_loss: true
  default_stop_loss: 0.15          # 15% stop loss
  enable_take_profit: true
  default_take_profit: 0.5         # 50% take profit
```

## 🔧 Advanced Usage

### **Custom Analysis Callbacks**

```python
from crypto_bot.solana.pump_fun_orchestrator import create_pump_fun_orchestrator

# Create orchestrator
orchestrator = create_pump_fun_orchestrator(config)

# Add custom callbacks
def on_high_probability_launch(analysis):
    if analysis.final_score >= 0.9:
        print(f"🚀 ULTRA HIGH PROBABILITY: {analysis.launch.token_symbol}")
        # Your custom logic here

orchestrator.add_analysis_callback(on_high_probability_launch)

# Start the service
await orchestrator.start()
```

### **Integration with Existing Systems**

```python
# Integrate with your existing pump sniper
from crypto_bot.solana.pump_sniper_orchestrator import PumpSniperOrchestrator

class CustomPumpFunOrchestrator(PumpFunOrchestrator):
    def __init__(self, config):
        super().__init__(config)
        self.pump_sniper = PumpSniperOrchestrator(config)
    
    async def _execute_launch(self, analysis):
        # Use your existing execution logic
        decision = await self.pump_sniper.analyze_launch(analysis.launch)
        if decision.should_execute:
            await self.pump_sniper.execute_decision(decision)
```

### **Custom Scoring Algorithms**

```python
# Extend the analyzer with custom scoring
class CustomPumpFunAnalyzer(PumpFunAnalyzer):
    async def _calculate_custom_score(self, analysis):
        # Your custom scoring logic
        custom_score = (
            analysis.social_buzz * 0.4 +
            analysis.whale_activity * 0.3 +
            (1 - analysis.rug_pull_risk) * 0.3
        )
        return custom_score
    
    async def _calculate_composite_score(self, analysis):
        # Override default scoring
        analysis.final_score = await self._calculate_custom_score(analysis)
        return analysis.final_score
```

## 📊 Performance Monitoring

### **Real-time Metrics**

The service provides comprehensive performance tracking:

- **Launch Detection Rate**: New launches detected per hour
- **Analysis Accuracy**: Success rate of pump predictions
- **Execution Performance**: Win rate and average returns
- **Risk Metrics**: Rug pull detection rate

### **Performance Dashboard**

```bash
# Enable dashboard in config
performance:
  enable_dashboard: true
  dashboard_port: 8080

# Access dashboard at http://localhost:8080
```

### **Performance Alerts**

```yaml
performance:
  enable_performance_alerts: true
  min_success_rate: 0.6        # Alert if success rate drops below 60%
  max_drawdown: 0.2            # Alert if drawdown exceeds 20%
```

## 🛡️ Risk Management

### **Multi-Layer Risk Assessment**

1. **Liquidity Risk**: Minimum liquidity thresholds
2. **Creator Risk**: Wallet balance and reputation checks
3. **Contract Risk**: Smart contract security analysis
4. **Market Risk**: Volatility and price stability
5. **Social Risk**: Community sentiment and growth

### **Automated Risk Controls**

- **Position Sizing**: Automatic position size calculation
- **Stop Losses**: Dynamic stop loss management
- **Take Profits**: Partial profit taking at multiple levels
- **Portfolio Limits**: Maximum exposure per token/strategy

## 🔌 API Integration

### **Pump.fun API**

```python
# Direct API access
import aiohttp

async def get_recent_pools():
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        async with session.get(
            "https://api.pump.fun/v1/pools/recent",
            headers=headers
        ) as response:
            return await response.json()
```

### **WebSocket Streams**

```python
# Subscribe to real-time updates
subscribe_message = {
    "action": "subscribe",
    "channels": [
        "pool_creations",
        "pool_updates",
        "token_launches",
        "liquidity_events"
    ]
}
```

## 🧪 Testing and Development

### **Test Mode**

```bash
# Run in test mode with mock data
python run_pump_fun_monitor.py --override development.test_mode true

# Validate configuration
python run_pump_fun_monitor.py --dry-run
```

### **Mock Data**

```yaml
development:
  enable_mock_data: true
  mock_data_file: "test_data/mock_launches.json"
```

### **Debug Mode**

```yaml
development:
  debug_mode: true
  verbose_logging: true
  enable_profiling: true
```

## 📈 Optimization and Tuning

### **Scoring Algorithm Tuning**

1. **Feature Weights**: Adjust based on historical performance
2. **Thresholds**: Optimize based on market conditions
3. **Time Windows**: Fine-tune optimal entry timing
4. **Risk Parameters**: Balance risk vs. reward

### **Performance Optimization**

- **Caching**: Implement Redis for high-frequency data
- **Database**: Use PostgreSQL for historical analysis
- **Scaling**: Deploy multiple instances for redundancy
- **Monitoring**: Implement health checks and alerts

## 🚨 Troubleshooting

### **Common Issues**

1. **API Rate Limits**: Adjust monitoring intervals
2. **WebSocket Disconnections**: Check network stability
3. **High Memory Usage**: Reduce cache sizes
4. **False Positives**: Adjust scoring thresholds

### **Debug Commands**

```bash
# Check service health
curl http://localhost:8080/health

# View logs
tail -f logs/pump_fun.log

# Monitor performance
python -m crypto_bot.solana.pump_fun_orchestrator --debug
```

### **Performance Tuning**

```yaml
# Optimize for speed
pump_fun_orchestrator:
  monitor_interval: 15.0      # Faster monitoring
  analysis_interval: 30.0     # Faster analysis
  execution_interval: 5.0     # Faster execution

# Optimize for accuracy
pump_fun_analyzer:
  min_confidence: 0.8         # Higher confidence threshold
  prediction_confidence_threshold: 0.85
```

## 🔮 Future Enhancements

### **Planned Features**

- **Multi-chain Support**: Extend to other blockchains
- **Advanced ML Models**: Deep learning and neural networks
- **Social Trading**: Copy successful traders
- **Portfolio Optimization**: Advanced portfolio management
- **Mobile App**: iOS and Android applications

### **Integration Opportunities**

- **DEX Aggregators**: 1inch, Paraswap integration
- **Cross-chain Bridges**: Multi-chain token launches
- **DeFi Protocols**: Yield farming and staking
- **NFT Markets**: NFT launch monitoring

## 📚 Additional Resources

### **Documentation**

- [Pump.fun API Documentation](https://docs.pump.fun)
- [Solana Development Guide](https://docs.solana.com)
- [Machine Learning Integration](docs/ml_integration.md)

### **Community**

- [Discord Server](https://discord.gg/your-server)
- [Telegram Channel](https://t.me/your-channel)
- [GitHub Issues](https://github.com/your-repo/issues)

### **Support**

For technical support and questions:
- Create a GitHub issue
- Join our Discord community
- Contact the development team

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Cryptocurrency trading involves substantial risk. Always conduct your own research and never invest more than you can afford to lose.

**📄 License**: This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
