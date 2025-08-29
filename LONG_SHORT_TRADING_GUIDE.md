# 🚀 Long and Short Trading Guide for LegacyCoinTrader

This guide covers how to use both long and short trading functionality in both live and paper trading modes on Kraken.

## 📋 Prerequisites

1. **Kraken Account**: Ensure you have a Kraken account with API access
2. **API Keys**: Set up your Kraken API keys in the `.env` file
3. **Margin Trading**: Enable margin trading on your Kraken account for short positions
4. **Sufficient Balance**: Ensure you have enough USDT/USD for margin requirements

## ⚙️ Configuration

### 1. Enable Short Trading

In your `crypto_bot/config.yaml`:

```yaml
# Enable short selling
allow_short: true

# Risk management for both long and short positions
risk:
  stop_loss_pct: 0.015  # 1.5% stop loss for both
  take_profit_pct: 0.07  # 7% take profit for both
  trailing_stop_pct: 0.02  # 2% trailing stop for both
  # Short-specific parameters
  short_margin_requirement: 1.5  # 150% margin requirement
  max_short_exposure: 0.3  # Maximum 30% in short positions

# Exit strategy
exit_strategy:
  min_gain_to_trail: 0.01  # Start trailing after 1% gain
  trailing_stop_pct: 0.02  # 2% trailing stop
  take_profit_pct: 0.05  # 5% take profit
```

### 2. Environment Variables

In your `crypto_bot/.env` file:

```bash
# Kraken API Configuration
KRAKEN_API_KEY=your_api_key_here
KRAKEN_API_SECRET=your_api_secret_here
KRAKEN_WS_TOKEN=your_websocket_token_here

# Trading Mode
EXECUTION_MODE=dry_run  # Use 'live' for real trading
MODE=cex

# Enable short trading
ALLOW_SHORT=true
```

## 🔄 Trading Modes

### Paper Trading Mode (`EXECUTION_MODE=dry_run`)

- **No real money**: Perfect for testing strategies
- **Full simulation**: Includes margin requirements and PnL tracking
- **Risk-free learning**: Test both long and short strategies

### Live Trading Mode (`EXECUTION_MODE=live`)

- **Real money**: Actual trades on Kraken
- **Margin requirements**: Must meet Kraken's margin requirements
- **Risk management**: Use proper position sizing and stop losses

## 📊 Position Types

### Long Positions (Buy)

```python
# Strategy generates "long" signal
# Bot executes "buy" order
# Profit when price goes up
# Loss when price goes down
```

**Example:**
- Entry: Buy BTC at $50,000
- Exit: Sell BTC at $52,500
- Profit: $2,500 (5% gain)

### Short Positions (Sell)

```python
# Strategy generates "short" signal  
# Bot executes "sell" order
# Profit when price goes down
# Loss when price goes up
```

**Example:**
- Entry: Sell ETH at $3,000
- Exit: Buy ETH at $2,850
- Profit: $150 (5% gain)

## 🎯 Strategy Signals

### Signal Generation

The bot analyzes market conditions and generates signals:

- **"long"** → Executes buy order
- **"short"** → Executes sell order  
- **"none"** → No action

### Signal Sources

1. **Technical Analysis**: RSI, MACD, Bollinger Bands
2. **Regime Classification**: Trending, sideways, volatile markets
3. **Pattern Recognition**: Support/resistance, breakouts
4. **Sentiment Analysis**: Market mood indicators

## 💰 Risk Management

### Position Sizing

```yaml
risk:
  trade_size_pct: 0.1  # 10% of balance per trade
  max_open_trades: 5   # Maximum concurrent positions
```

### Stop Losses

- **Long positions**: Exit if price falls below stop level
- **Short positions**: Exit if price rises above stop level
- **Hard stop**: Always enforced (1.5% default)
- **Trailing stop**: Dynamic stop that moves with profit

### Take Profits

- **Long positions**: Exit at 7% profit target
- **Short positions**: Exit at 7% profit target
- **Partial exits**: Scale out at 25%, 50%, 75%, 100% levels

## 🔧 Running the Bot

### 1. Start Paper Trading

```bash
cd crypto_bot
python main.py
```

When prompted, enter your paper trading balance (e.g., $10,000).

### 2. Start Live Trading

```bash
cd crypto_bot
# Ensure EXECUTION_MODE=live in .env
python main.py
```

### 3. Monitor Positions

The bot provides real-time updates:

- **Telegram notifications**: Trade entries, exits, PnL updates
- **Console output**: Position status, risk metrics
- **Log files**: Detailed trading history

## 📈 Example Trading Session

### Session 1: Paper Trading

```
🤖 CoinTrader2.0 started
Enter paper trading balance in USDT: 10000

📄 Paper CEX Trade Opened
BUY 0.0200 BTC/USDT
Price: $50000.00
Balance: $9000.00
Strategy: trend_bot

📄 Paper CEX Trade Opened  
SELL 3.3333 ETH/USDT
Price: $3000.00
Balance: $0.00
Strategy: breakout_bot

📄 Paper Trade Closed 💰
BUY 0.0200 BTC/USDT
Entry: $50000.00
Exit: $52500.00
PnL: $500.00
Balance: $9500.00
```

### Session 2: Live Trading

```
🤖 CoinTrader2.0 started
💰 Live Balance: $5000.00

💰 Live Trade Opened
BUY 0.0100 BTC/USDT
Price: $50000.00
Balance: $4500.00

💰 Live Trade Opened
SELL 1.6667 ETH/USDT  
Price: $3000.00
Balance: $0.00
```

## ⚠️ Important Considerations

### Short Trading Risks

1. **Unlimited Loss Potential**: Short positions can lose more than invested
2. **Margin Calls**: Insufficient margin can force position closure
3. **Squeeze Risk**: Short squeezes can cause rapid price increases
4. **Borrowing Costs**: Short positions may incur borrowing fees

### Risk Mitigation

1. **Position Sizing**: Never risk more than 1-2% per trade
2. **Stop Losses**: Always use stop losses for short positions
3. **Diversification**: Don't concentrate risk in one direction
4. **Monitoring**: Regularly check margin requirements

## 🧪 Testing Your Setup

Run the comprehensive test script:

```bash
python test_long_short_trading.py
```

This will test:
- ✅ Paper wallet long/short functionality
- ✅ Configuration validation
- ✅ Risk manager integration
- ✅ Kraken API connectivity

## 📞 Support

### Common Issues

1. **"Short selling disabled"**: Check `allow_short: true` in config
2. **"Insufficient margin"**: Increase balance or reduce position size
3. **"API authentication failed"**: Verify Kraken API keys
4. **"Position limit reached"**: Close existing positions first

### Getting Help

- Check log files in `crypto_bot/logs/`
- Review configuration in `crypto_bot/config.yaml`
- Test with paper trading first
- Use smaller position sizes initially

## 🎯 Best Practices

1. **Start Small**: Begin with paper trading and small live positions
2. **Test Strategies**: Validate strategies before committing capital
3. **Monitor Risk**: Regularly check portfolio exposure and PnL
4. **Stay Informed**: Keep up with market conditions and news
5. **Document Trades**: Track performance for strategy improvement

## 🚀 Advanced Features

### Custom Strategies

Create custom strategies that generate long/short signals:

```python
def custom_strategy(df, config):
    # Your analysis logic here
    if bullish_condition:
        return 0.8, "long", atr_value
    elif bearish_condition:
        return 0.8, "short", atr_value
    else:
        return 0.0, "none", atr_value
```

### Portfolio Management

- **Position correlation**: Avoid highly correlated positions
- **Sector rotation**: Balance exposure across different sectors
- **Risk parity**: Equalize risk contribution across positions

---

**Happy Trading! 🎉**

Remember: Start with paper trading, understand the risks, and always use proper risk management.
