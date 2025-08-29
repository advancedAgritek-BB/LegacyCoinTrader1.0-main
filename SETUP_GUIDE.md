# 🚀 LegacyCoinTrader Setup Guide

This guide will help you get LegacyCoinTrader up and running after downloading from GitHub.

## 📋 Prerequisites

- **Python 3.8+** (Python 3.9+ recommended)
- **Git** (for cloning the repository)
- **macOS, Linux, or Windows** (with WSL for Windows)

## 🔧 Quick Setup

### 1. Clone and Navigate
```bash
git clone https://github.com/advancedAgritek-BB/LegacyCoinTrader1.0-main.git
cd LegacyCoinTrader1.0-main
```

### 2. Use the Startup Script (Recommended)
The startup script automatically handles everything:
```bash
# Make executable (first time only)
chmod +x start.sh

# Start interactive shell
./start.sh

# Or run directly
./start.sh python -m crypto_bot.main
```

### 3. Manual Setup (Alternative)
If you prefer manual setup:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## ⚙️ Configuration

### 1. Environment Variables
Copy the template and fill in your API keys:
```bash
# Copy the template content from ENV_TEMPLATE.md
cp ENV_TEMPLATE.md crypto_bot/.env

# Edit the .env file with your actual API keys
nano crypto_bot/.env  # or use your preferred editor
```

**Required API Keys:**
- **Kraken**: `KRAKEN_API_KEY`, `KRAKEN_API_SECRET`
- **Telegram**: `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`
- **Solana**: `HELIUS_KEY`, `WALLET_ADDRESS`

### 2. Trading Configuration
Edit `crypto_bot/config.yaml` for your trading preferences:
```yaml
execution_mode: dry_run  # Start with dry_run for safety
exchange: kraken         # or coinbase
```

## 🚀 Running the Application

### Start the Main Trading Bot
```bash
python -m crypto_bot.main
```

### Start the Web Dashboard
```bash
python -m frontend.app
```

### Start Solana Sniper
```bash
python -m crypto_bot.solana.runner
```

## 🧪 Testing

### Verify Installation
```bash
# Test imports
python3 -c "import crypto_bot.main; print('✅ Main module works')"
python3 -c "import frontend.app; print('✅ Frontend works')"

# Run tests
python -m pytest tests/ -v
```

### Test Configuration
```bash
# Test wallet manager
python crypto_bot/wallet_manager.py
```

## 🔍 Troubleshooting

### Common Issues

**Import Errors:**
- Ensure virtual environment is activated
- Check PYTHONPATH is set correctly
- Verify all dependencies are installed

**API Connection Issues:**
- Verify API keys are correct
- Check internet connection
- Ensure API permissions are set correctly

**Configuration Errors:**
- Validate YAML syntax in config files
- Check environment variable names
- Ensure required fields are filled

### Getting Help

1. Check the logs in `crypto_bot/logs/`
2. Review the README.md for detailed documentation
3. Check specific feature READMEs (e.g., PUMP_SNIPER_README.md)

## 📚 Next Steps

1. **Paper Trading**: Start with `execution_mode: dry_run`
2. **Small Amounts**: Use small position sizes initially
3. **Monitor Performance**: Use the web dashboard to track results
4. **Adjust Strategy**: Modify config.yaml based on performance

## 🔒 Security Notes

- **Never commit API keys** to version control
- **Use environment variables** for sensitive data
- **Start with dry-run mode** to test strategies
- **Monitor logs** for any suspicious activity

## 📞 Support

For issues and questions:
- Check the documentation in the `docs/` folder
- Review the various README files
- Test with the provided test scripts

---

**Happy Trading! 🎯**
