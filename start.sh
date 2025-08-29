#!/bin/bash

# LegacyCoinTrader Startup Script
# This script ensures the virtual environment is activated before running the application

set -e

echo "🚀 Starting LegacyCoinTrader..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip if needed
pip install --upgrade pip > /dev/null 2>&1

# Ensure all dependencies are installed
echo "📦 Checking dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

# Set PYTHONPATH to ensure proper module loading
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "✅ Environment ready!"
echo ""
echo "Available commands:"
echo "  python -m crypto_bot.main              # Start main trading bot"
echo "  python -m frontend.app                 # Start web dashboard"
echo "  python crypto_bot/wallet_manager.py    # Setup API credentials"
echo "  python -m crypto_bot.solana.runner     # Start Solana sniper"
echo ""
echo "💡 Remember to:"
echo "   1. Copy crypto_bot/.env from ENV_TEMPLATE.md"
echo "   2. Run wallet_manager.py to configure API keys"
echo "   3. Edit config.yaml for your trading preferences"
echo ""

# Execute the provided command or start interactive shell
if [ $# -eq 0 ]; then
    echo "🔄 Starting interactive shell with activated environment..."
    echo "💡 Type 'python -m crypto_bot.main' to start the bot"
    exec bash --rcfile <(echo "PS1='(venv) \u@\h:\w\$ '")
else
    echo "▶️  Executing: $@"
    exec "$@"
fi
