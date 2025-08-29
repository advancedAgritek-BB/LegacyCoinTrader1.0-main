# .env File Template for LegacyCoinTrader
# Copy this content to crypto_bot/.env

# Complete environment configuration for the crypto bot
# Environment variables override values saved in user_config.yaml and config.yaml
# Uncomment and fill in the real values as needed

# Exchange Configuration
EXCHANGE=kraken
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_API_SECRET=your_kraken_api_secret_here
KRAKEN_WS_TOKEN=your_kraken_ws_token_here

# Alternative Exchange (Coinbase)
# COINBASE_API_KEY=your_coinbase_api_key
# COINBASE_API_SECRET=your_coinbase_secret
# COINBASE_PASSPHRASE=your_coinbase_passphrase

# Telegram Configuration
TELEGRAM_TOKEN=your_telegram_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
TELE_CHAT_ADMINS=your_telegram_chat_id_here

# Solana Configuration
HELIUS_KEY=your_helius_api_key_here
WALLET_ADDRESS=your_solana_wallet_address_here

# LunarCrush Sentiment Analysis (Optional)
# LUNARCRUSH_API_KEY=your_lunarcrush_api_key

# Google Cloud (Optional - for advanced logging)
# GOOGLE_CRED_JSON=path_to_google_credentials.json

# Trading Mode
MODE=cex

# Additional Configuration
# TESTING_MODE=true
# EXECUTION_MODE=dry_run

## Instructions:
# 1. Copy this entire content
# 2. Replace the content of crypto_bot/.env with this
# 3. Uncomment and fill in any missing values (like KRAKEN_WS_TOKEN)
# 4. Save the file
# 5. Restart your application

## What This Fixes:
# - Exchange API authentication
# - Telegram bot functionality  
# - Solana integration
# - Proper configuration loading
# - Missing environment variable prompts
