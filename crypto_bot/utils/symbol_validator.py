"""
Symbol validation and exchange routing utilities.

This module provides functions to determine the appropriate data source for symbols
and prevent querying centralized exchanges for Solana chain symbols that don't exist there.
"""

import base58
from typing import Dict, Set, Optional
from .logger import setup_logger, LOG_DIR

logger = setup_logger(__name__, LOG_DIR / "symbol_validator.log")

# Valid characters for Solana addresses
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# Exchange-specific symbol patterns
CEX_SYMBOL_PATTERNS = {
    "kraken": {
        "supported_quotes": {"USD", "USDT", "EUR", "BTC", "ETH"},
        "supported_bases": {"BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "MATIC", "AVAX", "UNI", "AAVE", "XRP", "LTC", "BCH", "EOS", "TRX", "XLM", "NEO", "VET", "ICX", "ONT", "QTUM", "ZIL", "BAT", "ZRX", "OMG", "KNC", "REP", "ZEC", "DASH", "XMR", "ETC", "WAVES", "STRAT", "LSK", "ARK", "STEEM", "GNT", "FUN", "SNT", "POWR", "BNT", "MANA", "SALT", "STORJ", "DNT", "LOOM", "REQ", "CVC", "RLC", "MKR", "ENJ", "REN", "BNB", "ADA", "DOT", "LINK", "MATIC", "AVAX", "UNI", "AAVE", "COMP", "YFI", "CRV", "BAL", "SUSHI", "1INCH", "ALPHA", "PERP", "RUNE", "FTM", "NEAR", "ALGO", "ATOM", "COSMOS", "OSMO", "JUNO", "SCRT", "KAVA", "ROSE", "FLOW", "ICP", "FIL", "AR", "HNT", "IOTA", "NANO", "VET", "ICX", "ONT", "QTUM", "ZIL", "BAT", "ZRX", "OMG", "KNC", "REP", "ZEC", "DASH", "XMR", "ETC", "WAVES", "STRAT", "LSK", "ARK", "STEEM", "GNT", "FUN", "SNT", "POWR", "BNT", "MANA", "SALT", "STORJ", "DNT", "LOOM", "REQ", "CVC", "RLC", "MKR", "ENJ", "REN", "BNB", "ADA", "DOT", "LINK", "MATIC", "AVAX", "UNI", "AAVE", "COMP", "YFI", "CRV", "BAL", "SUSHI", "1INCH", "ALPHA", "PERP", "RUNE", "FTM", "NEAR", "ALGO", "ATOM", "COSMOS", "OSMO", "JUNO", "SCRT", "KAVA", "ROSE", "FLOW", "ICP", "FIL", "AR", "HNT", "IOTA", "NANO"}
    },
    "coinbase": {
        "supported_quotes": {"USD", "USDC", "USDT", "BTC", "ETH"},
        "supported_bases": {"BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "MATIC", "AVAX", "UNI", "AAVE", "COMP", "YFI", "CRV", "BAL", "SUSHI", "1INCH", "ALPHA", "PERP", "RUNE", "FTM", "NEAR", "ALGO", "ATOM", "COSMOS", "OSMO", "JUNO", "SCRT", "KAVA", "ROSE", "FLOW", "ICP", "FIL", "AR", "HNT", "IOTA", "NANO", "VET", "ICX", "ONT", "QTUM", "ZIL", "BAT", "ZRX", "OMG", "KNC", "REP", "ZEC", "DASH", "XMR", "ETC", "WAVES", "STRAT", "LSK", "ARK", "STEEM", "GNT", "FUN", "SNT", "POWR", "BNT", "MANA", "SALT", "STORJ", "DNT", "LOOM", "REQ", "CVC", "RLC", "MKR", "ENJ", "REN", "BNB", "ADA", "DOT", "LINK", "MATIC", "AVAX", "UNI", "AAVE", "COMP", "YFI", "CRV", "BAL", "SUSHI", "1INCH", "ALPHA", "PERP", "RUNE", "FTM", "NEAR", "ALGO", "ATOM", "COSMOS", "OSMO", "JUNO", "SCRT", "KAVA", "ROSE", "FLOW", "ICP", "FIL", "AR", "HNT", "IOTA", "NANO"}
    },
    "binance": {
        "supported_quotes": {"USDT", "BUSD", "BTC", "ETH", "BNB"},
        "supported_bases": {"BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "MATIC", "AVAX", "UNI", "AAVE", "COMP", "YFI", "CRV", "BAL", "SUSHI", "1INCH", "ALPHA", "PERP", "RUNE", "FTM", "NEAR", "ALGO", "ATOM", "COSMOS", "OSMO", "JUNO", "SCRT", "KAVA", "ROSE", "FLOW", "ICP", "FIL", "AR", "HNT", "IOTA", "NANO", "VET", "ICX", "ONT", "QTUM", "ZIL", "BAT", "ZRX", "OMG", "KNC", "REP", "ZEC", "DASH", "XMR", "ETC", "WAVES", "STRAT", "LSK", "ARK", "STEEM", "GNT", "FUN", "SNT", "POWR", "BNT", "MANA", "SALT", "STORJ", "DNT", "LOOM", "REQ", "CVC", "RLC", "MKR", "ENJ", "REN", "BNB", "ADA", "DOT", "LINK", "MATIC", "AVAX", "UNI", "AAVE", "COMP", "YFI", "CRV", "BAL", "SUSHI", "1INCH", "ALPHA", "PERP", "RUNE", "FTM", "NEAR", "ALGO", "ATOM", "COSMOS", "OSMO", "JUNO", "SCRT", "KAVA", "ROSE", "FLOW", "ICP", "FIL", "AR", "HNT", "IOTA", "NANO"}
    }
}

# Solana-specific data sources
SOLANA_DATA_SOURCES = {
    "geckoterminal": "https://api.geckoterminal.com/api/v2/networks/solana",
    "jupiter": "https://price.jup.ag/v4",
    "birdeye": "https://public-api.birdeye.so",
    "dexscreener": "https://api.dexscreener.com/latest/dex/tokens"
}

# Major coins that exist on both CEX and DEX
MAJOR_COINS = {"BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "MATIC", "AVAX", "UNI", "AAVE"}


def is_valid_solana_token(token: str) -> bool:
    """
    Check if a token looks like a valid Solana mint address.
    
    Args:
        token: Token string to validate
        
    Returns:
        True if the token appears to be a valid Solana mint address
    """
    if not isinstance(token, str):
        return False
    
    if not (32 <= len(token) <= 44):
        return False
    
    try:
        decoded = base58.b58decode(token)
        return len(decoded) == 32 and all(c in BASE58_ALPHABET for c in token)
    except Exception:
        return False


def get_symbol_data_source(symbol: str, exchange_id: Optional[str] = None) -> str:
    """
    Determine the appropriate data source for a symbol.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTC/USD", "ABC123/USDC")
        exchange_id: Exchange identifier (e.g., "kraken", "coinbase")
        
    Returns:
        String indicating the data source:
        - "cex" for centralized exchange symbols
        - "solana" for Solana chain symbols
        - "unknown" for unrecognized symbols
    """
    if not isinstance(symbol, str) or "/" not in symbol:
        return "unknown"
    
    base, quote = symbol.split("/", 1)
    base = base.upper()
    quote = quote.upper()
    
    # Check if this is a Solana chain symbol
    if quote == "USDC" and is_valid_solana_token(base):
        return "solana"
    
    # Check if this is a known CEX symbol
    if exchange_id and exchange_id in CEX_SYMBOL_PATTERNS:
        exchange_patterns = CEX_SYMBOL_PATTERNS[exchange_id]
        if (base in exchange_patterns["supported_bases"] and 
            quote in exchange_patterns["supported_quotes"]):
            return "cex"
    
    # Fallback: check if base looks like a Solana address
    if is_valid_solana_token(base):
        return "solana"
    
    # Default to CEX for traditional symbols
    return "cex"


def is_symbol_supported_on_exchange(symbol: str, exchange_id: str) -> bool:
    """
    Check if a symbol is supported on a specific exchange.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTC/USD")
        exchange_id: Exchange identifier (e.g., "kraken", "coinbase")
    
    Returns:
        True if the symbol should be queried on this exchange
    """
    data_source = get_symbol_data_source(symbol, exchange_id)
    
    if data_source == "solana":
        # Solana symbols should not be queried on CEX exchanges
        logger.debug("Symbol %s identified as Solana chain symbol, not supported on %s", symbol, exchange_id)
        return False
    elif data_source == "cex":
        # CEX symbols can be queried on CEX exchanges
        return True
    else:
        # Unknown symbols - be conservative and don't query
        logger.debug("Symbol %s has unknown data source, not querying on %s", symbol, exchange_id)
        return False


def get_recommended_data_source(symbol: str, exchange_id: Optional[str] = None) -> str:
    """
    Get the recommended data source for OHLCV data for a given symbol.
    
    Args:
        symbol: Trading pair symbol
        exchange_id: Exchange identifier (optional)
    
    Returns:
        String indicating the recommended data source:
        - "geckoterminal" for Solana symbols
        - "coingecko" for major coins
        - "cex" for centralized exchange symbols
        - "unknown" for unrecognized symbols
    """
    data_source = get_symbol_data_source(symbol, exchange_id)
    
    if data_source == "solana":
        return "geckoterminal"
    elif data_source == "cex":
        base, _, _ = symbol.partition("/")
        if base in MAJOR_COINS:
            return "coingecko"
        else:
            return "cex"
    else:
        return "unknown"


def filter_symbols_by_exchange(symbols: list, exchange_id: str) -> tuple[list, list]:
    """
    Filter a list of symbols into those supported and not supported on an exchange.
    
    Args:
        symbols: List of trading pair symbols
        exchange_id: Exchange identifier
        
    Returns:
        Tuple of (supported_symbols, unsupported_symbols)
    """
    supported = []
    unsupported = []
    
    for symbol in symbols:
        if is_symbol_supported_on_exchange(symbol, exchange_id):
            supported.append(symbol)
        else:
            unsupported.append(symbol)
    
    if unsupported:
        logger.info(
            "Filtered %d symbols for %s: %d supported, %d unsupported (Solana chain symbols)",
            len(symbols), exchange_id, len(supported), len(unsupported)
        )
    
    return supported, unsupported


def validate_symbol_list(symbols: list, exchange_id: str) -> dict:
    """
    Validate a list of symbols and categorize them by data source.
    
    Args:
        symbols: List of trading pair symbols
        exchange_id: Exchange identifier
        
    Returns:
        Dictionary with categorized symbols
    """
    result = {
        "cex": [],
        "solana": [],
        "unknown": [],
        "total": len(symbols)
    }
    
    for symbol in symbols:
        data_source = get_symbol_data_source(symbol, exchange_id)
        if data_source in result:
            result[data_source].append(symbol)
        else:
            result["unknown"].append(symbol)
    
    logger.info(
        "Symbol validation for %s: %d CEX, %d Solana, %d unknown",
        exchange_id, len(result["cex"]), len(result["solana"]), len(result["unknown"])
    )
    
    return result
