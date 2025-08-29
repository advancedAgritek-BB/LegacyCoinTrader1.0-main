#!/usr/bin/env python3
"""
Test script for symbol validation functionality.

This script demonstrates how the symbol validation system prevents querying
CEX exchanges for Solana chain symbols that don't exist there.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'crypto_bot'))

from crypto_bot.utils.symbol_validator import (
    get_symbol_data_source,
    is_symbol_supported_on_exchange,
    filter_symbols_by_exchange,
    validate_symbol_list
)

def test_symbol_validation():
    """Test various symbol validation scenarios."""
    
    print("🔍 Testing Symbol Validation System")
    print("=" * 50)
    
    # Test cases
    test_symbols = [
        # CEX symbols (should be supported)
        "BTC/USD",
        "ETH/USDT", 
        "SOL/USD",
        "ADA/USDT",
        "DOT/USD",
        
        # Solana chain symbols (should NOT be supported on CEX)
        "ABC123456789012345678901234567890123456789/USDC",  # Fake Solana address
        "So11111111111111111111111111111111111111112/USDC",  # Real SOL mint
        "EPjFWdd5AufqSSqeM2q6ksjLpaEweidnGj9n92gtQgNf/USDC",  # USDC mint
        "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs/USDC",  # Random Solana token
        
        # Edge cases
        "INVALID_SYMBOL",
        "BTC/",  # Missing quote
        "/USD",  # Missing base
        "",      # Empty string
    ]
    
    exchanges = ["kraken", "coinbase", "binance"]
    
    for exchange in exchanges:
        print(f"\n📊 Testing {exchange.upper()} Exchange:")
        print("-" * 30)
        
        # Validate all symbols
        validation_result = validate_symbol_list(test_symbols, exchange)
        
        print(f"Total symbols: {validation_result['total']}")
        print(f"CEX symbols: {len(validation_result['cex'])}")
        print(f"Solana symbols: {len(validation_result['solana'])}")
        print(f"Unknown symbols: {len(validation_result['unknown'])}")
        
        # Filter symbols
        supported, unsupported = filter_symbols_by_exchange(test_symbols, exchange)
        
        print(f"\n✅ Supported on {exchange}: {len(supported)}")
        for symbol in supported[:5]:  # Show first 5
            print(f"  - {symbol}")
        if len(supported) > 5:
            print(f"  ... and {len(supported) - 5} more")
            
        print(f"\n❌ NOT supported on {exchange}: {len(unsupported)}")
        for symbol in unsupported[:5]:  # Show first 5
            data_source = get_symbol_data_source(symbol, exchange)
            print(f"  - {symbol} (data source: {data_source})")
        if len(unsupported) > 5:
            print(f"  ... and {len(unsupported) - 5} more")

def test_individual_symbols():
    """Test individual symbol validation logic."""
    
    print("\n🔍 Testing Individual Symbol Logic")
    print("=" * 50)
    
    test_cases = [
        ("BTC/USD", "kraken"),
        ("ABC123456789012345678901234567890123456789/USDC", "kraken"),
        ("SOL/USDT", "coinbase"),
        ("INVALID", "binance"),
        ("ETH/BTC", "kraken"),
    ]
    
    for symbol, exchange in test_cases:
        data_source = get_symbol_data_source(symbol, exchange)
        supported = is_symbol_supported_on_exchange(symbol, exchange)
        
        print(f"Symbol: {symbol:<40} | Exchange: {exchange:<10} | Data Source: {data_source:<8} | Supported: {supported}")

def test_solana_address_validation():
    """Test Solana address validation."""
    
    print("\n🔍 Testing Solana Address Validation")
    print("=" * 50)
    
    from crypto_bot.utils.symbol_validator import is_valid_solana_token
    
    test_addresses = [
        "So11111111111111111111111111111111111111112",  # SOL mint
        "EPjFWdd5AufqSSqeM2q6ksjLpaEweidnGj9n92gtQgNf",  # USDC mint
        "ABC123456789012345678901234567890123456789",  # Fake but valid format
        "invalid_address",
        "too_short",
        "way_too_long_address_that_exceeds_maximum_length_for_solana",
        "",
        None,
    ]
    
    for addr in test_addresses:
        is_valid = is_valid_solana_token(addr)
        print(f"Address: {str(addr):<50} | Valid: {is_valid}")

def main():
    """Run all tests."""
    try:
        test_symbol_validation()
        test_individual_symbols()
        test_solana_address_validation()
        
        print("\n✅ All tests completed successfully!")
        print("\n💡 Key Benefits:")
        print("  - Prevents querying CEX exchanges for Solana chain symbols")
        print("  - Automatically routes symbols to appropriate data sources")
        print("  - Reduces API errors and improves system reliability")
        print("  - Configurable validation rules per exchange")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
