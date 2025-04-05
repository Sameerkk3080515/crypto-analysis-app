"""
Configuration utilities for the Cryptocurrency Analysis Bot.
"""

import json
import os
from typing import Dict, List, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    "cryptocurrencies": [
        "BTC",  # Bitcoin
        "ETH",  # Ethereum
        "BNB",  # Binance Coin
        "SOL",  # Solana
        "XRP",  # Ripple
        "ADA",  # Cardano
        "DOGE", # Dogecoin
        "DOT",  # Polkadot
        "AVAX", # Avalanche
        "MATIC" # Polygon
    ],
    "market_data": {
        "binance_base_url": "https://data-api.binance.vision",
        "yahoo_finance_enabled": True,
        "price_history_days": 30,
        "update_interval_minutes": 60
    },
    "news_data": {
        "crypto_news_api_enabled": True,
        "twitter_api_enabled": True,
        "news_items_limit": 20,
        "sentiment_threshold": {
            "positive": 0.6,
            "negative": 0.4
        }
    },
    "technical_analysis": {
        "indicators": {
            "sma": [20, 50, 200],  # Simple Moving Average periods
            "ema": [12, 26],       # Exponential Moving Average periods
            "rsi": 14,             # Relative Strength Index period
            "macd": {
                "fast": 12,
                "slow": 26,
                "signal": 9
            },
            "bollinger_bands": {
                "period": 20,
                "std_dev": 2
            }
        },
        "signal_weights": {
            "trend": 0.4,
            "momentum": 0.3,
            "volatility": 0.3
        }
    },
    "recommendation": {
        "min_confidence": 0.7,
        "position_sizing": {
            "max_per_trade": 0.1,  # Maximum 10% of portfolio per trade
            "risk_factor": 0.02    # 2% risk per trade
        }
    }
}

class Config:
    """Configuration manager for the Cryptocurrency Analysis Bot."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration with default values or from a config file.
        
        Args:
            config_path: Path to the configuration JSON file (optional)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'config.json'
        )
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults if file doesn't exist.
        
        Returns:
            Dict containing configuration values
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
                return DEFAULT_CONFIG.copy()
        else:
            # Save default config for future use
            self.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
    
    def save_config(self, config: Dict[str, Any] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save (uses current config if None)
        """
        config_to_save = config if config is not None else self.config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
        except IOError as e:
            print(f"Error saving config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value to return if key is not found
            
        Returns:
            Configuration value or default
        """
        if '.' not in key:
            return self.config.get(key, default)
        
        # Handle nested keys with dot notation
        parts = key.split('.')
        value = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            value: Value to set
        """
        if '.' not in key:
            self.config[key] = value
            return
        
        # Handle nested keys with dot notation
        parts = key.split('.')
        config_section = self.config
        for part in parts[:-1]:
            if part not in config_section:
                config_section[part] = {}
            config_section = config_section[part]
        
        config_section[parts[-1]] = value
    
    def get_cryptocurrencies(self) -> List[str]:
        """
        Get the list of cryptocurrencies to track.
        
        Returns:
            List of cryptocurrency symbols
        """
        return self.get('cryptocurrencies', DEFAULT_CONFIG['cryptocurrencies'])
    
    def get_binance_base_url(self) -> str:
        """
        Get the Binance API base URL.
        
        Returns:
            Binance API base URL
        """
        return self.get('market_data.binance_base_url', 
                       DEFAULT_CONFIG['market_data']['binance_base_url'])
