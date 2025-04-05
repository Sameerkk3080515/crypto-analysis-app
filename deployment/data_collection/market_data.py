"""
Market data collection module for the Cryptocurrency Analysis Bot.

This module handles fetching cryptocurrency market data from various sources,
including Binance API and Yahoo Finance API.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import requests
import pandas as pd

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

class MarketDataCollector:
    """Collects cryptocurrency market data from various sources."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the market data collector.
        
        Args:
            config: Configuration object (creates a new one if None)
        """
        self.config = config or Config()
        self.base_url = self.config.get_binance_base_url()
        self.cryptocurrencies = self.config.get_cryptocurrencies()
    
    def get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for all tracked cryptocurrencies.
        
        Returns:
            Dictionary mapping cryptocurrency symbols to current prices
        """
        endpoint = f"{self.base_url}/api/v3/ticker/price"
        
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            
            all_prices = response.json()
            
            # Filter for our tracked cryptocurrencies and convert to USDT pairs
            prices = {}
            for item in all_prices:
                symbol = item['symbol']
                for crypto in self.cryptocurrencies:
                    if symbol == f"{crypto}USDT":
                        prices[crypto] = float(item['price'])
            
            return prices
        
        except requests.RequestException as e:
            print(f"Error fetching current prices: {e}")
            return {}
    
    def get_24h_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get 24-hour statistics for all tracked cryptocurrencies.
        
        Returns:
            Dictionary mapping cryptocurrency symbols to statistics
        """
        endpoint = f"{self.base_url}/api/v3/ticker/24hr"
        
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            
            all_stats = response.json()
            
            # Filter for our tracked cryptocurrencies and convert to USDT pairs
            stats = {}
            for item in all_stats:
                symbol = item['symbol']
                for crypto in self.cryptocurrencies:
                    if symbol == f"{crypto}USDT":
                        stats[crypto] = {
                            'price_change': float(item['priceChange']),
                            'price_change_percent': float(item['priceChangePercent']),
                            'weighted_avg_price': float(item['weightedAvgPrice']),
                            'prev_close_price': float(item['prevClosePrice']),
                            'last_price': float(item['lastPrice']),
                            'last_qty': float(item['lastQty']),
                            'bid_price': float(item['bidPrice']),
                            'bid_qty': float(item['bidQty']),
                            'ask_price': float(item['askPrice']),
                            'ask_qty': float(item['askQty']),
                            'open_price': float(item['openPrice']),
                            'high_price': float(item['highPrice']),
                            'low_price': float(item['lowPrice']),
                            'volume': float(item['volume']),
                            'quote_volume': float(item['quoteVolume']),
                            'count': int(item['count'])
                        }
            
            return stats
        
        except requests.RequestException as e:
            print(f"Error fetching 24-hour statistics: {e}")
            return {}
    
    def get_historical_klines(self, symbol: str, interval: str = '1d', 
                             limit: int = 30) -> pd.DataFrame:
        """
        Get historical candlestick data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            interval: Candlestick interval (e.g., '1d', '4h', '1h', '15m')
            limit: Number of candlesticks to retrieve (max 1000)
            
        Returns:
            DataFrame with historical price data
        """
        endpoint = f"{self.base_url}/api/v3/klines"
        
        params = {
            'symbol': f"{symbol}USDT",
            'interval': interval,
            'limit': min(limit, 1000)  # Binance API limit
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            klines = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                              'quote_asset_volume', 'taker_buy_base_asset_volume',
                              'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # Convert timestamps to datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            return df
        
        except requests.RequestException as e:
            print(f"Error fetching historical klines for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            limit: Depth of the order book (max 5000)
            
        Returns:
            Dictionary with order book data
        """
        endpoint = f"{self.base_url}/api/v3/depth"
        
        params = {
            'symbol': f"{symbol}USDT",
            'limit': min(limit, 5000)  # Binance API limit
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            order_book = response.json()
            
            # Convert string values to float
            for side in ['bids', 'asks']:
                order_book[side] = [[float(price), float(qty)] for price, qty in order_book[side]]
            
            return order_book
        
        except requests.RequestException as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return {'bids': [], 'asks': []}
    
    def get_market_data_for_analysis(self) -> Dict[str, Any]:
        """
        Collect all market data needed for analysis.
        
        Returns:
            Dictionary with comprehensive market data for all tracked cryptocurrencies
        """
        market_data = {
            'timestamp': datetime.now().isoformat(),
            'current_prices': self.get_current_prices(),
            'stats_24h': self.get_24h_stats(),
            'historical_data': {},
            'order_books': {}
        }
        
        # Get historical data and order books for each cryptocurrency
        for crypto in self.cryptocurrencies:
            # Get historical data
            df = self.get_historical_klines(
                crypto, 
                interval='1d',
                limit=self.config.get('market_data.price_history_days', 30)
            )
            
            if not df.empty:
                market_data['historical_data'][crypto] = df.to_dict('records')
            
            # Get order book
            market_data['order_books'][crypto] = self.get_order_book(crypto)
            
            # Avoid rate limiting
            time.sleep(0.1)
        
        return market_data
    
    def save_market_data(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save market data to a JSON file.
        
        Args:
            data: Market data to save
            filename: Output filename (generates a timestamped name if None)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"market_data_{timestamp}.json"
        
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        
        # Create data directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        
        # Convert DataFrame records back to lists for JSON serialization
        serializable_data = data.copy()
        
        if 'historical_data' in serializable_data:
            for crypto, records in serializable_data['historical_data'].items():
                for record in records:
                    for key, value in record.items():
                        if isinstance(value, pd.Timestamp):
                            record[key] = value.isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        return output_path


# Example usage
if __name__ == "__main__":
    collector = MarketDataCollector()
    
    # Get current prices
    prices = collector.get_current_prices()
    print("Current prices:")
    for crypto, price in prices.items():
        print(f"{crypto}: ${price:.2f}")
    
    # Get 24-hour statistics for Bitcoin
    stats = collector.get_24h_stats()
    if 'BTC' in stats:
        print("\nBitcoin 24-hour statistics:")
        for key, value in stats['BTC'].items():
            print(f"{key}: {value}")
    
    # Get and save comprehensive market data
    market_data = collector.get_market_data_for_analysis()
    output_path = collector.save_market_data(market_data)
    print(f"\nSaved market data to {output_path}")
