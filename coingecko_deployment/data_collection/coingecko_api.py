"""
CoinGecko API wrapper for cryptocurrency data collection.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json

class CoinGeckoAPI:
    """
    Wrapper class for CoinGecko API to fetch cryptocurrency data.
    """
    def __init__(self):
        """Initialize the CoinGecko API client."""
        self.base_url = "https://api.coingecko.com/api/v3"
        self.coin_mapping = {
            'BTC-USD': 'bitcoin',
            'ETH-USD': 'ethereum',
            'BNB-USD': 'binancecoin',
            'ADA-USD': 'cardano',
            'DOGE-USD': 'dogecoin',
            'XRP-USD': 'ripple',
            'DOT-USD': 'polkadot',
            'UNI-USD': 'uniswap',
            'LTC-USD': 'litecoin',
            'LINK-USD': 'chainlink'
        }
        self.reverse_mapping = {v: k for k, v in self.coin_mapping.items()}
        
    def _get_with_retry(self, url, params=None, max_retries=3, backoff_factor=1.5):
        """
        Make a GET request with retry logic.
        
        Args:
            url (str): URL to request
            params (dict): Query parameters
            max_retries (int): Maximum number of retries
            backoff_factor (float): Backoff factor for exponential backoff
            
        Returns:
            dict: Response JSON or None if failed
        """
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Too Many Requests
                    wait_time = backoff_factor ** retries
                    print(f"Rate limited. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    print(f"Error: HTTP {response.status_code}")
                    return None
            except Exception as e:
                print(f"Request error: {str(e)}")
                retries += 1
                time.sleep(backoff_factor ** retries)
        
        print(f"Failed after {max_retries} retries")
        return None
    
    def _convert_symbol_to_id(self, symbol):
        """
        Convert Yahoo Finance symbol to CoinGecko ID.
        
        Args:
            symbol (str): Yahoo Finance symbol (e.g., 'BTC-USD')
            
        Returns:
            str: CoinGecko ID (e.g., 'bitcoin')
        """
        if symbol in self.coin_mapping:
            return self.coin_mapping[symbol]
        else:
            # Try to guess the ID by removing the -USD suffix and converting to lowercase
            base_symbol = symbol.split('-')[0].lower()
            return base_symbol
    
    def get_current_price(self, symbol):
        """
        Get current price for a cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC-USD', 'ETH-USD')
            
        Returns:
            float: Current price
        """
        coin_id = self._convert_symbol_to_id(symbol)
        url = f"{self.base_url}/simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd',
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true'
        }
        
        data = self._get_with_retry(url, params)
        
        if data and coin_id in data:
            return data[coin_id]['usd']
        
        return None
    
    def get_crypto_price_history(self, symbol, days='30'):
        """
        Get price history for a cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC-USD', 'ETH-USD')
            days (str): Number of days of data to retrieve ('1', '7', '14', '30', '90', '180', '365', 'max')
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        coin_id = self._convert_symbol_to_id(symbol)
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        data = self._get_with_retry(url, params)
        
        if not data:
            return pd.DataFrame()
        
        try:
            # Extract price data
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            # Create DataFrame
            df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            
            # Convert timestamps to datetime
            df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
            df_volumes['timestamp'] = pd.to_datetime(df_volumes['timestamp'], unit='ms')
            
            # Merge price and volume data
            df = pd.merge(df_prices, df_volumes, on='timestamp')
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'timestamp': 'open_time',
                'price': 'close'
            })
            
            # Generate OHLC data (approximated from daily close prices)
            df['open'] = df['close'].shift(1)
            df['high'] = df['close'] * 1.01  # Approximate high as 1% above close
            df['low'] = df['close'] * 0.99   # Approximate low as 1% below close
            
            # Fill first row's open with its close value
            df.loc[0, 'open'] = df.loc[0, 'close']
            
            # Reorder columns to match expected format
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            print(f"Error processing data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_crypto_data(self, symbol, days='30'):
        """
        Get comprehensive data for a cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC-USD', 'ETH-USD')
            days (str): Number of days of data to retrieve
            
        Returns:
            dict: Cryptocurrency data including price history and metadata
        """
        coin_id = self._convert_symbol_to_id(symbol)
        
        # Get coin details
        url = f"{self.base_url}/coins/{coin_id}"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'false',
            'developer_data': 'false'
        }
        
        coin_data = self._get_with_retry(url, params)
        
        if not coin_data:
            return None
        
        # Get price history
        price_history = self.get_crypto_price_history(symbol, days)
        
        # Create metadata similar to Yahoo Finance format
        meta = {
            'currency': 'USD',
            'symbol': symbol,
            'exchangeName': 'CoinGecko',
            'instrumentType': 'CRYPTOCURRENCY',
            'regularMarketPrice': coin_data.get('market_data', {}).get('current_price', {}).get('usd'),
            'regularMarketDayHigh': coin_data.get('market_data', {}).get('high_24h', {}).get('usd'),
            'regularMarketDayLow': coin_data.get('market_data', {}).get('low_24h', {}).get('usd'),
            'regularMarketVolume': coin_data.get('market_data', {}).get('total_volume', {}).get('usd'),
            'longName': coin_data.get('name'),
            'shortName': coin_data.get('symbol', '').upper() + '-USD'
        }
        
        # Format similar to Yahoo Finance response
        result = {
            'meta': meta,
            'timestamp': price_history['open_time'].tolist() if not price_history.empty else [],
            'indicators': {
                'quote': [{
                    'open': price_history['open'].tolist() if not price_history.empty else [],
                    'high': price_history['high'].tolist() if not price_history.empty else [],
                    'low': price_history['low'].tolist() if not price_history.empty else [],
                    'close': price_history['close'].tolist() if not price_history.empty else [],
                    'volume': price_history['volume'].tolist() if not price_history.empty else []
                }]
            }
        }
        
        return result
    
    def get_multiple_crypto_data(self, symbols, days='30'):
        """
        Get data for multiple cryptocurrencies.
        
        Args:
            symbols (list): List of cryptocurrency symbols
            days (str): Number of days of data to retrieve
            
        Returns:
            dict: Dictionary with symbol as key and data as value
        """
        result = {}
        
        for symbol in symbols:
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
            data = self.get_crypto_price_history(symbol, days)
            if not data.empty:
                result[symbol] = data
        
        return result
    
    def get_market_data(self, vs_currency='usd', category='', order='market_cap_desc', per_page=100, page=1):
        """
        Get market data for multiple cryptocurrencies.
        
        Args:
            vs_currency (str): The target currency (default: 'usd')
            category (str): Filter by category
            order (str): Sort by field (default: 'market_cap_desc')
            per_page (int): Number of results per page (default: 100)
            page (int): Page number (default: 1)
            
        Returns:
            list: List of cryptocurrency market data
        """
        url = f"{self.base_url}/coins/markets"
        params = {
            'vs_currency': vs_currency,
            'category': category,
            'order': order,
            'per_page': per_page,
            'page': page,
            'sparkline': 'false'
        }
        
        return self._get_with_retry(url, params) or []
