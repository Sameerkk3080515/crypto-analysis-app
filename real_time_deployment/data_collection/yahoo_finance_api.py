"""
Yahoo Finance API wrapper for cryptocurrency data collection.
"""
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add path for data_api access
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

class YahooFinanceAPI:
    """
    Wrapper class for Yahoo Finance API to fetch cryptocurrency data.
    """
    def __init__(self):
        """Initialize the Yahoo Finance API client."""
        self.client = ApiClient()
        self.supported_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo']
        self.supported_ranges = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        
    def get_crypto_data(self, symbol, interval='1d', range='1mo'):
        """
        Fetch cryptocurrency data from Yahoo Finance.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC-USD', 'ETH-USD')
            interval (str): Data interval (e.g., '1d', '1h')
            range (str): Data range (e.g., '1mo', '3mo')
            
        Returns:
            dict: Cryptocurrency data including meta information and price history
        """
        if interval not in self.supported_intervals:
            raise ValueError(f"Interval must be one of {self.supported_intervals}")
        
        if range not in self.supported_ranges:
            raise ValueError(f"Range must be one of {self.supported_ranges}")
        
        try:
            response = self.client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'interval': interval,
                'range': range
            })
            
            if response and 'chart' in response and 'result' in response['chart'] and response['chart']['result']:
                return response['chart']['result'][0]
            else:
                raise ValueError(f"Invalid response format for {symbol}")
                
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_crypto_price_history(self, symbol, interval='1d', range='1mo'):
        """
        Get formatted price history for a cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC-USD', 'ETH-USD')
            interval (str): Data interval (e.g., '1d', '1h')
            range (str): Data range (e.g., '1mo', '3mo')
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        data = self.get_crypto_data(symbol, interval, range)
        
        if not data:
            return pd.DataFrame()
        
        try:
            timestamps = data['timestamp']
            quotes = data['indicators']['quote'][0]
            
            # Convert timestamps to datetime
            open_time = [datetime.fromtimestamp(ts) for ts in timestamps]
            
            # Create DataFrame
            df = pd.DataFrame({
                'open_time': open_time,
                'open': quotes['open'],
                'high': quotes['high'],
                'low': quotes['low'],
                'close': quotes['close'],
                'volume': quotes['volume']
            })
            
            return df
        
        except Exception as e:
            print(f"Error processing data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol):
        """
        Get current price for a cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC-USD', 'ETH-USD')
            
        Returns:
            float: Current price
        """
        data = self.get_crypto_data(symbol, interval='1d', range='1d')
        
        if not data or 'meta' not in data:
            return None
        
        return data['meta'].get('regularMarketPrice')
    
    def get_multiple_crypto_data(self, symbols, interval='1d', range='1mo'):
        """
        Get data for multiple cryptocurrencies.
        
        Args:
            symbols (list): List of cryptocurrency symbols
            interval (str): Data interval
            range (str): Data range
            
        Returns:
            dict: Dictionary with symbol as key and data as value
        """
        result = {}
        
        for symbol in symbols:
            data = self.get_crypto_price_history(symbol, interval, range)
            if not data.empty:
                result[symbol] = data
        
        return result
