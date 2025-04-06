import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import random

class BinanceDataCollector:
    """
    Class to collect data from Binance API for USDT trading pairs
    """
    
    def __init__(self, base_url="https://api.binance.com"):
        """
        Initialize the Binance data collector
        
        Args:
            base_url: Base URL for Binance API
        """
        self.base_url = base_url
        self.endpoints = {
            "exchange_info": "/api/v3/exchangeInfo",
            "ticker_24hr": "/api/v3/ticker/24hr",
            "klines": "/api/v3/klines",
            "ticker_price": "/api/v3/ticker/price",
            "ticker_book": "/api/v3/ticker/bookTicker"
        }
    
    def get_exchange_info(self):
        """
        Get exchange information from Binance
        
        Returns:
            Dictionary with exchange information
        """
        try:
            url = f"{self.base_url}{self.endpoints['exchange_info']}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting exchange info: {str(e)}")
            return {}
    
    def get_usdt_trading_pairs(self, max_pairs=None):
        """
        Get list of USDT trading pairs from Binance
        
        Args:
            max_pairs: Maximum number of pairs to return (None for all)
            
        Returns:
            List of USDT trading pair symbols
        """
        try:
            exchange_info = self.get_exchange_info()
            
            if not exchange_info or "symbols" not in exchange_info:
                return []
            
            # Filter for USDT trading pairs that are currently trading
            usdt_pairs = [
                symbol["symbol"] for symbol in exchange_info["symbols"]
                if symbol["symbol"].endswith("USDT") and 
                symbol["status"] == "TRADING" and
                symbol["isSpotTradingAllowed"] == True
            ]
            
            # Sort by symbol name
            usdt_pairs.sort()
            
            # Limit to max_pairs if specified
            if max_pairs is not None and max_pairs > 0:
                usdt_pairs = usdt_pairs[:max_pairs]
            
            return usdt_pairs
        except Exception as e:
            print(f"Error getting USDT trading pairs: {str(e)}")
            return []
    
    def get_ticker_24hr(self, symbol=None):
        """
        Get 24-hour ticker data from Binance
        
        Args:
            symbol: Trading pair symbol (None for all)
            
        Returns:
            Dictionary or list of dictionaries with ticker data
        """
        try:
            url = f"{self.base_url}{self.endpoints['ticker_24hr']}"
            
            if symbol:
                url += f"?symbol={symbol}"
            
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting ticker data: {str(e)}")
            return [] if symbol is None else {}
    
    def get_klines(self, symbol, interval="1h", limit=100, start_time=None, end_time=None):
        """
        Get klines (candlestick) data from Binance
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of klines to return (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            DataFrame with klines data
        """
        try:
            url = f"{self.base_url}{self.endpoints['klines']}?symbol={symbol}&interval={interval}&limit={limit}"
            
            if start_time:
                url += f"&startTime={start_time}"
            
            if end_time:
                url += f"&endTime={end_time}"
            
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            
            # Convert types
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            
            for col in ["open", "high", "low", "close", "volume", "quote_asset_volume",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]:
                df[col] = df[col].astype(float)
            
            df["number_of_trades"] = df["number_of_trades"].astype(int)
            
            return df
        except Exception as e:
            print(f"Error getting klines data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_ticker_price(self, symbol=None):
        """
        Get current price from Binance
        
        Args:
            symbol: Trading pair symbol (None for all)
            
        Returns:
            Dictionary or list of dictionaries with price data
        """
        try:
            url = f"{self.base_url}{self.endpoints['ticker_price']}"
            
            if symbol:
                url += f"?symbol={symbol}"
            
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting ticker price: {str(e)}")
            return [] if symbol is None else {}
    
    def get_ticker_book(self, symbol=None):
        """
        Get order book ticker from Binance
        
        Args:
            symbol: Trading pair symbol (None for all)
            
        Returns:
            Dictionary or list of dictionaries with order book data
        """
        try:
            url = f"{self.base_url}{self.endpoints['ticker_book']}"
            
            if symbol:
                url += f"?symbol={symbol}"
            
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting order book ticker: {str(e)}")
            return [] if symbol is None else {}
    
    def collect_comprehensive_data(self, max_pairs=20):
        """
        Collect comprehensive data for USDT trading pairs
        
        Args:
            max_pairs: Maximum number of pairs to collect data for
            
        Returns:
            Dictionary with comprehensive data for each trading pair
        """
        try:
            # Get USDT trading pairs
            usdt_pairs = self.get_usdt_trading_pairs(max_pairs=max_pairs)
            
            if not usdt_pairs:
                print("No USDT trading pairs found")
                return {}
            
            print(f"Collecting data for {len(usdt_pairs)} USDT trading pairs")
            
            # Get 24-hour ticker data for all pairs
            all_tickers = self.get_ticker_24hr()
            
            # Create dictionary to store data
            all_pairs_data = {}
            
            # Process each trading pair
            for symbol in usdt_pairs:
                try:
                    # Get ticker data for this symbol
                    ticker = next((t for t in all_tickers if t["symbol"] == symbol), None)
                    
                    if not ticker:
                        continue
                    
                    # Get current price
                    current_price = float(ticker["lastPrice"])
                    
                    # Get klines data for different intervals
                    klines_data = {}
                    for interval in ["15m", "1h", "4h", "1d"]:
                        klines_data[interval] = self.get_klines(symbol, interval=interval, limit=100)
                        # Add small delay to avoid rate limiting
                        time.sleep(0.1)
                    
                    # Get order book data
                    order_book = self.get_ticker_book(symbol)
                    
                    # Store data for this pair
                    all_pairs_data[symbol] = {
                        "symbol": symbol,
                        "ticker": ticker,
                        "current_price": current_price,
                        "klines": klines_data,
                        "order_book": order_book
                    }
                    
                    print(f"Collected data for {symbol}")
                    
                except Exception as e:
                    print(f"Error collecting data for {symbol}: {str(e)}")
                    continue
            
            return all_pairs_data
            
        except Exception as e:
            print(f"Error collecting comprehensive data: {str(e)}")
            return {}
    
    def simulate_comprehensive_data(self, max_pairs=20):
        """
        Simulate comprehensive data for USDT trading pairs when API access is limited
        
        Args:
            max_pairs: Maximum number of pairs to simulate data for
            
        Returns:
            Dictionary with simulated comprehensive data for each trading pair
        """
        try:
            # Define common USDT trading pairs
            common_pairs = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
                "XRPUSDT", "DOTUSDT", "UNIUSDT", "LTCUSDT", "LINKUSDT",
                "BCHUSDT", "XLMUSDT", "VETUSDT", "THETAUSDT", "FILUSDT",
                "TRXUSDT", "EOSUSDT", "AAVEUSDT", "NEOUSDT", "ATOMUSDT"
            ]
            
            # Limit to max_pairs
            usdt_pairs = common_pairs[:max_pairs]
            
            print(f"Simulating data for {len(usdt_pairs)} USDT trading pairs")
            
            # Create dictionary to store data
            all_pairs_data = {}
            
            # Current time
            now = datetime.now()
            
            # Process each trading pair
            for symbol in usdt_pairs:
                try:
                    # Generate base price based on symbol
                    if symbol == "BTCUSDT":
                        base_price = 50000 + random.uniform(-5000, 5000)
                    elif symbol == "ETHUSDT":
                        base_price = 3000 + random.uniform(-300, 300)
                    elif symbol == "BNBUSDT":
                        base_price = 500 + random.uniform(-50, 50)
                    else:
                        # Random price between 0.1 and 1000
                        base_price = random.uniform(0.1, 1000)
                    
                    # Generate ticker data
                    price_change = random.uniform(-5, 5)
                    price_change_percent = price_change
                    high_price = base_price * (1 + random.uniform(0, 0.05))
                    low_price = base_price * (1 - random.uniform(0, 0.05))
                    volume = random.uniform(1000, 1000000)
                    quote_volume = volume * base_price
                    
                    ticker = {
                        "symbol": symbol,
                        "priceChange": str(price_change),
                        "priceChangePercent": str(price_change_percent),
                        "weightedAvgPrice": str(base_price),
                        "prevClosePrice": str(base_price - price_change),
                        "lastPrice": str(base_price),
                        "lastQty": str(random.uniform(0.1, 10)),
                        "bidPrice": str(base_price * 0.999),
                        "bidQty": str(random.uniform(0.1, 10)),
                        "askPrice": str(base_price * 1.001),
                        "askQty": str(random.uniform(0.1, 10)),
                        "openPrice": str(base_price - price_change),
                        "highPrice": str(high_price),
                        "lowPrice": str(low_price),
                        "volume": str(volume),
                        "quoteVolume": str(quote_volume),
                        "openTime": int((now - timedelta(hours=24)).timestamp() * 1000),
                        "closeTime": int(now.timestamp() * 1000),
                        "firstId": 0,
                        "lastId": 0,
                        "count": int(random.uniform(10000, 100000))
                    }
                    
                    # Generate klines data for different intervals
                    klines_data = {}
                    for interval in ["15m", "1h", "4h", "1d"]:
                        # Determine number of candles based on interval
                        if interval == "15m":
                            num_candles = 96  # 24 hours
                            time_delta = timedelta(minutes=15)
                        elif interval == "1h":
                            num_candles = 100  # ~4 days
                            time_delta = timedelta(hours=1)
                        elif interval == "4h":
                            num_candles = 100  # ~16 days
                            time_delta = timedelta(hours=4)
                        else:  # 1d
                            num_candles = 100  # ~100 days
                            time_delta = timedelta(days=1)
                        
                        # Generate klines
                        klines = []
                        current_time = now - (num_candles * time_delta)
                        current_price = base_price * (1 - random.uniform(0, 0.2))  # Start a bit lower
                        
                        for i in range(num_candles):
                            # Generate price movement
                            price_change_pct = random.normalvariate(0, 0.02)  # Mean 0, std dev 2%
                            current_price *= (1 + price_change_pct)
                            
                            # Generate candle data
                            open_price = current_price
                            close_price = current_price * (1 + random.normalvariate(0, 0.01))
                            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.02))
                            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.02))
                            volume = random.uniform(100, 10000)
                            
                            # Create candle
                            candle = [
                                current_time,
                                open_price,
                                high_price,
                                low_price,
                                close_price,
                                volume,
                                current_time + time_delta,
                                volume * ((open_price + close_price) / 2),
                                int(random.uniform(100, 1000)),
                                volume * 0.4,
                                volume * 0.4 * ((open_price + close_price) / 2),
                                "0"
                            ]
                            
                            klines.append(candle)
                            current_time += time_delta
                            current_price = close_price
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(klines, columns=[
                            "open_time", "open", "high", "low", "close", "volume",
                            "close_time", "quote_asset_volume", "number_of_trades",
                            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                        ])
                        
                        # Convert times to datetime
                        df["open_time"] = pd.to_datetime(df["open_time"])
                        df["close_time"] = pd.to_datetime(df["close_time"])
                        
                        klines_data[interval] = df
                    
                    # Generate order book data
                    order_book = {
                        "symbol": symbol,
                        "bidPrice": str(base_price * 0.999),
                        "bidQty": str(random.uniform(0.1, 10)),
                        "askPrice": str(base_price * 1.001),
                        "askQty": str(random.uniform(0.1, 10))
                    }
                    
                    # Store data for this pair
                    all_pairs_data[symbol] = {
                        "symbol": symbol,
                        "ticker": ticker,
                        "current_price": base_price,
                        "klines": klines_data,
                        "order_book": order_book
                    }
                    
                    print(f"Simulated data for {symbol}")
                    
                except Exception as e:
                    print(f"Error simulating data for {symbol}: {str(e)}")
                    continue
            
            return all_pairs_data
            
        except Exception as e:
            print(f"Error simulating comprehensive data: {str(e)}")
            return {}
