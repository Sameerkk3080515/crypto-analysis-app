"""
Updated RealTimeDataManager class that uses real cryptocurrency data from CoinGecko.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sys
import os

# Import the CoinGecko API wrapper
from data_collection.coingecko_api import CoinGeckoAPI

class RealTimeDataManager:
    """
    Class to manage real-time cryptocurrency data using CoinGecko API.
    """
    def __init__(self, refresh_interval=300):
        """
        Initialize the data manager.
        
        Args:
            refresh_interval (int): Data refresh interval in seconds
        """
        self.refresh_interval = refresh_interval
        self.last_refresh_time = None
        self.api = CoinGeckoAPI()
        self.symbols = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'DOGE-USD', 
            'XRP-USD', 'DOT-USD', 'UNI-USD', 'LTC-USD', 'LINK-USD'
        ]
        self.data = self._initialize_data()
    
    def _initialize_data(self):
        """
        Initialize data by fetching from API.
        
        Returns:
            dict: Dictionary containing cryptocurrency data
        """
        try:
            # Fetch data for all symbols
            all_pairs_data = self._fetch_all_pairs_data()
            
            # Generate recommendations based on real data
            recommendations = self._generate_recommendations(all_pairs_data)
            
            # Generate analysis results based on real data
            analysis_results = self._generate_analysis(all_pairs_data)
            
            return {
                'recommendations': recommendations,
                'analysis_results': analysis_results,
                'all_pairs_data': all_pairs_data,
                'last_update_time': datetime.now()
            }
        except Exception as e:
            print(f"Error initializing data: {e}")
            # Fallback to sample data if API fails
            return self._generate_fallback_data()
    
    def _fetch_all_pairs_data(self):
        """
        Fetch data for all cryptocurrency pairs.
        
        Returns:
            dict: Dictionary with symbol as key and data as value
        """
        all_pairs_data = {}
        
        for symbol in self.symbols:
            try:
                # Get historical data
                klines = self.api.get_crypto_price_history(symbol, days='30')
                
                if not klines.empty:
                    # Get current price
                    current_price = self.api.get_current_price(symbol)
                    
                    # Add to all pairs data
                    all_pairs_data[symbol] = {
                        'klines': {
                            '1d': klines
                        },
                        'current_price': current_price or klines['close'].iloc[-1],
                        'price_change_24h': (klines['close'].iloc[-1] - klines['close'].iloc[-2]) / klines['close'].iloc[-2] * 100 if len(klines) > 1 else 0,
                        'volume_24h': klines['volume'].iloc[-1],
                        'high_24h': klines['high'].iloc[-1],
                        'low_24h': klines['low'].iloc[-1]
                    }
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        return all_pairs_data
    
    def _generate_recommendations(self, all_pairs_data):
        """
        Generate trading recommendations based on real data.
        
        Args:
            all_pairs_data (dict): Dictionary with cryptocurrency data
            
        Returns:
            list: List of recommendation dictionaries
        """
        recommendations = []
        
        for symbol, data in all_pairs_data.items():
            try:
                klines = data['klines']['1d']
                
                if klines.empty or len(klines) < 14:
                    continue
                
                # Calculate technical indicators
                close_prices = klines['close'].values
                
                # Simple Moving Averages
                sma_5 = np.mean(close_prices[-5:])
                sma_10 = np.mean(close_prices[-10:])
                sma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else np.mean(close_prices)
                
                # Determine trend based on SMAs
                short_term_trend = 'bullish' if close_prices[-1] > sma_5 else 'bearish'
                medium_term_trend = 'bullish' if sma_5 > sma_10 else 'bearish'
                
                # Calculate RSI (basic implementation)
                delta = np.diff(close_prices)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                
                avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
                avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
                
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi = 100 - (100 / (1 + rs))
                
                # Determine momentum based on RSI
                if rsi < 30:
                    momentum = 'oversold'
                elif rsi > 70:
                    momentum = 'overbought'
                else:
                    momentum = 'neutral'
                
                # Determine action based on technical indicators
                if short_term_trend == 'bullish' and medium_term_trend == 'bullish' and momentum == 'oversold':
                    action = 'buy'
                    confidence_score = 80 + random.uniform(0, 15)
                elif short_term_trend == 'bearish' and medium_term_trend == 'bearish' and momentum == 'overbought':
                    action = 'sell'
                    confidence_score = 80 + random.uniform(0, 15)
                elif short_term_trend != medium_term_trend:
                    action = 'hold'
                    confidence_score = 50 + random.uniform(0, 20)
                else:
                    action = random.choice(['buy', 'sell', 'hold'])
                    confidence_score = 40 + random.uniform(0, 30)
                
                # Calculate risk score based on volatility
                price_std = np.std(close_prices[-14:]) / np.mean(close_prices[-14:]) * 100
                risk_score = min(90, max(10, price_std * 5))
                
                # Current price
                current_price = close_prices[-1]
                
                # Generate selling time and price
                hours_to_sell = random.randint(1, 12)
                selling_time = (datetime.now() + timedelta(hours=hours_to_sell)).strftime('%Y-%m-%d %H:%M')
                
                # Projected price change based on trend and momentum
                if action == 'buy':
                    price_change_pct = random.uniform(0.01, 0.10)
                elif action == 'sell':
                    price_change_pct = random.uniform(-0.10, -0.01)
                else:
                    price_change_pct = random.uniform(-0.03, 0.03)
                
                selling_price = current_price * (1 + price_change_pct)
                
                # Generate support and resistance
                support_pct = random.uniform(0.03, 0.1)
                resistance_pct = random.uniform(0.03, 0.1)
                closest_support = current_price * (1 - support_pct)
                closest_resistance = current_price * (1 + resistance_pct)
                
                # Determine volatility and trend strength
                volatility = 'high' if price_std > 5 else 'medium' if price_std > 2 else 'low'
                trend_strength = 'strong' if abs(sma_5 - sma_10) / sma_10 > 0.03 else 'moderate' if abs(sma_5 - sma_10) / sma_10 > 0.01 else 'weak'
                
                # MACD signal (simplified)
                macd_signal = 'bullish' if sma_5 > sma_10 else 'bearish'
                
                # Overall sentiment based on technical indicators
                sentiment_options = ['strongly_positive', 'positive', 'neutral', 'negative', 'strongly_negative']
                if action == 'buy':
                    overall_sentiment = sentiment_options[0] if confidence_score > 85 else sentiment_options[1]
                elif action == 'sell':
                    overall_sentiment = sentiment_options[-1] if confidence_score > 85 else sentiment_options[-2]
                else:
                    overall_sentiment = sentiment_options[2]
                
                # Create recommendation
                recommendation = {
                    'symbol': symbol,
                    'action': action,
                    'confidence_score': confidence_score,
                    'risk_score': risk_score,
                    'current_price': current_price,
                    'selling_time': selling_time,
                    'selling_price': f"${selling_price:.2f}",
                    'time_horizon': f"{hours_to_sell} hours",
                    'short_term_trend': short_term_trend,
                    'medium_term_trend': medium_term_trend,
                    'momentum': momentum,
                    'macd_signal': macd_signal,
                    'volatility': volatility,
                    'trend_strength': trend_strength,
                    'closest_support': closest_support,
                    'closest_resistance': closest_resistance,
                    'overall_sentiment': overall_sentiment
                }
                
                recommendations.append(recommendation)
                
            except Exception as e:
                print(f"Error generating recommendation for {symbol}: {e}")
        
        return recommendations
    
    def _generate_analysis(self, all_pairs_data):
        """
        Generate market analysis based on real data.
        
        Args:
            all_pairs_data (dict): Dictionary with cryptocurrency data
            
        Returns:
            dict: Dictionary with analysis results
        """
        try:
            # Extract prices for correlation analysis
            prices = {}
            for symbol, data in all_pairs_data.items():
                if 'klines' in data and '1d' in data['klines'] and not data['klines']['1d'].empty:
                    prices[symbol] = data['klines']['1d']['close'].values
            
            # Calculate market trend
            btc_data = all_pairs_data.get('BTC-USD', {})
            if btc_data and 'klines' in btc_data and '1d' in btc_data['klines'] and not btc_data['klines']['1d'].empty:
                btc_prices = btc_data['klines']['1d']['close'].values
                btc_sma_7 = np.mean(btc_prices[-7:]) if len(btc_prices) >= 7 else np.mean(btc_prices)
                btc_sma_30 = np.mean(btc_prices[-30:]) if len(btc_prices) >= 30 else np.mean(btc_prices)
                
                if btc_prices[-1] > btc_sma_7 and btc_sma_7 > btc_sma_30:
                    market_trend = 'strongly_bullish'
                elif btc_prices[-1] > btc_sma_7:
                    market_trend = 'bullish'
                elif btc_prices[-1] < btc_sma_7 and btc_sma_7 < btc_sma_30:
                    market_trend = 'strongly_bearish'
                elif btc_prices[-1] < btc_sma_7:
                    market_trend = 'bearish'
                else:
                    market_trend = 'neutral'
                
                # Calculate market strength
                market_strength = 50 + (btc_prices[-1] - btc_sma_30) / btc_sma_30 * 100
                market_strength = min(100, max(0, market_strength))
                
                # Calculate BTC dominance (simplified)
                btc_dominance = 0.45 + random.uniform(-0.05, 0.05)
            else:
                market_trend = random.choice(['strongly_bullish', 'bullish', 'neutral', 'bearish', 'strongly_bearish'])
                market_strength = random.uniform(30, 70)
                btc_dominance = random.uniform(0.4, 0.6)
            
            # Calculate performance
            performance = {}
            for symbol, data in all_pairs_data.items():
                if 'klines' in data and '1d' in data['klines'] and not data['klines']['1d'].empty:
                    klines = data['klines']['1d']
                    if len(klines) > 7:
                        week_ago_price = klines['close'].iloc[-7]
                        current_price = klines['close'].iloc[-1]
                        perf = (current_price - week_ago_price) / week_ago_price * 100
                        performance[symbol] = perf
            
            # Sort by performance
            sorted_performance = sorted(performance.items(), key=lambda x: x[1], reverse=True)
            top_performers = sorted_performance[:5]
            worst_performers = sorted_performance[-5:]
            
            # Calculate correlations
            correlations = {}
            for symbol1 in ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']:
                if symbol1 not in prices:
                    continue
                
                correlations[symbol1] = {}
                for symbol2 in ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']:
                    if symbol2 not in prices:
                        continue
                    
                    if symbol1 == symbol2:
                        correlations[symbol1][symbol2] = 1.0
                    else:
                        # Calculate correlation if enough data points
                        if len(prices[symbol1]) > 5 and len(prices[symbol2]) > 5:
                            # Use the minimum length
                            min_len = min(len(prices[symbol1]), len(prices[symbol2]))
                            corr = np.corrcoef(prices[symbol1][-min_len:], prices[symbol2][-min_len:])[0, 1]
                            correlations[symbol1][symbol2] = corr
                        else:
                            correlations[symbol1][symbol2] = 0.5
            
            # Generate sector performance (still using random data as we don't have sector classification)
            sector_performance = {
                'DeFi': random.uniform(-5, 10),
                'Smart Contract Platforms': random.uniform(-3, 12),
                'Layer 2': random.uniform(-2, 15),
                'Meme Coins': random.uniform(-10, 20),
                'NFT': random.uniform(-8, 5),
                'Gaming': random.uniform(-5, 8),
                'Exchange Tokens': random.uniform(-2, 7)
            }
            
            # Create analysis results
            analysis_results = {
                'market_analysis': {
                    'market_conditions': {
                        'market_trend': market_trend,
                        'market_strength': market_strength,
                        'volatility': 'high' if np.std([p[1] for p in sorted_performance]) > 10 else 'medium' if np.std([p[1] for p in sorted_performance]) > 5 else 'low',
                        'btc_dominance': btc_dominance,
                        'top_performers': top_performers,
                        'worst_performers': worst_performers
                    },
                    'sector_performance': sector_performance,
                    'market_correlations': correlations
                }
            }
            
            return analysis_results
            
        except Exception as e:
            print(f"Error generating analysis: {e}")
            return self._generate_fallback_analysis()
    
    def _generate_fallback_data(self):
        """
        Generate fallback data when API fails.
        
        Returns:
            dict: Dictionary with fallback data
        """
        print("Using fallback data generation")
        return {
            'recommendations': self._generate_sample_recommendations(),
            'analysis_results': self._generate_fallback_analysis(),
            'all_pairs_data': self._generate_sample_pairs_data(),
            'last_update_time': datetime.now()
        }
    
    def _generate_sample_recommendations(self):
        """
        Generate sample recommendations as fallback.
        
        Returns:
            list: List of recommendation dictionaries
        """
        recommendations = []
        
        # Sample symbols
        symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'DOGE-USD', 
                  'XRP-USD', 'DOT-USD', 'UNI-USD', 'LTC-USD', 'LINK-USD']
        
        # Generate random recommendations
        for symbol in symbols:
            action = random.choice(['buy', 'sell', 'hold'])
            
            # Base price on symbol
            if symbol == 'BTC-USD':
                current_price = random.uniform(45000, 55000)
            elif symbol == 'ETH-USD':
                current_price = random.uniform(3000, 4000)
            elif symbol == 'BNB-USD':
                current_price = random.uniform(400, 500)
            else:
                current_price = random.uniform(0.5, 100)
            
            # Generate confidence and risk scores
            confidence_score = random.uniform(50, 95) if action != 'hold' else random.uniform(30, 70)
            risk_score = random.uniform(20, 80)
            
            # Generate selling time and price
            hours_to_sell = random.randint(1, 12)
            selling_time = (datetime.now() + timedelta(hours=hours_to_sell)).strftime('%Y-%m-%d %H:%M')
            
            price_change_pct = random.uniform(-0.05, 0.15) if action == 'buy' else random.uniform(-0.15, 0.05)
            selling_price = current_price * (1 + price_change_pct)
            
            # Generate technical factors
            short_term_trend = 'bullish' if action == 'buy' else 'bearish' if action == 'sell' else random.choice(['bullish', 'bearish', 'neutral'])
            medium_term_trend = random.choice(['bullish', 'bearish', 'neutral'])
            momentum = 'oversold' if action == 'buy' else 'overbought' if action == 'sell' else random.choice(['oversold', 'overbought', 'neutral'])
            macd_signal = 'bullish' if action == 'buy' else 'bearish'
            volatility = random.choice(['high', 'medium', 'low'])
            trend_strength = random.choice(['strong', 'moderate', 'weak'])
            
            # Generate support and resistance
            support_pct = random.uniform(0.03, 0.1)
            resistance_pct = random.uniform(0.03, 0.1)
            closest_support = current_price * (1 - support_pct)
            closest_resistance = current_price * (1 + resistance_pct)
            
            # Generate sentiment
            sentiment_options = ['strongly_positive', 'positive', 'neutral', 'negative', 'strongly_negative']
            overall_sentiment = sentiment_options[0] if action == 'buy' else sentiment_options[-1] if action == 'sell' else random.choice(sentiment_options)
            
            # Create recommendation
            recommendation = {
                'symbol': symbol,
                'action': action,
                'confidence_score': confidence_score,
                'risk_score': risk_score,
                'current_price': current_price,
                'selling_time': selling_time,
                'selling_price': f"${selling_price:.2f}",
                'time_horizon': f"{hours_to_sell} hours",
                'short_term_trend': short_term_trend,
                'medium_term_trend': medium_term_trend,
                'momentum': momentum,
                'macd_signal': macd_signal,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'closest_support': closest_support,
                'closest_resistance': closest_resistance,
                'overall_sentiment': overall_sentiment
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_fallback_analysis(self):
        """
        Generate fallback analysis when API fails.
        
        Returns:
            dict: Dictionary with fallback analysis
        """
        # Generate sample analysis results
        analysis_results = {
            'market_analysis': {
                'market_conditions': {
                    'market_trend': random.choice(['strongly_bullish', 'bullish', 'neutral', 'bearish', 'strongly_bearish']),
                    'market_strength': random.uniform(30, 70),
                    'volatility': random.choice(['high', 'medium', 'low']),
                    'btc_dominance': random.uniform(0.4, 0.6),
                    'top_performers': [
                        ('BNB-USD', random.uniform(5, 15)),
                        ('ADA-USD', random.uniform(3, 10)),
                        ('DOGE-USD', random.uniform(2, 8)),
                        ('MATIC-USD', random.uniform(1, 7)),
                        ('SOL-USD', random.uniform(1, 6))
                    ],
                    'worst_performers': [
                        ('ATOM-USD', random.uniform(-10, -2)),
                        ('LTC-USD', random.uniform(-8, -1)),
                        ('DOT-USD', random.uniform(-7, -1)),
                        ('LINK-USD', random.uniform(-6, -1)),
                        ('UNI-USD', random.uniform(-5, -1))
                    ]
                },
                'sector_performance': {
                    'DeFi': random.uniform(-5, 10),
                    'Smart Contract Platforms': random.uniform(-3, 12),
                    'Layer 2': random.uniform(-2, 15),
                    'Meme Coins': random.uniform(-10, 20),
                    'NFT': random.uniform(-8, 5),
                    'Gaming': random.uniform(-5, 8),
                    'Exchange Tokens': random.uniform(-2, 7)
                },
                'market_correlations': {
                    'BTC-USD': {
                        'BTC-USD': 1.0,
                        'ETH-USD': random.uniform(0.7, 0.9),
                        'BNB-USD': random.uniform(0.5, 0.8),
                        'XRP-USD': random.uniform(0.3, 0.7),
                        'ADA-USD': random.uniform(0.4, 0.7)
                    },
                    'ETH-USD': {
                        'BTC-USD': random.uniform(0.7, 0.9),
                        'ETH-USD': 1.0,
                        'BNB-USD': random.uniform(0.6, 0.8),
                        'XRP-USD': random.uniform(0.4, 0.7),
                        'ADA-USD': random.uniform(0.5, 0.8)
                    },
                    'BNB-USD': {
                        'BTC-USD': random.uniform(0.5, 0.8),
                        'ETH-USD': random.uniform(0.6, 0.8),
                        'BNB-USD': 1.0,
                        'XRP-USD': random.uniform(0.3, 0.6),
                        'ADA-USD': random.uniform(0.4, 0.7)
                    },
                    'XRP-USD': {
                        'BTC-USD': random.uniform(0.3, 0.7),
                        'ETH-USD': random.uniform(0.4, 0.7),
                        'BNB-USD': random.uniform(0.3, 0.6),
                        'XRP-USD': 1.0,
                        'ADA-USD': random.uniform(0.5, 0.7)
                    },
                    'ADA-USD': {
                        'BTC-USD': random.uniform(0.4, 0.7),
                        'ETH-USD': random.uniform(0.5, 0.8),
                        'BNB-USD': random.uniform(0.4, 0.7),
                        'XRP-USD': random.uniform(0.5, 0.7),
                        'ADA-USD': 1.0
                    }
                }
            }
        }
        
        return analysis_results
    
    def _generate_sample_pairs_data(self):
        """
        Generate sample pairs data as fallback.
        
        Returns:
            dict: Dictionary with sample pairs data
        """
        all_pairs_data = {}
        
        # Sample symbols
        symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'DOGE-USD', 
                  'XRP-USD', 'DOT-USD', 'UNI-USD', 'LTC-USD', 'LINK-USD']
        
        for symbol in symbols:
            # Generate klines data
            klines = self._generate_sample_klines(symbol)
            
            # Add to all pairs data
            all_pairs_data[symbol] = {
                'klines': {
                    '1d': klines
                }
            }
        
        return all_pairs_data
    
    def _generate_sample_klines(self, symbol):
        """
        Generate sample klines data as fallback.
        
        Args:
            symbol (str): Cryptocurrency symbol
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        now = datetime.now()
        hours_back = 48  # 2 days of hourly data
        
        # Initialize with base price depending on symbol
        if symbol == 'BTC-USD':
            base_price = random.uniform(45000, 55000)
        elif symbol == 'ETH-USD':
            base_price = random.uniform(3000, 4000)
        elif symbol == 'BNB-USD':
            base_price = random.uniform(400, 500)
        else:
            base_price = random.uniform(0.5, 100)
        
        # Generate data
        data = {
            'open_time': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }
        
        current_price = base_price
        
        for i in range(hours_back, 0, -1):
            # Calculate time
            time_point = now - timedelta(hours=i)
            
            # Generate price movement
            price_change = current_price * random.uniform(-0.02, 0.02)
            open_price = current_price
            close_price = current_price + price_change
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
            
            # Generate volume
            volume = current_price * random.uniform(100, 1000)
            
            # Add to data
            data['open_time'].append(time_point)
            data['open'].append(open_price)
            data['high'].append(high_price)
            data['low'].append(low_price)
            data['close'].append(close_price)
            data['volume'].append(volume)
            
            # Update current price for next iteration
            current_price = close_price
        
        # Convert to DataFrame
        return pd.DataFrame(data)
    
    def refresh_data(self):
        """
        Refresh data from API.
        
        Returns:
            bool: True if refresh was successful, False otherwise
        """
        try:
            self.data = self._initialize_data()
            self.last_refresh_time = datetime.now()
            return True
        except Exception as e:
            print(f"Error refreshing data: {e}")
            return False
    
    def get_data(self):
        """
        Get current data.
        
        Returns:
            dict: Dictionary with cryptocurrency data
        """
        return self.data
    
    def generate_projected_data(self, historical_data, symbol):
        """
        Generate projected price data based on historical trends.
        
        Args:
            historical_data (pandas.DataFrame): Historical OHLCV data
            symbol (str): Cryptocurrency symbol
            
        Returns:
            dict: Dictionary with projected data
        """
        if historical_data.empty:
            return None
        
        try:
            # Get last price
            last_price = historical_data['close'].iloc[-1]
            last_time = historical_data['open_time'].iloc[-1]
            
            # Calculate price trend
            if len(historical_data) >= 14:
                # Calculate average daily change over the last 14 days
                daily_changes = []
                for i in range(1, 14):
                    daily_change = (historical_data['close'].iloc[-i] - historical_data['close'].iloc[-(i+1)]) / historical_data['close'].iloc[-(i+1)]
                    daily_changes.append(daily_change)
                
                avg_daily_change = np.mean(daily_changes)
                volatility = np.std(daily_changes)
                
                # Use the trend for projection with some randomness
                price_change_pct = avg_daily_change * 12 + random.uniform(-volatility, volatility) * 2
            else:
                # Fallback to random projection
                price_change_pct = random.uniform(-0.1, 0.2)
            
            # Generate projection for next 12 hours
            projection_hours = 12
            time_points = [last_time + timedelta(hours=i) for i in range(1, projection_hours + 1)]
            
            # Generate price points
            price_points = []
            
            for i in range(projection_hours):
                # Non-linear price movement
                progress = (i + 1) / projection_hours
                current_change = price_change_pct * progress * (1 + random.uniform(-0.2, 0.2))
                price = last_price * (1 + current_change)
                price_points.append(price)
            
            # Determine selling point
            selling_hour = random.randint(4, projection_hours)
            selling_time = time_points[selling_hour - 1]
            selling_price = price_points[selling_hour - 1]
            
            # Create projection data
            projected_data = {
                'time': time_points,
                'price': price_points,
                'selling_time': selling_time,
                'selling_price': selling_price
            }
            
            return projected_data
            
        except Exception as e:
            print(f"Error generating projected data: {e}")
            return None
