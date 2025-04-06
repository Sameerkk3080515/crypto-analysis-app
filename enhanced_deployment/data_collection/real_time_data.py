import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import data collection and analysis modules
from data_collection.binance_data import BinanceDataCollector
from analysis.comprehensive_analysis import ComprehensiveAnalyzer

class RealTimeDataManager:
    """
    Class to manage real-time data collection and processing
    """
    
    def __init__(self, refresh_interval=300):
        """
        Initialize the real-time data manager
        
        Args:
            refresh_interval: Interval in seconds between data refreshes (default: 300 seconds / 5 minutes)
        """
        self.refresh_interval = refresh_interval
        self.data_collector = BinanceDataCollector()
        self.analyzer = ComprehensiveAnalyzer()
        self.all_pairs_data = {}
        self.analysis_results = {}
        self.recommendations = []
        self.last_update_time = None
        self.is_running = False
        self.refresh_thread = None
        self.lock = threading.Lock()
    
    def start_real_time_updates(self):
        """Start real-time data updates in a separate thread"""
        if not self.is_running:
            self.is_running = True
            self.refresh_thread = threading.Thread(target=self._refresh_loop)
            self.refresh_thread.daemon = True
            self.refresh_thread.start()
            print(f"Started real-time updates with {self.refresh_interval} second interval")
    
    def stop_real_time_updates(self):
        """Stop real-time data updates"""
        self.is_running = False
        if self.refresh_thread:
            self.refresh_thread.join(timeout=1.0)
            print("Stopped real-time updates")
    
    def _refresh_loop(self):
        """Internal method to continuously refresh data"""
        while self.is_running:
            try:
                self.refresh_data()
                time.sleep(self.refresh_interval)
            except Exception as e:
                print(f"Error in refresh loop: {str(e)}")
                time.sleep(self.refresh_interval)
    
    def refresh_data(self):
        """Refresh data and perform analysis"""
        try:
            print(f"Refreshing data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Collect data
            new_data = self.data_collector.collect_comprehensive_data(max_pairs=20)
            
            # Perform analysis
            new_analysis = self.analyzer.analyze_all_pairs(new_data)
            
            # Generate recommendations
            new_recommendations = self.generate_recommendations(new_analysis)
            
            # Update data with lock to prevent race conditions
            with self.lock:
                self.all_pairs_data = new_data
                self.analysis_results = new_analysis
                self.recommendations = new_recommendations
                self.last_update_time = datetime.now()
            
            print(f"Data refresh completed with {len(new_data)} pairs and {len(new_recommendations)} recommendations")
            return True
        except Exception as e:
            print(f"Error refreshing data: {str(e)}")
            return False
    
    def get_data(self):
        """Get the current data with lock to prevent race conditions"""
        with self.lock:
            return {
                'all_pairs_data': self.all_pairs_data,
                'analysis_results': self.analysis_results,
                'recommendations': self.recommendations,
                'last_update_time': self.last_update_time
            }
    
    def generate_recommendations(self, analysis_results):
        """
        Generate trading recommendations based on analysis results
        
        Args:
            analysis_results: Dictionary with analysis results
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        try:
            # Process each trading pair
            for symbol, pair_analysis in analysis_results['pairs'].items():
                # Skip non-USDT pairs
                if not symbol.endswith('USDT'):
                    continue
                
                # Extract current price
                current_price = pair_analysis.get('current_price', 0)
                
                # Extract technical analysis results
                if 'technical_analysis' in pair_analysis:
                    tech_analysis = pair_analysis['technical_analysis']
                    
                    # Get price action
                    if 'price_action' in tech_analysis:
                        price_action = tech_analysis['price_action']
                        short_term_trend = price_action.get('short_term_trend', 'neutral')
                        medium_term_trend = price_action.get('medium_term_trend', 'neutral')
                        momentum = price_action.get('momentum', 'neutral')
                        volatility = price_action.get('volatility', 'low')
                        macd_signal = price_action.get('macd_signal', 'neutral')
                        trend_strength = price_action.get('trend_strength', 'weak')
                    else:
                        short_term_trend = 'neutral'
                        medium_term_trend = 'neutral'
                        momentum = 'neutral'
                        volatility = 'low'
                        macd_signal = 'neutral'
                        trend_strength = 'weak'
                    
                    # Get support and resistance
                    if 'support_resistance' in tech_analysis:
                        support_resistance = tech_analysis['support_resistance']
                        closest_support = support_resistance.get('closest_support', current_price * 0.95)
                        closest_resistance = support_resistance.get('closest_resistance', current_price * 1.05)
                    else:
                        closest_support = current_price * 0.95
                        closest_resistance = current_price * 1.05
                else:
                    short_term_trend = 'neutral'
                    medium_term_trend = 'neutral'
                    momentum = 'neutral'
                    volatility = 'low'
                    macd_signal = 'neutral'
                    trend_strength = 'weak'
                    closest_support = current_price * 0.95
                    closest_resistance = current_price * 1.05
                
                # Extract sentiment analysis results
                if 'sentiment_analysis' in pair_analysis:
                    sentiment = pair_analysis['sentiment_analysis']
                    overall_sentiment = sentiment.get('overall_category', 'neutral')
                    combined_sentiment_score = sentiment.get('combined_sentiment', 0) * 100  # Convert to 0-100 scale
                else:
                    overall_sentiment = 'neutral'
                    combined_sentiment_score = 50
                
                # Calculate confidence score (0-100)
                # Combine technical and sentiment signals
                confidence_factors = []
                
                # Technical factors
                if short_term_trend == 'bullish':
                    confidence_factors.append(70)
                elif short_term_trend == 'bearish':
                    confidence_factors.append(30)
                else:
                    confidence_factors.append(50)
                
                if medium_term_trend == 'bullish':
                    confidence_factors.append(65)
                elif medium_term_trend == 'bearish':
                    confidence_factors.append(35)
                else:
                    confidence_factors.append(50)
                
                if momentum == 'overbought':
                    confidence_factors.append(30)  # Overbought is bearish
                elif momentum == 'oversold':
                    confidence_factors.append(70)  # Oversold is bullish
                else:
                    confidence_factors.append(50)
                
                if macd_signal == 'bullish':
                    confidence_factors.append(70)
                elif macd_signal == 'bearish':
                    confidence_factors.append(30)
                else:
                    confidence_factors.append(50)
                
                if trend_strength == 'strong':
                    confidence_factors.append(65)
                else:
                    confidence_factors.append(50)
                
                # Sentiment factors
                if overall_sentiment == 'very positive':
                    confidence_factors.append(80)
                elif overall_sentiment == 'positive':
                    confidence_factors.append(65)
                elif overall_sentiment == 'neutral':
                    confidence_factors.append(50)
                elif overall_sentiment == 'negative':
                    confidence_factors.append(35)
                elif overall_sentiment == 'very negative':
                    confidence_factors.append(20)
                
                # Calculate average confidence
                confidence_score = sum(confidence_factors) / len(confidence_factors)
                
                # Determine action (buy, sell, hold)
                if confidence_score >= 60:
                    action = 'buy'
                elif confidence_score <= 40:
                    action = 'sell'
                else:
                    action = 'hold'
                
                # Calculate risk score (0-100)
                # Higher volatility and stronger trends increase risk
                risk_factors = []
                
                if volatility == 'high':
                    risk_factors.append(80)
                elif volatility == 'medium':
                    risk_factors.append(50)
                else:
                    risk_factors.append(30)
                
                if trend_strength == 'strong':
                    risk_factors.append(70)
                else:
                    risk_factors.append(40)
                
                # Price relative to support/resistance
                price_position = (current_price - closest_support) / (closest_resistance - closest_support)
                if price_position > 0.8:
                    risk_factors.append(80)  # High risk near resistance
                elif price_position < 0.2:
                    risk_factors.append(30)  # Lower risk near support
                else:
                    risk_factors.append(50)
                
                # Calculate average risk
                risk_score = sum(risk_factors) / len(risk_factors)
                
                # Generate time horizon (15min to 12hr)
                # Base on volatility and trend strength
                if volatility == 'high' and trend_strength == 'strong':
                    time_horizon_hours = np.random.uniform(0.25, 2)  # 15min to 2hr
                elif volatility == 'high':
                    time_horizon_hours = np.random.uniform(0.5, 4)  # 30min to 4hr
                elif trend_strength == 'strong':
                    time_horizon_hours = np.random.uniform(1, 6)  # 1hr to 6hr
                else:
                    time_horizon_hours = np.random.uniform(2, 12)  # 2hr to 12hr
                
                # Format time horizon
                if time_horizon_hours < 1:
                    time_horizon = f"{int(time_horizon_hours * 60)} minutes"
                else:
                    time_horizon = f"{time_horizon_hours:.1f} hours"
                
                # Generate selling time (UTC+4)
                current_time = datetime.now()
                selling_time = current_time + timedelta(hours=time_horizon_hours)
                selling_time_str = selling_time.strftime('%I:%M %p')  # 12-hour format
                
                # Generate selling price
                if action == 'buy':
                    # For buy, selling price should be higher
                    price_change_pct = np.random.uniform(0.5, 5.0) / 100  # 0.5% to 5% increase
                    selling_price = current_price * (1 + price_change_pct)
                elif action == 'sell':
                    # For sell, current price is the selling price
                    selling_price = current_price
                else:
                    # For hold, selling price is slightly higher
                    price_change_pct = np.random.uniform(0.1, 1.0) / 100  # 0.1% to 1% increase
                    selling_price = current_price * (1 + price_change_pct)
                
                # Format selling price
                if selling_price < 0.01:
                    selling_price_str = f"${selling_price:.6f}"
                elif selling_price < 1:
                    selling_price_str = f"${selling_price:.4f}"
                elif selling_price < 100:
                    selling_price_str = f"${selling_price:.2f}"
                else:
                    selling_price_str = f"${selling_price:.2f}"
                
                # Create recommendation object
                recommendation = {
                    'symbol': symbol,
                    'action': action,
                    'current_price': current_price,
                    'confidence_score': confidence_score,
                    'risk_score': risk_score,
                    'time_horizon': time_horizon,
                    'selling_time': selling_time_str,
                    'selling_time_datetime': selling_time,
                    'selling_price': selling_price_str,
                    'selling_price_value': selling_price,
                    'short_term_trend': short_term_trend,
                    'medium_term_trend': medium_term_trend,
                    'momentum': momentum,
                    'volatility': volatility,
                    'macd_signal': macd_signal,
                    'trend_strength': trend_strength,
                    'overall_sentiment': overall_sentiment,
                    'closest_support': closest_support,
                    'closest_resistance': closest_resistance
                }
                
                recommendations.append(recommendation)
            
            # Sort recommendations by confidence score (descending)
            recommendations.sort(key=lambda x: x['confidence_score'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return []
    
    def generate_projected_data(self, historical_data, symbol):
        """
        Generate projected price data based on analysis results
        
        Args:
            historical_data: DataFrame with historical price data
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with projected price data
        """
        try:
            # Get the last available price and time
            last_price = historical_data['close'].iloc[-1]
            last_time = historical_data['open_time'].iloc[-1]
            
            # Generate future time points (12 hours ahead in 15-minute intervals)
            future_times = [last_time + timedelta(minutes=15*i) for i in range(1, 49)]
            
            # Extract relevant analysis results
            with self.lock:
                analysis_results = self.analysis_results
            
            if 'pairs' in analysis_results and symbol in analysis_results['pairs']:
                pair_analysis = analysis_results['pairs'][symbol]
                
                # Get technical analysis results
                if 'technical_analysis' in pair_analysis and 'price_action' in pair_analysis['technical_analysis']:
                    price_action = pair_analysis['technical_analysis']['price_action']
                    short_term_trend = price_action.get('short_term_trend', 'neutral')
                    momentum = price_action.get('momentum', 'neutral')
                    volatility = price_action.get('volatility', 'low')
                else:
                    short_term_trend = 'neutral'
                    momentum = 'neutral'
                    volatility = 'low'
                
                # Get sentiment analysis results
                if 'sentiment_analysis' in pair_analysis:
                    sentiment = pair_analysis['sentiment_analysis']
                    overall_sentiment = sentiment.get('overall_category', 'neutral')
                else:
                    overall_sentiment = 'neutral'
                
                # Determine trend direction and strength
                trend_direction = 1  # Default to slightly upward
                
                if short_term_trend == 'bullish' and overall_sentiment in ['positive', 'very positive']:
                    trend_direction = 2  # Strong upward
                elif short_term_trend == 'bearish' and overall_sentiment in ['negative', 'very negative']:
                    trend_direction = -2  # Strong downward
                elif short_term_trend == 'bullish' or overall_sentiment in ['positive', 'very positive']:
                    trend_direction = 1  # Moderate upward
                elif short_term_trend == 'bearish' or overall_sentiment in ['negative', 'very negative']:
                    trend_direction = -1  # Moderate downward
                
                # Determine volatility factor
                if volatility == 'high':
                    volatility_factor = 0.03
                elif volatility == 'medium':
                    volatility_factor = 0.015
                else:
                    volatility_factor = 0.008
                
                # Generate projected prices
                projected_prices = []
                current_price = last_price
                
                for i in range(len(future_times)):
                    # Calculate price change with some randomness
                    random_factor = np.random.normal(0, 1)
                    price_change = (trend_direction * 0.001 + random_factor * volatility_factor) * current_price
                    current_price += price_change
                    projected_prices.append(current_price)
                
                # Find matching recommendation for this symbol
                matching_recommendation = None
                with self.lock:
                    for rec in self.recommendations:
                        if rec['symbol'] == symbol:
                            matching_recommendation = rec
                            break
                
                # Determine selling time and price
                if matching_recommendation:
                    selling_time = matching_recommendation['selling_time_datetime']
                    selling_price = matching_recommendation['selling_price_value']
                else:
                    # For demonstration, we'll pick a point where the price is optimal based on the trend
                    if trend_direction > 0:
                        # For upward trend, find the highest price point
                        best_idx = projected_prices.index(max(projected_prices))
                    else:
                        # For downward trend, sell early
                        best_idx = min(8, len(projected_prices) - 1)
                    
                    selling_time = future_times[best_idx]
                    selling_price = projected_prices[best_idx]
                
                # Convert to DataFrame format
                projected_data = {
                    'time': future_times,
                    'price': projected_prices,
                    'selling_time': selling_time,
                    'selling_price': selling_price
                }
                
                return projected_data
                
            else:
                return None
                
        except Exception as e:
            print(f"Error generating projected data: {str(e)}")
            return None
    
    def save_data_to_cache(self, cache_file='data_cache.json'):
        """
        Save current data to cache file
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            Boolean indicating success
        """
        try:
            with self.lock:
                # Create a serializable version of the data
                cache_data = {
                    'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
                    'recommendations': self.recommendations
                }
            
            # Save to file
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            print(f"Saved data to cache file: {cache_file}")
            return True
        except Exception as e:
            print(f"Error saving data to cache: {str(e)}")
            return False
    
    def load_data_from_cache(self, cache_file='data_cache.json'):
        """
        Load data from cache file
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            Boolean indicating success
        """
        try:
            if not os.path.exists(cache_file):
                print(f"Cache file not found: {cache_file}")
                return False
            
            # Load from file
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            with self.lock:
                # Parse the data
                if 'last_update_time' in cache_data and cache_data['last_update_time']:
                    self.last_update_time = datetime.fromisoformat(cache_data['last_update_time'])
                
                if 'recommendations' in cache_data:
                    self.recommendations = cache_data['recommendations']
            
            print(f"Loaded data from cache file: {cache_file}")
            return True
        except Exception as e:
            print(f"Error loading data from cache: {str(e)}")
            return False
