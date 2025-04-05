"""
Technical analysis module for the Cryptocurrency Analysis Bot.

This module implements various technical analysis indicators and algorithms
to analyze cryptocurrency market data and identify trading signals.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

class TechnicalAnalyzer:
    """Performs technical analysis on cryptocurrency market data."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the technical analyzer.
        
        Args:
            config: Configuration object (creates a new one if None)
        """
        self.config = config or Config()
    
    def prepare_dataframe(self, market_data: Dict[str, Any], symbol: str) -> pd.DataFrame:
        """
        Prepare a DataFrame from market data for technical analysis.
        
        Args:
            market_data: Market data dictionary from MarketDataCollector
            symbol: Cryptocurrency symbol to analyze
            
        Returns:
            DataFrame with OHLCV data ready for analysis
        """
        if symbol not in market_data.get('historical_data', {}):
            print(f"No historical data available for {symbol}")
            return pd.DataFrame()
        
        # Convert historical data to DataFrame
        historical_data = market_data['historical_data'][symbol]
        
        if isinstance(historical_data, list):
            df = pd.DataFrame(historical_data)
        else:
            # If it's already a DataFrame
            df = historical_data
        
        # Ensure required columns exist
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns in historical data for {symbol}")
            return pd.DataFrame()
        
        # Convert timestamps if they're strings
        if isinstance(df['open_time'][0], str):
            df['open_time'] = pd.to_datetime(df['open_time'])
        
        # Sort by time
        df = df.sort_values('open_time')
        
        # Set index to timestamp
        df = df.set_index('open_time')
        
        return df
    
    def calculate_sma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages.
        
        Args:
            df: DataFrame with price data
            periods: List of periods for SMA calculation
            
        Returns:
            DataFrame with added SMA columns
        """
        result = df.copy()
        
        for period in periods:
            result[f'sma_{period}'] = result['close'].rolling(window=period).mean()
        
        return result
    
    def calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages.
        
        Args:
            df: DataFrame with price data
            periods: List of periods for EMA calculation
            
        Returns:
            DataFrame with added EMA columns
        """
        result = df.copy()
        
        for period in periods:
            result[f'ema_{period}'] = result['close'].ewm(span=period, adjust=False).mean()
        
        return result
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.
        
        Args:
            df: DataFrame with price data
            period: Period for RSI calculation
            
        Returns:
            DataFrame with added RSI column
        """
        result = df.copy()
        
        # Calculate price changes
        delta = result['close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return result
    
    def calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence.
        
        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            DataFrame with added MACD columns
        """
        result = df.copy()
        
        # Calculate fast and slow EMAs
        fast_ema = result['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = result['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        result['macd_line'] = fast_ema - slow_ema
        result['macd_signal'] = result['macd_line'].ewm(span=signal_period, adjust=False).mean()
        result['macd_histogram'] = result['macd_line'] - result['macd_signal']
        
        return result
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, 
                                std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with price data
            period: Period for moving average
            std_dev: Number of standard deviations for bands
            
        Returns:
            DataFrame with added Bollinger Bands columns
        """
        result = df.copy()
        
        # Calculate middle band (SMA)
        result['bb_middle'] = result['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        result['bb_std'] = result['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        result['bb_upper'] = result['bb_middle'] + (result['bb_std'] * std_dev)
        result['bb_lower'] = result['bb_middle'] - (result['bb_std'] * std_dev)
        
        return result
    
    def identify_support_resistance(self, df: pd.DataFrame, window: int = 10, 
                                  threshold: float = 0.02) -> pd.DataFrame:
        """
        Identify support and resistance levels.
        
        Args:
            df: DataFrame with price data
            window: Window size for local minima/maxima
            threshold: Threshold for level significance (percentage)
            
        Returns:
            DataFrame with support and resistance levels
        """
        result = df.copy()
        
        # Initialize support and resistance columns
        result['support'] = np.nan
        result['resistance'] = np.nan
        
        # Find local minima and maxima
        for i in range(window, len(result) - window):
            # Check if it's a local minimum (support)
            if all(result['low'].iloc[i] <= result['low'].iloc[i-window:i]) and \
               all(result['low'].iloc[i] <= result['low'].iloc[i+1:i+window+1]):
                result['support'].iloc[i] = result['low'].iloc[i]
            
            # Check if it's a local maximum (resistance)
            if all(result['high'].iloc[i] >= result['high'].iloc[i-window:i]) and \
               all(result['high'].iloc[i] >= result['high'].iloc[i+1:i+window+1]):
                result['resistance'].iloc[i] = result['high'].iloc[i]
        
        return result
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators
        """
        # Get indicator parameters from config
        sma_periods = self.config.get('technical_analysis.indicators.sma', [20, 50, 200])
        ema_periods = self.config.get('technical_analysis.indicators.ema', [12, 26])
        rsi_period = self.config.get('technical_analysis.indicators.rsi', 14)
        macd_config = self.config.get('technical_analysis.indicators.macd', 
                                     {'fast': 12, 'slow': 26, 'signal': 9})
        bb_config = self.config.get('technical_analysis.indicators.bollinger_bands', 
                                   {'period': 20, 'std_dev': 2})
        
        # Calculate indicators
        result = df.copy()
        result = self.calculate_sma(result, sma_periods)
        result = self.calculate_ema(result, ema_periods)
        result = self.calculate_rsi(result, rsi_period)
        result = self.calculate_macd(result, 
                                    macd_config['fast'], 
                                    macd_config['slow'], 
                                    macd_config['signal'])
        result = self.calculate_bollinger_bands(result, 
                                              bb_config['period'], 
                                              bb_config['std_dev'])
        result = self.identify_support_resistance(result)
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary with trading signals and their strengths
        """
        signals = {
            'trend': {
                'direction': 'neutral',
                'strength': 0.0,
                'indicators': {}
            },
            'momentum': {
                'direction': 'neutral',
                'strength': 0.0,
                'indicators': {}
            },
            'volatility': {
                'direction': 'neutral',
                'strength': 0.0,
                'indicators': {}
            },
            'support_resistance': {
                'support': None,
                'resistance': None,
                'distance_to_support': None,
                'distance_to_resistance': None
            },
            'overall': {
                'signal': 'neutral',
                'strength': 0.0
            }
        }
        
        # Skip if not enough data
        if len(df) < 200:  # Need at least 200 periods for reliable signals
            return signals
        
        # Get the latest data point
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Current price
        current_price = latest['close']
        
        # === Trend Analysis ===
        
        # SMA trend signals
        sma_signals = {}
        for period in [20, 50, 200]:
            if f'sma_{period}' in latest:
                sma_value = latest[f'sma_{period}']
                if not np.isnan(sma_value):
                    signal = 'bullish' if current_price > sma_value else 'bearish'
                    strength = abs(current_price - sma_value) / sma_value
                    sma_signals[f'sma_{period}'] = {'signal': signal, 'strength': strength}
        
        # SMA crossovers
        if 'sma_20' in latest and 'sma_50' in latest:
            if not np.isnan(latest['sma_20']) and not np.isnan(latest['sma_50']):
                if latest['sma_20'] > latest['sma_50'] and prev['sma_20'] <= prev['sma_50']:
                    sma_signals['sma_20_50_crossover'] = {'signal': 'bullish', 'strength': 0.8}
                elif latest['sma_20'] < latest['sma_50'] and prev['sma_20'] >= prev['sma_50']:
                    sma_signals['sma_20_50_crossover'] = {'signal': 'bearish', 'strength': 0.8}
        
        # Combine SMA signals
        if sma_signals:
            bullish_count = sum(1 for s in sma_signals.values() if s['signal'] == 'bullish')
            bearish_count = sum(1 for s in sma_signals.values() if s['signal'] == 'bearish')
            
            if bullish_count > bearish_count:
                signals['trend']['direction'] = 'bullish'
                signals['trend']['strength'] = sum(s['strength'] for s in sma_signals.values() 
                                                if s['signal'] == 'bullish') / bullish_count
            elif bearish_count > bullish_count:
                signals['trend']['direction'] = 'bearish'
                signals['trend']['strength'] = sum(s['strength'] for s in sma_signals.values() 
                                                if s['signal'] == 'bearish') / bearish_count
            
            signals['trend']['indicators']['sma'] = sma_signals
        
        # === Momentum Analysis ===
        
        # RSI signals
        rsi_signals = {}
        if 'rsi_14' in latest and not np.isnan(latest['rsi_14']):
            rsi = latest['rsi_14']
            if rsi < 30:
                rsi_signals['rsi_14'] = {'signal': 'bullish', 'strength': (30 - rsi) / 30}
            elif rsi > 70:
                rsi_signals['rsi_14'] = {'signal': 'bearish', 'strength': (rsi - 70) / 30}
            else:
                rsi_signals['rsi_14'] = {'signal': 'neutral', 'strength': 0.0}
        
        # MACD signals
        macd_signals = {}
        if all(k in latest for k in ['macd_line', 'macd_signal', 'macd_histogram']):
            if not np.isnan(latest['macd_histogram']):
                # MACD line crosses above signal line (bullish)
                if latest['macd_histogram'] > 0 and prev['macd_histogram'] <= 0:
                    macd_signals['macd_crossover'] = {'signal': 'bullish', 'strength': 0.7}
                # MACD line crosses below signal line (bearish)
                elif latest['macd_histogram'] < 0 and prev['macd_histogram'] >= 0:
                    macd_signals['macd_crossover'] = {'signal': 'bearish', 'strength': 0.7}
                
                # MACD line and signal line both above/below zero
                if latest['macd_line'] > 0 and latest['macd_signal'] > 0:
                    macd_signals['macd_position'] = {'signal': 'bullish', 'strength': 0.5}
                elif latest['macd_line'] < 0 and latest['macd_signal'] < 0:
                    macd_signals['macd_position'] = {'signal': 'bearish', 'strength': 0.5}
        
        # Combine momentum signals
        momentum_signals = {**rsi_signals, **macd_signals}
        if momentum_signals:
            bullish_count = sum(1 for s in momentum_signals.values() if s['signal'] == 'bullish')
            bearish_count = sum(1 for s in momentum_signals.values() if s['signal'] == 'bearish')
            
            if bullish_count > bearish_count:
                signals['momentum']['direction'] = 'bullish'
                signals['momentum']['strength'] = sum(s['strength'] for s in momentum_signals.values() 
                                                   if s['signal'] == 'bullish') / bullish_count
            elif bearish_count > bullish_count:
                signals['momentum']['direction'] = 'bearish'
                signals['momentum']['strength'] = sum(s['strength'] for s in momentum_signals.values() 
                                                   if s['signal'] == 'bearish') / bearish_count
            
            signals['momentum']['indicators'] = {
                'rsi': rsi_signals,
                'macd': macd_signals
            }
        
        # === Volatility Analysis ===
        
        # Bollinger Bands signals
        bb_signals = {}
        if all(k in latest for k in ['bb_upper', 'bb_lower', 'bb_middle']):
            if not np.isnan(latest['bb_upper']) and not np.isnan(latest['bb_lower']):
                # Price near upper band
                if current_price > latest['bb_middle']:
                    distance_to_upper = (latest['bb_upper'] - current_price) / (latest['bb_upper'] - latest['bb_middle'])
                    if distance_to_upper < 0.2:  # Close to upper band
                        bb_signals['bb_position'] = {'signal': 'bearish', 'strength': 1 - distance_to_upper}
                
                # Price near lower band
                elif current_price < latest['bb_middle']:
                    distance_to_lower = (current_price - latest['bb_lower']) / (latest['bb_middle'] - latest['bb_lower'])
                    if distance_to_lower < 0.2:  # Close to lower band
                        bb_signals['bb_position'] = {'signal': 'bullish', 'strength': 1 - distance_to_lower}
                
                # Bollinger Band width (volatility)
                band_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']
                bb_signals['bb_width'] = {'value': band_width}
                
                # Determine if volatility is increasing or decreasing
                prev_width = (prev['bb_upper'] - prev['bb_lower']) / prev['bb_middle']
                if band_width > prev_width:
                    bb_signals['bb_volatility'] = {'signal': 'increasing', 'strength': (band_width / prev_width) - 1}
                else:
                    bb_signals['bb_volatility'] = {'signal': 'decreasing', 'strength': 1 - (band_width / prev_width)}
        
        # Set volatility signals
        if bb_signals:
            if 'bb_position' in bb_signals:
                signals['volatility']['direction'] = bb_signals['bb_position']['signal']
                signals['volatility']['strength'] = bb_signals['bb_position']['strength']
            
            signals['volatility']['indicators']['bollinger_bands'] = bb_signals
        
        # === Support and Resistance Analysis ===
        
        # Find nearest support and resistance levels
        support_levels = df['support'].dropna().iloc[-20:].values
        resistance_levels = df['resistance'].dropna().iloc[-20:].values
        
        if len(support_levels) > 0:
            # Find support levels below current price
            valid_supports = support_levels[support_levels < current_price]
            if len(valid_supports) > 0:
                nearest_support = max(valid_supports)
                signals['support_resistance']['support'] = nearest_support
                signals['support_resistance']['distance_to_support'] = (current_price - nearest_support) / current_price
        
        if len(resistance_levels) > 0:
            # Find resistance levels above current price
            valid_resistances = resistance_levels[resistance_levels > current_price]
            if len(valid_resistances) > 0:
                nearest_resistance = min(valid_resistances)
                signals['support_resistance']['resistance'] = nearest_resistance
                signals['support_resistance']['distance_to_resistance'] = (nearest_resistance - current_price) / current_price
        
        # === Overall Signal ===
        
        # Get signal weights from config
        weights = {
            'trend': self.config.get('technical_analysis.signal_weights.trend', 0.4),
            'momentum': self.config.get('technical_analysis.signal_weights.momentum', 0.3),
            'volatility': self.config.get('technical_analysis.signal_weights.volatility', 0.3)
        }
        
        # Calculate weighted signal
        signal_scores = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0
        }
        
        for category in ['trend', 'momentum', 'volatility']:
            direction = signals[category]['direction']
            strength = signals[category]['strength']
            weight = weights[category]
            
            signal_scores[direction] += strength * weight
        
        # Determine overall signal
        max_score = max(signal_scores.values())
        if max_score > 0:
            overall_signal = max(signal_scores, key=signal_scores.get)
            signals['overall']['signal'] = overall_signal
            signals['overall']['strength'] = max_score
        
        return signals
    
    def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data for all tracked cryptocurrencies.
        
        Args:
            market_data: Market data dictionary from MarketDataCollector
            
        Returns:
            Dictionary with technical analysis results for each cryptocurrency
        """
        analysis_results = {
            'timestamp': market_data.get('timestamp'),
            'cryptocurrencies': {}
        }
        
        # Get list of cryptocurrencies
        cryptocurrencies = self.config.get_cryptocurrencies()
        
        # Analyze each cryptocurrency
        for symbol in cryptocurrencies:
            # Prepare DataFrame
            df = self.prepare_dataframe(market_data, symbol)
            
            if df.empty:
                continue
            
            # Calculate indicators
            df_with_indicators = self.calculate_all_indicators(df)
            
            # Generate signals
            signals = self.generate_signals(df_with_indicators)
            
            # Store results
            analysis_results['cryptocurrencies'][symbol] = {
                'current_price': market_data.get('current_prices', {}).get(symbol),
                'signals': signals,
                # Include last few rows of indicator data for visualization
                'indicator_data': df_with_indicators.tail(10).to_dict('records')
            }
        
        return analysis_results
    
    def save_analysis_results(self, results: Dict[str, Any], 
                             filename: Optional[str] = None) -> str:
        """
        Save analysis results to a JSON file.
        
        Args:
            results: Analysis results to save
            filename: Output filename (generates a timestamped name if None)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"technical_analysis_{timestamp}.json"
        
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        
        # Create data directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        
        # Convert DataFrame records and timestamps for JSON serialization
        serializable_results = results.copy()
        
        for symbol, data in serializable_results.get('cryptocurrencies', {}).items():
            if 'indicator_data' in data:
                for record in data['indicator_data']:
                    for key, value in record.items():
                        if isinstance(value, pd.Timestamp):
                            record[key] = value.isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return output_path


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_collection.market_data import MarketDataCollector
    
    # Collect market data
    market_collector = MarketDataCollector()
    market_data = market_collector.get_market_data_for_analysis()
    
    # Analyze market data
    analyzer = TechnicalAnalyzer()
    analysis_results = analyzer.analyze_market_data(market_data)
    
    # Print signals for each cryptocurrency
    for symbol, data in analysis_results['cryptocurrencies'].items():
        print(f"\n=== {symbol} Technical Analysis ===")
        print(f"Current Price: ${data['current_price']:.2f}")
        print(f"Overall Signal: {data['signals']['overall']['signal']} (Strength: {data['signals']['overall']['strength']:.2f})")
        print(f"Trend: {data['signals']['trend']['direction']} (Strength: {data['signals']['trend']['strength']:.2f})")
        print(f"Momentum: {data['signals']['momentum']['direction']} (Strength: {data['signals']['momentum']['strength']:.2f})")
        print(f"Volatility: {data['signals']['volatility']['direction']} (Strength: {data['signals']['volatility']['strength']:.2f})")
        
        if data['signals']['support_resistance']['support'] is not None:
            print(f"Nearest Support: ${data['signals']['support_resistance']['support']:.2f} "
                 f"(Distance: {data['signals']['support_resistance']['distance_to_support']:.2%})")
        
        if data['signals']['support_resistance']['resistance'] is not None:
            print(f"Nearest Resistance: ${data['signals']['support_resistance']['resistance']:.2f} "
                 f"(Distance: {data['signals']['support_resistance']['distance_to_resistance']:.2%})")
    
    # Save analysis results
    output_path = analyzer.save_analysis_results(analysis_results)
    print(f"\nSaved analysis results to {output_path}")
