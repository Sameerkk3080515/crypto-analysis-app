import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re
import math

class ComprehensiveAnalyzer:
    """
    Class to perform comprehensive analysis on cryptocurrency data
    """
    
    def __init__(self):
        """Initialize the comprehensive analyzer"""
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_analyzer = MarketAnalyzer()
    
    def analyze_all_pairs(self, all_pairs_data):
        """
        Analyze all trading pairs
        
        Args:
            all_pairs_data: Dictionary with data for all trading pairs
            
        Returns:
            Dictionary with analysis results
        """
        try:
            print(f"Analyzing {len(all_pairs_data)} trading pairs")
            
            # Create dictionary to store analysis results
            analysis_results = {
                'pairs': {},
                'market_analysis': {}
            }
            
            # Perform market-wide analysis
            analysis_results['market_analysis'] = self.market_analyzer.analyze_market(all_pairs_data)
            
            # Process each trading pair
            for symbol, pair_data in all_pairs_data.items():
                try:
                    # Skip pairs without necessary data
                    if 'klines' not in pair_data or not pair_data['klines']:
                        continue
                    
                    # Get current price
                    current_price = pair_data.get('current_price', 0)
                    
                    # Perform technical analysis
                    technical_analysis = self.technical_analyzer.analyze_pair(pair_data)
                    
                    # Perform sentiment analysis
                    sentiment_analysis = self.sentiment_analyzer.analyze_pair(symbol, pair_data)
                    
                    # Store analysis results for this pair
                    analysis_results['pairs'][symbol] = {
                        'symbol': symbol,
                        'current_price': current_price,
                        'technical_analysis': technical_analysis,
                        'sentiment_analysis': sentiment_analysis
                    }
                    
                    print(f"Analyzed {symbol}")
                    
                except Exception as e:
                    print(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            return analysis_results
            
        except Exception as e:
            print(f"Error analyzing all pairs: {str(e)}")
            return {'pairs': {}, 'market_analysis': {}}


class TechnicalAnalyzer:
    """
    Class to perform technical analysis on cryptocurrency data
    """
    
    def __init__(self):
        """Initialize the technical analyzer"""
        pass
    
    def analyze_pair(self, pair_data):
        """
        Perform technical analysis on a trading pair
        
        Args:
            pair_data: Dictionary with data for a trading pair
            
        Returns:
            Dictionary with technical analysis results
        """
        try:
            # Get klines data
            klines_data = pair_data.get('klines', {})
            
            # Check if we have necessary data
            if not klines_data or '1h' not in klines_data or klines_data['1h'].empty:
                return {}
            
            # Get different timeframe data
            df_15m = klines_data.get('15m', pd.DataFrame())
            df_1h = klines_data.get('1h', pd.DataFrame())
            df_4h = klines_data.get('4h', pd.DataFrame())
            df_1d = klines_data.get('1d', pd.DataFrame())
            
            # Use 1h data as default if others are not available
            if df_15m.empty:
                df_15m = df_1h
            
            if df_4h.empty:
                df_4h = df_1h
            
            if df_1d.empty:
                df_1d = df_1h
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df_1h)
            
            # Identify price patterns
            patterns = self._identify_patterns(df_1h, indicators)
            
            # Calculate support and resistance levels
            support_resistance = self._calculate_support_resistance(df_1h)
            
            # Analyze price action
            price_action = self._analyze_price_action(df_15m, df_1h, df_4h, df_1d, indicators)
            
            # Analyze volume profile
            volume_profile = self._analyze_volume_profile(df_1h)
            
            # Return combined technical analysis
            return {
                'indicators': indicators,
                'patterns': patterns,
                'support_resistance': support_resistance,
                'price_action': price_action,
                'volume_profile': volume_profile
            }
            
        except Exception as e:
            print(f"Error in technical analysis: {str(e)}")
            return {}
    
    def _calculate_indicators(self, df):
        """
        Calculate technical indicators
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with technical indicators
        """
        try:
            # Create copy of DataFrame to avoid modifying original
            df = df.copy()
            
            # Simple Moving Averages (SMA)
            df['sma_7'] = df['close'].rolling(window=7).mean()
            df['sma_25'] = df['close'].rolling(window=25).mean()
            df['sma_99'] = df['close'].rolling(window=99).mean()
            
            # Exponential Moving Averages (EMA)
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            # MACD
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Relative Strength Index (RSI)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Stochastic Oscillator
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # On-Balance Volume (OBV)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Return indicators
            return {
                'sma': {
                    'sma_7': latest['sma_7'],
                    'sma_25': latest['sma_25'],
                    'sma_99': latest['sma_99']
                },
                'ema': {
                    'ema_9': latest['ema_9'],
                    'ema_21': latest['ema_21']
                },
                'macd': {
                    'macd': latest['macd'],
                    'signal': latest['macd_signal'],
                    'histogram': latest['macd_histogram']
                },
                'rsi': latest['rsi'],
                'bollinger_bands': {
                    'upper': latest['bb_upper'],
                    'middle': latest['bb_middle'],
                    'lower': latest['bb_lower']
                },
                'atr': latest['atr'],
                'stochastic': {
                    'k': latest['stoch_k'],
                    'd': latest['stoch_d']
                },
                'obv': latest['obv']
            }
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return {}
    
    def _identify_patterns(self, df, indicators):
        """
        Identify price patterns
        
        Args:
            df: DataFrame with price data
            indicators: Dictionary with technical indicators
            
        Returns:
            Dictionary with identified patterns
        """
        try:
            # Create copy of DataFrame to avoid modifying original
            df = df.copy()
            
            # Get recent candles
            recent_candles = df.iloc[-5:].copy()
            
            # Identify candlestick patterns
            candlestick_patterns = self._identify_candlestick_patterns(recent_candles)
            
            # Identify chart patterns
            chart_patterns = self._identify_chart_patterns(df)
            
            # Identify indicator patterns
            indicator_patterns = self._identify_indicator_patterns(indicators)
            
            # Return combined patterns
            return {
                'candlestick_patterns': candlestick_patterns,
                'chart_patterns': chart_patterns,
                'indicator_patterns': indicator_patterns
            }
            
        except Exception as e:
            print(f"Error identifying patterns: {str(e)}")
            return {}
    
    def _identify_candlestick_patterns(self, candles):
        """
        Identify candlestick patterns
        
        Args:
            candles: DataFrame with recent candles
            
        Returns:
            Dictionary with identified candlestick patterns
        """
        try:
            patterns = []
            
            # Get the last few candles
            if len(candles) >= 3:
                last_candle = candles.iloc[-1]
                prev_candle = candles.iloc[-2]
                prev_prev_candle = candles.iloc[-3]
                
                # Calculate candle properties
                last_body_size = abs(last_candle['close'] - last_candle['open'])
                last_total_size = last_candle['high'] - last_candle['low']
                last_upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
                last_lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
                
                prev_body_size = abs(prev_candle['close'] - prev_candle['open'])
                prev_total_size = prev_candle['high'] - prev_candle['low']
                
                # Doji
                if last_body_size / last_total_size < 0.1 and last_total_size > 0:
                    patterns.append('doji')
                
                # Hammer
                if (last_lower_shadow > 2 * last_body_size and 
                    last_upper_shadow < 0.2 * last_body_size and
                    last_candle['close'] > last_candle['open']):
                    patterns.append('hammer')
                
                # Shooting Star
                if (last_upper_shadow > 2 * last_body_size and 
                    last_lower_shadow < 0.2 * last_body_size and
                    last_candle['close'] < last_candle['open']):
                    patterns.append('shooting_star')
                
                # Engulfing
                if (last_body_size > prev_body_size and
                    ((last_candle['close'] > last_candle['open'] and prev_candle['close'] < prev_candle['open']) or
                     (last_candle['close'] < last_candle['open'] and prev_candle['close'] > prev_candle['open']))):
                    if last_candle['close'] > last_candle['open']:
                        patterns.append('bullish_engulfing')
                    else:
                        patterns.append('bearish_engulfing')
                
                # Morning Star
                if (prev_prev_candle['close'] < prev_prev_candle['open'] and
                    prev_body_size / prev_total_size < 0.3 and
                    last_candle['close'] > last_candle['open'] and
                    last_candle['close'] > (prev_prev_candle['open'] + prev_prev_candle['close']) / 2):
                    patterns.append('morning_star')
                
                # Evening Star
                if (prev_prev_candle['close'] > prev_prev_candle['open'] and
                    prev_body_size / prev_total_size < 0.3 and
                    last_candle['close'] < last_candle['open'] and
                    last_candle['close'] < (prev_prev_candle['open'] + prev_prev_candle['close']) / 2):
                    patterns.append('evening_star')
            
            return patterns
            
        except Exception as e:
            print(f"Error identifying candlestick patterns: {str(e)}")
            return []
    
    def _identify_chart_patterns(self, df):
        """
        Identify chart patterns
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with identified chart patterns
        """
        try:
            patterns = {}
            
            # Need at least 30 candles for reliable pattern detection
            if len(df) < 30:
                return patterns
            
            # Get recent price data
            recent_df = df.iloc[-30:].copy()
            
            # Calculate highs and lows
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            closes = recent_df['close'].values
            
            # Double Top
            if self._is_double_top(highs, closes):
                patterns['double_top'] = True
            
            # Double Bottom
            if self._is_double_bottom(lows, closes):
                patterns['double_bottom'] = True
            
            # Head and Shoulders
            if self._is_head_and_shoulders(highs, closes):
                patterns['head_and_shoulders'] = True
            
            # Inverse Head and Shoulders
            if self._is_inverse_head_and_shoulders(lows, closes):
                patterns['inverse_head_and_shoulders'] = True
            
            # Ascending Triangle
            if self._is_ascending_triangle(highs, lows):
                patterns['ascending_triangle'] = True
            
            # Descending Triangle
            if self._is_descending_triangle(highs, lows):
                patterns['descending_triangle'] = True
            
            # Symmetrical Triangle
            if self._is_symmetrical_triangle(highs, lows):
                patterns['symmetrical_triangle'] = True
            
            return patterns
            
        except Exception as e:
            print(f"Error identifying chart patterns: {str(e)}")
            return {}
    
    def _is_double_top(self, highs, closes):
        """Check for double top pattern"""
        try:
            # Find local maxima
            peaks = []
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
            
            # Need at least 2 peaks
            if len(peaks) < 2:
                return False
            
            # Check if the two highest peaks are similar in height
            peaks.sort(key=lambda x: x[1], reverse=True)
            peak1_idx, peak1_val = peaks[0]
            peak2_idx, peak2_val = peaks[1]
            
            # Peaks should be similar in height (within 2%)
            if abs(peak1_val - peak2_val) / peak1_val > 0.02:
                return False
            
            # Peaks should be separated by at least 5 candles
            if abs(peak1_idx - peak2_idx) < 5:
                return False
            
            # Should be a significant drop between peaks
            trough_idx = min(peak1_idx, peak2_idx) + 1
            trough_end = max(peak1_idx, peak2_idx)
            trough_val = min(closes[trough_idx:trough_end])
            
            # Drop should be at least 3% from peak
            if (peak1_val - trough_val) / peak1_val < 0.03:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error checking double top: {str(e)}")
            return False
    
    def _is_double_bottom(self, lows, closes):
        """Check for double bottom pattern"""
        try:
            # Find local minima
            troughs = []
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    troughs.append((i, lows[i]))
            
            # Need at least 2 troughs
            if len(troughs) < 2:
                return False
            
            # Check if the two lowest troughs are similar in height
            troughs.sort(key=lambda x: x[1])
            trough1_idx, trough1_val = troughs[0]
            trough2_idx, trough2_val = troughs[1]
            
            # Troughs should be similar in height (within 2%)
            if abs(trough1_val - trough2_val) / trough1_val > 0.02:
                return False
            
            # Troughs should be separated by at least 5 candles
            if abs(trough1_idx - trough2_idx) < 5:
                return False
            
            # Should be a significant rise between troughs
            peak_idx = min(trough1_idx, trough2_idx) + 1
            peak_end = max(trough1_idx, trough2_idx)
            peak_val = max(closes[peak_idx:peak_end])
            
            # Rise should be at least 3% from trough
            if (peak_val - trough1_val) / trough1_val < 0.03:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error checking double bottom: {str(e)}")
            return False
    
    def _is_head_and_shoulders(self, highs, closes):
        """Check for head and shoulders pattern"""
        try:
            # Find local maxima
            peaks = []
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
            
            # Need at least 3 peaks
            if len(peaks) < 3:
                return False
            
            # Sort peaks by index
            peaks.sort(key=lambda x: x[0])
            
            # Check each set of 3 consecutive peaks
            for i in range(len(peaks) - 2):
                left_idx, left_val = peaks[i]
                head_idx, head_val = peaks[i+1]
                right_idx, right_val = peaks[i+2]
                
                # Head should be higher than shoulders
                if head_val <= left_val or head_val <= right_val:
                    continue
                
                # Shoulders should be similar in height (within 10%)
                if abs(left_val - right_val) / left_val > 0.1:
                    continue
                
                # Should be significant troughs between peaks
                left_trough = min(closes[left_idx+1:head_idx])
                right_trough = min(closes[head_idx+1:right_idx])
                
                # Troughs should be similar in height (within 5%)
                if abs(left_trough - right_trough) / left_trough > 0.05:
                    continue
                
                # Pattern confirmed
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking head and shoulders: {str(e)}")
            return False
    
    def _is_inverse_head_and_shoulders(self, lows, closes):
        """Check for inverse head and shoulders pattern"""
        try:
            # Find local minima
            troughs = []
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    troughs.append((i, lows[i]))
            
            # Need at least 3 troughs
            if len(troughs) < 3:
                return False
            
            # Sort troughs by index
            troughs.sort(key=lambda x: x[0])
            
            # Check each set of 3 consecutive troughs
            for i in range(len(troughs) - 2):
                left_idx, left_val = troughs[i]
                head_idx, head_val = troughs[i+1]
                right_idx, right_val = troughs[i+2]
                
                # Head should be lower than shoulders
                if head_val >= left_val or head_val >= right_val:
                    continue
                
                # Shoulders should be similar in height (within 10%)
                if abs(left_val - right_val) / left_val > 0.1:
                    continue
                
                # Should be significant peaks between troughs
                left_peak = max(closes[left_idx+1:head_idx])
                right_peak = max(closes[head_idx+1:right_idx])
                
                # Peaks should be similar in height (within 5%)
                if abs(left_peak - right_peak) / left_peak > 0.05:
                    continue
                
                # Pattern confirmed
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking inverse head and shoulders: {str(e)}")
            return False
    
    def _is_ascending_triangle(self, highs, lows):
        """Check for ascending triangle pattern"""
        try:
            # Need at least 10 candles
            if len(highs) < 10:
                return False
            
            # Find resistance level (horizontal line at top)
            top_points = []
            for i in range(len(highs)):
                if i > 0 and abs(highs[i] - highs[i-1]) / highs[i] < 0.01:
                    top_points.append(highs[i])
            
            if len(top_points) < 2:
                return False
            
            resistance = sum(top_points) / len(top_points)
            
            # Check for ascending support line
            support_points = []
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    support_points.append((i, lows[i]))
            
            if len(support_points) < 2:
                return False
            
            # Sort by index
            support_points.sort(key=lambda x: x[0])
            
            # Check if support line is ascending
            is_ascending = True
            for i in range(1, len(support_points)):
                if support_points[i][1] <= support_points[i-1][1]:
                    is_ascending = False
                    break
            
            return is_ascending
            
        except Exception as e:
            print(f"Error checking ascending triangle: {str(e)}")
            return False
    
    def _is_descending_triangle(self, highs, lows):
        """Check for descending triangle pattern"""
        try:
            # Need at least 10 candles
            if len(lows) < 10:
                return False
            
            # Find support level (horizontal line at bottom)
            bottom_points = []
            for i in range(len(lows)):
                if i > 0 and abs(lows[i] - lows[i-1]) / lows[i] < 0.01:
                    bottom_points.append(lows[i])
            
            if len(bottom_points) < 2:
                return False
            
            support = sum(bottom_points) / len(bottom_points)
            
            # Check for descending resistance line
            resistance_points = []
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    resistance_points.append((i, highs[i]))
            
            if len(resistance_points) < 2:
                return False
            
            # Sort by index
            resistance_points.sort(key=lambda x: x[0])
            
            # Check if resistance line is descending
            is_descending = True
            for i in range(1, len(resistance_points)):
                if resistance_points[i][1] >= resistance_points[i-1][1]:
                    is_descending = False
                    break
            
            return is_descending
            
        except Exception as e:
            print(f"Error checking descending triangle: {str(e)}")
            return False
    
    def _is_symmetrical_triangle(self, highs, lows):
        """Check for symmetrical triangle pattern"""
        try:
            # Need at least 10 candles
            if len(highs) < 10 or len(lows) < 10:
                return False
            
            # Find resistance points (lower highs)
            resistance_points = []
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    resistance_points.append((i, highs[i]))
            
            if len(resistance_points) < 2:
                return False
            
            # Sort by index
            resistance_points.sort(key=lambda x: x[0])
            
            # Check if resistance line is descending
            is_descending = True
            for i in range(1, len(resistance_points)):
                if resistance_points[i][1] >= resistance_points[i-1][1]:
                    is_descending = False
                    break
            
            # Find support points (higher lows)
            support_points = []
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    support_points.append((i, lows[i]))
            
            if len(support_points) < 2:
                return False
            
            # Sort by index
            support_points.sort(key=lambda x: x[0])
            
            # Check if support line is ascending
            is_ascending = True
            for i in range(1, len(support_points)):
                if support_points[i][1] <= support_points[i-1][1]:
                    is_ascending = False
                    break
            
            return is_descending and is_ascending
            
        except Exception as e:
            print(f"Error checking symmetrical triangle: {str(e)}")
            return False
    
    def _identify_indicator_patterns(self, indicators):
        """
        Identify patterns in technical indicators
        
        Args:
            indicators: Dictionary with technical indicators
            
        Returns:
            Dictionary with identified indicator patterns
        """
        try:
            patterns = {}
            
            # RSI patterns
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                patterns['rsi'] = 'overbought'
            elif rsi < 30:
                patterns['rsi'] = 'oversold'
            else:
                patterns['rsi'] = 'neutral'
            
            # MACD patterns
            macd = indicators.get('macd', {})
            macd_value = macd.get('macd', 0)
            macd_signal = macd.get('signal', 0)
            macd_histogram = macd.get('histogram', 0)
            
            if macd_value > macd_signal:
                patterns['macd'] = 'bullish'
            else:
                patterns['macd'] = 'bearish'
            
            if macd_histogram > 0 and macd_histogram > macd.get('histogram_prev', 0):
                patterns['macd_histogram'] = 'increasing_positive'
            elif macd_histogram > 0:
                patterns['macd_histogram'] = 'positive'
            elif macd_histogram < 0 and macd_histogram < macd.get('histogram_prev', 0):
                patterns['macd_histogram'] = 'decreasing_negative'
            else:
                patterns['macd_histogram'] = 'negative'
            
            # Bollinger Bands patterns
            bb = indicators.get('bollinger_bands', {})
            upper = bb.get('upper', 0)
            middle = bb.get('middle', 0)
            lower = bb.get('lower', 0)
            
            if upper and middle and lower:
                # Calculate bandwidth
                band_width = (upper - lower) / middle
                
                if band_width < 0.1:
                    patterns['bollinger_bands'] = 'squeeze'
                elif band_width > 0.5:
                    patterns['bollinger_bands'] = 'expansion'
                else:
                    patterns['bollinger_bands'] = 'normal'
            
            # Stochastic patterns
            stoch = indicators.get('stochastic', {})
            k = stoch.get('k', 50)
            d = stoch.get('d', 50)
            
            if k > 80 and d > 80:
                patterns['stochastic'] = 'overbought'
            elif k < 20 and d < 20:
                patterns['stochastic'] = 'oversold'
            elif k > d:
                patterns['stochastic'] = 'bullish_crossover'
            elif k < d:
                patterns['stochastic'] = 'bearish_crossover'
            else:
                patterns['stochastic'] = 'neutral'
            
            # Moving Average patterns
            sma = indicators.get('sma', {})
            ema = indicators.get('ema', {})
            
            sma_7 = sma.get('sma_7', 0)
            sma_25 = sma.get('sma_25', 0)
            sma_99 = sma.get('sma_99', 0)
            
            ema_9 = ema.get('ema_9', 0)
            ema_21 = ema.get('ema_21', 0)
            
            if sma_7 > sma_25 and sma_25 > sma_99:
                patterns['moving_averages'] = 'strong_uptrend'
            elif sma_7 < sma_25 and sma_25 < sma_99:
                patterns['moving_averages'] = 'strong_downtrend'
            elif sma_7 > sma_25:
                patterns['moving_averages'] = 'uptrend'
            elif sma_7 < sma_25:
                patterns['moving_averages'] = 'downtrend'
            else:
                patterns['moving_averages'] = 'neutral'
            
            if ema_9 > ema_21:
                patterns['ema_crossover'] = 'bullish'
            else:
                patterns['ema_crossover'] = 'bearish'
            
            return patterns
            
        except Exception as e:
            print(f"Error identifying indicator patterns: {str(e)}")
            return {}
    
    def _calculate_support_resistance(self, df):
        """
        Calculate support and resistance levels
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            # Create copy of DataFrame to avoid modifying original
            df = df.copy()
            
            # Get recent price data (last 100 candles)
            recent_df = df.iloc[-100:].copy()
            
            # Find local maxima and minima
            highs = []
            lows = []
            
            for i in range(2, len(recent_df) - 2):
                # Check for local maximum
                if (recent_df['high'].iloc[i] > recent_df['high'].iloc[i-1] and
                    recent_df['high'].iloc[i] > recent_df['high'].iloc[i-2] and
                    recent_df['high'].iloc[i] > recent_df['high'].iloc[i+1] and
                    recent_df['high'].iloc[i] > recent_df['high'].iloc[i+2]):
                    highs.append(recent_df['high'].iloc[i])
                
                # Check for local minimum
                if (recent_df['low'].iloc[i] < recent_df['low'].iloc[i-1] and
                    recent_df['low'].iloc[i] < recent_df['low'].iloc[i-2] and
                    recent_df['low'].iloc[i] < recent_df['low'].iloc[i+1] and
                    recent_df['low'].iloc[i] < recent_df['low'].iloc[i+2]):
                    lows.append(recent_df['low'].iloc[i])
            
            # Get current price
            current_price = recent_df['close'].iloc[-1]
            
            # Find closest support and resistance levels
            supports = [l for l in lows if l < current_price]
            resistances = [h for h in highs if h > current_price]
            
            closest_support = max(supports) if supports else current_price * 0.95
            closest_resistance = min(resistances) if resistances else current_price * 1.05
            
            # Calculate Fibonacci retracement levels
            high = recent_df['high'].max()
            low = recent_df['low'].min()
            diff = high - low
            
            fib_23_6 = high - 0.236 * diff
            fib_38_2 = high - 0.382 * diff
            fib_50_0 = high - 0.5 * diff
            fib_61_8 = high - 0.618 * diff
            
            # Return support and resistance levels
            return {
                'supports': sorted(supports),
                'resistances': sorted(resistances),
                'closest_support': closest_support,
                'closest_resistance': closest_resistance,
                'fibonacci_retracement': {
                    '23.6%': fib_23_6,
                    '38.2%': fib_38_2,
                    '50.0%': fib_50_0,
                    '61.8%': fib_61_8
                }
            }
            
        except Exception as e:
            print(f"Error calculating support and resistance: {str(e)}")
            return {}
    
    def _analyze_price_action(self, df_15m, df_1h, df_4h, df_1d, indicators):
        """
        Analyze price action across multiple timeframes
        
        Args:
            df_15m: DataFrame with 15-minute price data
            df_1h: DataFrame with 1-hour price data
            df_4h: DataFrame with 4-hour price data
            df_1d: DataFrame with 1-day price data
            indicators: Dictionary with technical indicators
            
        Returns:
            Dictionary with price action analysis
        """
        try:
            # Analyze short-term trend (15m and 1h)
            short_term_trend = self._determine_trend(df_15m, df_1h)
            
            # Analyze medium-term trend (4h)
            medium_term_trend = self._determine_trend(df_4h)
            
            # Analyze long-term trend (1d)
            long_term_trend = self._determine_trend(df_1d)
            
            # Determine trend strength
            trend_strength = self._determine_trend_strength(short_term_trend, medium_term_trend, long_term_trend)
            
            # Determine momentum
            momentum = self._determine_momentum(indicators)
            
            # Determine volatility
            volatility = self._determine_volatility(df_1h, indicators)
            
            # Get MACD signal
            macd_signal = indicators.get('macd', {}).get('macd', 0) > indicators.get('macd', {}).get('signal', 0)
            macd_signal_str = 'bullish' if macd_signal else 'bearish'
            
            # Return price action analysis
            return {
                'short_term_trend': short_term_trend,
                'medium_term_trend': medium_term_trend,
                'long_term_trend': long_term_trend,
                'trend_strength': trend_strength,
                'momentum': momentum,
                'volatility': volatility,
                'macd_signal': macd_signal_str
            }
            
        except Exception as e:
            print(f"Error analyzing price action: {str(e)}")
            return {}
    
    def _determine_trend(self, *dfs):
        """
        Determine trend based on moving averages
        
        Args:
            *dfs: One or more DataFrames with price data
            
        Returns:
            String indicating trend direction
        """
        try:
            trends = []
            
            for df in dfs:
                if df.empty:
                    continue
                
                # Calculate EMAs if not already present
                if 'ema_9' not in df.columns:
                    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
                
                if 'ema_21' not in df.columns:
                    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
                
                # Get latest values
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                
                # Check EMA relationship
                if latest['ema_9'] > latest['ema_21'] and latest['close'] > latest['ema_9']:
                    trends.append('bullish')
                elif latest['ema_9'] < latest['ema_21'] and latest['close'] < latest['ema_9']:
                    trends.append('bearish')
                elif latest['close'] > latest['ema_21']:
                    trends.append('moderately_bullish')
                elif latest['close'] < latest['ema_21']:
                    trends.append('moderately_bearish')
                else:
                    trends.append('neutral')
            
            # Determine overall trend
            if not trends:
                return 'neutral'
            
            bullish_count = trends.count('bullish') + 0.5 * trends.count('moderately_bullish')
            bearish_count = trends.count('bearish') + 0.5 * trends.count('moderately_bearish')
            
            if bullish_count > bearish_count:
                return 'bullish'
            elif bearish_count > bullish_count:
                return 'bearish'
            else:
                return 'neutral'
            
        except Exception as e:
            print(f"Error determining trend: {str(e)}")
            return 'neutral'
    
    def _determine_trend_strength(self, short_term_trend, medium_term_trend, long_term_trend):
        """
        Determine trend strength based on multiple timeframes
        
        Args:
            short_term_trend: String indicating short-term trend
            medium_term_trend: String indicating medium-term trend
            long_term_trend: String indicating long-term trend
            
        Returns:
            String indicating trend strength
        """
        try:
            # Check if trends align
            if short_term_trend == medium_term_trend and medium_term_trend == long_term_trend:
                return 'strong'
            elif short_term_trend == medium_term_trend or medium_term_trend == long_term_trend:
                return 'moderate'
            else:
                return 'weak'
            
        except Exception as e:
            print(f"Error determining trend strength: {str(e)}")
            return 'weak'
    
    def _determine_momentum(self, indicators):
        """
        Determine momentum based on indicators
        
        Args:
            indicators: Dictionary with technical indicators
            
        Returns:
            String indicating momentum
        """
        try:
            # Check RSI
            rsi = indicators.get('rsi', 50)
            
            if rsi > 70:
                return 'overbought'
            elif rsi < 30:
                return 'oversold'
            elif rsi > 60:
                return 'strong'
            elif rsi < 40:
                return 'weak'
            else:
                return 'neutral'
            
        except Exception as e:
            print(f"Error determining momentum: {str(e)}")
            return 'neutral'
    
    def _determine_volatility(self, df, indicators):
        """
        Determine volatility based on price data and indicators
        
        Args:
            df: DataFrame with price data
            indicators: Dictionary with technical indicators
            
        Returns:
            String indicating volatility
        """
        try:
            # Check ATR relative to price
            atr = indicators.get('atr', 0)
            current_price = df['close'].iloc[-1] if not df.empty else 0
            
            if current_price > 0:
                atr_percent = (atr / current_price) * 100
                
                if atr_percent > 5:
                    return 'high'
                elif atr_percent > 2:
                    return 'medium'
                else:
                    return 'low'
            
            # Check Bollinger Band width
            bb = indicators.get('bollinger_bands', {})
            upper = bb.get('upper', 0)
            lower = bb.get('lower', 0)
            middle = bb.get('middle', 0)
            
            if middle > 0:
                bb_width = (upper - lower) / middle
                
                if bb_width > 0.1:
                    return 'high'
                elif bb_width > 0.05:
                    return 'medium'
                else:
                    return 'low'
            
            return 'medium'
            
        except Exception as e:
            print(f"Error determining volatility: {str(e)}")
            return 'medium'
    
    def _analyze_volume_profile(self, df):
        """
        Analyze volume profile
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with volume profile analysis
        """
        try:
            # Create copy of DataFrame to avoid modifying original
            df = df.copy()
            
            # Get recent volume data (last 20 candles)
            recent_df = df.iloc[-20:].copy()
            
            # Calculate average volume
            avg_volume = recent_df['volume'].mean()
            
            # Calculate volume trend
            volume_trend = 'increasing' if recent_df['volume'].iloc[-1] > avg_volume else 'decreasing'
            
            # Check for volume spikes
            volume_spikes = []
            for i in range(len(recent_df)):
                if recent_df['volume'].iloc[i] > 2 * avg_volume:
                    volume_spikes.append(i)
            
            # Check volume-price relationship
            volume_price_relationship = []
            for i in range(1, len(recent_df)):
                price_change = recent_df['close'].iloc[i] - recent_df['close'].iloc[i-1]
                volume_change = recent_df['volume'].iloc[i] - recent_df['volume'].iloc[i-1]
                
                if price_change > 0 and volume_change > 0:
                    volume_price_relationship.append('bullish')
                elif price_change < 0 and volume_change > 0:
                    volume_price_relationship.append('bearish')
                else:
                    volume_price_relationship.append('neutral')
            
            # Determine overall volume-price relationship
            bullish_count = volume_price_relationship.count('bullish')
            bearish_count = volume_price_relationship.count('bearish')
            
            if bullish_count > bearish_count:
                overall_relationship = 'bullish'
            elif bearish_count > bullish_count:
                overall_relationship = 'bearish'
            else:
                overall_relationship = 'neutral'
            
            # Return volume profile analysis
            return {
                'average_volume': avg_volume,
                'volume_trend': volume_trend,
                'volume_spikes': len(volume_spikes),
                'volume_price_relationship': overall_relationship
            }
            
        except Exception as e:
            print(f"Error analyzing volume profile: {str(e)}")
            return {}


class SentimentAnalyzer:
    """
    Class to perform sentiment analysis on cryptocurrency data
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        pass
    
    def analyze_pair(self, symbol, pair_data):
        """
        Perform sentiment analysis on a trading pair
        
        Args:
            symbol: Trading pair symbol
            pair_data: Dictionary with data for a trading pair
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Extract base symbol (e.g., BTC from BTCUSDT)
            base_symbol = re.sub(r'USDT$', '', symbol)
            
            # Simulate news sentiment analysis
            news_sentiment = self._analyze_news_sentiment(base_symbol)
            
            # Simulate social media sentiment analysis
            social_sentiment = self._analyze_social_sentiment(base_symbol)
            
            # Combine sentiment scores
            combined_sentiment = self._combine_sentiment_scores(news_sentiment, social_sentiment)
            
            # Categorize overall sentiment
            overall_category = self._categorize_sentiment(combined_sentiment)
            
            # Return sentiment analysis results
            return {
                'news_sentiment': news_sentiment['overall_sentiment'],
                'news_articles': news_sentiment['articles'],
                'social_sentiment': social_sentiment,
                'combined_sentiment': combined_sentiment,
                'overall_category': overall_category
            }
            
        except Exception as e:
            print(f"Error in sentiment analysis for {symbol}: {str(e)}")
            return {}
    
    def _analyze_news_sentiment(self, symbol):
        """
        Analyze news sentiment for a cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Dictionary with news sentiment analysis
        """
        try:
            # Simulate news articles and sentiment
            num_articles = random.randint(3, 8)
            articles = []
            
            # Common news sources
            sources = [
                "CoinDesk", "CryptoSlate", "Cointelegraph", "Bitcoin.com",
                "Decrypt", "The Block", "CoinMarketCap", "Bitcoinist",
                "NewsBTC", "U.Today", "Crypto Briefing"
            ]
            
            # Positive headlines
            positive_headlines = [
                f"{symbol} Price Surges Amid Increased Adoption",
                f"{symbol} Breaks Resistance Level, Analysts Predict Further Gains",
                f"Major Exchange Adds Support for {symbol}, Price Rallies",
                f"Institutional Investors Show Growing Interest in {symbol}",
                f"{symbol} Development Team Announces Major Upgrade",
                f"Bullish Outlook for {symbol} as Market Sentiment Improves",
                f"{symbol} Gains Momentum Following Positive Regulatory News"
            ]
            
            # Negative headlines
            negative_headlines = [
                f"{symbol} Price Drops Amid Market Uncertainty",
                f"{symbol} Faces Selling Pressure as Traders Take Profits",
                f"Analysts Warn of Potential Correction for {symbol}",
                f"Regulatory Concerns Impact {symbol} Price",
                f"{symbol} Struggles to Maintain Support Level",
                f"Bearish Signals Emerge for {symbol} in Short Term",
                f"Market Volatility Affects {symbol} Trading Volume"
            ]
            
            # Neutral headlines
            neutral_headlines = [
                f"{symbol} Price Stabilizes Following Recent Fluctuations",
                f"Market Analysis: What's Next for {symbol}?",
                f"{symbol} Trading Volume Remains Steady",
                f"Experts Divided on {symbol} Price Direction",
                f"{symbol} Maintains Sideways Movement in Current Market",
                f"Technical Analysis: {symbol} at Critical Juncture",
                f"{symbol} Community Discusses Future Development Plans"
            ]
            
            # Generate random articles
            total_sentiment = 0
            
            for i in range(num_articles):
                # Determine sentiment (-1 to 1)
                sentiment = random.uniform(-1, 1)
                total_sentiment += sentiment
                
                # Select headline based on sentiment
                if sentiment > 0.3:
                    headline = random.choice(positive_headlines)
                elif sentiment < -0.3:
                    headline = random.choice(negative_headlines)
                else:
                    headline = random.choice(neutral_headlines)
                
                # Create article
                article = {
                    'title': headline,
                    'source': random.choice(sources),
                    'sentiment': sentiment,
                    'time': datetime.now() - timedelta(hours=random.randint(1, 24))
                }
                
                articles.append(article)
            
            # Calculate overall sentiment
            overall_sentiment = total_sentiment / num_articles if num_articles > 0 else 0
            
            return {
                'overall_sentiment': overall_sentiment,
                'articles': articles
            }
            
        except Exception as e:
            print(f"Error analyzing news sentiment: {str(e)}")
            return {'overall_sentiment': 0, 'articles': []}
    
    def _analyze_social_sentiment(self, symbol):
        """
        Analyze social media sentiment for a cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Dictionary with social media sentiment analysis
        """
        try:
            # Simulate social media sentiment
            twitter_sentiment = random.uniform(-1, 1)
            reddit_sentiment = random.uniform(-1, 1)
            telegram_sentiment = random.uniform(-1, 1)
            
            # Simulate mention count
            mention_count = random.randint(100, 10000)
            
            # Simulate sentiment change
            sentiment_change = random.uniform(-0.2, 0.2)
            
            return {
                'twitter_sentiment': twitter_sentiment,
                'reddit_sentiment': reddit_sentiment,
                'telegram_sentiment': telegram_sentiment,
                'mention_count': mention_count,
                'sentiment_change': sentiment_change
            }
            
        except Exception as e:
            print(f"Error analyzing social sentiment: {str(e)}")
            return {}
    
    def _combine_sentiment_scores(self, news_sentiment, social_sentiment):
        """
        Combine sentiment scores from different sources
        
        Args:
            news_sentiment: Dictionary with news sentiment analysis
            social_sentiment: Dictionary with social media sentiment analysis
            
        Returns:
            Combined sentiment score (-1 to 1)
        """
        try:
            # Extract sentiment scores
            news_score = news_sentiment.get('overall_sentiment', 0)
            twitter_score = social_sentiment.get('twitter_sentiment', 0)
            reddit_score = social_sentiment.get('reddit_sentiment', 0)
            telegram_score = social_sentiment.get('telegram_sentiment', 0)
            
            # Assign weights
            news_weight = 0.4
            twitter_weight = 0.3
            reddit_weight = 0.2
            telegram_weight = 0.1
            
            # Calculate weighted average
            combined_score = (
                news_score * news_weight +
                twitter_score * twitter_weight +
                reddit_score * reddit_weight +
                telegram_score * telegram_weight
            )
            
            return combined_score
            
        except Exception as e:
            print(f"Error combining sentiment scores: {str(e)}")
            return 0
    
    def _categorize_sentiment(self, sentiment_score):
        """
        Categorize sentiment score
        
        Args:
            sentiment_score: Sentiment score (-1 to 1)
            
        Returns:
            String indicating sentiment category
        """
        try:
            if sentiment_score > 0.6:
                return 'very positive'
            elif sentiment_score > 0.2:
                return 'positive'
            elif sentiment_score < -0.6:
                return 'very negative'
            elif sentiment_score < -0.2:
                return 'negative'
            else:
                return 'neutral'
            
        except Exception as e:
            print(f"Error categorizing sentiment: {str(e)}")
            return 'neutral'


class MarketAnalyzer:
    """
    Class to perform market-wide analysis on cryptocurrency data
    """
    
    def __init__(self):
        """Initialize the market analyzer"""
        pass
    
    def analyze_market(self, all_pairs_data):
        """
        Analyze overall market conditions
        
        Args:
            all_pairs_data: Dictionary with data for all trading pairs
            
        Returns:
            Dictionary with market analysis results
        """
        try:
            # Filter for USDT pairs
            usdt_pairs = {
                symbol: data for symbol, data in all_pairs_data.items()
                if symbol.endswith('USDT')
            }
            
            # Calculate market trend
            market_trend = self._calculate_market_trend(usdt_pairs)
            
            # Calculate market strength
            market_strength = self._calculate_market_strength(usdt_pairs)
            
            # Calculate market volatility
            market_volatility = self._calculate_market_volatility(usdt_pairs)
            
            # Calculate BTC dominance
            btc_dominance = self._calculate_btc_dominance(usdt_pairs)
            
            # Find top and worst performers
            top_performers = self._find_top_performers(usdt_pairs, 5)
            worst_performers = self._find_worst_performers(usdt_pairs, 5)
            
            # Calculate sector performance
            sector_performance = self._calculate_sector_performance(usdt_pairs)
            
            # Calculate market correlations
            market_correlations = self._calculate_market_correlations(usdt_pairs)
            
            # Return market analysis results
            return {
                'market_conditions': {
                    'market_trend': market_trend,
                    'market_strength': market_strength,
                    'volatility': market_volatility,
                    'btc_dominance': btc_dominance,
                    'top_performers': top_performers,
                    'worst_performers': worst_performers
                },
                'sector_performance': sector_performance,
                'market_correlations': market_correlations
            }
            
        except Exception as e:
            print(f"Error in market analysis: {str(e)}")
            return {}
    
    def _calculate_market_trend(self, usdt_pairs):
        """
        Calculate overall market trend
        
        Args:
            usdt_pairs: Dictionary with data for USDT trading pairs
            
        Returns:
            String indicating market trend
        """
        try:
            # Count bullish and bearish pairs
            bullish_count = 0
            bearish_count = 0
            
            for symbol, data in usdt_pairs.items():
                if 'ticker' in data:
                    price_change = float(data['ticker'].get('priceChangePercent', 0))
                    
                    if price_change > 0:
                        bullish_count += 1
                    elif price_change < 0:
                        bearish_count += 1
            
            total_pairs = bullish_count + bearish_count
            
            if total_pairs == 0:
                return 'neutral'
            
            bullish_percentage = (bullish_count / total_pairs) * 100
            
            if bullish_percentage > 70:
                return 'strongly_bullish'
            elif bullish_percentage > 55:
                return 'moderately_bullish'
            elif bullish_percentage < 30:
                return 'strongly_bearish'
            elif bullish_percentage < 45:
                return 'moderately_bearish'
            else:
                return 'neutral'
            
        except Exception as e:
            print(f"Error calculating market trend: {str(e)}")
            return 'neutral'
    
    def _calculate_market_strength(self, usdt_pairs):
        """
        Calculate market strength
        
        Args:
            usdt_pairs: Dictionary with data for USDT trading pairs
            
        Returns:
            Market strength percentage
        """
        try:
            # Calculate average price change
            total_change = 0
            count = 0
            
            for symbol, data in usdt_pairs.items():
                if 'ticker' in data:
                    price_change = float(data['ticker'].get('priceChangePercent', 0))
                    total_change += price_change
                    count += 1
            
            if count == 0:
                return 50  # Neutral
            
            # Convert to 0-100 scale
            avg_change = total_change / count
            market_strength = 50 + (avg_change * 5)  # Scale factor
            
            # Clamp to 0-100 range
            market_strength = max(0, min(100, market_strength))
            
            return market_strength
            
        except Exception as e:
            print(f"Error calculating market strength: {str(e)}")
            return 50
    
    def _calculate_market_volatility(self, usdt_pairs):
        """
        Calculate market volatility
        
        Args:
            usdt_pairs: Dictionary with data for USDT trading pairs
            
        Returns:
            String indicating market volatility
        """
        try:
            # Calculate average high-low range
            total_range = 0
            count = 0
            
            for symbol, data in usdt_pairs.items():
                if 'ticker' in data:
                    high = float(data['ticker'].get('highPrice', 0))
                    low = float(data['ticker'].get('lowPrice', 0))
                    open_price = float(data['ticker'].get('openPrice', 1))
                    
                    if open_price > 0:
                        range_percent = ((high - low) / open_price) * 100
                        total_range += range_percent
                        count += 1
            
            if count == 0:
                return 'medium'
            
            avg_range = total_range / count
            
            if avg_range > 8:
                return 'high'
            elif avg_range > 4:
                return 'medium'
            else:
                return 'low'
            
        except Exception as e:
            print(f"Error calculating market volatility: {str(e)}")
            return 'medium'
    
    def _calculate_btc_dominance(self, usdt_pairs):
        """
        Calculate BTC dominance
        
        Args:
            usdt_pairs: Dictionary with data for USDT trading pairs
            
        Returns:
            BTC dominance ratio (0-1)
        """
        try:
            # Calculate total market cap
            total_market_cap = 0
            btc_market_cap = 0
            
            for symbol, data in usdt_pairs.items():
                if 'ticker' in data:
                    price = float(data['ticker'].get('lastPrice', 0))
                    volume = float(data['ticker'].get('volume', 0))
                    
                    # Estimate market cap (price * volume as proxy)
                    market_cap = price * volume
                    total_market_cap += market_cap
                    
                    if symbol == 'BTCUSDT':
                        btc_market_cap = market_cap
            
            if total_market_cap == 0:
                return 0.4  # Default value
            
            btc_dominance = btc_market_cap / total_market_cap
            
            return btc_dominance
            
        except Exception as e:
            print(f"Error calculating BTC dominance: {str(e)}")
            return 0.4
    
    def _find_top_performers(self, usdt_pairs, limit=5):
        """
        Find top performing pairs
        
        Args:
            usdt_pairs: Dictionary with data for USDT trading pairs
            limit: Maximum number of pairs to return
            
        Returns:
            List of tuples (symbol, price_change)
        """
        try:
            # Calculate price changes
            price_changes = []
            
            for symbol, data in usdt_pairs.items():
                if 'ticker' in data:
                    price_change = float(data['ticker'].get('priceChangePercent', 0))
                    price_changes.append((symbol, price_change))
            
            # Sort by price change (descending)
            price_changes.sort(key=lambda x: x[1], reverse=True)
            
            # Return top performers
            return price_changes[:limit]
            
        except Exception as e:
            print(f"Error finding top performers: {str(e)}")
            return []
    
    def _find_worst_performers(self, usdt_pairs, limit=5):
        """
        Find worst performing pairs
        
        Args:
            usdt_pairs: Dictionary with data for USDT trading pairs
            limit: Maximum number of pairs to return
            
        Returns:
            List of tuples (symbol, price_change)
        """
        try:
            # Calculate price changes
            price_changes = []
            
            for symbol, data in usdt_pairs.items():
                if 'ticker' in data:
                    price_change = float(data['ticker'].get('priceChangePercent', 0))
                    price_changes.append((symbol, price_change))
            
            # Sort by price change (ascending)
            price_changes.sort(key=lambda x: x[1])
            
            # Return worst performers
            return price_changes[:limit]
            
        except Exception as e:
            print(f"Error finding worst performers: {str(e)}")
            return []
    
    def _calculate_sector_performance(self, usdt_pairs):
        """
        Calculate performance by sector
        
        Args:
            usdt_pairs: Dictionary with data for USDT trading pairs
            
        Returns:
            Dictionary with sector performance
        """
        try:
            # Define sectors
            sectors = {
                'defi': ['AAVEUSDT', 'COMPUSDT', 'UNIUSDT', 'SUSHIUSDT', 'CAKEUSDT', 'MKRUSDT'],
                'layer1': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'ADAUSDT', 'DOTUSDT'],
                'layer2': ['MATICUSDT', 'OPUSDT', 'ARBUSDT', 'IMXUSDT'],
                'exchange': ['BNBUSDT', 'FTMUSDT', 'OKBUSDT', 'HTUSDT', 'KCSUSDT'],
                'gaming': ['AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'ENJUSDT', 'GALAUSDT']
            }
            
            # Calculate sector performance
            sector_performance = {}
            
            for sector, symbols in sectors.items():
                total_change = 0
                count = 0
                
                for symbol in symbols:
                    if symbol in usdt_pairs and 'ticker' in usdt_pairs[symbol]:
                        price_change = float(usdt_pairs[symbol]['ticker'].get('priceChangePercent', 0))
                        total_change += price_change
                        count += 1
                
                if count > 0:
                    sector_performance[sector] = total_change / count
                else:
                    sector_performance[sector] = 0
            
            return sector_performance
            
        except Exception as e:
            print(f"Error calculating sector performance: {str(e)}")
            return {}
    
    def _calculate_market_correlations(self, usdt_pairs):
        """
        Calculate correlations between major cryptocurrencies
        
        Args:
            usdt_pairs: Dictionary with data for USDT trading pairs
            
        Returns:
            Dictionary with correlation data
        """
        try:
            # Define major pairs
            major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT']
            
            # Extract price data
            price_data = {}
            
            for symbol in major_pairs:
                if symbol in usdt_pairs and 'klines' in usdt_pairs[symbol] and '1h' in usdt_pairs[symbol]['klines']:
                    klines = usdt_pairs[symbol]['klines']['1h']
                    price_data[symbol] = klines['close'].values
            
            # Calculate correlations
            correlations = {}
            
            for symbol1 in price_data:
                correlations[symbol1] = {}
                
                for symbol2 in price_data:
                    if symbol1 == symbol2:
                        correlations[symbol1][symbol2] = 1.0
                        continue
                    
                    # Calculate correlation coefficient
                    data1 = price_data[symbol1]
                    data2 = price_data[symbol2]
                    
                    # Use minimum length
                    min_length = min(len(data1), len(data2))
                    
                    if min_length > 1:
                        # Calculate correlation
                        data1 = data1[-min_length:]
                        data2 = data2[-min_length:]
                        
                        # Calculate means
                        mean1 = sum(data1) / min_length
                        mean2 = sum(data2) / min_length
                        
                        # Calculate variances
                        var1 = sum((x - mean1) ** 2 for x in data1) / min_length
                        var2 = sum((x - mean2) ** 2 for x in data2) / min_length
                        
                        # Calculate covariance
                        cov = sum((data1[i] - mean1) * (data2[i] - mean2) for i in range(min_length)) / min_length
                        
                        # Calculate correlation
                        if var1 > 0 and var2 > 0:
                            correlation = cov / (math.sqrt(var1) * math.sqrt(var2))
                            correlations[symbol1][symbol2] = correlation
                        else:
                            correlations[symbol1][symbol2] = 0
                    else:
                        correlations[symbol1][symbol2] = 0
            
            return correlations
            
        except Exception as e:
            print(f"Error calculating market correlations: {str(e)}")
            return {}
