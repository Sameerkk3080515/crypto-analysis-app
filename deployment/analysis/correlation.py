"""
Correlation analysis module for the Cryptocurrency Analysis Bot.

This module identifies relationships between technical indicators, sentiment data,
and price movements to generate more accurate trading signals.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

class CorrelationEngine:
    """Analyzes correlations between different data points and identifies patterns."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the correlation engine.
        
        Args:
            config: Configuration object (creates a new one if None)
        """
        self.config = config or Config()
    
    def correlate_technical_sentiment(self, technical_analysis: Dict[str, Any], 
                                     sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Correlate technical and sentiment analysis results.
        
        Args:
            technical_analysis: Technical analysis results
            sentiment_analysis: Sentiment analysis results
            
        Returns:
            Dictionary with correlation results
        """
        correlation_results = {
            'timestamp': datetime.now().isoformat(),
            'cryptocurrencies': {}
        }
        
        # Get list of cryptocurrencies
        cryptocurrencies = self.config.get_cryptocurrencies()
        
        # Analyze each cryptocurrency
        for symbol in cryptocurrencies:
            # Skip if data is missing
            if (symbol not in technical_analysis.get('cryptocurrencies', {}) or
                symbol not in sentiment_analysis.get('sentiment', {})):
                continue
            
            # Get technical and sentiment data
            technical_data = technical_analysis['cryptocurrencies'][symbol]
            sentiment_data = sentiment_analysis['sentiment'][symbol]
            
            # Initialize correlation data
            correlation_data = {
                'signal_agreement': self._calculate_signal_agreement(technical_data, sentiment_data),
                'combined_signal': self._generate_combined_signal(technical_data, sentiment_data),
                'confidence_score': self._calculate_confidence_score(technical_data, sentiment_data)
            }
            
            # Add trend data if available
            if symbol in sentiment_analysis.get('trends', {}):
                correlation_data['sentiment_trend'] = sentiment_analysis['trends'][symbol]
            
            # Store correlation results
            correlation_results['cryptocurrencies'][symbol] = correlation_data
        
        return correlation_results
    
    def _calculate_signal_agreement(self, technical_data: Dict[str, Any], 
                                  sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate agreement between technical and sentiment signals.
        
        Args:
            technical_data: Technical analysis data for a cryptocurrency
            sentiment_data: Sentiment analysis data for a cryptocurrency
            
        Returns:
            Dictionary with signal agreement metrics
        """
        # Get technical signals
        technical_signal = technical_data['signals']['overall']['signal']
        technical_strength = technical_data['signals']['overall']['strength']
        
        # Get sentiment signal
        sentiment_direction = sentiment_data['direction']
        sentiment_strength = sentiment_data['strength']
        
        # Map sentiment direction to signal terminology
        sentiment_signal = sentiment_direction
        if sentiment_direction == 'bullish':
            sentiment_signal = 'bullish'
        elif sentiment_direction == 'bearish':
            sentiment_signal = 'bearish'
        else:
            sentiment_signal = 'neutral'
        
        # Determine agreement
        agreement = False
        if technical_signal == sentiment_signal:
            agreement = True
        
        # Calculate agreement strength
        agreement_strength = 0.0
        if agreement:
            agreement_strength = (technical_strength + sentiment_strength) / 2
        else:
            # Negative agreement strength indicates disagreement
            agreement_strength = -abs(technical_strength - sentiment_strength) / 2
        
        return {
            'agreement': agreement,
            'strength': agreement_strength,
            'technical_signal': technical_signal,
            'technical_strength': technical_strength,
            'sentiment_signal': sentiment_signal,
            'sentiment_strength': sentiment_strength
        }
    
    def _generate_combined_signal(self, technical_data: Dict[str, Any], 
                                sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a combined signal based on technical and sentiment data.
        
        Args:
            technical_data: Technical analysis data for a cryptocurrency
            sentiment_data: Sentiment analysis data for a cryptocurrency
            
        Returns:
            Dictionary with combined signal
        """
        # Get technical signals
        technical_signal = technical_data['signals']['overall']['signal']
        technical_strength = technical_data['signals']['overall']['strength']
        
        # Get sentiment signal
        sentiment_direction = sentiment_data['direction']
        sentiment_strength = sentiment_data['strength']
        
        # Map sentiment direction to signal terminology
        if sentiment_direction == 'bullish':
            sentiment_signal = 'bullish'
        elif sentiment_direction == 'bearish':
            sentiment_signal = 'bearish'
        else:
            sentiment_signal = 'neutral'
        
        # Weights for technical and sentiment signals
        technical_weight = 0.7  # Technical analysis has higher weight
        sentiment_weight = 0.3  # Sentiment analysis has lower weight
        
        # Calculate signal scores
        signal_scores = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0
        }
        
        # Add technical signal score
        signal_scores[technical_signal] += technical_strength * technical_weight
        
        # Add sentiment signal score
        signal_scores[sentiment_signal] += sentiment_strength * sentiment_weight
        
        # Determine combined signal
        combined_signal = max(signal_scores, key=signal_scores.get)
        combined_strength = signal_scores[combined_signal]
        
        return {
            'signal': combined_signal,
            'strength': combined_strength,
            'technical_contribution': technical_strength * technical_weight / combined_strength if combined_strength > 0 else 0,
            'sentiment_contribution': sentiment_strength * sentiment_weight / combined_strength if combined_strength > 0 else 0
        }
    
    def _calculate_confidence_score(self, technical_data: Dict[str, Any], 
                                  sentiment_data: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the combined signal.
        
        Args:
            technical_data: Technical analysis data for a cryptocurrency
            sentiment_data: Sentiment analysis data for a cryptocurrency
            
        Returns:
            Confidence score (0-1)
        """
        # Get signal agreement
        agreement_data = self._calculate_signal_agreement(technical_data, sentiment_data)
        
        # Base confidence on agreement
        if agreement_data['agreement']:
            # Higher confidence when signals agree
            base_confidence = 0.7 + (agreement_data['strength'] * 0.3)
        else:
            # Lower confidence when signals disagree
            base_confidence = 0.5 - (abs(agreement_data['strength']) * 0.2)
        
        # Adjust confidence based on data quality
        technical_confidence = min(1.0, technical_data['signals']['overall']['strength'] * 1.2)
        
        # Adjust confidence based on sentiment volume
        sentiment_volume = sentiment_data['volume']
        volume_factor = min(1.0, sentiment_volume / 20)  # Normalize volume (max at 20 mentions)
        
        # Calculate final confidence score
        confidence_score = base_confidence * 0.6 + technical_confidence * 0.3 + volume_factor * 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence_score))
    
    def identify_patterns(self, market_data: Dict[str, Any], 
                         technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify patterns in market data and technical indicators.
        
        Args:
            market_data: Market data dictionary
            technical_analysis: Technical analysis results
            
        Returns:
            Dictionary with identified patterns
        """
        pattern_results = {
            'timestamp': datetime.now().isoformat(),
            'cryptocurrencies': {}
        }
        
        # Get list of cryptocurrencies
        cryptocurrencies = self.config.get_cryptocurrencies()
        
        # Analyze each cryptocurrency
        for symbol in cryptocurrencies:
            # Skip if data is missing
            if (symbol not in market_data.get('historical_data', {}) or
                symbol not in technical_analysis.get('cryptocurrencies', {})):
                continue
            
            # Get historical data
            historical_data = market_data['historical_data'][symbol]
            
            # Convert to DataFrame if it's a list
            if isinstance(historical_data, list):
                df = pd.DataFrame(historical_data)
            else:
                df = historical_data
            
            # Skip if not enough data
            if len(df) < 20:
                continue
            
            # Identify patterns
            patterns = {
                'price_patterns': self._identify_price_patterns(df),
                'volume_patterns': self._identify_volume_patterns(df),
                'indicator_patterns': self._identify_indicator_patterns(
                    technical_analysis['cryptocurrencies'][symbol])
            }
            
            # Store pattern results
            pattern_results['cryptocurrencies'][symbol] = patterns
        
        return pattern_results
    
    def _identify_price_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify price patterns in historical data.
        
        Args:
            df: DataFrame with historical price data
            
        Returns:
            Dictionary with identified price patterns
        """
        patterns = {}
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            return patterns
        
        # Get recent price data
        recent_df = df.tail(20)
        
        # Check for trend
        close_prices = recent_df['close'].values
        first_price = close_prices[0]
        last_price = close_prices[-1]
        
        price_change = (last_price - first_price) / first_price
        
        if price_change > 0.05:
            patterns['trend'] = 'uptrend'
            patterns['trend_strength'] = min(1.0, price_change)
        elif price_change < -0.05:
            patterns['trend'] = 'downtrend'
            patterns['trend_strength'] = min(1.0, abs(price_change))
        else:
            patterns['trend'] = 'sideways'
            patterns['trend_strength'] = 0.0
        
        # Check for double top/bottom
        if len(close_prices) >= 10:
            # Find local maxima and minima
            maxima = []
            minima = []
            
            for i in range(2, len(close_prices) - 2):
                if (close_prices[i] > close_prices[i-1] and 
                    close_prices[i] > close_prices[i-2] and
                    close_prices[i] > close_prices[i+1] and
                    close_prices[i] > close_prices[i+2]):
                    maxima.append((i, close_prices[i]))
                
                if (close_prices[i] < close_prices[i-1] and 
                    close_prices[i] < close_prices[i-2] and
                    close_prices[i] < close_prices[i+1] and
                    close_prices[i] < close_prices[i+2]):
                    minima.append((i, close_prices[i]))
            
            # Check for double top
            if len(maxima) >= 2:
                top1_idx, top1_val = maxima[-2]
                top2_idx, top2_val = maxima[-1]
                
                # Check if tops are similar in height
                if abs(top1_val - top2_val) / top1_val < 0.03:
                    patterns['double_top'] = {
                        'confidence': 0.7,
                        'first_top': top1_val,
                        'second_top': top2_val
                    }
            
            # Check for double bottom
            if len(minima) >= 2:
                bottom1_idx, bottom1_val = minima[-2]
                bottom2_idx, bottom2_val = minima[-1]
                
                # Check if bottoms are similar in height
                if abs(bottom1_val - bottom2_val) / bottom1_val < 0.03:
                    patterns['double_bottom'] = {
                        'confidence': 0.7,
                        'first_bottom': bottom1_val,
                        'second_bottom': bottom2_val
                    }
        
        return patterns
    
    def _identify_volume_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify volume patterns in historical data.
        
        Args:
            df: DataFrame with historical price and volume data
            
        Returns:
            Dictionary with identified volume patterns
        """
        patterns = {}
        
        # Ensure required columns exist
        if 'volume' not in df.columns or 'close' not in df.columns:
            return patterns
        
        # Get recent volume data
        recent_df = df.tail(10)
        
        # Calculate average volume
        avg_volume = recent_df['volume'].mean()
        latest_volume = recent_df['volume'].iloc[-1]
        
        # Check for volume spike
        if latest_volume > avg_volume * 2:
            patterns['volume_spike'] = {
                'strength': min(1.0, latest_volume / avg_volume / 3),
                'ratio': latest_volume / avg_volume
            }
        
        # Check for volume trend
        volume_trend = np.polyfit(range(len(recent_df)), recent_df['volume'].values, 1)[0]
        
        if volume_trend > 0:
            patterns['volume_trend'] = 'increasing'
            patterns['volume_trend_strength'] = min(1.0, volume_trend / avg_volume * 10)
        else:
            patterns['volume_trend'] = 'decreasing'
            patterns['volume_trend_strength'] = min(1.0, abs(volume_trend) / avg_volume * 10)
        
        # Check for volume-price divergence
        price_trend = np.polyfit(range(len(recent_df)), recent_df['close'].values, 1)[0]
        
        if (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0):
            patterns['volume_price_divergence'] = {
                'type': 'bullish' if price_trend > 0 else 'bearish',
                'strength': min(1.0, (abs(price_trend) + abs(volume_trend)) / 2)
            }
        
        return patterns
    
    def _identify_indicator_patterns(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify patterns in technical indicators.
        
        Args:
            technical_data: Technical analysis data for a cryptocurrency
            
        Returns:
            Dictionary with identified indicator patterns
        """
        patterns = {}
        
        # Get indicator data
        indicator_data = technical_data.get('indicator_data', [])
        
        if not indicator_data:
            return patterns
        
        # Convert to DataFrame if it's a list
        if isinstance(indicator_data, list):
            df = pd.DataFrame(indicator_data)
        else:
            df = indicator_data
        
        # Check for RSI patterns
        if 'rsi_14' in df.columns:
            latest_rsi = df['rsi_14'].iloc[-1]
            
            if latest_rsi < 30:
                patterns['rsi_oversold'] = {
                    'value': latest_rsi,
                    'strength': (30 - latest_rsi) / 30
                }
            elif latest_rsi > 70:
                patterns['rsi_overbought'] = {
                    'value': latest_rsi,
                    'strength': (latest_rsi - 70) / 30
                }
        
        # Check for MACD patterns
        if all(col in df.columns for col in ['macd_line', 'macd_signal', 'macd_histogram']):
            latest_macd = df['macd_line'].iloc[-1]
            latest_signal = df['macd_signal'].iloc[-1]
            latest_hist = df['macd_histogram'].iloc[-1]
            prev_hist = df['macd_histogram'].iloc[-2] if len(df) > 1 else 0
            
            # MACD crossover
            if latest_hist > 0 and prev_hist <= 0:
                patterns['macd_bullish_crossover'] = {
                    'strength': min(1.0, abs(latest_hist) * 10)
                }
            elif latest_hist < 0 and prev_hist >= 0:
                patterns['macd_bearish_crossover'] = {
                    'strength': min(1.0, abs(latest_hist) * 10)
                }
            
            # MACD divergence
            if len(df) >= 5:
                macd_trend = np.polyfit(range(5), df['macd_line'].tail(5).values, 1)[0]
                price_trend = np.polyfit(range(5), df['close'].tail(5).values, 1)[0]
                
                if (macd_trend > 0 and price_trend < 0):
                    patterns['macd_bullish_divergence'] = {
                        'strength': min(1.0, (abs(macd_trend) + abs(price_trend)) / 2)
                    }
                elif (macd_trend < 0 and price_trend > 0):
                    patterns['macd_bearish_divergence'] = {
                        'strength': min(1.0, (abs(macd_trend) + abs(price_trend)) / 2)
                    }
        
        # Check for Bollinger Band patterns
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            latest_close = df['close'].iloc[-1]
            latest_upper = df['bb_upper'].iloc[-1]
            latest_lower = df['bb_lower'].iloc[-1]
            
            # Price near bands
            if latest_close > latest_upper * 0.98:
                patterns['price_near_upper_band'] = {
                    'strength': min(1.0, (latest_close - latest_upper * 0.98) / (latest_upper * 0.02))
                }
            elif latest_close < latest_lower * 1.02:
                patterns['price_near_lower_band'] = {
                    'strength': min(1.0, (latest_lower * 1.02 - latest_close) / (latest_lower * 0.02))
                }
            
            # Bollinger Band squeeze
            band_width = (latest_upper - latest_lower) / latest_middle
            avg_band_width = (df['bb_upper'] - df['bb_lower']).mean() / df['bb_middle'].mean()
            
            if band_width < avg_band_width * 0.8:
                patterns['bollinger_band_squeeze'] = {
                    'strength': min(1.0, (avg_band_width - band_width) / avg_band_width)
                }
        
        return patterns
    
    def correlate_all_data(self, market_data: Dict[str, Any], 
                          technical_analysis: Dict[str, Any],
                          sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Correlate all data sources to generate comprehensive analysis.
        
        Args:
            market_data: Market data dictionary
            technical_analysis: Technical analysis results
            sentiment_analysis: Sentiment analysis results
            
        Returns:
            Dictionary with comprehensive correlation results
        """
        # Correlate technical and sentiment analysis
        tech_sentiment_correlation = self.correlate_technical_sentiment(
            technical_analysis, sentiment_analysis)
        
        # Identify patterns
        patterns = self.identify_patterns(market_data, technical_analysis)
        
        # Combine results
        correlation_results = {
            'timestamp': datetime.now().isoformat(),
            'cryptocurrencies': {}
        }
        
        # Get list of cryptocurrencies
        cryptocurrencies = self.config.get_cryptocurrencies()
        
        # Combine data for each cryptocurrency
        for symbol in cryptocurrencies:
            # Skip if data is missing
            if (symbol not in tech_sentiment_correlation.get('cryptocurrencies', {}) or
                symbol not in patterns.get('cryptocurrencies', {})):
                continue
            
            # Get correlation data
            correlation_data = tech_sentiment_correlation['cryptocurrencies'][symbol]
            pattern_data = patterns['cryptocurrencies'][symbol]
            
            # Combine data
            combined_data = {
                'signal_agreement': correlation_data['signal_agreement'],
                'combined_signal': correlation_data['combined_signal'],
                'confidence_score': correlation_data['confidence_score'],
                'patterns': pattern_data
            }
            
            # Add sentiment trend if available
            if 'sentiment_trend' in correlation_data:
                combined_data['sentiment_trend'] = correlation_data['sentiment_trend']
            
            # Add key topics if available
            if symbol in sentiment_analysis.get('key_topics', {}):
                combined_data['key_topics'] = sentiment_analysis['key_topics'][symbol]
            
            # Store combined results
            correlation_results['cryptocurrencies'][symbol] = combined_data
        
        return correlation_results
    
    def save_correlation_results(self, results: Dict[str, Any], 
                               filename: Optional[str] = None) -> str:
        """
        Save correlation results to a JSON file.
        
        Args:
            results: Correlation results to save
            filename: Output filename (generates a timestamped name if None)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"correlation_analysis_{timestamp}.json"
        
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        
        # Create data directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return output_path


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_collection.market_data import MarketDataCollector
    from data_collection.news_data import NewsDataCollector
    from analysis.technical import TechnicalAnalyzer
    from analysis.sentiment import SentimentAnalyzer
    
    # Collect data
    market_collector = MarketDataCollector()
    news_collector = NewsDataCollector()
    
    market_data = market_collector.get_market_data_for_analysis()
    news_data = news_collector.get_news_data_for_analysis()
    
    # Perform analysis
    technical_analyzer = TechnicalAnalyzer()
    sentiment_analyzer = SentimentAnalyzer()
    
    technical_analysis = technical_analyzer.analyze_market_data(market_data)
    sentiment_analysis = sentiment_analyzer.analyze_news_data(news_data)
    
    # Correlate data
    correlation_engine = CorrelationEngine()
    correlation_results = correlation_engine.correlate_all_data(
        market_data, technical_analysis, sentiment_analysis)
    
    # Print correlation results for each cryptocurrency
    for symbol, data in correlation_results['cryptocurrencies'].items():
        print(f"\n=== {symbol} Correlation Analysis ===")
        
        # Print signal agreement
        agreement = data['signal_agreement']
        print(f"Signal Agreement: {'Yes' if agreement['agreement'] else 'No'} "
             f"(Strength: {agreement['strength']:.2f})")
        print(f"Technical Signal: {agreement['technical_signal']} (Strength: {agreement['technical_strength']:.2f})")
        print(f"Sentiment Signal: {agreement['sentiment_signal']} (Strength: {agreement['sentiment_strength']:.2f})")
        
        # Print combined signal
        combined = data['combined_signal']
        print(f"\nCombined Signal: {combined['signal']} (Strength: {combined['strength']:.2f})")
        print(f"Technical Contribution: {combined['technical_contribution']:.2%}")
        print(f"Sentiment Contribution: {combined['sentiment_contribution']:.2%}")
        
        # Print confidence score
        print(f"\nConfidence Score: {data['confidence_score']:.2%}")
        
        # Print patterns
        if 'patterns' in data:
            patterns = data['patterns']
            
            if 'price_patterns' in patterns:
                print("\nPrice Patterns:")
                for pattern, details in patterns['price_patterns'].items():
                    if isinstance(details, dict):
                        print(f"- {pattern}: {details.get('strength', 0):.2f}")
                    else:
                        print(f"- {pattern}: {details}")
            
            if 'volume_patterns' in patterns:
                print("\nVolume Patterns:")
                for pattern, details in patterns['volume_patterns'].items():
                    if isinstance(details, dict):
                        print(f"- {pattern}: {details.get('strength', 0):.2f}")
                    else:
                        print(f"- {pattern}: {details}")
            
            if 'indicator_patterns' in patterns:
                print("\nIndicator Patterns:")
                for pattern, details in patterns['indicator_patterns'].items():
                    if isinstance(details, dict):
                        print(f"- {pattern}: {details.get('strength', 0):.2f}")
                    else:
                        print(f"- {pattern}: {details}")
        
        # Print key topics
        if 'key_topics' in data:
            print("\nKey Topics:", ", ".join(data['key_topics']))
    
    # Save correlation results
    output_path = correlation_engine.save_correlation_results(correlation_results)
    print(f"\nSaved correlation results to {output_path}")
