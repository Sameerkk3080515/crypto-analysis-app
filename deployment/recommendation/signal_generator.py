"""
Signal generator module for the Cryptocurrency Analysis Bot.

This module converts analysis results into clear buy/sell/hold signals
with confidence levels and risk assessments.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

class SignalGenerator:
    """Generates trading signals based on analysis results."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the signal generator.
        
        Args:
            config: Configuration object (creates a new one if None)
        """
        self.config = config or Config()
        self.min_confidence = self.config.get('recommendation.min_confidence', 0.7)
    
    def generate_signals(self, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals from correlation analysis results.
        
        Args:
            correlation_results: Results from the correlation engine
            
        Returns:
            Dictionary with trading signals for each cryptocurrency
        """
        signals = {
            'timestamp': datetime.now().isoformat(),
            'cryptocurrencies': {}
        }
        
        # Process each cryptocurrency
        for symbol, data in correlation_results.get('cryptocurrencies', {}).items():
            # Skip if missing required data
            if 'combined_signal' not in data or 'confidence_score' not in data:
                continue
            
            # Get combined signal and confidence
            combined_signal = data['combined_signal']['signal']
            signal_strength = data['combined_signal']['strength']
            confidence_score = data['confidence_score']
            
            # Determine action based on signal and confidence
            action = 'hold'  # Default action
            
            if combined_signal == 'bullish' and confidence_score >= self.min_confidence:
                action = 'buy'
            elif combined_signal == 'bearish' and confidence_score >= self.min_confidence:
                action = 'sell'
            
            # Calculate risk level (1-5, where 1 is lowest risk)
            risk_level = self._calculate_risk_level(data)
            
            # Generate signal data
            signal_data = {
                'action': action,
                'confidence': confidence_score,
                'strength': signal_strength,
                'risk_level': risk_level,
                'supporting_factors': self._identify_supporting_factors(data),
                'risk_factors': self._identify_risk_factors(data)
            }
            
            # Store signal
            signals['cryptocurrencies'][symbol] = signal_data
        
        return signals
    
    def _calculate_risk_level(self, correlation_data: Dict[str, Any]) -> int:
        """
        Calculate risk level for a trading signal.
        
        Args:
            correlation_data: Correlation data for a cryptocurrency
            
        Returns:
            Risk level (1-5, where 1 is lowest risk)
        """
        # Start with medium risk
        risk_level = 3
        
        # Adjust based on confidence score
        confidence = correlation_data.get('confidence_score', 0.5)
        if confidence > 0.8:
            risk_level -= 1
        elif confidence < 0.6:
            risk_level += 1
        
        # Adjust based on signal agreement
        if 'signal_agreement' in correlation_data:
            agreement = correlation_data['signal_agreement'].get('agreement', False)
            if agreement:
                risk_level -= 1
            else:
                risk_level += 1
        
        # Adjust based on patterns
        if 'patterns' in correlation_data:
            patterns = correlation_data['patterns']
            
            # Check for high-risk patterns
            high_risk_patterns = [
                'double_top', 'double_bottom', 'macd_bearish_divergence',
                'macd_bullish_divergence', 'bollinger_band_squeeze'
            ]
            
            for category in ['price_patterns', 'indicator_patterns']:
                if category in patterns:
                    for pattern in high_risk_patterns:
                        if pattern in patterns[category]:
                            risk_level += 1
                            break
            
            # Check for volume-price divergence
            if 'volume_patterns' in patterns:
                if 'volume_price_divergence' in patterns['volume_patterns']:
                    risk_level += 1
        
        # Ensure risk level is between 1 and 5
        return max(1, min(5, risk_level))
    
    def _identify_supporting_factors(self, correlation_data: Dict[str, Any]) -> List[str]:
        """
        Identify factors supporting the trading signal.
        
        Args:
            correlation_data: Correlation data for a cryptocurrency
            
        Returns:
            List of supporting factors
        """
        supporting_factors = []
        
        # Get combined signal
        combined_signal = correlation_data.get('combined_signal', {}).get('signal', 'neutral')
        
        # Check for signal agreement
        if 'signal_agreement' in correlation_data:
            agreement = correlation_data['signal_agreement'].get('agreement', False)
            if agreement:
                supporting_factors.append("Technical and sentiment signals are in agreement")
        
        # Check for strong technical signal
        if 'signal_agreement' in correlation_data:
            technical_signal = correlation_data['signal_agreement'].get('technical_signal', 'neutral')
            technical_strength = correlation_data['signal_agreement'].get('technical_strength', 0)
            
            if technical_signal == combined_signal and technical_strength > 0.7:
                supporting_factors.append(f"Strong technical {technical_signal} signal")
        
        # Check for strong sentiment signal
        if 'signal_agreement' in correlation_data:
            sentiment_signal = correlation_data['signal_agreement'].get('sentiment_signal', 'neutral')
            sentiment_strength = correlation_data['signal_agreement'].get('sentiment_strength', 0)
            
            if sentiment_signal == combined_signal and sentiment_strength > 0.7:
                supporting_factors.append(f"Strong sentiment {sentiment_signal} signal")
        
        # Check for sentiment trend
        if 'sentiment_trend' in correlation_data:
            trend_direction = correlation_data['sentiment_trend'].get('direction', 'stable')
            
            if (combined_signal == 'bullish' and trend_direction == 'improving') or \
               (combined_signal == 'bearish' and trend_direction == 'deteriorating'):
                supporting_factors.append(f"Sentiment trend is {trend_direction}")
        
        # Check for supporting patterns
        if 'patterns' in correlation_data:
            patterns = correlation_data['patterns']
            
            # Check price patterns
            if 'price_patterns' in patterns:
                price_patterns = patterns['price_patterns']
                
                if combined_signal == 'bullish':
                    if 'trend' in price_patterns and price_patterns['trend'] == 'uptrend':
                        supporting_factors.append("Price is in an uptrend")
                    
                    if 'double_bottom' in price_patterns:
                        supporting_factors.append("Double bottom pattern detected")
                
                elif combined_signal == 'bearish':
                    if 'trend' in price_patterns and price_patterns['trend'] == 'downtrend':
                        supporting_factors.append("Price is in a downtrend")
                    
                    if 'double_top' in price_patterns:
                        supporting_factors.append("Double top pattern detected")
            
            # Check volume patterns
            if 'volume_patterns' in patterns:
                volume_patterns = patterns['volume_patterns']
                
                if combined_signal == 'bullish':
                    if 'volume_trend' in volume_patterns and volume_patterns['volume_trend'] == 'increasing':
                        supporting_factors.append("Trading volume is increasing")
                    
                    if 'volume_price_divergence' in volume_patterns and \
                       volume_patterns['volume_price_divergence']['type'] == 'bullish':
                        supporting_factors.append("Bullish volume-price divergence detected")
                
                elif combined_signal == 'bearish':
                    if 'volume_spike' in volume_patterns and 'volume_trend' in volume_patterns and \
                       volume_patterns['volume_trend'] == 'decreasing':
                        supporting_factors.append("Volume spike with decreasing trend detected")
                    
                    if 'volume_price_divergence' in volume_patterns and \
                       volume_patterns['volume_price_divergence']['type'] == 'bearish':
                        supporting_factors.append("Bearish volume-price divergence detected")
            
            # Check indicator patterns
            if 'indicator_patterns' in patterns:
                indicator_patterns = patterns['indicator_patterns']
                
                if combined_signal == 'bullish':
                    if 'rsi_oversold' in indicator_patterns:
                        supporting_factors.append("RSI indicates oversold conditions")
                    
                    if 'macd_bullish_crossover' in indicator_patterns:
                        supporting_factors.append("MACD bullish crossover detected")
                    
                    if 'macd_bullish_divergence' in indicator_patterns:
                        supporting_factors.append("MACD bullish divergence detected")
                    
                    if 'price_near_lower_band' in indicator_patterns:
                        supporting_factors.append("Price near lower Bollinger Band")
                
                elif combined_signal == 'bearish':
                    if 'rsi_overbought' in indicator_patterns:
                        supporting_factors.append("RSI indicates overbought conditions")
                    
                    if 'macd_bearish_crossover' in indicator_patterns:
                        supporting_factors.append("MACD bearish crossover detected")
                    
                    if 'macd_bearish_divergence' in indicator_patterns:
                        supporting_factors.append("MACD bearish divergence detected")
                    
                    if 'price_near_upper_band' in indicator_patterns:
                        supporting_factors.append("Price near upper Bollinger Band")
        
        return supporting_factors
    
    def _identify_risk_factors(self, correlation_data: Dict[str, Any]) -> List[str]:
        """
        Identify risk factors for the trading signal.
        
        Args:
            correlation_data: Correlation data for a cryptocurrency
            
        Returns:
            List of risk factors
        """
        risk_factors = []
        
        # Get combined signal
        combined_signal = correlation_data.get('combined_signal', {}).get('signal', 'neutral')
        
        # Check for signal disagreement
        if 'signal_agreement' in correlation_data:
            agreement = correlation_data['signal_agreement'].get('agreement', False)
            if not agreement:
                risk_factors.append("Technical and sentiment signals disagree")
        
        # Check for weak confidence
        confidence = correlation_data.get('confidence_score', 0.5)
        if confidence < 0.6:
            risk_factors.append("Low confidence in the signal")
        
        # Check for contradicting sentiment trend
        if 'sentiment_trend' in correlation_data:
            trend_direction = correlation_data['sentiment_trend'].get('direction', 'stable')
            
            if (combined_signal == 'bullish' and trend_direction == 'deteriorating') or \
               (combined_signal == 'bearish' and trend_direction == 'improving'):
                risk_factors.append(f"Sentiment trend is {trend_direction}, contradicting the signal")
        
        # Check for contradicting patterns
        if 'patterns' in correlation_data:
            patterns = correlation_data['patterns']
            
            # Check price patterns
            if 'price_patterns' in patterns:
                price_patterns = patterns['price_patterns']
                
                if combined_signal == 'bullish':
                    if 'trend' in price_patterns and price_patterns['trend'] == 'downtrend':
                        risk_factors.append("Price is in a downtrend, contradicting bullish signal")
                    
                    if 'double_top' in price_patterns:
                        risk_factors.append("Double top pattern detected, contradicting bullish signal")
                
                elif combined_signal == 'bearish':
                    if 'trend' in price_patterns and price_patterns['trend'] == 'uptrend':
                        risk_factors.append("Price is in an uptrend, contradicting bearish signal")
                    
                    if 'double_bottom' in price_patterns:
                        risk_factors.append("Double bottom pattern detected, contradicting bearish signal")
            
            # Check volume patterns
            if 'volume_patterns' in patterns:
                volume_patterns = patterns['volume_patterns']
                
                if combined_signal == 'bullish':
                    if 'volume_trend' in volume_patterns and volume_patterns['volume_trend'] == 'decreasing':
                        risk_factors.append("Trading volume is decreasing, contradicting bullish signal")
                    
                    if 'volume_price_divergence' in volume_patterns and \
                       volume_patterns['volume_price_divergence']['type'] == 'bearish':
                        risk_factors.append("Bearish volume-price divergence detected")
                
                elif combined_signal == 'bearish':
                    if 'volume_trend' in volume_patterns and volume_patterns['volume_trend'] == 'increasing':
                        risk_factors.append("Trading volume is increasing, contradicting bearish signal")
                    
                    if 'volume_price_divergence' in volume_patterns and \
                       volume_patterns['volume_price_divergence']['type'] == 'bullish':
                        risk_factors.append("Bullish volume-price divergence detected")
            
            # Check indicator patterns
            if 'indicator_patterns' in patterns:
                indicator_patterns = patterns['indicator_patterns']
                
                if combined_signal == 'bullish':
                    if 'rsi_overbought' in indicator_patterns:
                        risk_factors.append("RSI indicates overbought conditions, contradicting bullish signal")
                    
                    if 'macd_bearish_crossover' in indicator_patterns:
                        risk_factors.append("MACD bearish crossover detected, contradicting bullish signal")
                    
                    if 'price_near_upper_band' in indicator_patterns:
                        risk_factors.append("Price near upper Bollinger Band, contradicting bullish signal")
                
                elif combined_signal == 'bearish':
                    if 'rsi_oversold' in indicator_patterns:
                        risk_factors.append("RSI indicates oversold conditions, contradicting bearish signal")
                    
                    if 'macd_bullish_crossover' in indicator_patterns:
                        risk_factors.append("MACD bullish crossover detected, contradicting bearish signal")
                    
                    if 'price_near_lower_band' in indicator_patterns:
                        risk_factors.append("Price near lower Bollinger Band, contradicting bearish signal")
            
            # Check for volatility
            if 'indicator_patterns' in patterns:
                if 'bollinger_band_squeeze' in patterns['indicator_patterns']:
                    risk_factors.append("Bollinger Band squeeze detected, indicating potential volatility")
        
        return risk_factors
    
    def save_signals(self, signals: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save trading signals to a JSON file.
        
        Args:
            signals: Trading signals to save
            filename: Output filename (generates a timestamped name if None)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trading_signals_{timestamp}.json"
        
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        
        # Create data directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(signals, f, indent=2)
        
        return output_path
