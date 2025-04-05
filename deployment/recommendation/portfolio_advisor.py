"""
Portfolio advisor module for the Cryptocurrency Analysis Bot.

This module provides specific investment recommendations for Binance trading
based on trading signals and risk assessment.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

class PortfolioAdvisor:
    """Provides investment recommendations based on trading signals."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the portfolio advisor.
        
        Args:
            config: Configuration object (creates a new one if None)
        """
        self.config = config or Config()
        self.max_per_trade = self.config.get('recommendation.position_sizing.max_per_trade', 0.1)
        self.risk_factor = self.config.get('recommendation.position_sizing.risk_factor', 0.02)
    
    def generate_recommendations(self, signals: Dict[str, Any], 
                               market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate investment recommendations based on trading signals.
        
        Args:
            signals: Trading signals from SignalGenerator
            market_data: Market data from MarketDataCollector
            
        Returns:
            Dictionary with investment recommendations
        """
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'buy': [],
                'sell': [],
                'hold': []
            },
            'cryptocurrencies': {}
        }
        
        # Process each cryptocurrency
        for symbol, signal_data in signals.get('cryptocurrencies', {}).items():
            # Skip if missing required data
            if 'action' not in signal_data:
                continue
            
            # Get current price
            current_price = market_data.get('current_prices', {}).get(symbol)
            if current_price is None:
                continue
            
            # Get action and confidence
            action = signal_data['action']
            confidence = signal_data['confidence']
            risk_level = signal_data['risk_level']
            
            # Calculate position size
            position_size = self._calculate_position_size(confidence, risk_level)
            
            # Determine entry/exit points
            entry_exit_points = self._determine_entry_exit_points(
                symbol, action, current_price, market_data, signal_data)
            
            # Generate recommendation
            recommendation = {
                'action': action,
                'confidence': confidence,
                'risk_level': risk_level,
                'current_price': current_price,
                'position_size': position_size,
                'entry_point': entry_exit_points.get('entry'),
                'exit_point': entry_exit_points.get('exit'),
                'stop_loss': entry_exit_points.get('stop_loss'),
                'take_profit': entry_exit_points.get('take_profit'),
                'supporting_factors': signal_data.get('supporting_factors', []),
                'risk_factors': signal_data.get('risk_factors', []),
                'time_horizon': self._determine_time_horizon(signal_data, market_data.get('historical_data', {}).get(symbol, []))
            }
            
            # Add to summary
            recommendations['summary'][action].append(symbol)
            
            # Store recommendation
            recommendations['cryptocurrencies'][symbol] = recommendation
        
        # Sort summary lists by confidence
        for action in ['buy', 'sell', 'hold']:
            recommendations['summary'][action] = sorted(
                recommendations['summary'][action],
                key=lambda x: recommendations['cryptocurrencies'][x]['confidence'],
                reverse=True
            )
        
        return recommendations
    
    def _calculate_position_size(self, confidence: float, risk_level: int) -> float:
        """
        Calculate recommended position size as percentage of portfolio.
        
        Args:
            confidence: Signal confidence (0-1)
            risk_level: Risk level (1-5, where 1 is lowest risk)
            
        Returns:
            Recommended position size (0-1)
        """
        # Base position size on max_per_trade
        base_size = self.max_per_trade
        
        # Adjust for confidence
        confidence_factor = confidence
        
        # Adjust for risk level (higher risk = smaller position)
        risk_adjustment = 1 - ((risk_level - 1) / 5)
        
        # Calculate final position size
        position_size = base_size * confidence_factor * risk_adjustment
        
        # Ensure position size is within limits
        return max(0.01, min(self.max_per_trade, position_size))
    
    def _determine_entry_exit_points(self, symbol: str, action: str, current_price: float,
                                   market_data: Dict[str, Any], 
                                   signal_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Determine entry and exit points for a trade.
        
        Args:
            symbol: Cryptocurrency symbol
            action: Trading action (buy, sell, hold)
            current_price: Current price of the cryptocurrency
            market_data: Market data from MarketDataCollector
            signal_data: Trading signal data
            
        Returns:
            Dictionary with entry, exit, stop loss, and take profit points
        """
        result = {
            'entry': None,
            'exit': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        # If action is hold, no entry/exit points needed
        if action == 'hold':
            return result
        
        # Get historical data
        historical_data = market_data.get('historical_data', {}).get(symbol, [])
        
        # Get order book
        order_book = market_data.get('order_books', {}).get(symbol, {'bids': [], 'asks': []})
        
        # Get risk level and confidence
        risk_level = signal_data.get('risk_level', 3)
        confidence = signal_data.get('confidence', 0.5)
        
        # Calculate volatility from historical data
        volatility = self._calculate_volatility(historical_data)
        
        if action == 'buy':
            # Entry point: slightly below current price for buy orders
            entry_discount = 0.005 * (1 + (5 - risk_level) / 10)  # 0.5% - 1% discount based on risk
            entry_point = current_price * (1 - entry_discount)
            
            # Find nearest support level from order book
            support_level = self._find_support_level(order_book, current_price)
            
            # Stop loss: below support level or based on volatility
            if support_level:
                stop_loss = support_level * 0.98  # 2% below support
            else:
                # Higher risk = tighter stop loss
                stop_loss_percentage = 0.05 + (risk_level * 0.01)  # 6% - 10% stop loss
                stop_loss = entry_point * (1 - stop_loss_percentage)
            
            # Take profit: based on volatility and confidence
            take_profit_percentage = 0.1 + (confidence * 0.1)  # 10% - 20% take profit
            take_profit = entry_point * (1 + take_profit_percentage)
            
            # Exit point: halfway to take profit
            exit_point = entry_point * (1 + (take_profit_percentage / 2))
            
            result = {
                'entry': entry_point,
                'exit': exit_point,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        
        elif action == 'sell':
            # Entry point: slightly above current price for sell orders
            entry_premium = 0.005 * (1 + (5 - risk_level) / 10)  # 0.5% - 1% premium based on risk
            entry_point = current_price * (1 + entry_premium)
            
            # Find nearest resistance level from order book
            resistance_level = self._find_resistance_level(order_book, current_price)
            
            # Stop loss: above resistance level or based on volatility
            if resistance_level:
                stop_loss = resistance_level * 1.02  # 2% above resistance
            else:
                # Higher risk = tighter stop loss
                stop_loss_percentage = 0.05 + (risk_level * 0.01)  # 6% - 10% stop loss
                stop_loss = entry_point * (1 + stop_loss_percentage)
            
            # Take profit: based on volatility and confidence
            take_profit_percentage = 0.1 + (confidence * 0.1)  # 10% - 20% take profit
            take_profit = entry_point * (1 - take_profit_percentage)
            
            # Exit point: halfway to take profit
            exit_point = entry_point * (1 - (take_profit_percentage / 2))
            
            result = {
                'entry': entry_point,
                'exit': exit_point,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        
        return result
    
    def _calculate_volatility(self, historical_data: List[Dict[str, Any]]) -> float:
        """
        Calculate price volatility from historical data.
        
        Args:
            historical_data: List of historical price data points
            
        Returns:
            Volatility as a percentage
        """
        if not historical_data:
            return 0.05  # Default 5% volatility
        
        # Extract close prices
        if isinstance(historical_data, list) and len(historical_data) > 0:
            if isinstance(historical_data[0], dict) and 'close' in historical_data[0]:
                close_prices = [float(item['close']) for item in historical_data]
            else:
                return 0.05  # Default if data format is unexpected
        else:
            return 0.05  # Default if data is empty
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(close_prices)):
            daily_return = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
            returns.append(daily_return)
        
        # Calculate standard deviation of returns
        if returns:
            import numpy as np
            volatility = np.std(returns)
            return volatility
        else:
            return 0.05  # Default if not enough data
    
    def _find_support_level(self, order_book: Dict[str, List], current_price: float) -> Optional[float]:
        """
        Find nearest support level from order book.
        
        Args:
            order_book: Order book data
            current_price: Current price of the cryptocurrency
            
        Returns:
            Support level price or None if not found
        """
        bids = order_book.get('bids', [])
        
        if not bids:
            return None
        
        # Sort bids by price (descending)
        sorted_bids = sorted(bids, key=lambda x: float(x[0]), reverse=True)
        
        # Find significant bid levels (large volume)
        significant_bids = []
        
        for price, quantity in sorted_bids:
            price = float(price)
            quantity = float(quantity)
            
            # Only consider bids below current price
            if price >= current_price:
                continue
            
            # Check if this is a significant bid (large volume)
            if quantity > sum([float(b[1]) for b in sorted_bids]) / len(sorted_bids) * 2:
                significant_bids.append(price)
        
        # Return highest significant bid below current price
        if significant_bids:
            return max(significant_bids)
        
        # If no significant bids, return the highest bid below current price
        for price, _ in sorted_bids:
            price = float(price)
            if price < current_price:
                return price
        
        return None
    
    def _find_resistance_level(self, order_book: Dict[str, List], current_price: float) -> Optional[float]:
        """
        Find nearest resistance level from order book.
        
        Args:
            order_book: Order book data
            current_price: Current price of the cryptocurrency
            
        Returns:
            Resistance level price or None if not found
        """
        asks = order_book.get('asks', [])
        
        if not asks:
            return None
        
        # Sort asks by price (ascending)
        sorted_asks = sorted(asks, key=lambda x: float(x[0]))
        
        # Find significant ask levels (large volume)
        significant_asks = []
        
        for price, quantity in sorted_asks:
            price = float(price)
            quantity = float(quantity)
            
            # Only consider asks above current price
            if price <= current_price:
                continue
            
            # Check if this is a significant ask (large volume)
            if quantity > sum([float(a[1]) for a in sorted_asks]) / len(sorted_asks) * 2:
                significant_asks.append(price)
        
        # Return lowest significant ask above current price
        if significant_asks:
            return min(significant_asks)
        
        # If no significant asks, return the lowest ask above current price
        for price, _ in sorted_asks:
            price = float(price)
            if price > current_price:
                return price
        
        return None
    
    def _determine_time_horizon(self, signal_data: Dict[str, Any], 
                              historical_data: List[Dict[str, Any]]) -> str:
        """
        Determine recommended time horizon for the trade.
        
        Args:
            signal_data: Trading signal data
            historical_data: Historical price data
            
        Returns:
            Recommended time horizon (short-term, medium-term, long-term)
        """
        # Get confidence and risk level
        confidence = signal_data.get('confidence', 0.5)
        risk_level = signal_data.get('risk_level', 3)
        
        # Get supporting and risk factors
        supporting_factors = signal_data.get('supporting_factors', [])
        risk_factors = signal_data.get('risk_factors', [])
        
        # Check for specific patterns that indicate time horizon
        short_term_indicators = [
            "RSI indicates oversold conditions",
            "RSI indicates overbought conditions",
            "MACD bullish crossover detected",
            "MACD bearish crossover detected",
            "Price near upper Bollinger Band",
            "Price near lower Bollinger Band",
            "Bollinger Band squeeze detected"
        ]
        
        medium_term_indicators = [
            "Price is in an uptrend",
            "Price is in a downtrend",
            "Trading volume is increasing",
            "Trading volume is decreasing",
            "Double top pattern detected",
            "Double bottom pattern detected"
        ]
        
        long_term_indicators = [
            "Strong technical bullish signal",
            "Strong technical bearish signal",
            "Strong sentiment bullish signal",
            "Strong sentiment bearish signal"
        ]
        
        # Count indicators for each time horizon
        short_term_count = sum(1 for factor in supporting_factors if any(indicator in factor for indicator in short_term_indicators))
        medium_term_count = sum(1 for factor in supporting_factors if any(indicator in factor for indicator in medium_term_indicators))
        long_term_count = sum(1 for factor in supporting_factors if any(indicator in factor for indicator in long_term_indicators))
        
        # Determine time horizon based on indicator counts
        if short_term_count > medium_term_count and short_term_count > long_term_count:
            return "short-term (1-7 days)"
        elif medium_term_count > short_term_count and medium_term_count > long_term_count:
            return "medium-term (1-4 weeks)"
        elif long_term_count > short_term_count and long_term_count > medium_term_count:
            return "long-term (1-3 months)"
        
        # If no clear winner, determine based on confidence and risk
        if confidence > 0.8 and risk_level <= 2:
            return "long-term (1-3 months)"
        elif confidence > 0.6 and risk_level <= 3:
            return "medium-term (1-4 weeks)"
        else:
            return "short-term (1-7 days)"
    
    def save_recommendations(self, recommendations: Dict[str, Any], 
                           filename: Optional[str] = None) -> str:
        """
        Save investment recommendations to a JSON file.
        
        Args:
            recommendations: Investment recommendations to save
            filename: Output filename (generates a timestamped name if None)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"investment_recommendations_{timestamp}.json"
        
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        
        # Create data directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        return output_path


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_collection.market_data import MarketDataCollector
    from data_collection.news_data import NewsDataCollector
    from analysis.technical import TechnicalAnalyzer
    from analysis.sentiment import SentimentAnalyzer
    from analysis.correlation import CorrelationEngine
    from recommendation.signal_generator import SignalGenerator
    
    # Collect data
    market_collector = MarketDataCollector()
    news_collector = NewsDataCollector()
    
    market_data = market_collector.get_market_data_for_analysis()
    news_data = news_collector.get_news_data_for_analysis()
    
    # Perform analysis
    technical_analyzer = TechnicalAnalyzer()
    sentiment_analyzer = SentimentAnalyzer()
    correlation_engine = CorrelationEngine()
    
    technical_analysis = technical_analyzer.analyze_market_data(market_data)
    sentiment_analysis = sentiment_analyzer.analyze_news_data(news_data)
    correlation_results = correlation_engine.correlate_all_data(
        market_data, technical_analysis, sentiment_analysis)
    
    # Generate signals
    signal_generator = SignalGenerator()
    signals = signal_generator.generate_signals(correlation_results)
    
    # Generate recommendations
    portfolio_advisor = PortfolioAdvisor()
    recommendations = portfolio_advisor.generate_recommendations(signals, market_data)
    
    # Print recommendations
    print("=== Investment Recommendations ===\n")
    
    # Print summary
    print("Summary:")
    print(f"Buy: {', '.join(recommendations['summary']['buy'])}")
    print(f"Sell: {', '.join(recommendations['summary']['sell'])}")
    print(f"Hold: {', '.join(recommendations['summary']['hold'])}")
    
    # Print detailed recommendations
    print("\nDetailed Recommendations:")
    for symbol, data in recommendations['cryptocurrencies'].items():
        print(f"\n{symbol} - {data['action'].upper()} (Confidence: {data['confidence']:.2%}, Risk: {data['risk_level']}/5)")
        print(f"Current Price: ${data['current_price']:.2f}")
        
        if data['action'] != 'hold':
            print(f"Position Size: {data['position_size']:.2%} of portfolio")
            print(f"Entry Point: ${data['entry_point']:.2f}")
            print(f"Exit Point: ${data['exit_point']:.2f}")
            print(f"Stop Loss: ${data['stop_loss']:.2f}")
            print(f"Take Profit: ${data['take_profit']:.2f}")
        
        print(f"Time Horizon: {data['time_horizon']}")
        
        if data['supporting_factors']:
            print("\nSupporting Factors:")
            for factor in data['supporting_factors']:
                print(f"- {factor}")
        
        if data['risk_factors']:
            print("\nRisk Factors:")
            for factor in data['risk_factors']:
                print(f"- {factor}")
    
    # Save recommendations
    output_path = portfolio_advisor.save_recommendations(recommendations)
    print(f"\nSaved investment recommendations to {output_path}")
