"""
Command-line interface module for the Cryptocurrency Analysis Bot.

This module provides a simple command-line interface for running the bot
and viewing investment recommendations.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config
from data_collection.market_data import MarketDataCollector
from data_collection.news_data import NewsDataCollector
from analysis.technical import TechnicalAnalyzer
from analysis.sentiment import SentimentAnalyzer
from analysis.correlation import CorrelationEngine
from recommendation.signal_generator import SignalGenerator
from recommendation.portfolio_advisor import PortfolioAdvisor

class CommandLineInterface:
    """Command-line interface for the Cryptocurrency Analysis Bot."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the command-line interface.
        
        Args:
            config: Configuration object (creates a new one if None)
        """
        self.config = config or Config()
        
        # Initialize components
        self.market_collector = MarketDataCollector(self.config)
        self.news_collector = NewsDataCollector(self.config)
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.correlation_engine = CorrelationEngine(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.portfolio_advisor = PortfolioAdvisor(self.config)
        
        # Create data directory
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        os.makedirs(self.data_dir, exist_ok=True)
    
    def run_analysis(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            verbose: Whether to print progress messages
            
        Returns:
            Dictionary with investment recommendations
        """
        if verbose:
            print("Starting cryptocurrency analysis...")
            print("Collecting market data...")
        
        # Collect market data
        market_data = self.market_collector.get_market_data_for_analysis()
        market_data_path = self.market_collector.save_market_data(market_data)
        
        if verbose:
            print(f"Market data saved to {market_data_path}")
            print("Collecting news data...")
        
        # Collect news data
        news_data = self.news_collector.get_news_data_for_analysis()
        news_data_path = self.news_collector.save_news_data(news_data)
        
        if verbose:
            print(f"News data saved to {news_data_path}")
            print("Performing technical analysis...")
        
        # Perform technical analysis
        technical_analysis = self.technical_analyzer.analyze_market_data(market_data)
        technical_analysis_path = self.technical_analyzer.save_analysis_results(technical_analysis)
        
        if verbose:
            print(f"Technical analysis saved to {technical_analysis_path}")
            print("Performing sentiment analysis...")
        
        # Perform sentiment analysis
        sentiment_analysis = self.sentiment_analyzer.analyze_news_data(news_data)
        sentiment_analysis_path = self.sentiment_analyzer.save_analysis_results(sentiment_analysis)
        
        if verbose:
            print(f"Sentiment analysis saved to {sentiment_analysis_path}")
            print("Correlating analysis results...")
        
        # Correlate analysis results
        correlation_results = self.correlation_engine.correlate_all_data(
            market_data, technical_analysis, sentiment_analysis)
        correlation_path = self.correlation_engine.save_correlation_results(correlation_results)
        
        if verbose:
            print(f"Correlation results saved to {correlation_path}")
            print("Generating trading signals...")
        
        # Generate trading signals
        signals = self.signal_generator.generate_signals(correlation_results)
        signals_path = self.signal_generator.save_signals(signals)
        
        if verbose:
            print(f"Trading signals saved to {signals_path}")
            print("Generating investment recommendations...")
        
        # Generate investment recommendations
        recommendations = self.portfolio_advisor.generate_recommendations(signals, market_data)
        recommendations_path = self.portfolio_advisor.save_recommendations(recommendations)
        
        if verbose:
            print(f"Investment recommendations saved to {recommendations_path}")
            print("Analysis complete!")
        
        return recommendations
    
    def display_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """
        Display investment recommendations in a user-friendly format.
        
        Args:
            recommendations: Investment recommendations from PortfolioAdvisor
        """
        print("\n" + "="*80)
        print(" "*25 + "CRYPTOCURRENCY INVESTMENT RECOMMENDATIONS")
        print("="*80)
        
        # Print timestamp
        timestamp = datetime.fromisoformat(recommendations['timestamp'])
        print(f"\nAnalysis Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print summary
        print("\n" + "-"*80)
        print("SUMMARY OF RECOMMENDATIONS")
        print("-"*80)
        
        buy_list = recommendations['summary']['buy']
        sell_list = recommendations['summary']['sell']
        hold_list = recommendations['summary']['hold']
        
        if buy_list:
            print(f"\nBUY: {', '.join(buy_list)}")
        else:
            print("\nBUY: None")
        
        if sell_list:
            print(f"SELL: {', '.join(sell_list)}")
        else:
            print("SELL: None")
        
        if hold_list:
            print(f"HOLD: {', '.join(hold_list)}")
        else:
            print("HOLD: None")
        
        # Print detailed recommendations
        print("\n" + "-"*80)
        print("DETAILED RECOMMENDATIONS")
        print("-"*80)
        
        # Process buy recommendations first
        if buy_list:
            print("\nBUY RECOMMENDATIONS:")
            for symbol in buy_list:
                self._display_detailed_recommendation(symbol, recommendations['cryptocurrencies'][symbol])
        
        # Process sell recommendations
        if sell_list:
            print("\nSELL RECOMMENDATIONS:")
            for symbol in sell_list:
                self._display_detailed_recommendation(symbol, recommendations['cryptocurrencies'][symbol])
        
        # Process hold recommendations
        if hold_list:
            print("\nHOLD RECOMMENDATIONS:")
            for symbol in hold_list:
                self._display_detailed_recommendation(symbol, recommendations['cryptocurrencies'][symbol])
        
        print("\n" + "="*80)
        print(" "*15 + "NOTE: These recommendations are for informational purposes only.")
        print(" "*15 + "Always conduct your own research before making investment decisions.")
        print("="*80 + "\n")
    
    def _display_detailed_recommendation(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Display detailed recommendation for a single cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            data: Recommendation data for the cryptocurrency
        """
        action = data['action'].upper()
        confidence = data['confidence']
        risk_level = data['risk_level']
        current_price = data['current_price']
        
        # Print header
        print(f"\n{symbol} - {action}")
        print("-" * (len(symbol) + len(action) + 3))
        
        # Print basic info
        print(f"Confidence: {confidence:.2%}")
        print(f"Risk Level: {risk_level}/5")
        print(f"Current Price: ${current_price:.2f}")
        
        # Print trade details if not hold
        if action != "HOLD":
            position_size = data['position_size']
            entry_point = data['entry_point']
            exit_point = data['exit_point']
            stop_loss = data['stop_loss']
            take_profit = data['take_profit']
            
            print(f"Position Size: {position_size:.2%} of portfolio")
            print(f"Entry Point: ${entry_point:.2f}")
            print(f"Exit Point: ${exit_point:.2f}")
            print(f"Stop Loss: ${stop_loss:.2f} ({abs(stop_loss - entry_point) / entry_point:.2%} from entry)")
            print(f"Take Profit: ${take_profit:.2f} ({abs(take_profit - entry_point) / entry_point:.2%} from entry)")
        
        # Print time horizon
        print(f"Time Horizon: {data['time_horizon']}")
        
        # Print supporting factors
        if data['supporting_factors']:
            print("\nSupporting Factors:")
            for factor in data['supporting_factors']:
                print(f"✓ {factor}")
        
        # Print risk factors
        if data['risk_factors']:
            print("\nRisk Factors:")
            for factor in data['risk_factors']:
                print(f"! {factor}")
    
    def generate_report(self, recommendations: Dict[str, Any], 
                       output_file: Optional[str] = None) -> str:
        """
        Generate a detailed HTML report of investment recommendations.
        
        Args:
            recommendations: Investment recommendations from PortfolioAdvisor
            output_file: Output file path (generates a timestamped name if None)
            
        Returns:
            Path to the generated report
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.data_dir, f"crypto_recommendations_{timestamp}.html")
        
        # Generate HTML content
        html_content = self._generate_html_report(recommendations)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file
    
    def _generate_html_report(self, recommendations: Dict[str, Any]) -> str:
        """
        Generate HTML content for the report.
        
        Args:
            recommendations: Investment recommendations from PortfolioAdvisor
            
        Returns:
            HTML content as a string
        """
        # Get timestamp
        timestamp = datetime.fromisoformat(recommendations['timestamp'])
        formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Get summary lists
        buy_list = recommendations['summary']['buy']
        sell_list = recommendations['summary']['sell']
        hold_list = recommendations['summary']['hold']
        
        # Start HTML content
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Cryptocurrency Investment Recommendations</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-style: italic;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .summary {{
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 30px;
                }}
                .summary-section {{
                    margin-bottom: 15px;
                }}
                .recommendations {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .recommendation-card {{
                    flex: 1;
                    min-width: 300px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 20px;
                    margin-bottom: 20px;
                }}
                .buy {{
                    border-left: 5px solid #27ae60;
                }}
                .sell {{
                    border-left: 5px solid #e74c3c;
                }}
                .hold {{
                    border-left: 5px solid #f39c12;
                }}
                .crypto-symbol {{
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .action {{
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 3px;
                    color: white;
                    font-weight: bold;
                    margin-bottom: 15px;
                }}
                .action-buy {{
                    background-color: #27ae60;
                }}
                .action-sell {{
                    background-color: #e74c3c;
                }}
                .action-hold {{
                    background-color: #f39c12;
                }}
                .price-info {{
                    margin-bottom: 15px;
                }}
                .trade-details {{
                    background-color: #f9f9f9;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 15px;
                }}
                .factors {{
                    margin-top: 15px;
                }}
                .supporting-factor {{
                    color: #27ae60;
                    margin-bottom: 5px;
                }}
                .risk-factor {{
                    color: #e74c3c;
                    margin-bottom: 5px;
                }}
                .confidence-meter {{
                    height: 10px;
                    background-color: #ecf0f1;
                    border-radius: 5px;
                    margin-bottom: 15px;
                }}
                .confidence-level {{
                    height: 100%;
                    background-color: #3498db;
                    border-radius: 5px;
                }}
                .risk-meter {{
                    display: flex;
                    margin-bottom: 15px;
                }}
                .risk-segment {{
                    flex: 1;
                    height: 10px;
                    background-color: #ecf0f1;
                    margin-right: 2px;
                }}
                .risk-active {{
                    background-color: #e74c3c;
                }}
                .disclaimer {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    font-style: italic;
                    text-align: center;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Cryptocurrency Investment Recommendations</h1>
            </div>
            
            <div class="timestamp">
                Analysis Date: {formatted_timestamp}
            </div>
            
            <div class="summary">
                <h2>Summary of Recommendations</h2>
                <div class="summary-section">
                    <h3>Buy Recommendations</h3>
                    <p>{', '.join(buy_list) if buy_list else 'None'}</p>
                </div>
                <div class="summary-section">
                    <h3>Sell Recommendations</h3>
                    <p>{', '.join(sell_list) if sell_list else 'None'}</p>
                </div>
                <div class="summary-section">
                    <h3>Hold Recommendations</h3>
                    <p>{', '.join(hold_list) if hold_list else 'None'}</p>
                </div>
            </div>
            
            <h2>Detailed Recommendations</h2>
        """
        
        # Add buy recommendations
        if buy_list:
            html += '<h3>Buy Recommendations</h3><div class="recommendations">'
            for symbol in buy_list:
                html += self._generate_recommendation_card(symbol, recommendations['cryptocurrencies'][symbol], 'buy')
            html += '</div>'
        
        # Add sell recommendations
        if sell_list:
            html += '<h3>Sell Recommendations</h3><div class="recommendations">'
            for symbol in sell_list:
                html += self._generate_recommendation_card(symbol, recommendations['cryptocurrencies'][symbol], 'sell')
            html += '</div>'
        
        # Add hold recommendations
        if hold_list:
            html += '<h3>Hold Recommendations</h3><div class="recommendations">'
            for symbol in hold_list:
                html += self._generate_recommendation_card(symbol, recommendations['cryptocurrencies'][symbol], 'hold')
            html += '</div>'
        
        # Add disclaimer and close HTML
        html += """
            <div class="disclaimer">
                <p>These recommendations are for informational purposes only. Always conduct your own research before making investment decisions.</p>
                <p>Past performance is not indicative of future results. Cryptocurrency investments are subject to high market risk.</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_recommendation_card(self, symbol: str, data: Dict[str, Any], action_type: str) -> str:
        """
        Generate HTML for a recommendation card.
        
        Args:
            symbol: Cryptocurrency symbol
            data: Recommendation data for the cryptocurrency
            action_type: Type of action (buy, sell, hold)
            
        Returns:
            HTML content for the recommendation card
        """
        action = data['action'].upper()
        confidence = data['confidence']
        risk_level = data['risk_level']
        current_price = data['current_price']
        time_horizon = data['time_horizon']
        
        # Generate confidence meter
        confidence_html = f"""
        <div>
            <strong>Confidence: {confidence:.2%}</strong>
            <div class="confidence-meter">
                <div class="confidence-level" style="width: {confidence * 100}%;"></div>
            </div>
        </div>
        """
        
        # Generate risk meter
        risk_html = '<div><strong>Risk Level: {}/5</strong><div class="risk-meter">'.format(risk_level)
        for i in range(1, 6):
            risk_html += f'<div class="risk-segment{" risk-active" if i <= risk_level else ""}"></div>'
        risk_html += '</div></div>'
        
        # Generate trade details
        trade_html = ""
        if action != "HOLD":
            position_size = data['position_size']
            entry_point = data['entry_point']
            exit_point = data['exit_point']
            stop_loss = data['stop_loss']
            take_profit = data['take_profit']
            
            trade_html = f"""
            <div class="trade-details">
                <p><strong>Position Size:</strong> {position_size:.2%} of portfolio</p>
                <p><strong>Entry Point:</strong> ${entry_point:.2f}</p>
                <p><strong>Exit Point:</strong> ${exit_point:.2f}</p>
                <p><strong>Stop Loss:</strong> ${stop_loss:.2f} ({abs(stop_loss - entry_point) / entry_point:.2%} from entry)</p>
                <p><strong>Take Profit:</strong> ${take_profit:.2f} ({abs(take_profit - entry_point) / entry_point:.2%} from entry)</p>
            </div>
            """
        
        # Generate supporting factors
        supporting_factors_html = ""
        if data['supporting_factors']:
            supporting_factors_html = '<div class="factors"><strong>Supporting Factors:</strong>'
            for factor in data['supporting_factors']:
                supporting_factors_html += f'<div class="supporting-factor">✓ {factor}</div>'
            supporting_factors_html += '</div>'
        
        # Generate risk factors
        risk_factors_html = ""
        if data['risk_factors']:
            risk_factors_html = '<div class="factors"><strong>Risk Factors:</strong>'
            for factor in data['risk_factors']:
                risk_factors_html += f'<div class="risk-factor">! {factor}</div>'
            risk_factors_html += '</div>'
        
        # Generate complete card
        card_html = f"""
        <div class="recommendation-card {action_type}">
            <div class="crypto-symbol">{symbol}</div>
            <div class="action action-{action_type}">{action}</div>
            
            {confidence_html}
            {risk_html}
            
            <div class="price-info">
                <p><strong>Current Price:</strong> ${current_price:.2f}</p>
                <p><strong>Time Horizon:</strong> {time_horizon}</p>
            </div>
            
            {trade_html}
            {supporting_factors_html}
            {risk_factors_html}
        </div>
        """
        
        return card_html


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Analysis Bot')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    parser.add_argument('--report', '-r', action='store_true', help='Generate HTML report')
    parser.add_argument('--output', '-o', type=str, help='Output file for HTML report')
    return parser.parse_args()


def main():
    """Main entry point for the command-line interface."""
    args = parse_arguments()
    
    # Create CLI
    cli = CommandLineInterface()
    
    # Run analysis
    recommendations = cli.run_analysis(verbose=args.verbose)
    
    # Display recommendations
    cli.display_recommendations(recommendations)
    
    # Generate report if requested
    if args.report:
        report_path = cli.generate_report(recommendations, args.output)
        print(f"\nHTML report generated: {report_path}")


if __name__ == "__main__":
    main()
