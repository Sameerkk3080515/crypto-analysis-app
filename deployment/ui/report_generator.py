"""
Report generator module for the Cryptocurrency Analysis Bot.

This module generates detailed reports of cryptocurrency analysis and investment recommendations.
"""

import os
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

class ReportGenerator:
    """Generates detailed reports of cryptocurrency analysis and recommendations."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the report generator.
        
        Args:
            config: Configuration object (creates a new one if None)
        """
        self.config = config or Config()
        
        # Create output directories
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        self.reports_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'reports'
        )
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, 'images'), exist_ok=True)
    
    def generate_price_chart(self, symbol: str, historical_data: List[Dict[str, Any]], 
                           signals: Dict[str, Any] = None) -> str:
        """
        Generate a price chart for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            historical_data: Historical price data
            signals: Trading signals (optional)
            
        Returns:
            Path to the generated chart image
        """
        # Convert historical data to DataFrame
        if not historical_data:
            return ""
            
        df = pd.DataFrame(historical_data)
        
        # Ensure required columns exist
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return ""
        
        # Convert timestamps if they're strings
        if isinstance(df['open_time'][0], str):
            df['open_time'] = pd.to_datetime(df['open_time'])
        
        # Sort by time
        df = df.sort_values('open_time')
        
        # Create figure and axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        ax1.plot(df['open_time'], df['close'], label='Close Price')
        
        # Add moving averages if available
        for ma_period in [20, 50, 200]:
            if f'sma_{ma_period}' in df.columns:
                ax1.plot(df['open_time'], df[f'sma_{ma_period}'], 
                        label=f'SMA {ma_period}', alpha=0.7)
        
        # Add Bollinger Bands if available
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            ax1.plot(df['open_time'], df['bb_upper'], 'r--', alpha=0.3)
            ax1.plot(df['open_time'], df['bb_middle'], 'g--', alpha=0.3)
            ax1.plot(df['open_time'], df['bb_lower'], 'r--', alpha=0.3)
            ax1.fill_between(df['open_time'], df['bb_upper'], df['bb_lower'], 
                           color='gray', alpha=0.1)
        
        # Add buy/sell signals if available
        if signals and symbol in signals.get('cryptocurrencies', {}):
            signal_data = signals['cryptocurrencies'][symbol]
            action = signal_data.get('action', 'hold')
            
            if action == 'buy':
                ax1.axhline(y=signal_data.get('entry_point', 0), color='g', linestyle='--', alpha=0.7)
                ax1.axhline(y=signal_data.get('stop_loss', 0), color='r', linestyle='--', alpha=0.7)
                ax1.axhline(y=signal_data.get('take_profit', 0), color='b', linestyle='--', alpha=0.7)
                
                # Add annotations
                last_date = df['open_time'].iloc[-1]
                ax1.annotate('Entry', xy=(last_date, signal_data.get('entry_point', 0)), 
                           xytext=(5, 0), textcoords='offset points', color='g')
                ax1.annotate('Stop Loss', xy=(last_date, signal_data.get('stop_loss', 0)), 
                           xytext=(5, 0), textcoords='offset points', color='r')
                ax1.annotate('Take Profit', xy=(last_date, signal_data.get('take_profit', 0)), 
                           xytext=(5, 0), textcoords='offset points', color='b')
            
            elif action == 'sell':
                ax1.axhline(y=signal_data.get('entry_point', 0), color='r', linestyle='--', alpha=0.7)
                ax1.axhline(y=signal_data.get('stop_loss', 0), color='g', linestyle='--', alpha=0.7)
                ax1.axhline(y=signal_data.get('take_profit', 0), color='b', linestyle='--', alpha=0.7)
                
                # Add annotations
                last_date = df['open_time'].iloc[-1]
                ax1.annotate('Entry', xy=(last_date, signal_data.get('entry_point', 0)), 
                           xytext=(5, 0), textcoords='offset points', color='r')
                ax1.annotate('Stop Loss', xy=(last_date, signal_data.get('stop_loss', 0)), 
                           xytext=(5, 0), textcoords='offset points', color='g')
                ax1.annotate('Take Profit', xy=(last_date, signal_data.get('take_profit', 0)), 
                           xytext=(5, 0), textcoords='offset points', color='b')
        
        # Configure price chart
        ax1.set_title(f'{symbol} Price Chart')
        ax1.set_ylabel('Price (USD)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Plot volume
        ax2.bar(df['open_time'], df['volume'], color='blue', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        
        # Configure x-axis for both charts
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(self.reports_dir, 'images', f'{symbol}_chart.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def generate_technical_indicators_chart(self, symbol: str, 
                                          historical_data: List[Dict[str, Any]]) -> str:
        """
        Generate a chart of technical indicators for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            historical_data: Historical price data with indicators
            
        Returns:
            Path to the generated chart image
        """
        # Convert historical data to DataFrame
        if not historical_data:
            return ""
            
        df = pd.DataFrame(historical_data)
        
        # Ensure required columns exist
        required_columns = ['open_time', 'close']
        if not all(col in df.columns for col in required_columns):
            return ""
        
        # Convert timestamps if they're strings
        if isinstance(df['open_time'][0], str):
            df['open_time'] = pd.to_datetime(df['open_time'])
        
        # Sort by time
        df = df.sort_values('open_time')
        
        # Check which indicators are available
        has_rsi = 'rsi_14' in df.columns
        has_macd = all(col in df.columns for col in ['macd_line', 'macd_signal', 'macd_histogram'])
        
        # Determine number of subplots
        n_plots = 1 + has_rsi + has_macd
        
        # Create figure and axes
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
        
        # If only one subplot, convert axes to list for consistent indexing
        if n_plots == 1:
            axes = [axes]
        
        # Plot price
        ax_idx = 0
        axes[ax_idx].plot(df['open_time'], df['close'], label='Close Price')
        axes[ax_idx].set_title(f'{symbol} Price and Technical Indicators')
        axes[ax_idx].set_ylabel('Price (USD)')
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].legend(loc='upper left')
        
        # Plot RSI if available
        if has_rsi:
            ax_idx += 1
            axes[ax_idx].plot(df['open_time'], df['rsi_14'], label='RSI(14)', color='purple')
            axes[ax_idx].axhline(y=70, color='r', linestyle='--', alpha=0.5)
            axes[ax_idx].axhline(y=30, color='g', linestyle='--', alpha=0.5)
            axes[ax_idx].fill_between(df['open_time'], 70, 100, color='red', alpha=0.1)
            axes[ax_idx].fill_between(df['open_time'], 0, 30, color='green', alpha=0.1)
            axes[ax_idx].set_ylabel('RSI')
            axes[ax_idx].set_ylim(0, 100)
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].legend(loc='upper left')
        
        # Plot MACD if available
        if has_macd:
            ax_idx += 1
            axes[ax_idx].plot(df['open_time'], df['macd_line'], label='MACD Line', color='blue')
            axes[ax_idx].plot(df['open_time'], df['macd_signal'], label='Signal Line', color='red')
            
            # Plot histogram as bars
            positive = df['macd_histogram'] > 0
            negative = df['macd_histogram'] <= 0
            
            axes[ax_idx].bar(df.loc[positive, 'open_time'], df.loc[positive, 'macd_histogram'], 
                           color='green', alpha=0.5, width=1)
            axes[ax_idx].bar(df.loc[negative, 'open_time'], df.loc[negative, 'macd_histogram'], 
                           color='red', alpha=0.5, width=1)
            
            axes[ax_idx].set_ylabel('MACD')
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].legend(loc='upper left')
        
        # Configure x-axis for all charts
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(self.reports_dir, 'images', f'{symbol}_indicators.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def generate_sentiment_chart(self, symbol: str, sentiment_data: Dict[str, Any]) -> str:
        """
        Generate a sentiment chart for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            sentiment_data: Sentiment analysis data
            
        Returns:
            Path to the generated chart image
        """
        # Check if sentiment data is available for this symbol
        if not sentiment_data or symbol not in sentiment_data.get('sentiment', {}):
            return ""
        
        # Get sentiment distribution
        crypto_sentiment = sentiment_data['sentiment'][symbol]
        distribution = crypto_sentiment.get('distribution', {})
        
        if not distribution:
            return ""
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot sentiment distribution as pie chart
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [
            distribution.get('positive', 0),
            distribution.get('negative', 0),
            distribution.get('neutral', 0)
        ]
        colors = ['#27ae60', '#e74c3c', '#f39c12']
        explode = (0.1, 0.1, 0.1)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax1.axis('equal')
        ax1.set_title(f'{symbol} Sentiment Distribution')
        
        # Plot sentiment score as gauge chart
        sentiment_score = crypto_sentiment.get('score', 0)
        
        # Create gauge chart
        gauge_min, gauge_max = -1, 1
        gauge_range = gauge_max - gauge_min
        
        # Create background arc
        theta = np.linspace(np.pi, 0, 100)
        r = 0.8
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        ax2.plot(x, y, 'k-', lw=2)
        
        # Create colored arcs for different sentiment regions
        theta_neg = np.linspace(np.pi, 3*np.pi/4, 50)
        theta_neu = np.linspace(3*np.pi/4, np.pi/4, 50)
        theta_pos = np.linspace(np.pi/4, 0, 50)
        
        ax2.plot(r * np.cos(theta_neg), r * np.sin(theta_neg), 'r-', lw=10, alpha=0.3)
        ax2.plot(r * np.cos(theta_neu), r * np.sin(theta_neu), 'y-', lw=10, alpha=0.3)
        ax2.plot(r * np.cos(theta_pos), r * np.sin(theta_pos), 'g-', lw=10, alpha=0.3)
        
        # Add needle
        normalized_score = (sentiment_score - gauge_min) / gauge_range
        needle_angle = np.pi * (1 - normalized_score)
        needle_x = r * np.cos(needle_angle)
        needle_y = r * np.sin(needle_angle)
        
        ax2.plot([0, needle_x], [0, needle_y], 'k-', lw=2)
        ax2.add_patch(plt.Circle((0, 0), 0.05, color='k'))
        
        # Add labels
        ax2.text(-0.8, -0.1, 'Negative', fontsize=12, ha='center', va='center')
        ax2.text(0, -0.1, 'Neutral', fontsize=12, ha='center', va='center')
        ax2.text(0.8, -0.1, 'Positive', fontsize=12, ha='center', va='center')
        
        # Add score text
        score_text = f"Score: {sentiment_score:.2f}"
        ax2.text(0, -0.3, score_text, fontsize=14, ha='center', va='center', weight='bold')
        
        # Configure gauge chart
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-0.5, 1)
        ax2.axis('off')
        ax2.set_title(f'{symbol} Sentiment Score')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(self.reports_dir, 'images', f'{symbol}_sentiment.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def generate_detailed_report(self, symbol: str, market_data: Dict[str, Any],
                               technical_analysis: Dict[str, Any],
                               sentiment_analysis: Dict[str, Any],
                               correlation_results: Dict[str, Any],
                               recommendations: Dict[str, Any]) -> str:
        """
        Generate a detailed HTML report for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            market_data: Market data
            technical_analysis: Technical analysis results
            sentiment_analysis: Sentiment analysis results
            correlation_results: Correlation analysis results
            recommendations: Investment recommendations
            
        Returns:
            Path to the generated report
        """
        # Check if data is available for this symbol
        if (symbol not in market_data.get('historical_data', {}) or
            symbol not in technical_analysis.get('cryptocurrencies', {}) or
            symbol not in recommendations.get('cryptocurrencies', {})):
            return ""
        
        # Get data for this symbol
        historical_data = market_data['historical_data'][symbol]
        technical_data = technical_analysis['cryptocurrencies'][symbol]
        recommendation_data = recommendations['cryptocurrencies'][symbol]
        
        # Generate charts
        price_chart_path = self.generate_price_chart(symbol, historical_data, recommendations)
        indicators_chart_path = self.generate_technical_indicators_chart(symbol, historical_data)
        sentiment_chart_path = self.generate_sentiment_chart(symbol, sentiment_analysis)
        
        # Get relative paths for HTML
        price_chart_rel = os.path.relpath(price_chart_path, self.reports_dir) if price_chart_path else ""
        indicators_chart_rel = os.path.relpath(indicators_chart_path, self.reports_dir) if indicators_chart_path else ""
        sentiment_chart_rel = os.path.relpath(sentiment_chart_path, self.reports_dir) if sentiment_chart_path else ""
        
        # Get timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get action and other recommendation data
        action = recommendation_data['action'].upper()
        confidence = recommendation_data['confidence']
        risk_level = recommendation_data['risk_level']
        current_price = recommendation_data['current_price']
        time_horizon = recommendation_data['time_horizon']
        supporting_factors = recommendation_data.get('supporting_factors', [])
        risk_factors = recommendation_data.get('risk_factors', [])
        
        # Generate HTML content
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{symbol} Cryptocurrency Analysis</title>
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
                .action {{
                    display: inline-block;
                    padding: 10px 20px;
                    border-radius: 5px;
                    color: white;
                    font-weight: bold;
                    font-size: 18px;
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
                .chart-container {{
                    margin-bottom: 30px;
                }}
                .chart {{
                    width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .metrics {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    flex: 1;
                    min-width: 200px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                }}
                .metric-title {{
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .metric-value {{
                    font-size: 24px;
                    color: #2c3e50;
                }}
                .trade-details {{
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 30px;
                }}
                .factors-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .factors {{
                    flex: 1;
                    min-width: 300px;
                }}
                .factor {{
                    margin-bottom: 10px;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .supporting-factor {{
                    background-color: #e8f8f5;
                    border-left: 4px solid #27ae60;
                }}
                .risk-factor {{
                    background-color: #fdedec;
                    border-left: 4px solid #e74c3c;
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
                <h1>{symbol} Cryptocurrency Analysis</h1>
            </div>
            
            <div class="timestamp">
                Analysis Date: {timestamp}
            </div>
            
            <div class="summary">
                <h2>Investment Recommendation</h2>
                <div class="action action-{action.lower()}">{action}</div>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p><strong>Risk Level:</strong> {risk_level}/5</p>
                <p><strong>Current Price:</strong> ${current_price:.2f}</p>
                <p><strong>Time Horizon:</strong> {time_horizon}</p>
            </div>
        """
        
        # Add trade details if not hold
        if action != "HOLD":
            position_size = recommendation_data['position_size']
            entry_point = recommendation_data['entry_point']
            exit_point = recommendation_data['exit_point']
            stop_loss = recommendation_data['stop_loss']
            take_profit = recommendation_data['take_profit']
            
            html += f"""
            <div class="trade-details">
                <h2>Trade Details</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-title">Position Size</div>
                        <div class="metric-value">{position_size:.2%}</div>
                        <div>of portfolio</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Entry Point</div>
                        <div class="metric-value">${entry_point:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Exit Point</div>
                        <div class="metric-value">${exit_point:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Stop Loss</div>
                        <div class="metric-value">${stop_loss:.2f}</div>
                        <div>({abs(stop_loss - entry_point) / entry_point:.2%} from entry)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Take Profit</div>
                        <div class="metric-value">${take_profit:.2f}</div>
                        <div>({abs(take_profit - entry_point) / entry_point:.2%} from entry)</div>
                    </div>
                </div>
            </div>
            """
        
        # Add price chart
        if price_chart_rel:
            html += f"""
            <div class="chart-container">
                <h2>Price Chart</h2>
                <img src="{price_chart_rel}" alt="{symbol} Price Chart" class="chart">
            </div>
            """
        
        # Add technical indicators chart
        if indicators_chart_rel:
            html += f"""
            <div class="chart-container">
                <h2>Technical Indicators</h2>
                <img src="{indicators_chart_rel}" alt="{symbol} Technical Indicators" class="chart">
            </div>
            """
        
        # Add sentiment chart
        if sentiment_chart_rel:
            html += f"""
            <div class="chart-container">
                <h2>Sentiment Analysis</h2>
                <img src="{sentiment_chart_rel}" alt="{symbol} Sentiment Analysis" class="chart">
            </div>
            """
        
        # Add supporting and risk factors
        html += """
            <h2>Analysis Factors</h2>
            <div class="factors-container">
        """
        
        # Add supporting factors
        html += """
                <div class="factors">
                    <h3>Supporting Factors</h3>
        """
        
        if supporting_factors:
            for factor in supporting_factors:
                html += f"""
                    <div class="factor supporting-factor">
                        {factor}
                    </div>
                """
        else:
            html += """
                    <p>No supporting factors identified.</p>
            """
        
        html += """
                </div>
        """
        
        # Add risk factors
        html += """
                <div class="factors">
                    <h3>Risk Factors</h3>
        """
        
        if risk_factors:
            for factor in risk_factors:
                html += f"""
                    <div class="factor risk-factor">
                        {factor}
                    </div>
                """
        else:
            html += """
                    <p>No risk factors identified.</p>
            """
        
        html += """
                </div>
            </div>
        """
        
        # Add disclaimer and close HTML
        html += """
            <div class="disclaimer">
                <p>This analysis is for informational purposes only. Always conduct your own research before making investment decisions.</p>
                <p>Past performance is not indicative of future results. Cryptocurrency investments are subject to high market risk.</p>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        output_path = os.path.join(self.reports_dir, f'{symbol}_analysis.html')
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def generate_summary_report(self, market_data: Dict[str, Any],
                              technical_analysis: Dict[str, Any],
                              sentiment_analysis: Dict[str, Any],
                              correlation_results: Dict[str, Any],
                              recommendations: Dict[str, Any]) -> str:
        """
        Generate a summary HTML report for all cryptocurrencies.
        
        Args:
            market_data: Market data
            technical_analysis: Technical analysis results
            sentiment_analysis: Sentiment analysis results
            correlation_results: Correlation analysis results
            recommendations: Investment recommendations
            
        Returns:
            Path to the generated report
        """
        # Get timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get summary lists
        buy_list = recommendations['summary']['buy']
        sell_list = recommendations['summary']['sell']
        hold_list = recommendations['summary']['hold']
        
        # Generate detailed reports for each cryptocurrency
        detailed_reports = {}
        for symbol in recommendations.get('cryptocurrencies', {}):
            report_path = self.generate_detailed_report(
                symbol, market_data, technical_analysis, 
                sentiment_analysis, correlation_results, recommendations)
            
            if report_path:
                detailed_reports[symbol] = os.path.relpath(report_path, self.reports_dir)
        
        # Generate HTML content
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
                .view-details {{
                    display: inline-block;
                    margin-top: 10px;
                    padding: 5px 10px;
                    background-color: #3498db;
                    color: white;
                    text-decoration: none;
                    border-radius: 3px;
                }}
                .view-details:hover {{
                    background-color: #2980b9;
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
                Analysis Date: {timestamp}
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
                if symbol in recommendations['cryptocurrencies']:
                    html += self._generate_recommendation_card_html(
                        symbol, recommendations['cryptocurrencies'][symbol], 
                        'buy', detailed_reports.get(symbol, ''))
            html += '</div>'
        
        # Add sell recommendations
        if sell_list:
            html += '<h3>Sell Recommendations</h3><div class="recommendations">'
            for symbol in sell_list:
                if symbol in recommendations['cryptocurrencies']:
                    html += self._generate_recommendation_card_html(
                        symbol, recommendations['cryptocurrencies'][symbol], 
                        'sell', detailed_reports.get(symbol, ''))
            html += '</div>'
        
        # Add hold recommendations
        if hold_list:
            html += '<h3>Hold Recommendations</h3><div class="recommendations">'
            for symbol in hold_list:
                if symbol in recommendations['cryptocurrencies']:
                    html += self._generate_recommendation_card_html(
                        symbol, recommendations['cryptocurrencies'][symbol], 
                        'hold', detailed_reports.get(symbol, ''))
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
        
        # Save HTML report
        output_path = os.path.join(self.reports_dir, 'crypto_recommendations_summary.html')
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def _generate_recommendation_card_html(self, symbol: str, data: Dict[str, Any], 
                                         action_type: str, detailed_report: str) -> str:
        """
        Generate HTML for a recommendation card.
        
        Args:
            symbol: Cryptocurrency symbol
            data: Recommendation data for the cryptocurrency
            action_type: Type of action (buy, sell, hold)
            detailed_report: Path to detailed report
            
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
            entry_point = data['entry_point']
            stop_loss = data['stop_loss']
            take_profit = data['take_profit']
            
            trade_html = f"""
            <div class="trade-details">
                <p><strong>Entry Point:</strong> ${entry_point:.2f}</p>
                <p><strong>Stop Loss:</strong> ${stop_loss:.2f}</p>
                <p><strong>Take Profit:</strong> ${take_profit:.2f}</p>
            </div>
            """
        
        # Generate view details link
        details_link = ""
        if detailed_report:
            details_link = f'<a href="{detailed_report}" class="view-details">View Detailed Analysis</a>'
        
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
            {details_link}
        </div>
        """
        
        return card_html
    
    def generate_all_reports(self, market_data: Dict[str, Any],
                           technical_analysis: Dict[str, Any],
                           sentiment_analysis: Dict[str, Any],
                           correlation_results: Dict[str, Any],
                           recommendations: Dict[str, Any]) -> str:
        """
        Generate all reports for the cryptocurrency analysis.
        
        Args:
            market_data: Market data
            technical_analysis: Technical analysis results
            sentiment_analysis: Sentiment analysis results
            correlation_results: Correlation analysis results
            recommendations: Investment recommendations
            
        Returns:
            Path to the summary report
        """
        # Generate summary report
        summary_report_path = self.generate_summary_report(
            market_data, technical_analysis, sentiment_analysis, 
            correlation_results, recommendations)
        
        return summary_report_path


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
    from recommendation.portfolio_advisor import PortfolioAdvisor
    
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
    
    # Generate signals and recommendations
    signal_generator = SignalGenerator()
    portfolio_advisor = PortfolioAdvisor()
    
    signals = signal_generator.generate_signals(correlation_results)
    recommendations = portfolio_advisor.generate_recommendations(signals, market_data)
    
    # Generate reports
    report_generator = ReportGenerator()
    summary_report_path = report_generator.generate_all_reports(
        market_data, technical_analysis, sentiment_analysis, 
        correlation_results, recommendations)
    
    print(f"Summary report generated: {summary_report_path}")
    print(f"All reports saved to: {report_generator.reports_dir}")
