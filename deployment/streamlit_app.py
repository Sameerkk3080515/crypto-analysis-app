import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from datetime import datetime
import base64
from io import BytesIO

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import bot modules
from utils.config import Config
from data_collection.market_data import MarketDataCollector
from data_collection.news_data import NewsDataCollector
from analysis.technical import TechnicalAnalyzer
from analysis.sentiment import SentimentAnalyzer
from analysis.correlation import CorrelationEngine
from recommendation.signal_generator import SignalGenerator
from recommendation.portfolio_advisor import PortfolioAdvisor
from ui.report_generator import ReportGenerator

# Set page configuration
st.set_page_config(
    page_title="Crypto Investment Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        font-size: 16px;
    }
    .crypto-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .buy-card {
        border-left: 5px solid #27ae60;
    }
    .sell-card {
        border-left: 5px solid #e74c3c;
    }
    .hold-card {
        border-left: 5px solid #f39c12;
    }
    .action-button {
        padding: 8px 16px;
        color: white;
        border-radius: 4px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 15px;
    }
    .action-buy {
        background-color: #27ae60;
    }
    .action-sell {
        background-color: #e74c3c;
    }
    .action-hold {
        background-color: #f39c12;
    }
    .confidence-meter {
        height: 10px;
        background-color: #ecf0f1;
        border-radius: 5px;
        margin: 10px 0;
    }
    .confidence-level {
        height: 100%;
        background-color: #3498db;
        border-radius: 5px;
    }
    .risk-meter {
        display: flex;
        margin: 10px 0;
    }
    .risk-segment {
        flex: 1;
        height: 10px;
        background-color: #ecf0f1;
        margin-right: 2px;
    }
    .risk-active {
        background-color: #e74c3c;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #3498db;
        cursor: help;
    }
    .disclaimer {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        font-style: italic;
        text-align: center;
        margin-top: 30px;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def generate_confidence_meter(confidence):
    """Generate HTML for confidence meter"""
    html = f"""
    <div>
        <strong>Confidence: {confidence:.0%}</strong>
        <div class="confidence-meter">
            <div class="confidence-level" style="width: {confidence * 100}%;"></div>
        </div>
    </div>
    """
    return html

def generate_risk_meter(risk_level):
    """Generate HTML for risk meter"""
    html = f'<div><strong>Risk Level: {risk_level}/5</strong><div class="risk-meter">'
    for i in range(1, 6):
        html += f'<div class="risk-segment{" risk-active" if i <= risk_level else ""}"></div>'
    html += '</div></div>'
    return html

def get_tooltip(term, explanation):
    """Generate HTML for tooltip"""
    return f'<span class="tooltip" title="{explanation}">{term}</span>'

def plot_price_chart(symbol, historical_data, recommendation=None):
    """Generate price chart for a cryptocurrency"""
    if not historical_data:
        return None
        
    # Convert to DataFrame if it's a list
    if isinstance(historical_data, list):
        df = pd.DataFrame(historical_data)
    else:
        df = historical_data
    
    # Ensure required columns exist
    required_columns = ['open_time', 'close']
    if not all(col in df.columns for col in required_columns):
        return None
    
    # Convert timestamps if they're strings
    if isinstance(df['open_time'][0], str):
        df['open_time'] = pd.to_datetime(df['open_time'])
    
    # Sort by time
    df = df.sort_values('open_time')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot price
    ax.plot(df['open_time'], df['close'], label='Price', color='#3498db')
    
    # Add moving averages if available
    if 'sma_20' in df.columns:
        ax.plot(df['open_time'], df['sma_20'], label='20-day Average', color='#27ae60', alpha=0.7)
    
    if 'sma_50' in df.columns:
        ax.plot(df['open_time'], df['sma_50'], label='50-day Average', color='#f39c12', alpha=0.7)
    
    # Add buy/sell signals if available
    if recommendation and recommendation.get('action') != 'hold':
        action = recommendation.get('action')
        entry_point = recommendation.get('entry_point')
        stop_loss = recommendation.get('stop_loss')
        take_profit = recommendation.get('take_profit')
        
        # Add horizontal lines
        if entry_point:
            ax.axhline(y=entry_point, color='g' if action == 'buy' else 'r', 
                      linestyle='--', alpha=0.7, label='Entry Point')
        
        if stop_loss:
            ax.axhline(y=stop_loss, color='r', linestyle='--', alpha=0.7, label='Stop Loss')
        
        if take_profit:
            ax.axhline(y=take_profit, color='g', linestyle='--', alpha=0.7, label='Take Profit')
    
    # Configure chart
    ax.set_title(f'{symbol} Price Chart', fontsize=16)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    return fig

def run_analysis():
    """Run the complete analysis pipeline"""
    with st.spinner('Analyzing cryptocurrency markets and news... This may take a few minutes.'):
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        os.makedirs('reports/images', exist_ok=True)
        
        # Initialize components
        config = Config()
        market_collector = MarketDataCollector(config)
        news_collector = NewsDataCollector(config)
        technical_analyzer = TechnicalAnalyzer(config)
        sentiment_analyzer = SentimentAnalyzer(config)
        correlation_engine = CorrelationEngine(config)
        signal_generator = SignalGenerator(config)
        portfolio_advisor = PortfolioAdvisor(config)
        
        # Collect data
        market_data = market_collector.get_market_data_for_analysis()
        news_data = news_collector.get_news_data_for_analysis()
        
        # Perform analysis
        technical_analysis = technical_analyzer.analyze_market_data(market_data)
        sentiment_analysis = sentiment_analyzer.analyze_news_data(news_data)
        correlation_results = correlation_engine.correlate_all_data(
            market_data, technical_analysis, sentiment_analysis)
        
        # Generate signals
        signals = signal_generator.generate_signals(correlation_results)
        
        # Generate recommendations
        recommendations = portfolio_advisor.generate_recommendations(signals, market_data)
        
        # Save results to session state
        st.session_state['market_data'] = market_data
        st.session_state['news_data'] = news_data
        st.session_state['technical_analysis'] = technical_analysis
        st.session_state['sentiment_analysis'] = sentiment_analysis
        st.session_state['correlation_results'] = correlation_results
        st.session_state['signals'] = signals
        st.session_state['recommendations'] = recommendations
        st.session_state['analysis_complete'] = True
        st.session_state['analysis_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def display_welcome():
    """Display welcome screen"""
    st.markdown("<h1 style='text-align: center;'>Cryptocurrency Investment Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em;'>Get beginner-friendly investment recommendations based on market data and news analysis</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://img.freepik.com/free-vector/cryptocurrency-bitcoin-golden-coin-background_1017-31505.jpg", use_column_width=True)
        
        if st.button("Get Started", key="welcome_button", help="Run analysis to get cryptocurrency recommendations"):
            st.session_state['show_welcome'] = False
            run_analysis()
    
    st.markdown("""
    <div style='text-align: center; margin-top: 2em;'>
        <h3>How it works:</h3>
        <div style='display: flex; justify-content: space-around; flex-wrap: wrap; margin: 2em 0;'>
            <div style='width: 200px; margin: 1em;'>
                <div style='font-size: 3em; color: #3498db;'>📊</div>
                <h4>Analyze Market Data</h4>
                <p>We collect and analyze price trends and trading patterns</p>
            </div>
            <div style='width: 200px; margin: 1em;'>
                <div style='font-size: 3em; color: #3498db;'>📰</div>
                <h4>Review News</h4>
                <p>We analyze news and social media sentiment</p>
            </div>
            <div style='width: 200px; margin: 1em;'>
                <div style='font-size: 3em; color: #3498db;'>🤖</div>
                <h4>Generate Recommendations</h4>
                <p>Our AI provides simple buy, sell, or hold advice</p>
            </div>
            <div style='width: 200px; margin: 1em;'>
                <div style='font-size: 3em; color: #3498db;'>💰</div>
                <h4>Make Decisions</h4>
                <p>You get clear guidance on what to do next</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='disclaimer'>
        <p>This tool is for educational purposes only. Always do your own research before investing.</p>
        <p>Cryptocurrency investments involve high risk and potential loss of capital.</p>
    </div>
    """, unsafe_allow_html=True)

def display_recommendation_card(symbol, data, historical_data=None):
    """Display a recommendation card for a cryptocurrency"""
    action = data['action'].upper()
    confidence = data['confidence']
    risk_level = data['risk_level']
    current_price = data['current_price']
    time_horizon = data['time_horizon']
    
    # Determine card class based on action
    card_class = f"crypto-card {'buy-card' if action == 'BUY' else 'sell-card' if action == 'SELL' else 'hold-card'}"
    
    # Start card
    st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
    
    # Header section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"<h2>{symbol}</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='action-button action-{action.lower()}'>{action}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h3>${current_price:.2f}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>Current Price</p>", unsafe_allow_html=True)
    
    # Confidence and risk meters
    st.markdown(generate_confidence_meter(confidence), unsafe_allow_html=True)
    st.markdown(generate_risk_meter(risk_level), unsafe_allow_html=True)
    
    # Time horizon with tooltip
    horizon_explanation = "How long you should consider holding this position before re-evaluating"
    st.markdown(f"<p><strong>Time Horizon:</strong> {get_tooltip(time_horizon, horizon_explanation)}</p>", unsafe_allow_html=True)
    
    # Trade details if not hold
    if action != "HOLD":
        with st.expander("Trade Details", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            position_size = data['position_size']
            entry_point = data['entry_point']
            stop_loss = data['stop_loss']
            take_profit = data['take_profit']
            
            position_explanation = "The recommended percentage of your portfolio to invest in this cryptocurrency"
            entry_explanation = "The suggested price at which to buy or sell this cryptocurrency"
            stop_explanation = "The price at which you should exit the position to limit losses"
            profit_explanation = "The target price at which you should consider taking profits"
            
            with col1:
                st.markdown(f"<p><strong>{get_tooltip('Position Size', position_explanation)}:</strong> {position_size:.0%}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>{get_tooltip('Entry Point', entry_explanation)}:</strong> ${entry_point:.2f}</p>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<p><strong>{get_tooltip('Stop Loss', stop_explanation)}:</strong> ${stop_loss:.2f}</p>", unsafe_allow_html=True)
                loss_percentage = abs(stop_loss - entry_point) / entry_point
                st.markdown(f"<p>({loss_percentage:.1%} from entry)</p>", unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"<p><strong>{get_tooltip('Take Profit', profit_explanation)}:</strong> ${take_profit:.2f}</p>", unsafe_allow_html=True)
                profit_percentage = abs(take_profit - entry_point) / entry_point
                st.markdown(f"<p>({profit_percentage:.1%} from entry)</p>", unsafe_allow_html=True)
    
    # Price chart
    if historical_data is not None:
        with st.expander("Price Chart", expanded=True):
            fig = plot_price_chart(symbol, historical_data, data)
            if fig:
                st.pyplot(fig)
            else:
                st.info("Price chart not available for this cryptocurrency")
    
    # Supporting and risk factors
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Why This Recommendation?", expanded=False):
            supporting_factors = data.get('supporting_factors', [])
            if supporting_factors:
                for factor in supporting_factors:
                    st.markdown(f"✅ {factor}")
            else:
                st.info("No supporting factors identified")
    
    with col2:
        with st.expander("Risk Factors", expanded=False):
            risk_factors = data.get('risk_factors', [])
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"⚠️ {factor}")
            else:
                st.info("No risk factors identified")
    
    # Beginner explanation
    with st.expander("What Does This Mean?", expanded=False):
        if action == "BUY":
            st.markdown("""
            **This recommendation means:**
            - The analysis suggests this cryptocurrency may increase in value
            - Consider buying at the suggested entry point
            - Set a stop loss to protect yourself from large losses
            - Consider taking profits at the take profit level
            
            **Before you buy:**
            - Only invest money you can afford to lose
            - Consider starting with a small amount to learn
            - Be prepared for price volatility
            """)
        elif action == "SELL":
            st.markdown("""
            **This recommendation means:**
            - The analysis suggests this cryptocurrency may decrease in value
            - If you own this cryptocurrency, consider selling at the suggested price
            - If you don't own it, you can ignore this recommendation
            
            **Before you sell:**
            - Consider tax implications of selling
            - Don't panic sell based on short-term price movements
            - Remember that selling locks in any gains or losses
            """)
        else:  # HOLD
            st.markdown("""
            **This recommendation means:**
            - The analysis is neutral on this cryptocurrency
            - If you own it, consider keeping your position for now
            - If you don't own it, there's no strong reason to buy
            
            **While holding:**
            - Continue to monitor the market
            - Be ready to act if the recommendation changes
            - Consider setting price alerts for significant movements
            """)
    
    # End card
    st.markdown("</div>", unsafe_allow_html=True)

def display_analysis_dashboard():
    """Display the main analysis dashboard"""
    if not st.session_state.get('analysis_complete', False):
        run_analysis()
    
    recommendations = st.session_state.get('recommendations', {})
    market_data = st.session_state.get('market_data', {})
    
    # Get recommendation lists
    buy_list = recommendations.get('summary', {}).get('buy', [])
    sell_list = recommendations.get('summary', {}).get('sell', [])
    hold_list = recommendations.get('summary', {}).get('hold', [])
    
    # Display timestamp
    analysis_time = st.session_state.get('analysis_timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    st.markdown(f"<p style='text-align: center;'>Analysis completed: {analysis_time}</p>", unsafe_allow_html=True)
    
    # Create tabs
    tabs = ["Overview", "Buy Recommendations", "Sell Recommendations", "Hold Recommendations", "Learn About Trading"]
    selected_tab = st.tabs(tabs)
    
    # Overview Tab
    with selected_tab[0]:
        st.markdown("<h2>Market Overview</h2>", unsafe_allow_html=True)
        
        # Market summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>Top Recommendations</h3>", unsafe_allow_html=True)
            
            if buy_list:
                st.markdown("<h4>Buy</h4>", unsafe_allow_html=True)
                for symbol in buy_list[:3]:  # Top 3 buy recommendations
                    if symbol in recommendations.get('cryptocurrencies', {}):
                        confidence = recommendations['cryptocurrencies'][symbol]['confidence']
                        st.markdown(f"- {symbol} (Confidence: {confidence:.0%})")
            
            if sell_list:
                st.markdown("<h4>Sell</h4>", unsafe_allow_html=True)
                for symbol in sell_list[:3]:  # Top 3 sell recommendations
                    if symbol in recommendations.get('cryptocurrencies', {}):
                        confidence = recommendations['cryptocurrencies'][symbol]['confidence']
                        st.markdown(f"- {symbol} (Confidence: {confidence:.0%})")
        
        with col2:
            st.markdown("<h3>Market Sentiment</h3>", unsafe_allow_html=True)
            
            # Calculate overall market sentiment
            buy_count = len(buy_list)
            sell_count = len(sell_list)
            hold_count = len(hold_list)
            total_count = buy_count + sell_count + hold_count
            
            if total_count > 0:
                buy_percentage = buy_count / total_count
                sell_percentage = sell_count / total_count
                hold_percentage = hold_count / total_count
                
                # Determine overall sentiment
                if buy_percentage > 0.5:
                    sentiment = "Bullish"
                    sentiment_color = "#27ae60"
                elif sell_percentage > 0.5:
                    sentiment = "Bearish"
                    sentiment_color = "#e74c3c"
                else:
                    sentiment = "Neutral"
                    sentiment_color = "#f39c12"
                
                st.markdown(f"<h4 style='color: {sentiment_color};'>Overall: {sentiment}</h4>", unsafe_allow_html=True)
                
                # Display sentiment distribution
                st.markdown("Distribution of recommendations:")
                st.progress(buy_percentage, text=f"Buy: {buy_percentage:.0%}")
                st.progress(sell_percentage, text=f"Sell: {sell_percentage:.0%}")
                st.progress(hold_percentage, text=f"Hold: {hold_percentage:.0%}")
            else:
                st.info("No sentiment data available")
        
        # Run new analysis button
        if st.button("Run New Analysis"):
            run_analysis()
            st.experimental_rerun()
    
    # Buy Recommendations Tab
    with selected_tab[1]:
        st.markdown("<h2>Buy Recommendations</h2>", unsafe_allow_html=True)
        
        if buy_list:
            for symbol in buy_list:
                if symbol in recommendations.get('cryptocurrencies', {}):
                    historical_data = market_data.get('historical_data', {}).get(symbol, None)
                    display_recommendation_card(symbol, recommendations['cryptocurrencies'][symbol], historical_data)
        else:
            st.info("No buy recommendations at this time")
    
    # Sell Recommendations Tab
    with selected_tab[2]:
        st.markdown("<h2>Sell Recommendations</h2>", unsafe_allow_html=True)
        
        if sell_list:
            for symbol in sell_list:
                if symbol in recommendations.get('cryptocurrencies', {}):
                    historical_data = market_data.get('historical_data', {}).get(symbol, None)
                    display_recommendation_card(symbol, recommendations['cryptocurrencies'][symbol], historical_data)
        else:
            st.info("No sell recommendations at this time")
    
    # Hold Recommendations Tab
    with selected_tab[3]:
        st.markdown("<h2>Hold Recommendations</h2>", unsafe_allow_html=True)
        
        if hold_list:
            for symbol in hold_list:
                if symbol in recommendations.get('cryptocurrencies', {}):
                    historical_data = market_data.get('historical_data', {}).get(symbol, None)
                    display_recommendation_card(symbol, recommendations['cryptocurrencies'][symbol], historical_data)
        else:
            st.info("No hold recommendations at this time")
    
    # Learn About Trading Tab
    with selected_tab[4]:
        st.markdown("<h2>Cryptocurrency Trading Basics</h2>", unsafe_allow_html=True)
        
        with st.expander("What is Cryptocurrency?", expanded=False):
            st.markdown("""
            **Cryptocurrency** is a digital or virtual currency that uses cryptography for security and operates on a technology called blockchain.
            
            **Key features:**
            - Decentralized (not controlled by any central authority)
            - Digital (exists only electronically)
            - Limited supply (many cryptocurrencies have a maximum supply)
            - Secured by cryptography (making it difficult to counterfeit)
            
            **Popular cryptocurrencies:**
            - Bitcoin (BTC): The first and most valuable cryptocurrency
            - Ethereum (ETH): Known for its smart contract functionality
            - Binance Coin (BNB): Native token of the Binance exchange
            - And many others (there are thousands of cryptocurrencies)
            """)
        
        with st.expander("Trading Terminology", expanded=False):
            st.markdown("""
            ### Basic Terms
            
            - **Buy**: Purchasing a cryptocurrency with the expectation its value will increase
            - **Sell**: Selling a cryptocurrency to realize profits or prevent further losses
            - **Hold**: Keeping a cryptocurrency in your portfolio without buying or selling
            - **Bull Market**: A market where prices are rising or expected to rise
            - **Bear Market**: A market where prices are falling or expected to fall
            
            ### Technical Analysis Terms
            
            - **Moving Average**: Average price over a specific time period, helps identify trends
            - **Support Level**: Price level where a cryptocurrency tends to stop falling
            - **Resistance Level**: Price level where a cryptocurrency tends to stop rising
            - **Volume**: Amount of cryptocurrency traded in a given period
            
            ### Risk Management Terms
            
            - **Stop Loss**: A predetermined price at which you'll sell to limit losses
            - **Take Profit**: A predetermined price at which you'll sell to secure profits
            - **Position Sizing**: How much of your portfolio you allocate to a specific investment
            - **Diversification**: Spreading investments across multiple cryptocurrencies to reduce risk
            """)
        
        with st.expander("How to Use Binance", expanded=False):
            st.markdown("""
            ### Getting Started with Binance
            
            1. **Create an Account**
               - Visit [Binance.com](https://www.binance.com)
               - Click "Register" and follow the instructions
               - Complete identity verification (KYC)
            
            2. **Deposit Funds**
               - Click on "Wallet" > "Fiat and Spot"
               - Click "Deposit" and select your preferred method
               - Follow the instructions to complete your deposit
            
            3. **Buying Cryptocurrency**
               - Go to "Trade" > "Spot"
               - Search for the cryptocurrency you want to buy (e.g., BTC/USDT)
               - Enter the amount you want to buy
               - Review and confirm your order
            
            4. **Setting Stop Loss and Take Profit**
               - When placing an order, select "Stop Limit" order type
               - Enter your stop price and limit price
               - For take profit, place a separate "Limit" sell order at your target price
            
            5. **Monitoring Your Portfolio**
               - Go to "Wallet" > "Fiat and Spot" to see your holdings
               - Use the "Trade" > "Spot" view to monitor price movements
            """)
        
        with st.expander("Risk Management Tips", expanded=False):
            st.markdown("""
            ### Essential Risk Management for Beginners
            
            1. **Only invest what you can afford to lose**
               - Cryptocurrency is highly volatile and risky
               - Never invest rent money, emergency funds, or money needed for essential expenses
            
            2. **Start small**
               - Begin with small amounts while you're learning
               - Increase position sizes gradually as you gain experience
            
            3. **Use stop losses**
               - Always set stop losses to limit potential losses
               - A common approach is setting stop losses at 5-10% below your entry price
            
            4. **Diversify your investments**
               - Don't put all your money in one cryptocurrency
               - Spread investments across different cryptocurrencies and asset classes
            
            5. **Don't chase pumps**
               - Avoid buying cryptocurrencies that have already increased significantly in price
               - FOMO (Fear Of Missing Out) often leads to buying at the top
            
            6. **Have a plan**
               - Decide in advance when you'll buy, sell, or hold
               - Stick to your plan and avoid emotional decisions
            
            7. **Keep records**
               - Track all your trades for tax purposes
               - Analyze your performance to improve your strategy
            """)
        
        with st.expander("Frequently Asked Questions", expanded=False):
            st.markdown("""
            ### FAQ for Beginner Traders
            
            **Q: How much money do I need to start trading cryptocurrency?**  
            A: You can start with as little as $10-$20 on most exchanges. It's best to start small while learning.
            
            **Q: How often should I check my investments?**  
            A: For long-term investing, weekly or monthly is sufficient. For active trading, you'll need to monitor more frequently, but avoid obsessive checking which can lead to emotional decisions.
            
            **Q: Should I buy when prices are falling?**  
            A: This strategy (called "buying the dip") can work, but ensure you're not catching a falling knife. Look for signs of price stabilization before buying.
            
            **Q: When should I sell my cryptocurrency?**  
            A: Sell when you've reached your predetermined profit target, when your investment thesis has changed, or when your stop loss is triggered.
            
            **Q: Is it better to make many small trades or fewer larger trades?**  
            A: For beginners, fewer, well-researched trades are usually better than frequent trading, which can lead to higher fees and emotional decisions.
            
            **Q: How do I know which cryptocurrency to buy?**  
            A: Research fundamentals, use tools like this analysis bot, and focus on established cryptocurrencies with real-world use cases and strong development teams.
            
            **Q: Do I need to pay taxes on cryptocurrency trading?**  
            A: In most countries, yes. Cryptocurrency trades are typically subject to capital gains tax. Consult a tax professional for advice specific to your situation.
            """)
    
    # Disclaimer
    st.markdown("""
    <div class='disclaimer'>
        <p>This tool is for educational purposes only. Always do your own research before investing.</p>
        <p>Cryptocurrency investments involve high risk and potential loss of capital.</p>
    </div>
    """, unsafe_allow_html=True)

# Main app logic
def main():
    # Initialize session state
    if 'show_welcome' not in st.session_state:
        st.session_state['show_welcome'] = True
    
    if 'analysis_complete' not in st.session_state:
        st.session_state['analysis_complete'] = False
    
    # Display appropriate screen
    if st.session_state['show_welcome'] and not st.session_state['analysis_complete']:
        display_welcome()
    else:
        display_analysis_dashboard()

if __name__ == "__main__":
    main()
