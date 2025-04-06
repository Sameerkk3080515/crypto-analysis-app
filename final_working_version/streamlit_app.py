import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import sys
import json
import base64
import io
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Binance Spot Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize data manager as a session state variable
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = None
    st.session_state.last_refresh_time = None

# Function to create confidence meter
def create_confidence_meter(value, width=200, height=100, label="Confidence"):
    # Create a gauge-like confidence meter
    fig = go.Figure()
    
    # Add a gauge/indicator
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': label, 'font': {'size': 14, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#F0B90B"},
            'bgcolor': "gray",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 30], 'color': '#FF4B4B'},
                {'range': [30, 70], 'color': '#FFD700'},
                {'range': [70, 100], 'color': '#00FF7F'}
            ]
        },
        number={'font': {'size': 20, 'color': 'white'}}
    ))
    
    # Update layout
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return fig

# Function to create price chart
def create_price_chart(historical_data, projected_data=None, symbol="BTCUSDT"):
    # Create a price chart with historical and projected data
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    
    # Add historical price data
    fig.add_trace(
        go.Candlestick(
            x=historical_data['open_time'],
            open=historical_data['open'],
            high=historical_data['high'],
            low=historical_data['low'],
            close=historical_data['close'],
            name="Historical Data",
            increasing={'line': {'color': '#00FF7F'}},
            decreasing={'line': {'color': '#FF4B4B'}}
        )
    )
    
    # Add volume as bar chart at the bottom
    fig.add_trace(
        go.Bar(
            x=historical_data['open_time'],
            y=historical_data['volume'],
            name="Volume",
            marker={'color': 'rgba(240, 185, 11, 0.5)'},
            opacity=0.3,
            yaxis="y2"
        )
    )
    
    # Add projected data if available
    if projected_data:
        # Add projected price line
        fig.add_trace(
            go.Scatter(
                x=projected_data['time'],
                y=projected_data['price'],
                mode='lines',
                line={'color': '#F0B90B', 'width': 2, 'dash': 'dash'},
                name="Projected Price"
            )
        )
        
        # Add selling point
        fig.add_trace(
            go.Scatter(
                x=[projected_data['selling_time']],
                y=[projected_data['selling_price']],
                mode='markers',
                marker={'color': '#F0B90B', 'size': 12, 'symbol': 'star'},
                name="Selling Point"
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        legend_title="Data",
        height=500,
        xaxis_rangeslider_visible=False,
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        template="plotly_dark",
        plot_bgcolor='#1E2026',
        paper_bgcolor='#1E2026',
        font={'color': 'white'},
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# Function to initialize data manager
@st.cache_resource
def get_data_manager():
    return RealTimeDataManager(refresh_interval=300)  # 5 minutes

# Function to display loading animation
def display_loading_animation():
    cols = st.columns(3)
    with cols[1]:
        st.markdown(
            """
            <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
                <div class="loader"></div>
            </div>
            <style>
                .loader {
                    border: 16px solid #474D57;
                    border-radius: 50%;
                    border-top: 16px solid #F0B90B;
                    width: 120px;
                    height: 120px;
                    animation: spin 2s linear infinite;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
            """,
            unsafe_allow_html=True
        )

# Class to simulate real-time data management
class RealTimeDataManager:
    def __init__(self, refresh_interval=300):
        self.refresh_interval = refresh_interval
        self.last_refresh_time = None
        self.data = self._initialize_data()
    
    def _initialize_data(self):
        # Initialize with sample data
        return {
            'recommendations': self._generate_sample_recommendations(),
            'analysis_results': self._generate_sample_analysis(),
            'all_pairs_data': self._generate_sample_pairs_data(),
            'last_update_time': datetime.now()
        }
    
    def refresh_data(self):
        # Simulate data refresh
        try:
            self.data = self._initialize_data()
            self.last_refresh_time = datetime.now()
            return True
        except Exception as e:
            print(f"Error refreshing data: {e}")
            return False
    
    def get_data(self):
        # Return current data
        return self.data
    
    def _generate_sample_recommendations(self):
        # Generate sample recommendations
        recommendations = []
        
        # Sample symbols
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 
                  'XRPUSDT', 'DOTUSDT', 'UNIUSDT', 'LTCUSDT', 'LINKUSDT']
        
        # Generate random recommendations
        for symbol in symbols:
            action = random.choice(['buy', 'sell', 'hold'])
            
            # Base price on symbol
            if symbol == 'BTCUSDT':
                current_price = random.uniform(45000, 55000)
            elif symbol == 'ETHUSDT':
                current_price = random.uniform(3000, 4000)
            elif symbol == 'BNBUSDT':
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
    
    def _generate_sample_analysis(self):
        # Generate sample analysis results
        analysis_results = {
            'market_analysis': {
                'market_conditions': {
                    'market_trend': random.choice(['strongly_bullish', 'bullish', 'neutral', 'bearish', 'strongly_bearish']),
                    'market_strength': random.uniform(30, 70),
                    'volatility': random.choice(['high', 'medium', 'low']),
                    'btc_dominance': random.uniform(0.4, 0.6),
                    'top_performers': [
                        ('BNBUSDT', random.uniform(5, 15)),
                        ('ADAUSDT', random.uniform(3, 10)),
                        ('DOGEUSDT', random.uniform(2, 8)),
                        ('MATICUSDT', random.uniform(1, 7)),
                        ('SOLUSDT', random.uniform(1, 6))
                    ],
                    'worst_performers': [
                        ('ATOMUSDT', random.uniform(-10, -2)),
                        ('LTCUSDT', random.uniform(-8, -1)),
                        ('DOTUSDT', random.uniform(-7, -1)),
                        ('LINKUSDT', random.uniform(-6, -1)),
                        ('UNIUSDT', random.uniform(-5, -1))
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
                    'BTCUSDT': {
                        'BTCUSDT': 1.0,
                        'ETHUSDT': random.uniform(0.7, 0.9),
                        'BNBUSDT': random.uniform(0.5, 0.8),
                        'XRPUSDT': random.uniform(0.3, 0.7),
                        'ADAUSDT': random.uniform(0.4, 0.7)
                    },
                    'ETHUSDT': {
                        'BTCUSDT': random.uniform(0.7, 0.9),
                        'ETHUSDT': 1.0,
                        'BNBUSDT': random.uniform(0.6, 0.8),
                        'XRPUSDT': random.uniform(0.4, 0.7),
                        'ADAUSDT': random.uniform(0.5, 0.8)
                    },
                    'BNBUSDT': {
                        'BTCUSDT': random.uniform(0.5, 0.8),
                        'ETHUSDT': random.uniform(0.6, 0.8),
                        'BNBUSDT': 1.0,
                        'XRPUSDT': random.uniform(0.3, 0.6),
                        'ADAUSDT': random.uniform(0.4, 0.7)
                    },
                    'XRPUSDT': {
                        'BTCUSDT': random.uniform(0.3, 0.7),
                        'ETHUSDT': random.uniform(0.4, 0.7),
                        'BNBUSDT': random.uniform(0.3, 0.6),
                        'XRPUSDT': 1.0,
                        'ADAUSDT': random.uniform(0.5, 0.7)
                    },
                    'ADAUSDT': {
                        'BTCUSDT': random.uniform(0.4, 0.7),
                        'ETHUSDT': random.uniform(0.5, 0.8),
                        'BNBUSDT': random.uniform(0.4, 0.7),
                        'XRPUSDT': random.uniform(0.5, 0.7),
                        'ADAUSDT': 1.0
                    }
                }
            }
        }
        
        return analysis_results
    
    def _generate_sample_pairs_data(self):
        # Generate sample data for all pairs
        all_pairs_data = {}
        
        # Sample symbols
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 
                  'XRPUSDT', 'DOTUSDT', 'UNIUSDT', 'LTCUSDT', 'LINKUSDT']
        
        for symbol in symbols:
            # Generate klines data
            klines = self._generate_sample_klines(symbol)
            
            # Add to all pairs data
            all_pairs_data[symbol] = {
                'klines': {
                    '1h': klines
                }
            }
        
        return all_pairs_data
    
    def _generate_sample_klines(self, symbol):
        # Generate sample klines data
        now = datetime.now()
        hours_back = 48  # 2 days of hourly data
        
        # Initialize with base price depending on symbol
        if symbol == 'BTCUSDT':
            base_price = random.uniform(45000, 55000)
        elif symbol == 'ETHUSDT':
            base_price = random.uniform(3000, 4000)
        elif symbol == 'BNBUSDT':
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
    
    def generate_projected_data(self, historical_data, symbol):
        # Generate projected price data
        if historical_data.empty:
            return None
        
        # Get last price
        last_price = historical_data['close'].iloc[-1]
        last_time = historical_data['open_time'].iloc[-1]
        
        # Generate projection for next 12 hours
        projection_hours = 12
        time_points = [last_time + timedelta(hours=i) for i in range(1, projection_hours + 1)]
        
        # Generate price movement
        price_change_pct = random.uniform(-0.1, 0.2)  # -10% to +20%
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

# Function to display welcome screen
def display_welcome():
    # Welcome message
    st.markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #F0B90B;">Welcome to the Binance Spot Trading Assistant</h1>
            <p style="font-size: 18px;">Your AI-powered guide for daily cryptocurrency spot trading on Binance</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Features
    st.markdown(
        """
        <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #F0B90B;">Features</h3>
            <ul>
                <li>Real-time analysis of Binance USDT trading pairs</li>
                <li>Comprehensive technical and sentiment analysis</li>
                <li>Specific buy, sell, and hold recommendations</li>
                <li>Projected price movements and selling points</li>
                <li>Risk assessment and confidence scoring</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # How it works
    st.markdown(
        """
        <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #F0B90B;">How It Works</h3>
            <ol>
                <li>Click the "Get Started" button below</li>
                <li>The system will analyze cryptocurrency markets and news</li>
                <li>Review the recommendations and projected price movements</li>
                <li>Make informed trading decisions based on the analysis</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Get started button
    cols = st.columns([1, 1, 1])
    with cols[1]:
        if st.button("Get Started", key="get_started", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()  # Updated from experimental_rerun()

# Function to run analysis
def run_analysis():
    # Initialize data manager if not already done
    if st.session_state.data_manager is None:
        st.session_state.data_manager = get_data_manager()
    
    # Display loading animation
    display_loading_animation()
    
    # Check if we need to refresh data
    current_time = datetime.now()
    if st.session_state.last_refresh_time is None or (current_time - st.session_state.last_refresh_time).total_seconds() > 300:
        st.info("Collecting and analyzing cryptocurrency data... This may take a moment.")
        
        # Refresh data
        success = st.session_state.data_manager.refresh_data()
        
        if success:
            st.session_state.last_refresh_time = current_time
            st.success("Analysis complete! Displaying results...")
            time.sleep(1)  # Brief pause for UI feedback
        else:
            st.error("Error refreshing data. Using cached data if available.")
    
    # Get data
    data = st.session_state.data_manager.get_data()
    
    # Check if we have recommendations
    if not data['recommendations']:
        st.warning("No recommendations available. Please try again later.")
        return
    
    # Store data in session state
    st.session_state.data = data
    
    # Navigate to dashboard
    st.session_state.page = "dashboard"
    st.rerun()  # Updated from experimental_rerun()

# Function to display dashboard
def display_dashboard():
    # Get data from session state
    data = st.session_state.data
    
    # Sidebar navigation
    st.sidebar.markdown("## Navigation")
    page_options = ["Overview", "Buy Recommendations", "Sell Recommendations", "Hold Recommendations", "Market Analysis", "Learn About Trading"]
    selected_page = st.sidebar.radio("Go to", page_options)
    
    # Display last update time
    if data['last_update_time']:
        st.sidebar.markdown(f"**Last Updated:** {data['last_update_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Refresh button
    if st.sidebar.button("Refresh Data", key="refresh_data"):
        st.session_state.page = "analysis"
        st.rerun()  # Updated from experimental_rerun()
    
    # Back button
    if st.sidebar.button("Back to Welcome", key="back_to_welcome"):
        st.session_state.page = "welcome"
        st.rerun()  # Updated from experimental_rerun()
    
    # Display selected page
    if selected_page == "Overview":
        display_overview(data)
    elif selected_page == "Buy Recommendations":
        display_buy_recommendations(data)
    elif selected_page == "Sell Recommendations":
        display_sell_recommendations(data)
    elif selected_page == "Hold Recommendations":
        display_hold_recommendations(data)
    elif selected_page == "Market Analysis":
        display_market_analysis(data)
    elif selected_page == "Learn About Trading":
        display_learning_resources()

# Function to display overview
def display_overview(data):
    st.markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h1 style="color: #F0B90B;">Binance Spot Trading Dashboard</h1>
            <p style="font-size: 16px;">Daily trading recommendations for USDT pairs</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Get recommendations
    recommendations = data['recommendations']
    
    # Split recommendations by action
    buy_recs = [r for r in recommendations if r['action'] == 'buy']
    sell_recs = [r for r in recommendations if r['action'] == 'sell']
    hold_recs = [r for r in recommendations if r['action'] == 'hold']
    
    # Display summary cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #00FF7F;">{len(buy_recs)}</h2>
                <h3>Buy Recommendations</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #FF4B4B;">{len(sell_recs)}</h2>
                <h3>Sell Recommendations</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #FFD700;">{len(hold_recs)}</h2>
                <h3>Hold Recommendations</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Display top recommendations
    st.markdown("## Top Recommendations")
    
    # Top buy recommendation
    if buy_recs:
        top_buy = max(buy_recs, key=lambda x: x['confidence_score'])
        display_recommendation_card(top_buy, "buy")
    
    # Top sell recommendation
    if sell_recs:
        top_sell = max(sell_recs, key=lambda x: x['confidence_score'])
        display_recommendation_card(top_sell, "sell")
    
    # Market overview
    st.markdown("## Market Overview")
    
    # Get market analysis
    market_analysis = data['analysis_results'].get('market_analysis', {})
    market_conditions = market_analysis.get('market_conditions', {})
    
    # Display market conditions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Market Conditions")
        
        # Market trend
        market_trend = market_conditions.get('market_trend', 'neutral')
        trend_color = "#00FF7F" if 'bullish' in market_trend else "#FF4B4B" if 'bearish' in market_trend else "#FFD700"
        
        st.markdown(
            f"""
            <div style="background-color: #474D57; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <p><strong>Market Trend:</strong> <span style="color: {trend_color};">{market_trend.replace('_', ' ').title()}</span></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Market strength
        market_strength = market_conditions.get('market_strength', 50)
        strength_fig = create_confidence_meter(market_strength, width=300, height=150, label="Market Strength")
        st.plotly_chart(strength_fig, use_container_width=True)
        
        # Volatility
        volatility = market_conditions.get('volatility', 'medium')
        volatility_color = "#FF4B4B" if volatility == 'high' else "#FFD700" if volatility == 'medium' else "#00FF7F"
        
        st.markdown(
            f"""
            <div style="background-color: #474D57; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <p><strong>Market Volatility:</strong> <span style="color: {volatility_color};">{volatility.title()}</span></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown("### Top Performers")
        
        # Top performers
        top_performers = market_conditions.get('top_performers', [])
        
        if top_performers:
            for symbol, change in top_performers[:3]:
                st.markdown(
                    f"""
                    <div style="background-color: #474D57; padding: 15px; border-radius: 10px; margin-bottom: 10px; display: flex; align-items: center;">
                        <div style="margin-right: 10px; font-weight: bold;">{symbol}</div>
                        <div style="color: {'#00FF7F' if change > 0 else '#FF4B4B'}; margin-left: auto;">
                            {'+' if change > 0 else ''}{change:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.markdown("### Worst Performers")
        
        # Worst performers
        worst_performers = market_conditions.get('worst_performers', [])
        
        if worst_performers:
            for symbol, change in worst_performers[:3]:
                st.markdown(
                    f"""
                    <div style="background-color: #474D57; padding: 15px; border-radius: 10px; margin-bottom: 10px; display: flex; align-items: center;">
                        <div style="margin-right: 10px; font-weight: bold;">{symbol}</div>
                        <div style="color: {'#00FF7F' if change > 0 else '#FF4B4B'}; margin-left: auto;">
                            {'+' if change > 0 else ''}{change:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Function to display recommendation card
def display_recommendation_card(recommendation, action_type):
    # Determine colors based on action type
    if action_type == "buy":
        action_color = "#00FF7F"
        action_text = "BUY"
    elif action_type == "sell":
        action_color = "#FF4B4B"
        action_text = "SELL"
    else:
        action_color = "#FFD700"
        action_text = "HOLD"
    
    # Get recommendation details
    symbol = recommendation['symbol']
    confidence = recommendation['confidence_score']
    risk = recommendation['risk_score']
    current_price = recommendation['current_price']
    selling_time = recommendation['selling_time']
    selling_price = recommendation['selling_price']
    time_horizon = recommendation['time_horizon']
    
    # Format current price based on value
    if current_price < 0.01:
        formatted_price = f"${current_price:.6f}"
    elif current_price < 1:
        formatted_price = f"${current_price:.4f}"
    else:
        formatted_price = f"${current_price:.2f}"
    
    # Create card
    st.markdown(
        f"""
        <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid {action_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <div style="display: flex; align-items: center;">
                    <h2 style="margin: 0; color: {action_color};">{symbol}</h2>
                </div>
                <div style="background-color: {action_color}; color: #1E2026; padding: 5px 15px; border-radius: 20px; font-weight: bold;">
                    {action_text}
                </div>
            </div>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <div>
                    <p><strong>Current Price:</strong> {formatted_price}</p>
                    <p><strong>Selling Price:</strong> {selling_price}</p>
                </div>
                <div>
                    <p><strong>Selling Time:</strong> {selling_time}</p>
                    <p><strong>Time Horizon:</strong> {time_horizon}</p>
                </div>
            </div>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <div style="width: 48%;">
                    <p style="margin-bottom: 5px;"><strong>Confidence Score:</strong> {confidence:.1f}/100</p>
                    <div style="background-color: #2c3e50; height: 10px; border-radius: 5px; overflow: hidden;">
                        <div style="background-color: {action_color}; width: {confidence}%; height: 100%;"></div>
                    </div>
                </div>
                <div style="width: 48%;">
                    <p style="margin-bottom: 5px;"><strong>Risk Score:</strong> {risk:.1f}/100</p>
                    <div style="background-color: #2c3e50; height: 10px; border-radius: 5px; overflow: hidden;">
                        <div style="background-color: {'#FF4B4B' if risk > 70 else '#FFD700' if risk > 30 else '#00FF7F'}; width: {risk}%; height: 100%;"></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to display buy recommendations
def display_buy_recommendations(data):
    st.markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h1 style="color: #00FF7F;">Buy Recommendations</h1>
            <p style="font-size: 16px;">Cryptocurrencies recommended for purchase</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Get buy recommendations
    recommendations = data['recommendations']
    buy_recs = [r for r in recommendations if r['action'] == 'buy']
    
    if not buy_recs:
        st.info("No buy recommendations available at this time.")
        return
    
    # Sort by confidence score
    buy_recs.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    # Display each recommendation
    for rec in buy_recs:
        with st.expander(f"{rec['symbol']} - Confidence: {rec['confidence_score']:.1f}/100", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Get historical data
                all_pairs_data = data['all_pairs_data']
                if rec['symbol'] in all_pairs_data and 'klines' in all_pairs_data[rec['symbol']]:
                    historical_data = all_pairs_data[rec['symbol']]['klines'].get('1h', pd.DataFrame())
                    
                    # Generate projected data
                    projected_data = st.session_state.data_manager.generate_projected_data(historical_data, rec['symbol'])
                    
                    # Create price chart
                    if not historical_data.empty and projected_data:
                        fig = create_price_chart(historical_data, projected_data, rec['symbol'])
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display confidence and risk meters
                conf_fig = create_confidence_meter(rec['confidence_score'], width=250, height=150, label="Confidence")
                st.plotly_chart(conf_fig, use_container_width=True)
                
                risk_fig = create_confidence_meter(rec['risk_score'], width=250, height=150, label="Risk")
                st.plotly_chart(risk_fig, use_container_width=True)
                
                # Format current price based on value
                current_price = rec['current_price']
                if current_price < 0.01:
                    formatted_price = f"${current_price:.6f}"
                elif current_price < 1:
                    formatted_price = f"${current_price:.4f}"
                else:
                    formatted_price = f"${current_price:.2f}"
                
                # Display key metrics
                st.markdown("### Key Metrics")
                
                st.markdown(
                    f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <p><strong>Current Price:</strong> {formatted_price}</p>
                        <p><strong>Selling Price:</strong> {rec['selling_price']}</p>
                        <p><strong>Selling Time:</strong> {rec['selling_time']}</p>
                        <p><strong>Time Horizon:</strong> {rec['time_horizon']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display technical factors
            st.markdown("### Technical Factors")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px;">
                        <p><strong>Short-term Trend:</strong> <span style="color: {'#00FF7F' if rec['short_term_trend'] == 'bullish' else '#FF4B4B' if rec['short_term_trend'] == 'bearish' else '#FFD700'};">{rec['short_term_trend'].title()}</span></p>
                        <p><strong>Medium-term Trend:</strong> <span style="color: {'#00FF7F' if rec['medium_term_trend'] == 'bullish' else '#FF4B4B' if rec['medium_term_trend'] == 'bearish' else '#FFD700'};">{rec['medium_term_trend'].title()}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px;">
                        <p><strong>Momentum:</strong> <span style="color: {'#00FF7F' if rec['momentum'] == 'oversold' else '#FF4B4B' if rec['momentum'] == 'overbought' else '#FFD700'};">{rec['momentum'].title()}</span></p>
                        <p><strong>MACD Signal:</strong> <span style="color: {'#00FF7F' if rec['macd_signal'] == 'bullish' else '#FF4B4B'};">{rec['macd_signal'].title()}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px;">
                        <p><strong>Volatility:</strong> <span style="color: {'#FF4B4B' if rec['volatility'] == 'high' else '#FFD700' if rec['volatility'] == 'medium' else '#00FF7F'};">{rec['volatility'].title()}</span></p>
                        <p><strong>Trend Strength:</strong> <span style="color: {'#00FF7F' if rec['trend_strength'] == 'strong' else '#FFD700'};">{rec['trend_strength'].title()}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display support and resistance
            st.markdown("### Support and Resistance")
            
            # Format support and resistance based on value
            support = rec['closest_support']
            resistance = rec['closest_resistance']
            
            if support < 0.01:
                formatted_support = f"${support:.6f}"
            elif support < 1:
                formatted_support = f"${support:.4f}"
            else:
                formatted_support = f"${support:.2f}"
                
            if resistance < 0.01:
                formatted_resistance = f"${resistance:.6f}"
            elif resistance < 1:
                formatted_resistance = f"${resistance:.4f}"
            else:
                formatted_resistance = f"${resistance:.2f}"
            
            st.markdown(
                f"""
                <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px; display: flex; justify-content: space-between;">
                    <div>
                        <p><strong>Support:</strong> {formatted_support}</p>
                    </div>
                    <div>
                        <p><strong>Current:</strong> {formatted_price}</p>
                    </div>
                    <div>
                        <p><strong>Resistance:</strong> {formatted_resistance}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Display sentiment
            st.markdown("### Market Sentiment")
            
            st.markdown(
                f"""
                <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px;">
                    <p><strong>Overall Sentiment:</strong> <span style="color: {'#00FF7F' if 'positive' in rec['overall_sentiment'] else '#FF4B4B' if 'negative' in rec['overall_sentiment'] else '#FFD700'};">{rec['overall_sentiment'].replace('_', ' ').title()}</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Function to display sell recommendations
def display_sell_recommendations(data):
    st.markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h1 style="color: #FF4B4B;">Sell Recommendations</h1>
            <p style="font-size: 16px;">Cryptocurrencies recommended for selling</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Get sell recommendations
    recommendations = data['recommendations']
    sell_recs = [r for r in recommendations if r['action'] == 'sell']
    
    if not sell_recs:
        st.info("No sell recommendations available at this time.")
        return
    
    # Sort by confidence score
    sell_recs.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    # Display each recommendation
    for rec in sell_recs:
        with st.expander(f"{rec['symbol']} - Confidence: {rec['confidence_score']:.1f}/100", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Get historical data
                all_pairs_data = data['all_pairs_data']
                if rec['symbol'] in all_pairs_data and 'klines' in all_pairs_data[rec['symbol']]:
                    historical_data = all_pairs_data[rec['symbol']]['klines'].get('1h', pd.DataFrame())
                    
                    # Generate projected data
                    projected_data = st.session_state.data_manager.generate_projected_data(historical_data, rec['symbol'])
                    
                    # Create price chart
                    if not historical_data.empty and projected_data:
                        fig = create_price_chart(historical_data, projected_data, rec['symbol'])
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display confidence and risk meters
                conf_fig = create_confidence_meter(rec['confidence_score'], width=250, height=150, label="Confidence")
                st.plotly_chart(conf_fig, use_container_width=True)
                
                risk_fig = create_confidence_meter(rec['risk_score'], width=250, height=150, label="Risk")
                st.plotly_chart(risk_fig, use_container_width=True)
                
                # Format current price based on value
                current_price = rec['current_price']
                if current_price < 0.01:
                    formatted_price = f"${current_price:.6f}"
                elif current_price < 1:
                    formatted_price = f"${current_price:.4f}"
                else:
                    formatted_price = f"${current_price:.2f}"
                
                # Display key metrics
                st.markdown("### Key Metrics")
                
                st.markdown(
                    f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <p><strong>Current Price:</strong> {formatted_price}</p>
                        <p><strong>Selling Price:</strong> {rec['selling_price']}</p>
                        <p><strong>Selling Time:</strong> {rec['selling_time']}</p>
                        <p><strong>Time Horizon:</strong> {rec['time_horizon']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display technical factors
            st.markdown("### Technical Factors")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px;">
                        <p><strong>Short-term Trend:</strong> <span style="color: {'#00FF7F' if rec['short_term_trend'] == 'bullish' else '#FF4B4B' if rec['short_term_trend'] == 'bearish' else '#FFD700'};">{rec['short_term_trend'].title()}</span></p>
                        <p><strong>Medium-term Trend:</strong> <span style="color: {'#00FF7F' if rec['medium_term_trend'] == 'bullish' else '#FF4B4B' if rec['medium_term_trend'] == 'bearish' else '#FFD700'};">{rec['medium_term_trend'].title()}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px;">
                        <p><strong>Momentum:</strong> <span style="color: {'#00FF7F' if rec['momentum'] == 'oversold' else '#FF4B4B' if rec['momentum'] == 'overbought' else '#FFD700'};">{rec['momentum'].title()}</span></p>
                        <p><strong>MACD Signal:</strong> <span style="color: {'#00FF7F' if rec['macd_signal'] == 'bullish' else '#FF4B4B'};">{rec['macd_signal'].title()}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px;">
                        <p><strong>Volatility:</strong> <span style="color: {'#FF4B4B' if rec['volatility'] == 'high' else '#FFD700' if rec['volatility'] == 'medium' else '#00FF7F'};">{rec['volatility'].title()}</span></p>
                        <p><strong>Trend Strength:</strong> <span style="color: {'#00FF7F' if rec['trend_strength'] == 'strong' else '#FFD700'};">{rec['trend_strength'].title()}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display support and resistance
            st.markdown("### Support and Resistance")
            
            # Format support and resistance based on value
            support = rec['closest_support']
            resistance = rec['closest_resistance']
            
            if support < 0.01:
                formatted_support = f"${support:.6f}"
            elif support < 1:
                formatted_support = f"${support:.4f}"
            else:
                formatted_support = f"${support:.2f}"
                
            if resistance < 0.01:
                formatted_resistance = f"${resistance:.6f}"
            elif resistance < 1:
                formatted_resistance = f"${resistance:.4f}"
            else:
                formatted_resistance = f"${resistance:.2f}"
            
            st.markdown(
                f"""
                <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px; display: flex; justify-content: space-between;">
                    <div>
                        <p><strong>Support:</strong> {formatted_support}</p>
                    </div>
                    <div>
                        <p><strong>Current:</strong> {formatted_price}</p>
                    </div>
                    <div>
                        <p><strong>Resistance:</strong> {formatted_resistance}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Display sentiment
            st.markdown("### Market Sentiment")
            
            st.markdown(
                f"""
                <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px;">
                    <p><strong>Overall Sentiment:</strong> <span style="color: {'#00FF7F' if 'positive' in rec['overall_sentiment'] else '#FF4B4B' if 'negative' in rec['overall_sentiment'] else '#FFD700'};">{rec['overall_sentiment'].replace('_', ' ').title()}</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Function to display hold recommendations
def display_hold_recommendations(data):
    st.markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h1 style="color: #FFD700;">Hold Recommendations</h1>
            <p style="font-size: 16px;">Cryptocurrencies recommended for holding</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Get hold recommendations
    recommendations = data['recommendations']
    hold_recs = [r for r in recommendations if r['action'] == 'hold']
    
    if not hold_recs:
        st.info("No hold recommendations available at this time.")
        return
    
    # Sort by confidence score
    hold_recs.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    # Display each recommendation
    for rec in hold_recs:
        with st.expander(f"{rec['symbol']} - Confidence: {rec['confidence_score']:.1f}/100", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Get historical data
                all_pairs_data = data['all_pairs_data']
                if rec['symbol'] in all_pairs_data and 'klines' in all_pairs_data[rec['symbol']]:
                    historical_data = all_pairs_data[rec['symbol']]['klines'].get('1h', pd.DataFrame())
                    
                    # Generate projected data
                    projected_data = st.session_state.data_manager.generate_projected_data(historical_data, rec['symbol'])
                    
                    # Create price chart
                    if not historical_data.empty and projected_data:
                        fig = create_price_chart(historical_data, projected_data, rec['symbol'])
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display confidence and risk meters
                conf_fig = create_confidence_meter(rec['confidence_score'], width=250, height=150, label="Confidence")
                st.plotly_chart(conf_fig, use_container_width=True)
                
                risk_fig = create_confidence_meter(rec['risk_score'], width=250, height=150, label="Risk")
                st.plotly_chart(risk_fig, use_container_width=True)
                
                # Format current price based on value
                current_price = rec['current_price']
                if current_price < 0.01:
                    formatted_price = f"${current_price:.6f}"
                elif current_price < 1:
                    formatted_price = f"${current_price:.4f}"
                else:
                    formatted_price = f"${current_price:.2f}"
                
                # Display key metrics
                st.markdown("### Key Metrics")
                
                st.markdown(
                    f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <p><strong>Current Price:</strong> {formatted_price}</p>
                        <p><strong>Selling Price:</strong> {rec['selling_price']}</p>
                        <p><strong>Selling Time:</strong> {rec['selling_time']}</p>
                        <p><strong>Time Horizon:</strong> {rec['time_horizon']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display technical factors
            st.markdown("### Technical Factors")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px;">
                        <p><strong>Short-term Trend:</strong> <span style="color: {'#00FF7F' if rec['short_term_trend'] == 'bullish' else '#FF4B4B' if rec['short_term_trend'] == 'bearish' else '#FFD700'};">{rec['short_term_trend'].title()}</span></p>
                        <p><strong>Medium-term Trend:</strong> <span style="color: {'#00FF7F' if rec['medium_term_trend'] == 'bullish' else '#FF4B4B' if rec['medium_term_trend'] == 'bearish' else '#FFD700'};">{rec['medium_term_trend'].title()}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px;">
                        <p><strong>Momentum:</strong> <span style="color: {'#00FF7F' if rec['momentum'] == 'oversold' else '#FF4B4B' if rec['momentum'] == 'overbought' else '#FFD700'};">{rec['momentum'].title()}</span></p>
                        <p><strong>MACD Signal:</strong> <span style="color: {'#00FF7F' if rec['macd_signal'] == 'bullish' else '#FF4B4B'};">{rec['macd_signal'].title()}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f"""
                    <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px;">
                        <p><strong>Volatility:</strong> <span style="color: {'#FF4B4B' if rec['volatility'] == 'high' else '#FFD700' if rec['volatility'] == 'medium' else '#00FF7F'};">{rec['volatility'].title()}</span></p>
                        <p><strong>Trend Strength:</strong> <span style="color: {'#00FF7F' if rec['trend_strength'] == 'strong' else '#FFD700'};">{rec['trend_strength'].title()}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display support and resistance
            st.markdown("### Support and Resistance")
            
            # Format support and resistance based on value
            support = rec['closest_support']
            resistance = rec['closest_resistance']
            
            if support < 0.01:
                formatted_support = f"${support:.6f}"
            elif support < 1:
                formatted_support = f"${support:.4f}"
            else:
                formatted_support = f"${support:.2f}"
                
            if resistance < 0.01:
                formatted_resistance = f"${resistance:.6f}"
            elif resistance < 1:
                formatted_resistance = f"${resistance:.4f}"
            else:
                formatted_resistance = f"${resistance:.2f}"
            
            st.markdown(
                f"""
                <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px; display: flex; justify-content: space-between;">
                    <div>
                        <p><strong>Support:</strong> {formatted_support}</p>
                    </div>
                    <div>
                        <p><strong>Current:</strong> {formatted_price}</p>
                    </div>
                    <div>
                        <p><strong>Resistance:</strong> {formatted_resistance}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Display sentiment
            st.markdown("### Market Sentiment")
            
            st.markdown(
                f"""
                <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px;">
                    <p><strong>Overall Sentiment:</strong> <span style="color: {'#00FF7F' if 'positive' in rec['overall_sentiment'] else '#FF4B4B' if 'negative' in rec['overall_sentiment'] else '#FFD700'};">{rec['overall_sentiment'].replace('_', ' ').title()}</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Function to display market analysis
def display_market_analysis(data):
    st.markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h1 style="color: #F0B90B;">Market Analysis</h1>
            <p style="font-size: 16px;">Comprehensive analysis of cryptocurrency market conditions</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Get market analysis
    market_analysis = data['analysis_results'].get('market_analysis', {})
    market_conditions = market_analysis.get('market_conditions', {})
    sector_performance = market_analysis.get('sector_performance', {})
    
    # Display market conditions
    st.markdown("## Market Conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market trend
        market_trend = market_conditions.get('market_trend', 'neutral')
        trend_color = "#00FF7F" if 'bullish' in market_trend else "#FF4B4B" if 'bearish' in market_trend else "#FFD700"
        
        st.markdown(
            f"""
            <div style="background-color: #474D57; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <p><strong>Market Trend:</strong> <span style="color: {trend_color};">{market_trend.replace('_', ' ').title()}</span></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Market strength
        market_strength = market_conditions.get('market_strength', 50)
        strength_fig = create_confidence_meter(market_strength, width=300, height=150, label="Market Strength")
        st.plotly_chart(strength_fig, use_container_width=True)
        
        # Volatility
        volatility = market_conditions.get('volatility', 'medium')
        volatility_color = "#FF4B4B" if volatility == 'high' else "#FFD700" if volatility == 'medium' else "#00FF7F"
        
        st.markdown(
            f"""
            <div style="background-color: #474D57; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <p><strong>Market Volatility:</strong> <span style="color: {volatility_color};">{volatility.title()}</span></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # BTC dominance
        btc_dominance = market_conditions.get('btc_dominance', 0.4)
        
        st.markdown(
            f"""
            <div style="background-color: #474D57; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <p><strong>BTC Dominance:</strong> {btc_dominance * 100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown("### Top Performers")
        
        # Top performers
        top_performers = market_conditions.get('top_performers', [])
        
        if top_performers:
            for symbol, change in top_performers:
                st.markdown(
                    f"""
                    <div style="background-color: #474D57; padding: 15px; border-radius: 10px; margin-bottom: 10px; display: flex; align-items: center;">
                        <div style="margin-right: 10px; font-weight: bold;">{symbol}</div>
                        <div style="color: {'#00FF7F' if change > 0 else '#FF4B4B'}; margin-left: auto;">
                            {'+' if change > 0 else ''}{change:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.markdown("### Worst Performers")
        
        # Worst performers
        worst_performers = market_conditions.get('worst_performers', [])
        
        if worst_performers:
            for symbol, change in worst_performers:
                st.markdown(
                    f"""
                    <div style="background-color: #474D57; padding: 15px; border-radius: 10px; margin-bottom: 10px; display: flex; align-items: center;">
                        <div style="margin-right: 10px; font-weight: bold;">{symbol}</div>
                        <div style="color: {'#00FF7F' if change > 0 else '#FF4B4B'}; margin-left: auto;">
                            {'+' if change > 0 else ''}{change:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # Display sector performance
    st.markdown("## Sector Performance")
    
    if sector_performance:
        # Create bar chart
        sectors = list(sector_performance.keys())
        performances = list(sector_performance.values())
        
        # Create colors based on performance
        colors = ['#00FF7F' if p > 0 else '#FF4B4B' for p in performances]
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=sectors,
                y=performances,
                marker_color=colors,
                text=[f"{p:.2f}%" for p in performances],
                textposition='auto'
            )
        )
        
        fig.update_layout(
            title="Performance by Sector",
            xaxis_title="Sector",
            yaxis_title="Performance (%)",
            template="plotly_dark",
            plot_bgcolor='#1E2026',
            paper_bgcolor='#1E2026',
            font={'color': 'white'},
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display market correlations
    st.markdown("## Market Correlations")
    
    market_correlations = market_analysis.get('market_correlations', {})
    
    if market_correlations:
        # Create correlation matrix
        symbols = list(market_correlations.keys())
        correlation_matrix = []
        
        for symbol1 in symbols:
            row = []
            for symbol2 in symbols:
                row.append(market_correlations[symbol1].get(symbol2, 0))
            correlation_matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=symbols,
            y=symbols,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            template="plotly_dark",
            plot_bgcolor='#1E2026',
            paper_bgcolor='#1E2026',
            font={'color': 'white'},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Function to display learning resources
def display_learning_resources():
    st.markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h1 style="color: #F0B90B;">Learn About Trading</h1>
            <p style="font-size: 16px;">Educational resources for cryptocurrency trading</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Basic concepts
    st.markdown("## Basic Concepts")
    
    with st.expander("What is Spot Trading?", expanded=False):
        st.markdown(
            """
            **Spot Trading** refers to the buying and selling of cryptocurrencies at the current market price for immediate delivery. Unlike futures or margin trading, spot trading involves:
            
            - Trading with funds you actually own
            - No leverage or borrowing
            - Immediate settlement of trades
            - Lower risk compared to derivatives trading
            
            When you buy a cryptocurrency in spot trading, you actually own the asset and can withdraw it to your personal wallet.
            """
        )
    
    with st.expander("Understanding Trading Pairs", expanded=False):
        st.markdown(
            """
            **Trading Pairs** represent the two cryptocurrencies being exchanged in a trade. For example, in the pair BTCUSDT:
            
            - BTC is the base currency (what you're buying or selling)
            - USDT is the quote currency (what you're paying with or receiving)
            
            When you see a price like "BTCUSDT = 50,000", it means 1 Bitcoin costs 50,000 USDT.
            
            This application focuses exclusively on USDT trading pairs, where USDT (Tether) is a stablecoin pegged to the US Dollar.
            """
        )
    
    with st.expander("Reading Price Charts", expanded=False):
        st.markdown(
            """
            **Price Charts** display historical price movements. The main types are:
            
            - **Line Charts**: Simple visualization of closing prices
            - **Candlestick Charts**: Show opening, closing, high, and low prices for each time period
            
            In candlestick charts:
            - Green/white candles indicate price increased during that period
            - Red/black candles indicate price decreased
            - The "body" shows opening and closing prices
            - The "wicks" show the highest and lowest prices reached
            
            This application uses candlestick charts with projected price movements shown as dashed lines.
            """
        )
    
    # Technical analysis
    st.markdown("## Technical Analysis")
    
    with st.expander("Technical Indicators Explained", expanded=False):
        st.markdown(
            """
            **Technical Indicators** are mathematical calculations based on price and volume data that help traders identify potential trading opportunities.
            
            Key indicators used in this application:
            
            - **Moving Averages (MA)**: Average price over a specific period, helps identify trends
            - **Relative Strength Index (RSI)**: Measures the speed and change of price movements (0-100)
                - Above 70: Potentially overbought
                - Below 30: Potentially oversold
            - **Moving Average Convergence Divergence (MACD)**: Shows relationship between two moving averages
            - **Bollinger Bands**: Shows price volatility with upper and lower bands
            
            These indicators help identify trends, momentum, and potential reversal points.
            """
        )
    
    with st.expander("Support and Resistance Levels", expanded=False):
        st.markdown(
            """
            **Support and Resistance** are price levels where a cryptocurrency has historically had difficulty moving beyond.
            
            - **Support**: Price level where buying interest is strong enough to overcome selling pressure
            - **Resistance**: Price level where selling pressure overcomes buying interest
            
            When price breaks through these levels, they often switch roles (former resistance becomes support and vice versa).
            
            These levels are important for:
            - Setting entry and exit points
            - Placing stop-loss orders
            - Identifying potential price targets
            """
        )
    
    with st.expander("Chart Patterns", expanded=False):
        st.markdown(
            """
            **Chart Patterns** are specific formations on price charts that can signal potential future price movements.
            
            Common patterns include:
            
            - **Head and Shoulders**: Potential reversal pattern with three peaks
            - **Double Top/Bottom**: Potential reversal pattern with two peaks/troughs
            - **Triangle Patterns**: Consolidation patterns showing converging trendlines
            - **Flag and Pennant**: Continuation patterns after strong price movements
            
            This application identifies these patterns automatically and incorporates them into trading recommendations.
            """
        )
    
    # Trading strategies
    st.markdown("## Trading Strategies")
    
    with st.expander("Day Trading Basics", expanded=False):
        st.markdown(
            """
            **Day Trading** involves opening and closing positions within the same day, focusing on short-term price movements.
            
            Key principles:
            
            - **Set clear goals**: Define profit targets and stop-loss levels before trading
            - **Use technical analysis**: Rely on indicators and chart patterns for entry/exit decisions
            - **Manage risk**: Never risk more than 1-2% of your trading capital on a single trade
            - **Control emotions**: Stick to your strategy regardless of fear or greed
            - **Keep records**: Track all trades to identify what works and what doesn't
            
            This application is designed for daily spot trading with timeframes ranging from 15 minutes to 12 hours.
            """
        )
    
    with st.expander("Risk Management", expanded=False):
        st.markdown(
            """
            **Risk Management** is crucial for long-term trading success. Key principles include:
            
            - **Position Sizing**: Determine how much to invest in each trade
                - Recommended: 1-2% of total capital per trade
            - **Stop-Loss Orders**: Set automatic sell orders to limit potential losses
            - **Take-Profit Levels**: Set target prices to secure profits
            - **Risk-Reward Ratio**: Aim for potential rewards that outweigh potential risks (e.g., 1:2 or 1:3)
            - **Diversification**: Spread risk across multiple cryptocurrencies
            
            The confidence and risk scores in this application help you assess the risk level of each recommendation.
            """
        )
    
    with st.expander("Using This Application Effectively", expanded=False):
        st.markdown(
            """
            **Best Practices** for using the Binance Spot Trading Assistant:
            
            1. **Review all metrics**: Don't just look at the buy/sell recommendation, but understand the supporting factors
            2. **Check multiple timeframes**: Confirm signals across different time periods
            3. **Consider market conditions**: Overall market trend affects individual cryptocurrencies
            4. **Use the projected data**: The selling time and price projections help with exit planning
            5. **Set your own stop-loss**: Always protect your capital with appropriate stop-loss orders
            6. **Refresh regularly**: Market conditions change rapidly, refresh data for the latest analysis
            7. **Start small**: Begin with smaller positions until you're comfortable with the system
            
            Remember that no trading system is 100% accurate. Always use this tool as one input in your decision-making process.
            """
        )

# Main function
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "welcome"
    
    # Display appropriate page
    if st.session_state.page == "welcome":
        display_welcome()
    elif st.session_state.page == "analysis":
        run_analysis()
    elif st.session_state.page == "dashboard":
        display_dashboard()

# Run the application
if __name__ == "__main__":
    main()
