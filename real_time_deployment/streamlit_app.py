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

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the real-time data manager
from data_collection.real_time_data_manager import RealTimeDataManager

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
def create_price_chart(historical_data, projected_data=None, symbol="BTC-USD"):
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
        yaxis_title="Price (USD)",
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
            <p>This assistant analyzes cryptocurrency data using:</p>
            <ul>
                <li><strong>Technical Analysis:</strong> Price patterns, trends, and indicators</li>
                <li><strong>Market Conditions:</strong> Overall market sentiment and correlations</li>
                <li><strong>Risk Assessment:</strong> Volatility and potential downside</li>
            </ul>
            <p>All recommendations are specifically tailored for daily spot trading on Binance.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Disclaimer
    st.markdown(
        """
        <div style="background-color: #474D57; padding: 20px; border-radius: 10px;">
            <h3 style="color: #F0B90B;">Disclaimer</h3>
            <p>This tool is for informational purposes only and does not constitute financial advice. Always do your own research before making investment decisions. Cryptocurrency trading involves significant risk.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to display overview
def display_overview(data):
    # Display market overview
    st.markdown(
        """
        <h2 style="color: #F0B90B;">Market Overview</h2>
        """,
        unsafe_allow_html=True
    )
    
    # Get market analysis
    market_analysis = data['analysis_results']['market_analysis']
    market_conditions = market_analysis['market_conditions']
    
    # Create columns for market metrics
    cols = st.columns(4)
    
    # Market trend
    with cols[0]:
        trend = market_conditions['market_trend']
        trend_color = "#00FF7F" if "bullish" in trend else "#FF4B4B" if "bearish" in trend else "#F0B90B"
        st.markdown(
            f"""
            <div style="background-color: #474D57; padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="color: white; margin-bottom: 10px;">Market Trend</h4>
                <p style="font-size: 20px; color: {trend_color}; font-weight: bold;">{trend.replace('_', ' ').title()}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Market strength
    with cols[1]:
        strength = market_conditions['market_strength']
        st.markdown(
            f"""
            <div style="background-color: #474D57; padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="color: white; margin-bottom: 10px;">Market Strength</h4>
                <p style="font-size: 20px; color: #F0B90B; font-weight: bold;">{strength:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Volatility
    with cols[2]:
        volatility = market_conditions['volatility']
        volatility_color = "#FF4B4B" if volatility == "high" else "#F0B90B" if volatility == "medium" else "#00FF7F"
        st.markdown(
            f"""
            <div style="background-color: #474D57; padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="color: white; margin-bottom: 10px;">Volatility</h4>
                <p style="font-size: 20px; color: {volatility_color}; font-weight: bold;">{volatility.title()}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # BTC Dominance
    with cols[3]:
        btc_dominance = market_conditions['btc_dominance'] * 100
        st.markdown(
            f"""
            <div style="background-color: #474D57; padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="color: white; margin-bottom: 10px;">BTC Dominance</h4>
                <p style="font-size: 20px; color: #F0B90B; font-weight: bold;">{btc_dominance:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Display top recommendations
    st.markdown(
        """
        <h2 style="color: #F0B90B; margin-top: 30px;">Top Recommendations</h2>
        """,
        unsafe_allow_html=True
    )
    
    # Get recommendations
    recommendations = data['recommendations']
    
    # Filter by action
    buy_recs = [r for r in recommendations if r['action'] == 'buy']
    sell_recs = [r for r in recommendations if r['action'] == 'sell']
    
    # Sort by confidence score
    buy_recs = sorted(buy_recs, key=lambda x: x['confidence_score'], reverse=True)
    sell_recs = sorted(sell_recs, key=lambda x: x['confidence_score'], reverse=True)
    
    # Get top recommendations
    top_buy = buy_recs[0] if buy_recs else None
    top_sell = sell_recs[0] if sell_recs else None
    
    # Create columns for top recommendations
    cols = st.columns(2)
    
    # Display top buy recommendation
    with cols[0]:
        if top_buy:
            display_recommendation_card(top_buy, "buy")
        else:
            st.markdown(
                """
                <div style="background-color: #474D57; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #F0B90B;">No Buy Recommendations</h3>
                    <p>There are currently no buy recommendations based on market conditions.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Display top sell recommendation
    with cols[1]:
        if top_sell:
            display_recommendation_card(top_sell, "sell")
        else:
            st.markdown(
                """
                <div style="background-color: #474D57; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #F0B90B;">No Sell Recommendations</h3>
                    <p>There are currently no sell recommendations based on market conditions.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Function to display recommendation card
def display_recommendation_card(rec, action_type):
    # Set colors based on action type
    if action_type == "buy":
        action_color = "#00FF7F"
        action_text = "BUY"
    else:
        action_color = "#FF4B4B"
        action_text = "SELL"
    
    # Format current price based on value
    current_price = rec['current_price']
    if current_price < 0.01:
        price_display = f"${current_price:.6f}"
    elif current_price < 1:
        price_display = f"${current_price:.4f}"
    else:
        price_display = f"${current_price:.2f}"
    
    # Create card header
    st.markdown(
        f"""
        <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="color: white; margin: 0;">{rec['symbol']}</h3>
                <span style="background-color: {action_color}; color: black; padding: 5px 15px; border-radius: 5px; font-weight: bold;">{action_text}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                <div>
                    <p><strong>Current Price:</strong> {price_display}</p>
                    <p><strong>Selling Time:</strong> {rec['selling_time']}</p>
                    <p><strong>Selling Price:</strong> {rec['selling_price']}</p>
                </div>
                <div>
                    <p><strong>Time Horizon:</strong> {rec['time_horizon']}</p>
                    <p><strong>Short-Term Trend:</strong> {rec['short_term_trend'].title()}</p>
                    <p><strong>Momentum:</strong> {rec['momentum'].title()}</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create columns for confidence and risk meters
    cols = st.columns(2)
    
    # Display confidence meter
    with cols[0]:
        confidence_fig = create_confidence_meter(rec['confidence_score'], width=200, height=120, label="Confidence")
        st.plotly_chart(confidence_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Display risk meter
    with cols[1]:
        risk_fig = create_confidence_meter(rec['risk_score'], width=200, height=120, label="Risk")
        st.plotly_chart(risk_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Close the card
    st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to display buy recommendations
def display_buy_recommendations(data):
    st.markdown(
        """
        <h2 style="color: #F0B90B;">Buy Recommendations</h2>
        <p>Cryptocurrencies recommended for purchase based on technical analysis and market conditions.</p>
        """,
        unsafe_allow_html=True
    )
    
    # Get recommendations
    recommendations = data['recommendations']
    
    # Filter buy recommendations
    buy_recs = [r for r in recommendations if r['action'] == 'buy']
    
    # Sort by confidence score
    buy_recs = sorted(buy_recs, key=lambda x: x['confidence_score'], reverse=True)
    
    if not buy_recs:
        st.markdown(
            """
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #F0B90B;">No Buy Recommendations</h3>
                <p>There are currently no buy recommendations based on market conditions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return
    
    # Display each recommendation
    for i, rec in enumerate(buy_recs):
        st.markdown(
            f"""
            <h3 style="color: #F0B90B; margin-top: 30px;">{i+1}. {rec['symbol']}</h3>
            """,
            unsafe_allow_html=True
        )
        
        # Create columns for details and chart
        cols = st.columns([2, 3])
        
        # Display recommendation details
        with cols[0]:
            # Format current price based on value
            current_price = rec['current_price']
            if current_price < 0.01:
                price_display = f"${current_price:.6f}"
            elif current_price < 1:
                price_display = f"${current_price:.4f}"
            else:
                price_display = f"${current_price:.2f}"
                
            st.markdown(
                f"""
                <div style="background-color: #474D57; padding: 20px; border-radius: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h4 style="color: white; margin: 0;">Details</h4>
                        <span style="background-color: #00FF7F; color: black; padding: 5px 15px; border-radius: 5px; font-weight: bold;">BUY</span>
                    </div>
                    <p><strong>Current Price:</strong> {price_display}</p>
                    <p><strong>Confidence Score:</strong> {rec['confidence_score']:.1f}%</p>
                    <p><strong>Risk Score:</strong> {rec['risk_score']:.1f}%</p>
                    <p><strong>Selling Time:</strong> {rec['selling_time']}</p>
                    <p><strong>Selling Price:</strong> {rec['selling_price']}</p>
                    <p><strong>Time Horizon:</strong> {rec['time_horizon']}</p>
                    
                    <h4 style="color: #F0B90B; margin-top: 20px;">Technical Factors</h4>
                    <p><strong>Short-Term Trend:</strong> {rec['short_term_trend'].title()}</p>
                    <p><strong>Medium-Term Trend:</strong> {rec['medium_term_trend'].title()}</p>
                    <p><strong>Momentum:</strong> {rec['momentum'].title()}</p>
                    <p><strong>MACD Signal:</strong> {rec['macd_signal'].title()}</p>
                    <p><strong>Volatility:</strong> {rec['volatility'].title()}</p>
                    <p><strong>Trend Strength:</strong> {rec['trend_strength'].title()}</p>
                    
                    <h4 style="color: #F0B90B; margin-top: 20px;">Support & Resistance</h4>
                    <p><strong>Support Level:</strong> ${rec['closest_support']:.2f}</p>
                    <p><strong>Resistance Level:</strong> ${rec['closest_resistance']:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Display price chart
        with cols[1]:
            # Get historical data
            symbol = rec['symbol']
            if symbol in data['all_pairs_data'] and 'klines' in data['all_pairs_data'][symbol] and '1d' in data['all_pairs_data'][symbol]['klines']:
                historical_data = data['all_pairs_data'][symbol]['klines']['1d']
                
                # Generate projected data
                if st.session_state.data_manager:
                    projected_data = st.session_state.data_manager.generate_projected_data(historical_data, rec['symbol'])
                    
                    # Create and display chart
                    fig = create_price_chart(historical_data, projected_data, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Data manager not initialized")

# Function to display sell recommendations
def display_sell_recommendations(data):
    st.markdown(
        """
        <h2 style="color: #F0B90B;">Sell Recommendations</h2>
        <p>Cryptocurrencies recommended for selling based on technical analysis and market conditions.</p>
        """,
        unsafe_allow_html=True
    )
    
    # Get recommendations
    recommendations = data['recommendations']
    
    # Filter sell recommendations
    sell_recs = [r for r in recommendations if r['action'] == 'sell']
    
    # Sort by confidence score
    sell_recs = sorted(sell_recs, key=lambda x: x['confidence_score'], reverse=True)
    
    if not sell_recs:
        st.markdown(
            """
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #F0B90B;">No Sell Recommendations</h3>
                <p>There are currently no sell recommendations based on market conditions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return
    
    # Display each recommendation
    for i, rec in enumerate(sell_recs):
        st.markdown(
            f"""
            <h3 style="color: #F0B90B; margin-top: 30px;">{i+1}. {rec['symbol']}</h3>
            """,
            unsafe_allow_html=True
        )
        
        # Create columns for details and chart
        cols = st.columns([2, 3])
        
        # Display recommendation details
        with cols[0]:
            # Format current price based on value
            current_price = rec['current_price']
            if current_price < 0.01:
                price_display = f"${current_price:.6f}"
            elif current_price < 1:
                price_display = f"${current_price:.4f}"
            else:
                price_display = f"${current_price:.2f}"
                
            st.markdown(
                f"""
                <div style="background-color: #474D57; padding: 20px; border-radius: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h4 style="color: white; margin: 0;">Details</h4>
                        <span style="background-color: #FF4B4B; color: black; padding: 5px 15px; border-radius: 5px; font-weight: bold;">SELL</span>
                    </div>
                    <p><strong>Current Price:</strong> {price_display}</p>
                    <p><strong>Confidence Score:</strong> {rec['confidence_score']:.1f}%</p>
                    <p><strong>Risk Score:</strong> {rec['risk_score']:.1f}%</p>
                    <p><strong>Selling Time:</strong> {rec['selling_time']}</p>
                    <p><strong>Selling Price:</strong> {rec['selling_price']}</p>
                    <p><strong>Time Horizon:</strong> {rec['time_horizon']}</p>
                    
                    <h4 style="color: #F0B90B; margin-top: 20px;">Technical Factors</h4>
                    <p><strong>Short-Term Trend:</strong> {rec['short_term_trend'].title()}</p>
                    <p><strong>Medium-Term Trend:</strong> {rec['medium_term_trend'].title()}</p>
                    <p><strong>Momentum:</strong> {rec['momentum'].title()}</p>
                    <p><strong>MACD Signal:</strong> {rec['macd_signal'].title()}</p>
                    <p><strong>Volatility:</strong> {rec['volatility'].title()}</p>
                    <p><strong>Trend Strength:</strong> {rec['trend_strength'].title()}</p>
                    
                    <h4 style="color: #F0B90B; margin-top: 20px;">Support & Resistance</h4>
                    <p><strong>Support Level:</strong> ${rec['closest_support']:.2f}</p>
                    <p><strong>Resistance Level:</strong> ${rec['closest_resistance']:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Display price chart
        with cols[1]:
            # Get historical data
            symbol = rec['symbol']
            if symbol in data['all_pairs_data'] and 'klines' in data['all_pairs_data'][symbol] and '1d' in data['all_pairs_data'][symbol]['klines']:
                historical_data = data['all_pairs_data'][symbol]['klines']['1d']
                
                # Generate projected data
                if st.session_state.data_manager:
                    projected_data = st.session_state.data_manager.generate_projected_data(historical_data, rec['symbol'])
                    
                    # Create and display chart
                    fig = create_price_chart(historical_data, projected_data, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Data manager not initialized")

# Function to display hold recommendations
def display_hold_recommendations(data):
    st.markdown(
        """
        <h2 style="color: #F0B90B;">Hold Recommendations</h2>
        <p>Cryptocurrencies recommended for holding based on technical analysis and market conditions.</p>
        """,
        unsafe_allow_html=True
    )
    
    # Get recommendations
    recommendations = data['recommendations']
    
    # Filter hold recommendations
    hold_recs = [r for r in recommendations if r['action'] == 'hold']
    
    # Sort by confidence score
    hold_recs = sorted(hold_recs, key=lambda x: x['confidence_score'], reverse=True)
    
    if not hold_recs:
        st.markdown(
            """
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #F0B90B;">No Hold Recommendations</h3>
                <p>There are currently no hold recommendations based on market conditions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return
    
    # Create a grid layout for hold recommendations
    cols = st.columns(2)
    
    # Display each recommendation
    for i, rec in enumerate(hold_recs):
        with cols[i % 2]:
            # Format current price based on value
            current_price = rec['current_price']
            if current_price < 0.01:
                price_display = f"${current_price:.6f}"
            elif current_price < 1:
                price_display = f"${current_price:.4f}"
            else:
                price_display = f"${current_price:.2f}"
                
            st.markdown(
                f"""
                <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h3 style="color: white; margin: 0;">{rec['symbol']}</h3>
                        <span style="background-color: #F0B90B; color: black; padding: 5px 15px; border-radius: 5px; font-weight: bold;">HOLD</span>
                    </div>
                    <p><strong>Current Price:</strong> {price_display}</p>
                    <p><strong>Confidence Score:</strong> {rec['confidence_score']:.1f}%</p>
                    <p><strong>Risk Score:</strong> {rec['risk_score']:.1f}%</p>
                    <p><strong>Short-Term Trend:</strong> {rec['short_term_trend'].title()}</p>
                    <p><strong>Medium-Term Trend:</strong> {rec['medium_term_trend'].title()}</p>
                    <p><strong>Momentum:</strong> {rec['momentum'].title()}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Function to display market analysis
def display_market_analysis(data):
    st.markdown(
        """
        <h2 style="color: #F0B90B;">Market Analysis</h2>
        <p>Comprehensive analysis of current cryptocurrency market conditions.</p>
        """,
        unsafe_allow_html=True
    )
    
    # Get market analysis
    market_analysis = data['analysis_results']['market_analysis']
    market_conditions = market_analysis['market_conditions']
    sector_performance = market_analysis['sector_performance']
    market_correlations = market_analysis['market_correlations']
    
    # Create tabs for different analysis sections
    tabs = st.tabs(["Market Conditions", "Sector Performance", "Correlations"])
    
    # Market Conditions tab
    with tabs[0]:
        # Create columns for market metrics
        cols = st.columns(4)
        
        # Market trend
        with cols[0]:
            trend = market_conditions['market_trend']
            trend_color = "#00FF7F" if "bullish" in trend else "#FF4B4B" if "bearish" in trend else "#F0B90B"
            st.markdown(
                f"""
                <div style="background-color: #474D57; padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="color: white; margin-bottom: 10px;">Market Trend</h4>
                    <p style="font-size: 20px; color: {trend_color}; font-weight: bold;">{trend.replace('_', ' ').title()}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Market strength
        with cols[1]:
            strength = market_conditions['market_strength']
            st.markdown(
                f"""
                <div style="background-color: #474D57; padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="color: white; margin-bottom: 10px;">Market Strength</h4>
                    <p style="font-size: 20px; color: #F0B90B; font-weight: bold;">{strength:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Volatility
        with cols[2]:
            volatility = market_conditions['volatility']
            volatility_color = "#FF4B4B" if volatility == "high" else "#F0B90B" if volatility == "medium" else "#00FF7F"
            st.markdown(
                f"""
                <div style="background-color: #474D57; padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="color: white; margin-bottom: 10px;">Volatility</h4>
                    <p style="font-size: 20px; color: {volatility_color}; font-weight: bold;">{volatility.title()}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # BTC Dominance
        with cols[3]:
            btc_dominance = market_conditions['btc_dominance'] * 100
            st.markdown(
                f"""
                <div style="background-color: #474D57; padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="color: white; margin-bottom: 10px;">BTC Dominance</h4>
                    <p style="font-size: 20px; color: #F0B90B; font-weight: bold;">{btc_dominance:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Top and worst performers
        st.markdown("<h3 style='color: #F0B90B; margin-top: 30px;'>Top Performers</h3>", unsafe_allow_html=True)
        
        # Create columns for top performers
        cols = st.columns(5)
        
        # Display top performers
        for i, (symbol, perf) in enumerate(market_conditions['top_performers']):
            with cols[i]:
                perf_color = "#00FF7F" if perf > 0 else "#FF4B4B"
                st.markdown(
                    f"""
                    <div style="background-color: #474D57; padding: 15px; border-radius: 10px; text-align: center;">
                        <h4 style="color: white; margin-bottom: 10px;">{symbol}</h4>
                        <p style="font-size: 18px; color: {perf_color}; font-weight: bold;">{perf:+.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.markdown("<h3 style='color: #F0B90B; margin-top: 30px;'>Worst Performers</h3>", unsafe_allow_html=True)
        
        # Create columns for worst performers
        cols = st.columns(5)
        
        # Display worst performers
        for i, (symbol, perf) in enumerate(market_conditions['worst_performers']):
            with cols[i]:
                perf_color = "#00FF7F" if perf > 0 else "#FF4B4B"
                st.markdown(
                    f"""
                    <div style="background-color: #474D57; padding: 15px; border-radius: 10px; text-align: center;">
                        <h4 style="color: white; margin-bottom: 10px;">{symbol}</h4>
                        <p style="font-size: 18px; color: {perf_color}; font-weight: bold;">{perf:+.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # Sector Performance tab
    with tabs[1]:
        st.markdown("<h3 style='color: #F0B90B;'>Sector Performance</h3>", unsafe_allow_html=True)
        
        # Create bar chart for sector performance
        sectors = list(sector_performance.keys())
        performances = list(sector_performance.values())
        
        # Create colors based on performance
        colors = ["#00FF7F" if p > 0 else "#FF4B4B" for p in performances]
        
        # Create figure
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=sectors,
                y=performances,
                marker_color=colors,
                text=[f"{p:+.2f}%" for p in performances],
                textposition="auto"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Cryptocurrency Sector Performance",
            xaxis_title="Sector",
            yaxis_title="Performance (%)",
            template="plotly_dark",
            plot_bgcolor='#1E2026',
            paper_bgcolor='#1E2026',
            font={'color': 'white'},
            height=500
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlations tab
    with tabs[2]:
        st.markdown("<h3 style='color: #F0B90B;'>Market Correlations</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <p>Correlation between major cryptocurrencies. Values close to 1.0 indicate strong positive correlation, while values close to -1.0 indicate strong negative correlation.</p>
            """,
            unsafe_allow_html=True
        )
        
        # Create correlation matrix
        corr_symbols = list(market_correlations.keys())
        corr_matrix = []
        
        for symbol1 in corr_symbols:
            row = []
            for symbol2 in corr_symbols:
                row.append(market_correlations[symbol1].get(symbol2, 0))
            corr_matrix.append(row)
        
        # Create heatmap
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                x=corr_symbols,
                y=corr_symbols,
                colorscale='YlOrRd',
                zmin=0,
                zmax=1,
                text=[[f"{val:.2f}" for val in row] for row in corr_matrix],
                texttemplate="%{text}",
                textfont={"size": 12}
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Cryptocurrency Correlation Matrix",
            template="plotly_dark",
            plot_bgcolor='#1E2026',
            paper_bgcolor='#1E2026',
            font={'color': 'white'},
            height=500
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)

# Function to display educational resources
def display_educational_resources():
    st.markdown(
        """
        <h2 style="color: #F0B90B;">Educational Resources</h2>
        <p>Learn more about cryptocurrency trading and technical analysis.</p>
        """,
        unsafe_allow_html=True
    )
    
    # Create tabs for different educational sections
    tabs = st.tabs(["Trading Basics", "Technical Analysis", "Risk Management"])
    
    # Trading Basics tab
    with tabs[0]:
        st.markdown(
            """
            <h3 style="color: #F0B90B;">Cryptocurrency Trading Basics</h3>
            
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: #F0B90B;">What is Spot Trading?</h4>
                <p>Spot trading refers to the buying and selling of cryptocurrencies at the current market price with immediate delivery. Unlike futures or margin trading, spot trading doesn't involve leverage or borrowing.</p>
                <p>Key characteristics of spot trading:</p>
                <ul>
                    <li>You own the actual cryptocurrency</li>
                    <li>No expiration date on positions</li>
                    <li>Lower risk compared to leveraged trading</li>
                    <li>Suitable for beginners and long-term investors</li>
                </ul>
            </div>
            
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: #F0B90B;">Understanding Trading Pairs</h4>
                <p>Cryptocurrency trading pairs represent the exchange rate between two different cryptocurrencies or between a cryptocurrency and a fiat currency.</p>
                <p>For example:</p>
                <ul>
                    <li><strong>BTC/USDT</strong>: Bitcoin priced in Tether (a stablecoin pegged to USD)</li>
                    <li><strong>ETH/BTC</strong>: Ethereum priced in Bitcoin</li>
                    <li><strong>BNB/USDT</strong>: Binance Coin priced in Tether</li>
                </ul>
                <p>The first currency in the pair is the base currency (what you're buying or selling), and the second is the quote currency (what you're using to buy or sell).</p>
            </div>
            
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px;">
                <h4 style="color: #F0B90B;">Order Types</h4>
                <p>Different order types allow you to execute trades in various ways:</p>
                <ul>
                    <li><strong>Market Order</strong>: Buy or sell immediately at the current market price</li>
                    <li><strong>Limit Order</strong>: Buy or sell at a specified price or better</li>
                    <li><strong>Stop-Limit Order</strong>: Combines stop orders and limit orders to buy or sell when the price reaches a specified level</li>
                    <li><strong>OCO (One Cancels the Other)</strong>: Combines a limit order with a stop-limit order</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Technical Analysis tab
    with tabs[1]:
        st.markdown(
            """
            <h3 style="color: #F0B90B;">Technical Analysis Indicators</h3>
            
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: #F0B90B;">Moving Averages</h4>
                <p>Moving averages smooth out price data to create a single flowing line, making it easier to identify the direction of the trend.</p>
                <ul>
                    <li><strong>Simple Moving Average (SMA)</strong>: Average of closing prices over a specific period</li>
                    <li><strong>Exponential Moving Average (EMA)</strong>: Gives more weight to recent prices</li>
                </ul>
                <p><strong>How to use:</strong> When a short-term MA crosses above a long-term MA, it's often considered a bullish signal (golden cross). When a short-term MA crosses below a long-term MA, it's often considered a bearish signal (death cross).</p>
            </div>
            
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: #F0B90B;">Relative Strength Index (RSI)</h4>
                <p>RSI is a momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100.</p>
                <ul>
                    <li>RSI above 70 is considered overbought (potential sell signal)</li>
                    <li>RSI below 30 is considered oversold (potential buy signal)</li>
                </ul>
                <p><strong>How to use:</strong> Look for divergences between RSI and price, which can signal potential reversals. For example, if price makes a new high but RSI doesn't, it might indicate weakening momentum.</p>
            </div>
            
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: #F0B90B;">MACD (Moving Average Convergence Divergence)</h4>
                <p>MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price.</p>
                <ul>
                    <li>MACD Line: Difference between 12-period and 26-period EMAs</li>
                    <li>Signal Line: 9-period EMA of the MACD Line</li>
                    <li>Histogram: Difference between MACD Line and Signal Line</li>
                </ul>
                <p><strong>How to use:</strong> When the MACD Line crosses above the Signal Line, it's a bullish signal. When it crosses below, it's a bearish signal. The histogram helps visualize the strength of the trend.</p>
            </div>
            
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px;">
                <h4 style="color: #F0B90B;">Support and Resistance</h4>
                <p>Support and resistance levels are price points where a cryptocurrency has historically had difficulty falling below (support) or rising above (resistance).</p>
                <p><strong>How to use:</strong></p>
                <ul>
                    <li>Buy near support levels when the overall trend is upward</li>
                    <li>Sell near resistance levels when the overall trend is downward</li>
                    <li>When a support or resistance level is broken, it often becomes the opposite (former support becomes resistance and vice versa)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Risk Management tab
    with tabs[2]:
        st.markdown(
            """
            <h3 style="color: #F0B90B;">Risk Management Strategies</h3>
            
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: #F0B90B;">Position Sizing</h4>
                <p>Position sizing refers to determining how much of your capital to allocate to a single trade.</p>
                <p><strong>Guidelines:</strong></p>
                <ul>
                    <li>Never risk more than 1-2% of your total trading capital on a single trade</li>
                    <li>Adjust position size based on volatility (smaller positions for higher volatility)</li>
                    <li>Consider your risk-to-reward ratio when determining position size</li>
                </ul>
            </div>
            
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: #F0B90B;">Stop-Loss Orders</h4>
                <p>A stop-loss order is designed to limit your loss on a position by automatically selling when the price reaches a predetermined level.</p>
                <p><strong>Best practices:</strong></p>
                <ul>
                    <li>Always use stop-loss orders for every trade</li>
                    <li>Place stop-losses at logical levels (below support for buys, above resistance for sells)</li>
                    <li>Avoid placing stop-losses at obvious levels where many traders might place them</li>
                    <li>Consider using trailing stop-losses to lock in profits as the price moves in your favor</li>
                </ul>
            </div>
            
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: #F0B90B;">Risk-to-Reward Ratio</h4>
                <p>The risk-to-reward ratio compares the potential loss (risk) to the potential profit (reward) of a trade.</p>
                <p><strong>Example:</strong> If you buy Bitcoin at $50,000 with a stop-loss at $48,000 and a target of $55,000, your risk is $2,000 and your potential reward is $5,000, giving a risk-to-reward ratio of 1:2.5.</p>
                <p><strong>Guidelines:</strong></p>
                <ul>
                    <li>Aim for a minimum risk-to-reward ratio of 1:2 (potential profit twice the potential loss)</li>
                    <li>Higher risk-to-reward ratios allow for a lower win rate while still being profitable</li>
                    <li>Avoid trades with risk-to-reward ratios below 1:1</li>
                </ul>
            </div>
            
            <div style="background-color: #474D57; padding: 20px; border-radius: 10px;">
                <h4 style="color: #F0B90B;">Diversification</h4>
                <p>Diversification involves spreading your investments across different cryptocurrencies to reduce risk.</p>
                <p><strong>Strategies:</strong></p>
                <ul>
                    <li>Invest in cryptocurrencies from different sectors (DeFi, NFT, Layer 1, etc.)</li>
                    <li>Include both established cryptocurrencies (BTC, ETH) and promising newer projects</li>
                    <li>Consider correlation between cryptocurrencies (avoid those that move together)</li>
                    <li>Rebalance your portfolio periodically to maintain your desired allocation</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

# Function to display dashboard
def display_dashboard():
    # Display header
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0;">
            <h1 style="color: #F0B90B; margin: 0;">Binance Spot Trading Assistant</h1>
            <p style="color: white; margin: 0;">Last Updated: {}</p>
        </div>
        <hr style="border-color: #474D57; margin-bottom: 20px;">
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        unsafe_allow_html=True
    )
    
    # Initialize or get data manager
    if not st.session_state.data_manager:
        with st.spinner("Initializing data manager..."):
            st.session_state.data_manager = get_data_manager()
            st.session_state.last_refresh_time = datetime.now()
    
    # Add refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh Data"):
            with st.spinner("Refreshing data..."):
                success = st.session_state.data_manager.refresh_data()
                if success:
                    st.session_state.last_refresh_time = datetime.now()
                    st.success("Data refreshed successfully!")
                else:
                    st.error("Failed to refresh data. Using cached data.")
    
    # Display last refresh time
    with col1:
        if st.session_state.last_refresh_time:
            st.markdown(
                f"""
                <div style="text-align: right;">
                    <p style="color: #888; margin: 0; font-size: 14px;">Last refreshed: {st.session_state.last_refresh_time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Get data
    data = st.session_state.data_manager.get_data()
    
    # Create tabs for different sections
    tabs = st.tabs(["Overview", "Buy Recommendations", "Sell Recommendations", "Hold Recommendations", "Market Analysis", "Educational Resources"])
    
    # Overview tab
    with tabs[0]:
        display_overview(data)
    
    # Buy Recommendations tab
    with tabs[1]:
        display_buy_recommendations(data)
    
    # Sell Recommendations tab
    with tabs[2]:
        display_sell_recommendations(data)
    
    # Hold Recommendations tab
    with tabs[3]:
        display_hold_recommendations(data)
    
    # Market Analysis tab
    with tabs[4]:
        display_market_analysis(data)
    
    # Educational Resources tab
    with tabs[5]:
        display_educational_resources()

# Main function
def main():
    # Add custom CSS
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0B0E11;
            color: white;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #474D57;
            border-radius: 4px;
            color: white;
            padding: 10px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #F0B90B;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.image("https://cryptologos.cc/logos/binance-coin-bnb-logo.png", width=100)
        st.markdown("<h2 style='color: #F0B90B;'>Navigation</h2>", unsafe_allow_html=True)
        
        # Navigation options
        page = st.radio(
            "Select a page",
            ["Dashboard", "Welcome"],
            label_visibility="collapsed"
        )
        
        st.markdown("<hr style='border-color: #474D57;'>", unsafe_allow_html=True)
        
        # About section
        st.markdown("<h3 style='color: #F0B90B;'>About</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            This Binance Spot Trading Assistant provides real-time cryptocurrency analysis and trading recommendations.
            
            Data is sourced from Yahoo Finance API and updated regularly to provide the most current market insights.
            
            **Note:** This tool is for informational purposes only and does not constitute financial advice.
            """,
            unsafe_allow_html=True
        )
    
    # Display selected page
    if page == "Dashboard":
        display_dashboard()
    else:
        display_welcome()

# Run the app
if __name__ == "__main__":
    main()
