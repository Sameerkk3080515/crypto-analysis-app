"""
News data collection module for the Cryptocurrency Analysis Bot.

This module handles fetching cryptocurrency news and social media data from various sources,
including Crypto News API and Twitter API.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import requests
import pandas as pd

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

class NewsDataCollector:
    """Collects cryptocurrency news and social media data from various sources."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the news data collector.
        
        Args:
            config: Configuration object (creates a new one if None)
        """
        self.config = config or Config()
        self.cryptocurrencies = self.config.get_cryptocurrencies()
        
        # For demonstration purposes, we'll use a placeholder API key
        # In a real implementation, this would be stored securely in the config
        self.crypto_news_api_key = "YOUR_CRYPTO_NEWS_API_KEY"
        
    def get_crypto_news(self, symbols: Optional[List[str]] = None, 
                       items_limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get cryptocurrency news from Crypto News API.
        
        Args:
            symbols: List of cryptocurrency symbols to get news for (uses all tracked if None)
            items_limit: Maximum number of news items to retrieve
            
        Returns:
            List of news items with metadata
        """
        if not self.config.get('news_data.crypto_news_api_enabled', True):
            print("Crypto News API is disabled in configuration")
            return []
            
        symbols = symbols or self.cryptocurrencies
        tickers = ','.join(symbols)
        
        # Crypto News API endpoint
        url = "https://cryptonews-api.com/api/v1"
        
        params = {
            'tickers': tickers,
            'items': min(items_limit, 50),  # API limit
            'token': self.crypto_news_api_key
        }
        
        try:
            # In a real implementation, this would make an actual API call
            # For demonstration, we'll simulate a response
            # response = requests.get(url, params=params)
            # response.raise_for_status()
            # news_data = response.json()
            
            # Simulated response
            news_data = self._simulate_crypto_news_response(symbols, items_limit)
            
            return news_data
            
        except Exception as e:
            print(f"Error fetching news from Crypto News API: {e}")
            return []
    
    def _simulate_crypto_news_response(self, symbols: List[str], 
                                      items_limit: int) -> List[Dict[str, Any]]:
        """
        Simulate a response from Crypto News API for demonstration purposes.
        
        Args:
            symbols: List of cryptocurrency symbols
            items_limit: Maximum number of news items
            
        Returns:
            Simulated news data
        """
        current_time = datetime.now()
        
        # Create a few sample news items for each symbol
        news_items = []
        
        for symbol in symbols:
            # Generate 1-3 news items per symbol
            num_items = min(3, items_limit // len(symbols))
            
            for i in range(num_items):
                # Alternate between positive, negative, and neutral sentiment
                sentiment = ["positive", "negative", "neutral"][i % 3]
                
                # Create timestamp within the last 24 hours
                hours_ago = i * 8
                timestamp = (current_time - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
                
                # Generate news item
                news_item = {
                    "news_url": f"https://example.com/crypto/{symbol.lower()}/article{i}",
                    "image_url": f"https://example.com/images/{symbol.lower()}{i}.jpg",
                    "title": f"{symbol} {self._get_title_by_sentiment(sentiment, symbol)}",
                    "text": f"{self._get_text_by_sentiment(sentiment, symbol)}",
                    "source_name": ["CryptoNews", "CoinDesk", "Decrypt"][i % 3],
                    "date": timestamp,
                    "topics": self._get_topics_by_symbol(symbol),
                    "sentiment": sentiment,
                    "type": "article",
                    "tickers": [symbol]
                }
                
                news_items.append(news_item)
        
        return news_items[:items_limit]
    
    def _get_title_by_sentiment(self, sentiment: str, symbol: str) -> str:
        """Generate a title based on sentiment and symbol."""
        if sentiment == "positive":
            return f"Shows Strong Bullish Signals as Adoption Increases"
        elif sentiment == "negative":
            return f"Faces Selling Pressure Amid Market Uncertainty"
        else:
            return f"Stabilizes as Market Participants Await Further Direction"
    
    def _get_text_by_sentiment(self, sentiment: str, symbol: str) -> str:
        """Generate news text based on sentiment and symbol."""
        if sentiment == "positive":
            return (f"{symbol} has been showing strong bullish signals in recent trading sessions "
                   f"as institutional adoption continues to increase. Analysts predict further "
                   f"upside potential as market sentiment improves.")
        elif sentiment == "negative":
            return (f"{symbol} is experiencing selling pressure amid broader market uncertainty. "
                   f"Technical indicators suggest a potential downward trend if support levels "
                   f"don't hold in the coming days.")
        else:
            return (f"{symbol} has stabilized in a tight trading range as market participants "
                   f"await further direction. Trading volume has decreased, indicating a period "
                   f"of consolidation before the next significant move.")
    
    def _get_topics_by_symbol(self, symbol: str) -> List[str]:
        """Generate relevant topics based on the cryptocurrency symbol."""
        common_topics = ["cryptocurrency", "blockchain", "trading"]
        
        symbol_specific = {
            "BTC": ["bitcoin", "mining", "halving"],
            "ETH": ["ethereum", "smart contracts", "defi"],
            "BNB": ["binance", "exchange", "bnb chain"],
            "SOL": ["solana", "high performance", "dapps"],
            "XRP": ["ripple", "payments", "cross-border"],
            "ADA": ["cardano", "proof of stake", "smart contracts"],
            "DOGE": ["dogecoin", "meme coin", "elon musk"],
            "DOT": ["polkadot", "interoperability", "parachains"],
            "AVAX": ["avalanche", "defi", "scalability"],
            "MATIC": ["polygon", "layer 2", "scaling solution"]
        }
        
        return common_topics + (symbol_specific.get(symbol, []))
    
    def get_twitter_data(self, query: Optional[str] = None, 
                        count: int = 20) -> List[Dict[str, Any]]:
        """
        Get cryptocurrency-related tweets from Twitter API.
        
        Args:
            query: Search query (uses cryptocurrency symbols if None)
            count: Maximum number of tweets to retrieve
            
        Returns:
            List of tweets with metadata
        """
        if not self.config.get('news_data.twitter_api_enabled', True):
            print("Twitter API is disabled in configuration")
            return []
            
        if query is None:
            # Create a search query using cryptocurrency symbols
            query = " OR ".join([f"#{crypto}" for crypto in self.cryptocurrencies])
        
        try:
            # In a real implementation, this would use the Twitter API from the datasource module
            # For demonstration, we'll simulate a response
            tweets = self._simulate_twitter_response(query, count)
            
            return tweets
            
        except Exception as e:
            print(f"Error fetching data from Twitter API: {e}")
            return []
    
    def _simulate_twitter_response(self, query: str, count: int) -> List[Dict[str, Any]]:
        """
        Simulate a response from Twitter API for demonstration purposes.
        
        Args:
            query: Search query
            count: Maximum number of tweets
            
        Returns:
            Simulated tweet data
        """
        current_time = datetime.now()
        
        # Extract cryptocurrency symbols from the query
        symbols = []
        for crypto in self.cryptocurrencies:
            if f"#{crypto}" in query:
                symbols.append(crypto)
        
        if not symbols:
            symbols = self.cryptocurrencies[:3]  # Use first 3 cryptocurrencies
        
        # Create sample tweets
        tweets = []
        
        for i in range(min(count, 20)):
            # Select a symbol
            symbol = symbols[i % len(symbols)]
            
            # Alternate between positive, negative, and neutral sentiment
            sentiment_type = i % 3
            
            # Create timestamp within the last 12 hours
            minutes_ago = i * 30
            timestamp = (current_time - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S")
            
            # Generate tweet content based on sentiment
            if sentiment_type == 0:  # Positive
                content = f"Really bullish on #{symbol} right now! The technical indicators are looking great and adoption is increasing. 🚀 #crypto #bullish"
                sentiment = "positive"
            elif sentiment_type == 1:  # Negative
                content = f"Not feeling good about #{symbol} at these levels. The chart looks bearish and volume is dropping. Might be time to take profits. #crypto #bearish"
                sentiment = "negative"
            else:  # Neutral
                content = f"Watching #{symbol} closely at these levels. Could go either way depending on market conditions. Stay vigilant. #crypto #trading"
                sentiment = "neutral"
            
            # Create tweet object
            tweet = {
                "id": f"tweet_{i}_{int(time.time())}",
                "created_at": timestamp,
                "text": content,
                "user": {
                    "screen_name": f"crypto_trader_{i}",
                    "followers_count": 1000 + (i * 500),
                    "verified": i % 5 == 0  # Every 5th user is verified
                },
                "retweet_count": i * 3,
                "favorite_count": i * 7,
                "entities": {
                    "hashtags": [{"text": symbol}, {"text": "crypto"}],
                    "symbols": [{"text": symbol}]
                },
                "sentiment": sentiment
            }
            
            tweets.append(tweet)
        
        return tweets
    
    def analyze_sentiment(self, news_items: List[Dict[str, Any]], 
                         tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment from news and social media data.
        
        Args:
            news_items: List of news items
            tweets: List of tweets
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Initialize sentiment counters for each cryptocurrency
        sentiment_data = {crypto: {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "total": 0,
            "score": 0.0
        } for crypto in self.cryptocurrencies}
        
        # Process news items
        for item in news_items:
            # Get tickers mentioned in the news item
            tickers = item.get('tickers', [])
            sentiment = item.get('sentiment', 'neutral')
            
            for ticker in tickers:
                if ticker in sentiment_data:
                    sentiment_data[ticker][sentiment] += 1
                    sentiment_data[ticker]['total'] += 1
        
        # Process tweets
        for tweet in tweets:
            # Extract symbols from hashtags and cashtags
            symbols = []
            
            # Check hashtags
            for hashtag in tweet.get('entities', {}).get('hashtags', []):
                tag = hashtag.get('text', '').upper()
                if tag in self.cryptocurrencies:
                    symbols.append(tag)
            
            # Check symbols/cashtags
            for symbol in tweet.get('entities', {}).get('symbols', []):
                tag = symbol.get('text', '').upper()
                if tag in self.cryptocurrencies:
                    symbols.append(tag)
            
            # If no specific cryptocurrency is mentioned, skip
            if not symbols:
                continue
            
            # Get sentiment
            sentiment = tweet.get('sentiment', 'neutral')
            
            # Update sentiment data for each mentioned cryptocurrency
            for symbol in symbols:
                if symbol in sentiment_data:
                    sentiment_data[symbol][sentiment] += 1
                    sentiment_data[symbol]['total'] += 1
        
        # Calculate sentiment scores (range from -1 to 1)
        for crypto, data in sentiment_data.items():
            if data['total'] > 0:
                data['score'] = (data['positive'] - data['negative']) / data['total']
        
        return sentiment_data
    
    def get_news_data_for_analysis(self) -> Dict[str, Any]:
        """
        Collect all news and social media data needed for analysis.
        
        Returns:
            Dictionary with comprehensive news and sentiment data
        """
        # Get news from Crypto News API
        news_limit = self.config.get('news_data.news_items_limit', 20)
        news_items = self.get_crypto_news(items_limit=news_limit)
        
        # Get data from Twitter API
        tweets = self.get_twitter_data(count=news_limit)
        
        # Analyze sentiment
        sentiment_data = self.analyze_sentiment(news_items, tweets)
        
        # Compile all data
        news_data = {
            'timestamp': datetime.now().isoformat(),
            'news_items': news_items,
            'tweets': tweets,
            'sentiment_analysis': sentiment_data
        }
        
        return news_data
    
    def save_news_data(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save news data to a JSON file.
        
        Args:
            data: News data to save
            filename: Output filename (generates a timestamped name if None)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"news_data_{timestamp}.json"
        
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        
        # Create data directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path


# Example usage
if __name__ == "__main__":
    collector = NewsDataCollector()
    
    # Get news data
    news_data = collector.get_news_data_for_analysis()
    
    # Print sentiment analysis
    print("Sentiment Analysis:")
    for crypto, data in news_data['sentiment_analysis'].items():
        if data['total'] > 0:
            print(f"{crypto}: Score={data['score']:.2f} (Positive={data['positive']}, "
                 f"Negative={data['negative']}, Neutral={data['neutral']}, Total={data['total']})")
    
    # Save news data
    output_path = collector.save_news_data(news_data)
    print(f"\nSaved news data to {output_path}")
