"""
Sentiment analysis module for the Cryptocurrency Analysis Bot.

This module processes news and social media data to analyze sentiment
and generate insights about market mood and trends.
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

class SentimentAnalyzer:
    """Analyzes sentiment from cryptocurrency news and social media data."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Configuration object (creates a new one if None)
        """
        self.config = config or Config()
        self.sentiment_threshold = self.config.get('news_data.sentiment_threshold', 
                                                 {'positive': 0.6, 'negative': 0.4})
    
    def analyze_news_sentiment(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment from news data.
        
        Args:
            news_data: News data dictionary from NewsDataCollector
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Extract sentiment analysis from news data
        sentiment_analysis = news_data.get('sentiment_analysis', {})
        
        # Process sentiment data
        processed_sentiment = {}
        
        for crypto, data in sentiment_analysis.items():
            if data['total'] == 0:
                continue
                
            # Calculate sentiment metrics
            sentiment_score = data['score']
            sentiment_volume = data['total']
            sentiment_direction = 'neutral'
            
            if sentiment_score >= self.sentiment_threshold['positive']:
                sentiment_direction = 'bullish'
            elif sentiment_score <= -self.sentiment_threshold['negative']:
                sentiment_direction = 'bearish'
            
            # Calculate sentiment strength (0-1)
            sentiment_strength = abs(sentiment_score)
            
            # Store processed sentiment
            processed_sentiment[crypto] = {
                'direction': sentiment_direction,
                'score': sentiment_score,
                'strength': sentiment_strength,
                'volume': sentiment_volume,
                'distribution': {
                    'positive': data['positive'] / data['total'],
                    'negative': data['negative'] / data['total'],
                    'neutral': data['neutral'] / data['total']
                }
            }
        
        return processed_sentiment
    
    def analyze_sentiment_trends(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment trends from news and social media data.
        
        Args:
            news_data: News data dictionary from NewsDataCollector
            
        Returns:
            Dictionary with sentiment trend analysis
        """
        # Extract news items and tweets
        news_items = news_data.get('news_items', [])
        tweets = news_data.get('tweets', [])
        
        # Combine all content with timestamps
        all_content = []
        
        for item in news_items:
            if 'date' in item and 'sentiment' in item and 'tickers' in item:
                all_content.append({
                    'timestamp': item['date'],
                    'sentiment': item['sentiment'],
                    'tickers': item['tickers'],
                    'source': 'news'
                })
        
        for tweet in tweets:
            if 'created_at' in tweet and 'sentiment' in tweet:
                # Extract cryptocurrency symbols from tweet
                symbols = []
                
                # Check hashtags
                for hashtag in tweet.get('entities', {}).get('hashtags', []):
                    tag = hashtag.get('text', '').upper()
                    if tag in self.config.get_cryptocurrencies():
                        symbols.append(tag)
                
                # Check symbols/cashtags
                for symbol in tweet.get('entities', {}).get('symbols', []):
                    tag = symbol.get('text', '').upper()
                    if tag in self.config.get_cryptocurrencies():
                        symbols.append(tag)
                
                if symbols:
                    all_content.append({
                        'timestamp': tweet['created_at'],
                        'sentiment': tweet['sentiment'],
                        'tickers': symbols,
                        'source': 'twitter'
                    })
        
        # Convert to DataFrame for easier analysis
        if not all_content:
            return {}
            
        df = pd.DataFrame(all_content)
        
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Initialize trend analysis
        trend_analysis = {}
        
        # Analyze trends for each cryptocurrency
        for crypto in self.config.get_cryptocurrencies():
            # Filter content for this cryptocurrency
            crypto_df = df[df['tickers'].apply(lambda x: crypto in x)]
            
            if len(crypto_df) < 5:  # Need at least 5 data points for trend analysis
                continue
            
            # Convert sentiment to numeric values
            sentiment_values = []
            for sentiment in crypto_df['sentiment']:
                if sentiment == 'positive':
                    sentiment_values.append(1.0)
                elif sentiment == 'negative':
                    sentiment_values.append(-1.0)
                else:
                    sentiment_values.append(0.0)
            
            crypto_df['sentiment_value'] = sentiment_values
            
            # Calculate rolling sentiment
            window_size = min(5, len(crypto_df))
            crypto_df['rolling_sentiment'] = crypto_df['sentiment_value'].rolling(window=window_size).mean()
            
            # Determine sentiment trend
            if len(crypto_df) >= 2 * window_size:
                recent_sentiment = crypto_df['rolling_sentiment'].iloc[-window_size:].mean()
                previous_sentiment = crypto_df['rolling_sentiment'].iloc[-2*window_size:-window_size].mean()
                
                sentiment_change = recent_sentiment - previous_sentiment
                
                trend_direction = 'stable'
                if sentiment_change > 0.2:
                    trend_direction = 'improving'
                elif sentiment_change < -0.2:
                    trend_direction = 'deteriorating'
                
                trend_analysis[crypto] = {
                    'direction': trend_direction,
                    'change': sentiment_change,
                    'recent_sentiment': recent_sentiment,
                    'previous_sentiment': previous_sentiment,
                    'data_points': len(crypto_df)
                }
        
        return trend_analysis
    
    def identify_key_topics(self, news_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Identify key topics for each cryptocurrency from news data.
        
        Args:
            news_data: News data dictionary from NewsDataCollector
            
        Returns:
            Dictionary mapping cryptocurrencies to lists of key topics
        """
        # Extract news items
        news_items = news_data.get('news_items', [])
        
        # Initialize topic counter for each cryptocurrency
        topic_counter = {crypto: {} for crypto in self.config.get_cryptocurrencies()}
        
        # Count topics for each cryptocurrency
        for item in news_items:
            if 'topics' in item and 'tickers' in item:
                topics = item['topics']
                tickers = item['tickers']
                
                for ticker in tickers:
                    if ticker in topic_counter:
                        for topic in topics:
                            if topic in topic_counter[ticker]:
                                topic_counter[ticker][topic] += 1
                            else:
                                topic_counter[ticker][topic] = 1
        
        # Extract top topics for each cryptocurrency
        key_topics = {}
        
        for crypto, topics in topic_counter.items():
            if not topics:
                continue
                
            # Sort topics by frequency
            sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 5 topics
            key_topics[crypto] = [topic for topic, count in sorted_topics[:5]]
        
        return key_topics
    
    def analyze_news_data(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze news and social media data for all tracked cryptocurrencies.
        
        Args:
            news_data: News data dictionary from NewsDataCollector
            
        Returns:
            Dictionary with sentiment analysis results for each cryptocurrency
        """
        analysis_results = {
            'timestamp': news_data.get('timestamp', datetime.now().isoformat()),
            'sentiment': self.analyze_news_sentiment(news_data),
            'trends': self.analyze_sentiment_trends(news_data),
            'key_topics': self.identify_key_topics(news_data)
        }
        
        return analysis_results
    
    def save_analysis_results(self, results: Dict[str, Any], 
                             filename: Optional[str] = None) -> str:
        """
        Save sentiment analysis results to a JSON file.
        
        Args:
            results: Analysis results to save
            filename: Output filename (generates a timestamped name if None)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sentiment_analysis_{timestamp}.json"
        
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
    from data_collection.news_data import NewsDataCollector
    
    # Collect news data
    news_collector = NewsDataCollector()
    news_data = news_collector.get_news_data_for_analysis()
    
    # Analyze news data
    analyzer = SentimentAnalyzer()
    analysis_results = analyzer.analyze_news_data(news_data)
    
    # Print sentiment for each cryptocurrency
    print("=== Sentiment Analysis ===")
    for crypto, data in analysis_results['sentiment'].items():
        print(f"\n{crypto}:")
        print(f"Direction: {data['direction']}")
        print(f"Score: {data['score']:.2f}")
        print(f"Strength: {data['strength']:.2f}")
        print(f"Volume: {data['volume']} mentions")
        print(f"Distribution: Positive={data['distribution']['positive']:.2%}, "
             f"Negative={data['distribution']['negative']:.2%}, "
             f"Neutral={data['distribution']['neutral']:.2%}")
    
    # Print sentiment trends
    print("\n=== Sentiment Trends ===")
    for crypto, data in analysis_results['trends'].items():
        print(f"\n{crypto}:")
        print(f"Trend Direction: {data['direction']}")
        print(f"Sentiment Change: {data['change']:.2f}")
        print(f"Recent Sentiment: {data['recent_sentiment']:.2f}")
        print(f"Previous Sentiment: {data['previous_sentiment']:.2f}")
        print(f"Data Points: {data['data_points']}")
    
    # Print key topics
    print("\n=== Key Topics ===")
    for crypto, topics in analysis_results['key_topics'].items():
        print(f"\n{crypto}: {', '.join(topics)}")
    
    # Save analysis results
    output_path = analyzer.save_analysis_results(analysis_results)
    print(f"\nSaved sentiment analysis results to {output_path}")
