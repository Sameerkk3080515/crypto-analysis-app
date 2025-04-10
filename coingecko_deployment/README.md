
# Binance Spot Trading Assistant with Real-Time Data

This application provides cryptocurrency trading recommendations based on real-time data from CoinGecko API.

## Features

- Real-time cryptocurrency price data from CoinGecko
- Technical analysis based on actual market data
- Buy, sell, and hold recommendations with confidence scores
- Market analysis with trend indicators and correlations
- Price projections and selling point recommendations
- Educational resources for cryptocurrency trading

## Deployment

This application is designed to be deployed on Streamlit Cloud:

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your GitHub repository
4. Deploy the application

## Local Development

To run this application locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Data Sources

This application uses the CoinGecko API to retrieve real-time cryptocurrency data, replacing the previous version that used randomly generated data. CoinGecko provides free access to cryptocurrency market data with reasonable rate limits.
        