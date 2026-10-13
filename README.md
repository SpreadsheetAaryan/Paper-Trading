# Paper Trading App

A Flask web app for practicing stock trading with fake money. Users register, get $10,000 in virtual cash, and buy and sell real tickers at live Yahoo Finance prices. It also has a stock search page with a one-year price chart, a market news feed, and an experimental LSTM next-day price prediction.

Demo video: https://youtu.be/zd0G3uwaul4

Built in January 2025 as an early learning project. The login, registration, `apology` page and `usd` helper follow the structure of the CS50x Finance problem set; the trading on live `yfinance` prices, news, search and LSTM pieces were added on top.

## Features

- Register, log in and log out (passwords stored as Werkzeug hashes, server-side sessions)
- Buy and sell shares at the latest close from `yfinance`, with cash and holdings tracked in SQLite
- Portfolio page with current price and gain or loss per holding
- Paginated transaction history
- Stock search with one year of closing prices
- Market news from NewsAPI
- LSTM price prediction (see below)

## About the LSTM prediction

**This is a learning exercise, not a trading signal.** It was my first time building a time-series model, and it has not been backtested or evaluated against a baseline, so its output should not be used to make trading decisions.

How it works (`run.py`, `StockPredictor`):

1. Download two years of daily closing prices for the ticker with `yfinance`.
2. Scale them to [0, 1] with `MinMaxScaler`.
3. Feed the last 60 days into a Keras model: LSTM(50) -> Dropout(0.2) -> LSTM(50) -> Dropout(0.2) -> Dense(1).
4. Inverse-scale the output to get a predicted next-day closing price.

The weights are loaded from `stm.h5`. The data-preparation code is in `app/models/stock_model.py`, but the training run that produced the weights is not in this repo. Known limitations: the scaler is re-fit on each ticker at prediction time, and a single next-day point estimate says nothing about uncertainty.

## Run it locally

Requires Python 3.10 to 3.12 (for TensorFlow 2.16).

```bash
git clone https://github.com/SpreadsheetAaryan/Paper-Trading
cd Paper-Trading
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export NEWS_API_KEY=your_newsapi_key   # free key from https://newsapi.org
export API_KEY=anything                # checked at startup but not used; yfinance needs no key

python run.py
```

Open http://127.0.0.1:5555. The SQLite database (`stock_data.db`) and the `flask_session/` folder are created on first run and are not tracked in git.

The app runs Flask's debug server on all interfaces, so only use it on a trusted local network.

## Project layout

```
run.py                     the Flask app: routes, database setup, StockPredictor
app/models/stock_model.py  data preparation for the LSTM (sequence building, scaling)
stm.h5                     trained LSTM weights
templates/                 Jinja templates
requirements.txt
```

## Stack

Python, Flask, Flask-Session, SQLite, yfinance, NewsAPI, TensorFlow/Keras, scikit-learn, HTML/CSS (Jinja templates).
