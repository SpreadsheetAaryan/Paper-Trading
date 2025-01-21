# README.txt

## Project Title: Stock Trading and Prediction Platform

### Description
This project is a **web-based stock trading and prediction platform** built using Flask. It provides users with the ability to:
- Manage their stock portfolio by buying and selling stocks.
- View historical transactions and portfolio performance.
- Predict future stock prices using an LSTM-based machine learning model.
- Access the latest stock-related news articles.

The application is publicly hosted using **GitHub Pages** for documentation and user guides, while the backend runs locally or on a server.

---

### Features
- **User Authentication:** Secure registration, login, and logout functionality.
- **Stock Trading:** Buy and sell stocks with real-time price updates.
- **Portfolio Overview:** Display current holdings, including stock symbols, shares, prices, and total value.
- **Stock Price Prediction:** Predict future prices of selected stocks using an LSTM model.
- **Historical Transactions:** View a paginated history of all transactions (buy/sell).
- **Stock News:** Fetch and display the latest financial news articles.
- **Search Stocks:** Search for specific stocks to view historical price data over the past year.

---

### Technologies Used
- **Backend Framework:** Flask
- **Database:** SQLite
- **APIs:**
  - Yahoo Finance (via `yfinance` library) for stock data.
  - NewsAPI for financial news.
- **Machine Learning:**
  - TensorFlow for LSTM-based stock price prediction.
  - Scikit-learn's `MinMaxScaler` for data preprocessing.
- **Frontend:** HTML templates rendered with Flask's `render_template`.
- **Hosting:** GitHub Pages (for documentation).

---
