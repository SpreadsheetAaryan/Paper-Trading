# README.txt

## Project Demo: https://youtu.be/zd0G3uwaul4?si=89-RomV_qXvx4VIf

---

## Paper Trading Application

### Description
This project is a **web-based stock trading and prediction platform** built using Flask. It combines stock market data retrieval, AI-powered stock price predictions, and a user-friendly interface for portfolio management. Users can:
- Buy and sell stocks while tracking their portfolio value.
- Predict future stock prices using a pre-trained LSTM model.
- View transaction history and access the latest financial news.

The AI component leverages an LSTM (Long Short-Term Memory) model for time-series forecasting, with `MinMaxScaler` used for data preprocessing.

---

### Features
- **User Authentication**: Secure registration, login, and logout functionality.
- **Stock Trading**: Buy and sell stocks with real-time price updates from Yahoo Finance.
- **Portfolio Management**: Track your holdings, cash balance, and overall portfolio value.
- **AI Stock Price Prediction**:
  - Uses an LSTM model trained on historical stock data for future price forecasting.
  - Preprocesses data using `MinMaxScaler` to normalize input features.
  - Predicts future prices based on the last 60 days of stock closing prices.
- **Historical Transactions**: View a paginated history of all buy/sell transactions.
- **Financial News**: Fetches the latest stock market news using NewsAPI.
- **Stock Search**: Look up historical price data for specific stocks over the past year.

---

### Installation and Setup
Follow these steps to install and set up the application:

1. **Clone the Repository**:
git clone <repository_url>
cd <repository_directory>

2. **Install Dependencies**:
Ensure you have Python 3.x installed, then run:
pip install -r requirements.txt

3. **Set Up Environment Variables**:
Create a `.env` file in the root directory with the following content:
API_KEY=<Your_Yahoo_Finance_API_Key>
NEWS_API_KEY=<Your_News_API_Key>

4. **Initialize the Database**:
The database will be automatically created when you first run the application.


Access the application in your browser at `http://127.0.0.1:5555`.

---

### How the AI Works
The AI component of this project uses an LSTM neural network model to predict future stock prices based on historical data. Below is an overview of how it works:

1. **Data Collection**:
- Stock price data is retrieved using Yahoo Finance (`yfinance` library).
- The last two years of closing prices are used for predictions.

2. **Data Preprocessing**:
- Prices are normalized to a range of [0, 1] using `MinMaxScaler` from Scikit-learn.
- A sequence of the last 60 days' closing prices is prepared as input for the model.

3. **LSTM Model Architecture**:
- Two LSTM layers with 50 units each are used to capture temporal dependencies in stock prices.
- Dropout layers are added for regularization to prevent overfitting.
- A Dense layer outputs a single predicted price.

4. **Prediction Process**:
- The pre-trained model (`stm.h5`) is loaded at runtime.
- The scaled input sequence is fed into the LSTM model to generate predictions.
- The output is inverse-transformed back to the original scale using `MinMaxScaler`.

---

### Folder Structure

project/
│
├── templates/ # HTML templates for rendering views
├── static/ # Static files (CSS, JS, images)
├── app.py # Main Flask application file
├── stm.h5 # Pre-trained LSTM model weights
├── requirements.txt # Python dependencies
├── .env # Environment variables file (not included in GitHub)
└── README.txt # Project documentation (this file)


---

### Dependencies
The project requires the following Python libraries:
- Flask (for web development)
- Flask-Session (for session management)
- SQLite (for database management)
- TensorFlow (for building and running the LSTM model)
- Scikit-learn (for data preprocessing with `MinMaxScaler`)
- yfinance (for fetching stock market data)
- requests (for API calls)

Install all dependencies using:
pip install -r requirements.txt


---

### Usage Instructions
1. Open your browser and navigate to `http://127.0.0.1:5555`.
2. Register or log in to access features such as buying/selling stocks and viewing your portfolio.
3. Use the `/predict` route to enter a stock symbol and view AI-generated predictions for its future price.
4. Access financial news via `/news` or search for specific stocks using `/search`.
5. Review your transaction history through `/history`.

---

### License
This project is licensed under [MIT License].
