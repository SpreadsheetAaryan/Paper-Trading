from flask import Flask, jsonify, request, render_template, session, redirect, g
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tempfile import mkdtemp
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
import os
import requests
import urllib.parse
from flask_session import Session
from datetime import datetime
from flask_paginate import Pagination, get_page_parameter
from flask_sqlalchemy import SQLAlchemy
import os
import requests

import sqlite3

app = Flask(__name__, template_folder='templates')

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

if not os.environ.get("API_KEY"):
    raise RuntimeError("API_KEY not set")
if not os.environ.get("NEWS_API_KEY"):
    raise RuntimeError("NEWS_API_KEY not set")

DATABASE = 'stock_data.db'
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE, check_same_thread=False)
        g.db.row_factory = sqlite3.Row
    return g.db

def create_database():
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            hash TEXT NOT NULL,
            cash REAL DEFAULT 10000.0
        )
    ''')

    db.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            shares INTEGER NOT NULL,
            price REAL NOT NULL,
            name TEXT NOT NULL,
            time TIMESTAMP NOT NULL,
            type TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    db.execute('''
        CREATE TABLE IF NOT EXISTS purchases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            shares INTEGER NOT NULL,
            price REAL NOT NULL,
            name TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    db.commit()

@app.before_request
def initialize_database():
    if not hasattr(g, 'db_initialized'):
        create_database()
        g.db_initialized = True

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

def apology(message, code=400):
    """Render message as an apology to user."""
    def escape(s):
        """
        Escape special characters.

        https://github.com/jacebrowning/memegen#special-characters
        """
        for old, new in [("-", "--"), (" ", "-"), ("_", "__"), ("?", "~q"),
                         ("%", "~p"), ("#", "~h"), ("/", "~s"), ("\"", "''")]:
            s = s.replace(old, new)
        return s
    return render_template("apology.html", top=code, bottom=escape(message), message=message), code

def login_required(f):
    """
    Decorate routes to require login.

    https://flask.palletsprojects.com/en/1.1.x/patterns/viewdecorators/
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

def usd(value):
    """Format value as USD."""
    return f"${value:,.2f}"

custom_objects = {
    'huber_loss': tf.keras.losses.Huber(),
    'mae': tf.keras.metrics.MeanAbsoluteError()
}

class StockPredictor:
    def __init__(self, model_path='stm.h5', sequence_length=60):
        # Load the entire model
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])

        self.model.load_weights(model_path)
        
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def prepare_data(self, symbol):
        # Fetch stock data
        stock = yf.Ticker(symbol)
        data = stock.history(period='2y')['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequence for prediction
        sequence = scaled_data[-self.sequence_length:]
        sequence = np.reshape(sequence, (1, self.sequence_length, 1))
        return sequence
    
    def predict(self, symbol):
        sequence = self.prepare_data(symbol)
        prediction_scaled = self.model.predict(sequence)
        prediction = self.scaler.inverse_transform(prediction_scaled)
        return float(prediction[0][0])

@app.route('/ai', methods=['GET', 'POST'])
def ai():
    if request.method == 'GET':
        return render_template('ai.html')
    else:
        pass

@app.route('/predict', methods=['GET', 'POST'])
def predict_stock():
    if request.method == 'POST':
        symbol = request.form['symbol']
        stock = yf.Ticker(symbol)
        data = stock.history(period='2y')
        
        if data.empty:
            return jsonify({'error': 'Invalid stock symbol'}), 400
        
        predictor = StockPredictor()
        try:
            prediction = predictor.predict(symbol)
            prediction = round(prediction, 2)
            return render_template('ai.html', symbol=symbol, prediction=prediction)
        except Exception as e:
            return jsonify({'error': str(e)})
    elif request.method == 'GET':
        return render_template('predict.html')
    
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'GET':
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT shares, symbol, price, name FROM purchases WHERE user_id = ?', (session['user_id'],))
        stocks = cursor.fetchall()
        cursor.execute('SELECT cash FROM users WHERE id = ?', (session['user_id'],))
        cash = cursor.fetchone()['cash']

        prices = []
        total_vals = []
        value = cash
        cash = usd(cash)
        for stock in stocks:
            p = yf.Ticker(stock['symbol']).history(period='1d')['Close'].values[0]
            prices.append(usd(p))

            total_val = p * stock['shares']
            total_val -= stock['price']
            value += total_val
            total_vals.append(usd(total_val))
        
        zipped = zip(stocks, prices, total_vals)
        value = usd(value)
        
        return render_template('index.html', stocks=stocks, usd=usd, cash=cash, zipped=zipped, value=value)
    
@app.route('/buy', methods=['POST'])
@login_required
def buy():
    if request.method == 'POST':
        symbol = request.form['symbol']
        shares = request.form['shares']
        stock = yf.Ticker(symbol)

        if not stock:
            return apology("enter a symbol")
        if not shares:
            return apology("enter the number of shares")

        try:
            shares = int(request.form.get("shares"))
        except ValueError:
            return apology("enter a positive number")

        if shares < 1:
            return apology("enter a positive number")
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT cash FROM users WHERE id = ?', (session['user_id'],))
        cash = cursor.fetchone()['cash']
        price = stock.history(period='1d')['Close'].values[0]
        total = price * int(shares)
        if cash < total:
            return apology('Not enough cash')
        else:
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('UPDATE users SET cash = ? WHERE id = ?', (cash - total, session['user_id']))
            cursor.execute('SELECT shares FROM purchases WHERE user_id = ? AND symbol = ?', (session['user_id'], symbol))
            current_shares = cursor.fetchone()
            if current_shares:
                cursor.execute('SELECT price FROM purchases WHERE user_id = ? AND symbol = ?', (session['user_id'], symbol))
                p = cursor.fetchone()['price']
                cursor.execute('UPDATE purchases SET shares = ? WHERE user_id = ? AND symbol = ?', (current_shares['shares'] + int(shares), session['user_id'], symbol))
                cursor.execute('UPDATE purchases SET price = ? WHERE user_id = ? AND symbol = ?', (p + total, session['user_id'], symbol))
            else:
                cursor.execute('INSERT INTO purchases (user_id, symbol, shares, price, name) VALUES (?, ?, ?, ?, ?)', (session['user_id'], symbol, shares, total, stock.info['shortName']))
            cursor.execute('INSERT INTO history (user_id, symbol, shares, price, name, time, type) VALUES (?, ?, ?, ?, ?, ?, ?)', (session['user_id'], symbol, shares, price, stock.info['shortName'], current_timestamp, 'BUY'))
            db.commit()
            return redirect('/')
        
@app.route('/sell', methods=['POST'])
@login_required
def sell():
    if request.method == 'POST':
        symbol = request.form['symbol']
        shares = request.form['shares']
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT cash FROM users WHERE id = ?', (session['user_id'],))
        cash = cursor.fetchone()['cash']
        stock = yf.Ticker(symbol)
        price = stock.history(period='1d')['Close'].values[0]
        total = price * int(shares)
        cursor.execute('UPDATE users SET cash = ? WHERE id = ?', (cash + total, session['user_id']))
        cursor.execute('SELECT shares FROM purchases WHERE user_id = ? AND symbol = ?', (session['user_id'], symbol))
        current_shares = cursor.fetchone()['shares']
        if int(shares) == current_shares:
            cursor.execute('DELETE FROM purchases WHERE user_id = ? AND symbol = ?', (session['user_id'], symbol))
        elif int(shares) > current_shares:
            return apology('Not enough shares')
        else:
            cursor.execute('SELECT price FROM purchases WHERE user_id = ? AND symbol = ?', (session['user_id'], symbol))
            p = cursor.fetchone()['price']
            cursor.execute('UPDATE purchases SET shares = ? WHERE user_id = ? AND symbol = ?', (current_shares - int(shares), session['user_id'], symbol))
            cursor.execute('UPDATE purchases SET price = ? WHERE user_id = ? AND symbol = ?', (p - total, session['user_id'], symbol))
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('INSERT INTO history (user_id, symbol, shares, price, name, time, type) VALUES (?, ?, ?, ?, ?, ?, ?)', (session['user_id'], symbol, shares, price, stock.info['shortName'], current_timestamp, 'SELL'))
        db.commit()
        return redirect('/')

    
@app.route('/news', methods=['GET', 'POST'])
def news():
    if request.method == 'GET':
        url = f'https://newsapi.org/v2/everything?q=stocks&apiKey={NEWS_API_KEY}'
        response = requests.get(url)
        news_data = response.json()

        if news_data['status'] != 'ok':
            return apology('Failed to fetch news')

        articles = news_data['articles']
        return render_template('news.html', articles=articles)
    
@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    if request.method == 'POST':
        symbol = request.form['symbol']
        stock = yf.Ticker(symbol)
        data = stock.history(period='1y')  # Fetch 1 year of historical data

        if data.empty:
            return jsonify({'error': 'Invalid stock symbol'}), 400

        dates = data.index.strftime('%Y-%m-%d').tolist()
        prices = data['Close'].tolist()
        name = stock.info['shortName']
        price = data['Close'].iloc[-1]

        print(f"Dates: {dates}")
        print(f"Prices: {prices}")

        return render_template('search.html', symbol=symbol, name=name, price=price, dates=dates, prices=prices, usd=usd)
    else:
        return render_template('search.html')
    
@app.route('/history')
@login_required
def history():
    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 5
    start = (page - 1) * per_page
    end = start + per_page

    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM history WHERE user_id = ? ORDER BY time DESC', (session['user_id'],))
    historys = cursor.fetchall()

    total_pages = (len(historys) + per_page - 1) // per_page
    if total_pages == 0:
        total_pages = 1
    historys = historys[start:end]

    return render_template('history.html', historys=historys, usd=usd, page=page, total_pages=total_pages)

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 403)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 403)

        # Query database for username
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (request.form.get("username"),))
        rows = cursor.fetchall()

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            return apology("invalid username and/or password", 403)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")
    
@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username:
            return apology("enter a username")
        elif not password:
            return apology("enter a password")

        hash = generate_password_hash(password)
        db = get_db()
        try:
            db.execute("INSERT INTO users (username, hash) VALUES (?, ?);", (username, hash))
            db.commit()
            return redirect("/")
        except sqlite3.IntegrityError:
            return apology("username already exists")
    else:
        return render_template("login.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)


    
