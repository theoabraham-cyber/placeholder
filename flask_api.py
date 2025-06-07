from flask import Flask, request, jsonify
import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd

app = Flask(__name__)

def calculate_historical_volatility(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        raise ValueError(f"No historical data found for {ticker}")
    daily_returns = hist['Close'].pct_change().dropna()
    return daily_returns.std() * np.sqrt(252)

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0, 0.0
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        intrinsic = max(S - K, 0)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        intrinsic = max(K - S, 0)
    
    return price, price - intrinsic

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = {}
    try:
        hist = stock.history(period="1d")
        if not hist.empty:
            data["price"] = hist["Close"].iloc[-1]
        data["volatility"] = calculate_historical_volatility(ticker)
        data["beta"] = stock.info.get("beta", None)
        data["pe_ratio"] = stock.info.get("trailingPE", None)
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data: {str(e)}")

@app.route('/calculate', methods=['GET'])
def calculate():
    ticker = request.args.get('ticker', '').upper()
    try:
        strike = float(request.args.get('strike'))
        time = float(request.args.get('time'))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid strike price or time to expiration"}), 400
    
    r = 0.05  # Risk-free rate (5%)
    
    try:
        stock_data = get_stock_data(ticker)
        if not stock_data.get("price") or not stock_data.get("volatility"):
            return jsonify({"error": "Failed to fetch stock data"}), 500
        
        S = stock_data["price"]
        sigma = stock_data["volatility"]
        
        call_price, call_pnl = black_scholes(S, strike, time, r, sigma, "call")
        put_price, put_pnl = black_scholes(S, strike, time, r, sigma, "put")
        
        response = {
            "ticker": ticker,
            "stock_price": round(S, 4),
            "volatility": round(sigma, 4),
            "beta": stock_data.get("beta"),
            "pe_ratio": stock_data.get("pe_ratio"),
            "call_price": round(call_price, 4),
            "call_pnl": round(call_pnl, 4),
            "put_price": round(put_price, 4),
            "put_pnl": round(put_pnl, 4)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)