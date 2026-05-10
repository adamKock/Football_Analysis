import yfinance as yf
import pandas as pd

# Load S&P 500 tickers
csv = "constituents.csv"
df = pd.read_csv(csv)
symbols = df["Symbol"].tolist()


def get_200_day_sma(ticker):
    try:
        stock = yf.Ticker(ticker)

        df = stock.history(period="2y", interval="1d", auto_adjust=False)

        # Calculate 200-day SMA
        df["200_SMA"] = df["Close"].rolling(window=200).mean()
        latest_price = df["Close"].iloc[-1]
        latest_sma = df["200_SMA"].iloc[-1]

        return {
            "ticker": ticker,
            "price": latest_price,
            "200_sma": latest_sma,
            "above_sma": latest_price > latest_sma
        }

    except Exception as e:
        print(f"Error with {ticker}: {e}")
        return None


results = []

for symbol in symbols[:50]:  # limit for speed
    data = get_200_day_sma(symbol)
    if data:
        results.append(data)


result_df = pd.DataFrame(results)

print(result_df.sort_values("above_sma", ascending=False))