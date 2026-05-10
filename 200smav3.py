import yfinance as yf
import pandas as pd


csv = "constituents.csv"
df = pd.read_csv(csv)
symbols = df["Symbol"].tolist()


#Get Stock and its history 
#Calculate 200day sma by using rolling mean
#Return ticker price sma and if price is above or below sma
#For loop through each of the tickers in symbol csv then call the 200sma function
#Put that return value into a list 
#Then create a df from the list and print the results sorted by price above or below sma


def get_200_day_sma(ticker):
    try:
        stock=yf.Ticker(ticker)
        df = stock.history(period="2y", interval="1d")


        df["200SMA"] = df["Close"].rolling(window=200).mean()
        current_price = df["Close"].iloc[-1]
        current_sma = df["200SMA"].iloc[-1]

        return{
            "Ticker":ticker,
            "Current Price":current_price,
            "200 SMA":current_sma,
            "Is SMA Above or Belo Current Price":current_price>current_sma
        }
    except Exception as e:
        print(f"Error with {ticker}: {e}")
        return None


def backtest_sma_strategy(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y", interval="1d")

        df["200SMA"] = df["Close"].rolling(window=200).mean()
        df = df.dropna()

        df["Signal"] = df["Close"] > df["200SMA"]
        df["Signal"] = df["Signal"].shift(1)

        df["Returns"] = df["Close"].pct_change()
        df["Strategy_Returns"] = df["Returns"] * df["Signal"]

        cumulative_return = (1 + df["Strategy_Returns"]).cumprod().iloc[-1] - 1
        buy_hold_return = (1 + df["Returns"]).cumprod().iloc[-1] - 1
        

        return {
            "Ticker": ticker,
            "Strategy Return": cumulative_return,
            "Buy & Hold Return": buy_hold_return
        }

    except Exception as e:
        print(f"Error with {ticker}: {e}")
        return None

results=[]

for symbol in symbols:
    sma = backtest_sma_strategy(symbol)
    if sma:
        results.append(sma)


results_df = pd.DataFrame(results)
results_df["Outperformance"] = results_df["Strategy Return"] - results_df["Buy & Hold Return"]

results_df["Better Strategy?"] = results_df["Outperformance"] > 0

positive_sma_results_df = pd.DataFrame(results_df[results_df["Better Strategy?"] == True])
print(positive_sma_results_df.shape)






   
