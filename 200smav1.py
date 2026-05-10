import yfinance as yf
import pandas as pd
import numpy as np

#We want to get the stock tickers from the csv file

#Gets S&P 500 stocks
#Pulls weekly price data
#Calculates the 200-week moving average
#Shows where price is vs that average

csv = "constituents.csv"
df = pd.read_csv(csv)
symbols=df["Symbol"].values.tolist()

def get_weekly_stock_data(ticker):
    stock_prices=[]
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y", interval="1d")
    df.index = df.index + pd.offsets.Week(weekday=4)
    df = df.tail(200)

    for count,row in df.iterrows():
        last_price = row["Close"]
        stock_prices.append(last_price)
       
   
    return stock_prices




def calculate_200_sma(stock_prices):
    sum_counter = 0.0
    for price in stock_prices:
        sum_counter += price

    sum_counter= sum_counter/len(stock_prices)

    return sum_counter


stock_prices = get_weekly_stock_data("NVDA")
price_total=calculate_200_sma(stock_prices)

print(price_total)
    






    #Step-by-Step CalculationGather Data: 
    # Obtain the closing price of the asset at the end of every week for the last 200 weeks.
    # Sum Prices: Add all 200 weekly closing prices together.
    # Divide: Divide the total sum by 200.
    # Update: As a new week ends, add the new closing price, remove the oldest price, and recalculate to keep the average "moving"






