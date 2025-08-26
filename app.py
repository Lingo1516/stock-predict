import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import ta
from datetime import datetime, timedelta

@st.cache_data
def predict_next_5(stock):
    end   = datetime.today().date()
    start = end - timedelta(days=250)

    df   = yf.download(stock, start=start, end=end + timedelta(days=1), interval="1d", auto_adjust=True)
    twii = yf.download("^TWII", start=start, end=end + timedelta(days=1), interval="1d", auto_adjust=True)
    sp   = yf.download("^GSPC", start=start, end=end + timedelta(days=1), interval="1d", auto_adjust=True)

    if df.empty:
        return None, None

    df['TWII_Close'] = twii['Close'].reindex(df.index).ffill()
    df['SP500_Close'] = sp['Close'].reindex(df.index).ffill()

    close = df['Close'].squeeze()
    df['MA10']  = close.rolling(10).mean()
    df['MA20']  = close.rolling(20).mean()
    df['RSI']   = ta.momentum.rsi(close, 14)
    macd = ta.trend.MACD(close)
    df['MACD']  = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['Prev_Close']  = close.shift(1)

    feats = ['Prev_Close','MA10','MA20','Volume','RSI','MACD','MACD_Signal','TWII_Close','SP500_Close']
    df_clean = df.dropna()
    if len(df_clean) < 30:
        return None, None

    X_latest = df_clean[feats].iloc[-1:].values
    preds = {}
    for d in range(1, 6):
        y = close.shift(-d).dropna()
        model = LinearRegression().fit(df_clean[feats][:len(y)], y)
        preds[f'T+{d}'] = float(model.predict(X_latest)[0])

    last = float(close.iloc[-1])
    dates = [(end + pd.tseries.offsets.BusinessDay(i)).date() for i in range(1, 6)]
    return last, dict(zip(dates, preds.values()))

st.set_page_config(page_title="5日股價預測", page_icon="📈")
st.title("📈 5 日股價預測")
code = st.text_input("輸入股票代號（例：3714.TW 或 2330.TW）", "3714.TW")
if st.button("預測"):
    last, forecast = predict_next_5(code.strip())
    if last is None:
        st.error("❌ 無法下載資料，請檢查代號或網路。")
    else:
        st.success(f"最後收盤：{last:.2f}")
        for d, p in forecast.items():
            st.write(f"{d}：{p:.2f}")