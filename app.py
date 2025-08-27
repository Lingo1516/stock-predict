import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import ta
from datetime import datetime, timedelta

@st.cache_data
def predict_next_5(stock):
    end = datetime.today().date()
    start = end - timedelta(days=450)  # 使用400天的歷史數據

    # 下載資料
    df = yf.download(stock, start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True)
    twii = yf.download("^TWII", start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True)
    sp = yf.download("^GSPC", start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True)

    if df.empty or twii.empty or sp.empty:
        st.error(f"無法下載資料：{stock}, ^TWII, 或 ^GSPC")
        return None, None

    # 處理多重索引
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        twii.columns = [col[0] for col in twii.columns]
        sp.columns = [col[0] for col in sp.columns]

    # 確保 close 是一維
    close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]

    df['TWII_Close'] = twii['Close'].reindex(df.index).ffill()
    df['SP500_Close'] = sp['Close'].reindex(df.index).ffill()

    # 計算技術指標
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['Prev_Close'] = close.shift(1)

    feats = ['Prev_Close', 'MA10', 'MA20', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'TWII_Close', 'SP500_Close']
    
    # 檢查特徵是否存在
    missing_feats = [f for f in feats if f not in df.columns]
    if missing_feats:
        st.error(f"缺少的特徵: {missing_feats}")
        return None, None

    df = df.dropna()
    if len(df) < 30:
        st.error(f"資料不足，僅有 {len(df)} 行數據")
        return None, None

    # 確保 X_latest 是二維數組
    X_latest = df[feats].iloc[-1:].values
    preds = {}
    for d in range(1, 6):
        tmp = df.copy()
        tmp['y'] = close.shift(-d)
        tmp = tmp.dropna()
        model = LinearRegression().fit(tmp[feats], tmp['y'])
        preds[f'T+{d}'] = float(model.predict(X_latest)[0])

    last = float(close.iloc[-1])
    dates = [(end + pd.offsets.BDay(d)).date() for d in range(1, 6)]
    return last, dict(zip(dates, preds.values()))

st.title("📈 5 日股價預測")
code = st.text_input("股票代號", "3714.TW")
if st.button("預測"):
    last, forecast = predict_next_5(code.strip())
    if last is None:
        st.error("無法下載資料或資料不足")
    else:
        st.success(f"最後收盤：{last:.2f}")
        for d, p in forecast.items():
            st.write(f"{d}：{p:.2f}")
