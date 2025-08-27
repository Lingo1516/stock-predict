import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import ta
from datetime import datetime, timedelta
import time

@st.cache_data
def predict_next_5(stock, days=400, decay_factor=0.005):
    end = pd.Timestamp(datetime.today().date())
    start = end - pd.Timedelta(days=days)

    # 嘗試下載資料，添加重試邏輯
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(stock, start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True)
            twii = yf.download("^TWII", start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True)
            sp = yf.download("^GSPC", start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True)
            if not (df.empty or twii.empty or sp.empty):
                break
        except Exception as e:
            st.warning(f"嘗試 {attempt + 1}/{max_retries} 下載失敗: {e}")
            time.sleep(2)  # 延遲 2 秒後重試
        if attempt == max_retries - 1:
            st.error(f"無法下載資料：{stock}, ^TWII, 或 ^GSPC")
            return None, None, None

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
        return None, None, None

    df = df.dropna()
    if len(df) < 30:
        st.error(f"資料不足，僅有 {len(df)} 行數據")
        return None, None, None

    # 定義特徵權重
    feature_weights = np.array([0.25, 0.15, 0.10, 0.05, 0.15, 0.10, 0.10, 0.05, 0.05])  # 對應 feats 順序

    # 計算基於日期的時間權重
    dates = df.index
    if not isinstance(dates, pd.DatetimeIndex):
        st.error("索引不是有效的日期格式")
        return None, None, None
    time_diffs = [(end - date).days for date in dates]
    time_weights = np.array([np.exp(-decay_factor * diff) for diff in time_diffs])
    time_weights = time_weights / np.sum(time_weights)  # 正規化

    # 應用特徵權重到整個數據集
    df_weighted = df[feats].copy()
    df_weighted[feats] = df_weighted[feats].multiply(feature_weights, axis=1)

    # 確保 X_latest 應用權重
    X_latest = df_weighted[feats].iloc[-1:].values

    preds = {}
    for d in range(1, 6):
        tmp = df.copy()
        tmp['y'] = close.shift(-d)
        tmp = tmp.dropna()
        
        # 應用權重到 tmp 的特徵
        tmp_weighted = tmp[feats].multiply(feature_weights, axis=1)
        X_train = tmp_weighted.values
        y_train = tmp['y'].values

        # 應用時間權重到訓練數據
        sample_weight = time_weights[:len(tmp)]
        model = LinearRegression()
        model.fit(X_train, y_train, sample_weight=sample_weight)
        
        # 預測並確保非負
        pred = model.predict(X_latest)[0]
        preds[f'T+{d}'] = max(0, pred)  # 股價不可為負

    last = float(close.iloc[-1])
    dates = [(end + pd.offsets.BDay(d)).date() for d in range(1, 6)]
    return last, dict(zip(dates, preds.values())), preds

def get_trade_advice(last, preds):
    price_changes = [preds[f'T+{d}'] - last for d in range(1, 6)]
    avg_change = np.mean(price_changes)
    return "買" if avg_change > 0 else "賣"

st.title("📈 5 日股價預測")
code = st.text_input("股票代號", "3714.TW")
days = st.slider("歷史數據天數", 100, 500, 400, step=50)
decay_factor = st.slider("時間衰減因子", 0.001, 0.01, 0.005, step=0.001)
if st.button("預測"):
    last, forecast, preds = predict_next_5(code.strip(), days, decay_factor)
    if last is None:
        st.error("無法下載資料或資料不足")
    else:
        st.success(f"最後收盤：{last:.2f}")
        for d, p in forecast.items():
            st.write(f"{d}：{p:.2f}")
        advice = get_trade_advice(last, preds)
        st.write(f"**交易建議**：{advice}")
