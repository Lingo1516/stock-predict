import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import ta
from datetime import datetime, timedelta
import time

@st.cache_data
def predict_next_5(stock, days, decay_factor):
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
            time.sleep(2)
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

    # 填充外部指數
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

    # 加入滯後特徵（例如 `Prev_Close`）來捕捉時間序列的影響
    for i in range(1, 6):
        df[f'Prev_Close_Lag{i}'] = close.shift(i)

    # 加入成交量特徵來捕捉波動性
    df['Volume'] = df['Volume'].shift(1)  # 使用前一天的成交量

    feats = ['Prev_Close', 'MA10', 'MA20', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'TWII_Close', 'SP500_Close'] + [f'Prev_Close_Lag{i}' for i in range(1, 6)]
    
    # 檢查特徵
    missing_feats = [f for f in feats if f not in df.columns]
    if missing_feats:
        st.error(f"缺少的特徵: {missing_feats}")
        return None, None, None

    df = df.dropna()
    if len(df) < 30:
        st.error(f"資料不足，僅有 {len(df)} 行數據")
        return None, None, None

    # 定義 last 變量
    last = float(close.iloc[-1])

    # 標準化特徵
    df_standardized = (df[feats] - df[feats].mean()) / df[feats].std()

    # 計算時間權重
    dates = df.index
    time_diffs = [(end - date).days for date in dates]
    time_weights = np.array([np.exp(-decay_factor * diff) for diff in time_diffs])
    time_weights = time_weights / np.sum(time_weights)

    # 準備訓練數據
    X = df_standardized[feats].values
    y = close.values
    X_latest = df_standardized[feats].iloc[-1:].values

    # 刪除包含 NaN 的行以保證 X 和 y 長度一致
    df_cleaned = df.dropna(subset=['Close', 'Prev_Close', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 'Volume', 'TWII_Close', 'SP500_Close'])
    X_cleaned = df_cleaned[feats].values
    y_cleaned = df_cleaned['Close'].values

    # 檢查 X 和 y 長度是否一致
    if len(X_cleaned) != len(y_cleaned):
        st.error(f"特徵矩陣 X 和目標變數 y 的長度不一致，X 長度: {len(X_cleaned)}, y 長度: {len(y_cleaned)}")
        return None, None, None

    # 訓練隨機森林模型
    X_train, X_val, y_train, y_val = train_test_split(X_cleaned, y_cleaned, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=1000, max_depth=20, random_state=42)  # 增加樹的數量和深度
    model.fit(X_train, y_train, sample_weight=time_weights[:len(X_train)])

    # 預測未來 5 天
    preds = {}
    for d in range(1, 6):
        pred = model.predict(X_latest)[0]

        # 去掉預測範圍限制，讓股價能夠自由波動
        preds[f'T+{d}'] = float(pred)

    dates = [(end + pd.offsets.BDay(d)).date() for d in range(1, 6)]

    # 驗證模型表現
    y_pred_val = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)
    st.write(f"模型的均方誤差 (MSE): {mse:.2f}")

    return last, dict(zip(dates, preds.values())), preds

def get_trade_advice(last, preds):
    price_changes = [preds[f'T+{d}'] - last for d in range(1, 6)]
    avg_change = np.mean(price_changes)
    return "買" if avg_change > 0 else "賣"

# Streamlit 介面
st.title("📈 5 日股價預測")
code = st.text_input("股票代號", "3714.TW")

# 選擇模式
mode = st.selectbox("選擇模式", ["中期模式", "短期模式", "長期模式"])
if mode == "短期模式":
    days, decay_factor = 100, 0.008
elif mode == "長期模式":
    days, decay_factor = 400, 0.002
else:  # 中期模式
    days, decay_factor = 200, 0.005

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
