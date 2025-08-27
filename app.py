import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import ta
from datetime import datetime
import time
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator
from statsmodels.tsa.arima.model import ARIMA  # 引入ARIMA模型
from keras.models import Sequential
from keras.layers import LSTM, Dense  # 引入LSTM模型
from sklearn.preprocessing import MinMaxScaler  # 用於LSTM預處理

# 股票代號到中文名稱簡易對照字典，可自行擴充
stock_name_dict = {
    "2330.TW": "台灣積體電路製造股份有限公司",
    "2317.TW": "鴻海精密工業股份有限公司",
    "2412.TW": "中華電信股份有限公司",
}

@st.cache_data
def predict_next_5(stock, days, decay_factor):
    try:
        # 設定時間範圍
        end = pd.Timestamp(datetime.today().date())
        start = end - pd.Timedelta(days=days)
        max_retries = 3
        df, twii, sp = None, None, None

        # 下載資料並加強錯誤處理
        for attempt in range(max_retries):
            try:
                df = yf.download(stock, start=start, end=end + pd.Timedelta(days=1),
                                 interval="1d", auto_adjust=True, progress=False)
                twii = yf.download("^TWII", start=start, end=end + pd.Timedelta(days=1),
                                  interval="1d", auto_adjust=True, progress=False)
                sp = yf.download("^GSPC", start=start, end=end + pd.Timedelta(days=1),
                                interval="1d", auto_adjust=True, progress=False)
                if not (df.empty or twii.empty or sp.empty):
                    break
            except Exception as e:
                st.warning(f"嘗試 {attempt + 1}/{max_retries} 下載失敗: {e}")
                time.sleep(2)

            if attempt == max_retries - 1:
                st.error(f"無法下載資料：{stock}")
                return None, None, None

        # 資料量不足與缺失值處理
        if df is None or len(df) < 50:
            st.error(f"資料不足，僅有 {len(df) if df is not None else 0} 行數據")
            return None, None, None

        df = df.fillna(method='bfill').fillna(method='ffill')

        # 計算技術指標，剔除冗餘特徵
        close = df['Close'].squeeze()
        df['TWII_Close'] = twii['Close'].reindex(df.index, method='ffill').fillna(method='bfill')
        df['SP500_Close'] = sp['Close'].reindex(df.index, method='ffill').fillna(method='bfill')

        df['MA5'] = close.rolling(5, min_periods=1).mean()
        df['MA10'] = close.rolling(10, min_periods=1).mean()
        df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()

        bb_indicator = BollingerBands(close, window=20, window_dev=2)
        df['BB_High'] = bb_indicator.bollinger_hband()
        df['BB_Low'] = bb_indicator.bollinger_lband()

        adx_indicator = ADXIndicator(df['High'], df['Low'], close, window=14)
        df['ADX'] = adx_indicator.adx()

        # 融入宏觀經濟數據（此處以假數據為例，實際應引入API）
        df['GDP_Growth'] = 2.5  # 假設GDP增長為2.5%
        df['Unemployment'] = 4.0  # 假設失業率為4%

        # 新增基礎面數據（例如市盈率PE，這里以假的示例為例）
        ticker_info = yf.Ticker(stock).info
        df['PE_Ratio'] = ticker_info.get("trailingPE", 0)

        # 加強特徵選擇
        feats = ['Prev_Close', 'MA5', 'MA10', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'ADX', 'GDP_Growth', 'Unemployment', 'PE_Ratio']
        df_clean = df[feats + ['Close']].dropna()

        # 資料不足時處理
        if len(df_clean) < 30:
            st.error(f"清理後資料不足，僅有 {len(df_clean)} 行數據")
            return None, None, None

        X = df_clean[feats].values
        y = df_clean['Close'].values

        # 訓練模型：使用LSTM和ARIMA模型進行預測
        # ARIMA模型
        arima_model = ARIMA(y, order=(5, 1, 0))  # 預設ARIMA參數
        arima_model_fit = arima_model.fit()

        # LSTM模型（時間序列模型）
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # LSTM輸入格式

        lstm_model = Sequential()
        lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
        lstm_model.add(LSTM(50, return_sequences=False))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_scaled, y, epochs=10, batch_size=32)

        # 使用LSTM和ARIMA進行預測
        arima_pred = arima_model_fit.forecast(steps=5)
        lstm_pred = lstm_model.predict(X_scaled[-5:].reshape(1, 5, X_scaled.shape[2]))

        final_pred = (arima_pred + lstm_pred.flatten()) / 2  # 混合預測

        # 計算RMSE
        mse = mean_squared_error(y, arima_pred)
        rmse = np.sqrt(mse)

        st.info(f"模型驗證 - RMSE: {rmse:.2f}")
        
        predictions = {}
        for i, pred in enumerate(final_pred):
            predictions[f'T+{i + 1}'] = pred

        return y[-1], predictions, final_pred

    except Exception as e:
        st.error(f"預測過程發生錯誤: {e}")
        return None, None, None
