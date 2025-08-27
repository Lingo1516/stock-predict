import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import ta
from datetime import datetime
import time
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator, CCIIndicator
from ta.momentum import StochasticOscillator
from ta.volume import OnBalanceVolume

# 股票代號到中文名稱簡易對照字典，可自行擴充
stock_name_dict = {
    "2330.TW": "台灣積體電路製造股份有限公司",
    "2317.TW": "鴻海精密工業股份有限公司",
    "2412.TW": "中華電信股份有限公司",
}

@st.cache_data
def predict_next_5(stock, days, decay_factor):
    """
    下載股票數據，計算技術指標，並使用隨機森林模型預測未來5天的股價。
    Args:
        stock (str): 股票代號，例如 "2330.TW"。
        days (int): 要下載的歷史天數。
        decay_factor (float): 權重衰減因子，用於強調近期數據的重要性。
    Returns:
        tuple: (當前股價, 未來預測價格字典, 預測價格列表)。
    """
    try:
        end = pd.Timestamp(datetime.today().date())
        start = end - pd.Timedelta(days=days)
        max_retries = 3
        df, twii, sp = None, None, None

        # 嘗試下載資料，最多重試3次
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
                st.error(f"無法下載資料：{stock}。請檢查股票代號或網路連線。")
                return None, None, None

        if df is None or len(df) < 50:
            st.error(f"資料不足，僅有 {len(df) if df is not None else 0} 行數據，無法進行預測。")
            return None, None, None

        # 處理資料框架欄位名稱
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = [col[0] for col in twii.columns]
        if isinstance(sp.columns, pd.MultiIndex):
            sp.columns = [col[0] for col in sp.columns]

        # 確保 'Close', 'High', 'Low' 欄位存在
        if not all(col in df.columns for col in ['Close', 'High', 'Low']):
            st.error("股票數據中缺少必要的欄位 (Close, High, Low)。")
            return None, None, None

        close = df['Close'].squeeze()
        
        # 整合市場指數
        df['TWII_Close'] = twii['Close'].reindex(df.index, method='ffill').fillna(method='bfill')
        df['SP500_Close'] = sp['Close'].reindex(df.index, method='ffill').fillna(method='bfill')

        # === 核心技術指標計算 ===
        # 移動平均線
        df['MA5'] = close.rolling(5, min_periods=1).mean()
        df['MA10'] = close.rolling(10, min_periods=1).mean()
        df['MA20'] = close.rolling(20, min_periods=1).mean()

        # 相對強弱指標 (RSI)
        df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        
        # 移動平均收斂散度 (MACD)
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # 布林帶 (Bollinger Bands)
        bb_indicator = BollingerBands(close, window=20, window_dev=2)
        df['BB_High'] = bb_indicator.bollinger_hband()
        df['BB_Low'] = bb_indicator.bollinger_lband()
        
        # 平均趨向指標 (ADX)
        adx_indicator = ADXIndicator(df['High'], df['Low'], close, window=14)
        df['ADX'] = adx_indicator.adx()
        
        # === 新增技術指標 ===
        # 隨機指標 (Stochastic Oscillator)
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=close, window=14, smooth_window=3)
        df['STOCH_K'] = stoch.stoch()
        df['STOCH_D'] = stoch.stoch_signal()

        # 商品通道指標 (CCI)
        cci = CCIIndicator(high=df['High'], low=df['Low'], close=close, window=20)
        df['CCI'] = cci.cci()

        # 能量潮 (On-Balance Volume)
        if 'Volume' in df.columns and not df['Volume'].isnull().all():
            obv = OnBalanceVolume(close, df['Volume'])
            df['OBV'] = obv.on_balance_volume()
            # 增加OBV的移動平均作為趨勢特徵
            df['OBV_MA'] = df['OBV'].rolling(10, min_periods=1).mean()
        else:
            df['OBV'] = 0
            df['OBV_MA'] = 0

        # === 其他特徵 ===
        df['Prev_Close'] = close.shift(1)
        for i in range(1, 4):
            df[f'Prev_Close_Lag{i}'] = close.shift(i)

        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(10, min_periods=1).mean()
        else:
            df['Volume_MA'] = 0

        df['Volatility'] = close.rolling(10, min_periods=1).std()

        # === 定義特徵集 ===
        feats = [
            'Prev_Close', 'MA5', 'MA10', 'MA20', 'Volume_MA', 'RSI', 'MACD',
            'MACD_Signal', 'TWII_Close', 'SP500_Close', 'Volatility', 'BB_High',
            'BB_Low', 'ADX', 'STOCH_K', 'STOCH_D', 'CCI', 'OBV', 'OBV_MA'
        ] + [f'Prev_Close_Lag{i}' for i in range(1, 4)]
        
        # 檢查是否有缺失的特徵
        missing_feats = [f for f in feats if f not in df.columns]
        if missing_feats:
            st.error(f"缺少特徵: {missing_feats}")
            return None, None, None

        df_clean = df[feats + ['Close']].dropna()
        if len(df_clean) < 30:
            st.error(f"清理後資料不足，僅有 {len(df_clean)} 行數據，無法進行預測。")
            return None, None, None

        X = df_clean[feats].values
        y = df_clean['Close'].values

        # 資料正規化
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1 # 防止除以0

        X_normalized = (X - X_mean) / X_std
        weights = np.exp(-decay_factor * np.arange(len(X))[::-1])
        weights = weights / np.sum(weights)

        # 訓練/驗證資料分割
        split_idx = int(len(X_normalized) * 0.8)
        X_train, X_val = X_normalized[:split_idx], X_normalized[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        train_weights = weights[:split_idx]

        # 建立並訓練隨機森林模型
        models = []
        model_params = [
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42},
            {'n_estimators': 80, 'max_depth': 8, 'min_samples_split': 3, 'min_samples_leaf': 1, 'random_state': 123},
            {'n_estimators': 120, 'max_depth': 12, 'min_samples_split': 7, 'min_samples_leaf': 3, 'random_state': 456}
        ]
        for params in model_params:
            rf_model = RandomForestRegressor(**params, n_jobs=-1)
            rf_model.fit(X_train, y_train, sample_weight=train_weights)
            models.append(('RF', rf_model))

        # 進行未來5天股價預測
        last_features = X_normalized[-1:].copy()
        last_close = float(y[-1])
        predictions = {}
        future_dates = []
        current_date = end
        for i in range(5):
            current_date = current_date + pd.offsets.BDay(1)
            future_dates.append(current_date.date())

        current_features = last_features.copy()
        predicted_prices = [last_close]
        max_deviation_pct = 0.10 # 最大偏離限制 ±10%

        for i, date in enumerate(future_dates):
            day_predictions = []
            for model_name, model in models:
                pred = model.predict(current_features)[0]
                variation = np.random.normal(0, pred * 0.002)
                day_predictions.append(pred + variation)

            weights_ensemble = [0.5, 0.3, 0.2]
            ensemble_pred = np.average(day_predictions, weights=weights_ensemble)
            
            historical_volatility = np.std(y[-30:]) / np.mean(y[-30:])
            volatility_adjustment = np.random.normal(0, ensemble_pred * historical_volatility * 0.05)
            final_pred = ensemble_pred + volatility_adjustment
            
            # 限制預測價格在合理範圍內
            upper_limit = last_close * (1 + max_deviation_pct)
            lower_limit = last_close * (1 - max_deviation_pct)
            final_pred = min(max(final_pred, lower_limit), upper_limit)

            predictions[date] = float(final_pred)
            predicted_prices.append(final_pred)

            # 更新特徵用於下一天的預測
            if i < 4:
                new_features = current_features[0].copy()
                prev_close_idx = feats.index('Prev_Close')
                new_features[prev_close_idx] = (final_pred - X_mean[prev_close_idx]) / X_std[prev_close_idx]

                # 更新滯後特徵
                for j in range(1, min(4, len(predicted_prices))):
                    if f'Prev_Close_Lag{j}' in feats:
                        lag_idx = feats.index(f'Prev_Close_Lag{j}')
                        if len(predicted_prices) > j:
                            lag_price = predicted_prices[-(j + 1)]
                            new_features[lag_idx] = (lag_price - X_mean[lag_idx]) / X_std[lag_idx]

                # 更新移動平均線
                for ma in [5, 10]:
                    if f'MA{ma}' in feats and len(predicted_prices) >= ma + 1:
                        ma_idx = feats.index(f'MA{ma}')
                        recent_ma = np.mean(predicted_prices[-min(ma, len(predicted_prices)) - 1:-1])
                        new_features[ma_idx] = (recent_ma - X_mean[ma_idx]) / X_std[ma_idx]
                
                # 更新波動率
                if 'Volatility' in feats and len(predicted_prices) >= 3:
                    volatility_idx = feats.index('Volatility')
                    recent_volatility = np.std(predicted_prices[-min(10, len(predicted_prices)):])
                    new_features[volatility_idx] = (recent_volatility - X_mean[volatility_idx]) / X_std[volatility_idx]

                current_features = new_features.reshape(1, -1)

        preds = {f'T+{i + 1}': pred for i, pred in enumerate(predictions.values())}

        # 顯示模型驗證資訊
        if len(X_val) > 0:
            y_pred_val = models[0][1].predict(X_val)
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            st.info(f"模型驗證 - RMSE: {rmse:.2f} (約 {rmse / last_close * 100:.1f}%)")
            feature_importance = models[0][1].feature_importances_
            top_features = sorted(zip(feats, feature_importance), key=lambda x: x[1], reverse=True)[:5]
            st.info(f"重要特徵: {', '.join([f'{feat}({imp:.3f})' for feat, imp in top_features])}")

        return last_close, predictions, preds

    except Exception as e:
        st.error(f"預測過程發生錯誤: {str(e)}")
        return None, None, None


def get_trade_advice(last, preds):
    """根據預測結果提供交易建議。"""
    if not preds:
        return "無法判斷"
    price_changes = [preds[f'T+{d}'] - last for d in range(1, 6)]
    avg_change = np.mean(price_changes)
    change_percent = (avg_change / last) * 100
    if change_percent > 2:
        return f"強烈買入 (預期上漲 {change_percent:.1f}%)"
    elif change_percent > 0.5:
        return f"買入 (預期上漲 {change_percent:.1f}%)"
    elif change_percent < -2:
        return f"強烈賣出 (預期下跌 {abs(change_percent):.1f}%)"
    elif change_percent < -0.5:
        return f"賣出 (預期下跌 {abs(change_percent):.1f}%)"
    else:
        return f"持有 (預期變動 {change_percent:.1f}%)"


# Streamlit UI
st.set_page_config(page_title="股價預測系統", layout="centered", initial_sidebar_state="auto")
st.title("📈 5 日股價預測系統")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    code = st.text_input("請輸入股票代號（僅輸入數字部分即可）", "2330")
with col2:
    mode = st.selectbox("預測模式", ["中期模式", "短期模式", "長期模式"])

mode_info = {
    "短期模式": ("使用 100 天歷史資料，高敏感度", 100, 0.008),
    "中期模式": ("使用 200 天歷史資料，平衡敏感度", 200, 0.005),
    "長期模式": ("使用 400 天歷史資料，低敏感度", 400, 0.002)
}
st.info(f"**{mode}**: {mode_info[mode][0]}")
days, decay_factor = mode_info[mode][1], mode_info[mode][2]

if st.button("🔮 開始預測", type="primary"):
    full_code = code.strip()
    if not full_code.upper().endswith(".TW"):
        full_code = f"{full_code}.TW"
    with st.spinner("正在下載資料並進行預測..."):
        last, forecast, preds = predict_next_5(full_code, days, decay_factor)

    if last is None:
        st.error("❌ 預測失敗，請檢查股票代號或網路連線")
    else:
        st.success("✅ 預測完成！")

        # 顯示中英文股票名稱
        try:
            ticker_info = yf.Ticker(full_code).info
            company_name = ticker_info.get('shortName') or ticker_info.get('longName') or "無法取得名稱"
        except Exception:
            company_name = "無法取得名稱"

        ch_name = stock_name_dict.get(full_code, "無中文名稱")
        st.write(f"📌 股票名稱：**{ch_name} ({company_name})**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("當前股價", f"${last:.2f}")
            advice = get_trade_advice(last, preds)
            if "買入" in advice:
                st.success(f"📈 **交易建議**: {advice}")
            elif "賣出" in advice:
                st.error(f"📉 **交易建議**: {advice}")
            else:
                st.warning(f"📊 **交易建議**: {advice}")

        with col2:
            st.subheader("📅 未來 5 日預測")
            for date, price in forecast.items():
                change = price - last
                change_pct = (change / last) * 100
                if change > 0:
                    st.write(f"**{date}**: ${price:.2f} (+{change:.2f}, +{change_pct:.1f}%)")
                else:
                    st.write(f"**{date}**: ${price:.2f} ({change:.2f}, {change_pct:.1f}%)")

            # 顯示最佳買賣點
            min_date = min(forecast, key=forecast.get)
            min_price = forecast[min_date]
            max_date = max(forecast, key=forecast.get)
            max_price = forecast[max_date]

            st.markdown("### 📌 預測期間最佳買賣點")
            st.write(f"最佳買點：**{min_date}**，預測價格：${min_price:.2f}")
            st.write(f"最佳賣點：**{max_date}**，預測價格：${max_price:.2f}")

        st.subheader("📈 預測趨勢")
        chart_data = pd.DataFrame({
            '日期': ['今日'] + list(forecast.keys()),
            '股價': [last] + list(forecast.values())
        })
        st.line_chart(chart_data.set_index('日期'))

st.markdown("---")
st.caption("⚠️ 此預測僅供參考，投資有風險，請謹慎決策")
