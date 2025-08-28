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
from ta.trend import ADXIndicator

# 股票代號到中文名稱簡易對照字典
stock_name_dict = {
    "2330.TW": "台灣積體電路製造股份有限公司",
    "2317.TW": "鴻海精密工業股份有限公司",
    "2412.TW": "中華電信股份有限公司",
}

@st.cache_data
def predict_next_5(stock, days, decay_factor):
    try:
        end = pd.Timestamp(datetime.today().date())
        start = end - pd.Timedelta(days=days)
        max_retries = 3
        df, twii, sp = None, None, None

        for attempt in range(max_retries):
            try:
                # 下載資料
                df = yf.download(stock, start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True, progress=False)
                twii = yf.download("^TWII", start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True, progress=False)
                sp = yf.download("^GSPC", start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True, progress=False)
                if not (df.empty or twii.empty or sp.empty):
                    break
            except Exception as e:
                st.warning(f"嘗試 {attempt + 1}/{max_retries} 下載失敗: {e}")
                time.sleep(2)

            if attempt == max_retries - 1 and (df is None or df.empty):
                st.error(f"無法下載資料：{stock}")
                return None, None, None, None

        if df is None or len(df) < 50:
            st.error(f"資料不足，僅有 {len(df) if df is not None else 0} 行數據")
            return None, None, None, None

        # 處理欄位名稱
        for frame in [df, twii, sp]:
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = [col[0] for col in frame.columns]

        if not all(col in df.columns for col in ['High', 'Low', 'Close']):
            st.error("下載的資料缺少 'High', 'Low', 或 'Close' 欄位。")
            return None, None, None, None

        close = df['Close']
        df['TWII_Close'] = twii['Close'].reindex(df.index, method='ffill').fillna(method='bfill')
        df['SP500_Close'] = sp['Close'].reindex(df.index, method='ffill').fillna(method='bfill')

        # 計算技術指標
        df['MA5'] = close.rolling(5, min_periods=1).mean()
        df['MA10'] = close.rolling(10, min_periods=1).mean()
        df['MA20'] = close.rolling(20, min_periods=1).mean()
        df['RSI'] = ta.momentum.RSIIndicator(close, n=14).rsi()
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        bb_indicator = BollingerBands(close, n=20, ndev=2)
        df['BB_High'] = bb_indicator.bollinger_hband()
        df['BB_Low'] = bb_indicator.bollinger_lband()
        adx_indicator = ADXIndicator(df['High'], df['Low'], close, n=14)
        df['ADX'] = adx_indicator.adx()
        df['Prev_Close'] = close.shift(1)
        for i in range(1, 4):
            df[f'Prev_Close_Lag{i}'] = close.shift(i)
        df['Volume_MA'] = df['Volume'].rolling(10, min_periods=1).mean() if 'Volume' in df.columns else 0
        df['Volatility'] = close.rolling(10, min_periods=1).std()

        # 準備特徵
        feats = ['Prev_Close', 'MA5', 'MA10', 'MA20', 'Volume_MA', 'RSI', 'MACD',
                 'MACD_Signal', 'TWII_Close', 'SP500_Close', 'Volatility', 'BB_High',
                 'BB_Low', 'ADX'] + [f'Prev_Close_Lag{i}' for i in range(1, 4)]
        
        df_clean = df[feats + ['Close']].dropna()
        if len(df_clean) < 30:
            st.error(f"清理後資料不足，僅有 {len(df_clean)} 行數據")
            return None, None, None, None

        # 訓練模型
        X = df_clean[feats].values
        y = df_clean['Close'].values
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X_normalized = (X - X_mean) / X_std
        weights = np.exp(-decay_factor * np.arange(len(X))[::-1])
        weights = weights / np.sum(weights)
        split_idx = int(len(X_normalized) * 0.8)
        X_train, X_val, y_train, y_val, train_weights = X_normalized[:split_idx], X_normalized[split_idx:], y[:split_idx], y[split_idx:], weights[:split_idx]

        models = []
        model_params = [
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42},
            {'n_estimators': 80, 'max_depth': 8, 'min_samples_split': 3, 'min_samples_leaf': 1, 'random_state': 123},
            {'n_estimators': 120, 'max_depth': 12, 'min_samples_split': 7, 'min_samples_leaf': 3, 'random_state': 456}
        ]
        for params in model_params:
            model = RandomForestRegressor(**params, n_jobs=-1)
            model.fit(X_train, y_train, sample_weight=train_weights)
            models.append(model)

        # 進行預測
        last_features_normalized = X_normalized[-1:].copy()
        last_close = float(y[-1])
        predictions = {}
        future_dates = pd.bdate_range(start=df_clean.index[-1], periods=6)[1:]
        current_features_normalized = last_features_normalized.copy()
        predicted_prices = [last_close]

        for i, date in enumerate(future_dates):
            day_predictions = [model.predict(current_features_normalized)[0] for model in models]
            ensemble_pred = np.average(day_predictions, weights=[0.5, 0.3, 0.2])
            
            # 加入波動性調整
            historical_volatility = np.std(y[-30:]) / np.mean(y[-30:])
            volatility_adjustment = np.random.normal(0, ensemble_pred * historical_volatility * 0.05)
            final_pred = ensemble_pred + volatility_adjustment
            
            # 限制價格範圍
            max_deviation_pct = 0.10
            upper_limit = last_close * (1 + max_deviation_pct)
            lower_limit = last_close * (1 - max_deviation_pct)
            final_pred = min(max(final_pred, lower_limit), upper_limit)
            
            predictions[date.date()] = float(final_pred)
            predicted_prices.append(final_pred)

            # 更新下一天的特徵 (迭代預測)
            if i < len(future_dates) - 1:
                new_features = current_features_normalized[0].copy()
                # 遍歷所有需要更新的特徵
                for feat_name in ['Prev_Close', 'MA5', 'MA10', 'Volatility'] + [f'Prev_Close_Lag{j}' for j in range(1, 4)]:
                    if feat_name in feats:
                        idx = feats.index(feat_name)
                        if feat_name == 'Prev_Close':
                            value = final_pred
                        elif feat_name.startswith('Prev_Close_Lag'):
                            lag = int(feat_name[-1])
                            if len(predicted_prices) > lag:
                                value = predicted_prices[-(lag + 1)]
                            else: continue
                        elif feat_name == 'MA5':
                            value = np.mean(predicted_prices[-min(5, len(predicted_prices)):])
                        elif feat_name == 'MA10':
                            value = np.mean(predicted_prices[-min(10, len(predicted_prices)):])
                        elif feat_name == 'Volatility':
                             if len(predicted_prices) >= 2:
                                value = np.std(predicted_prices[-min(10, len(predicted_prices)):])
                             else: continue
                        
                        new_features[idx] = (value - X_mean[idx]) / X_std[idx]
                current_features_normalized = new_features.reshape(1, -1)
        
        preds = {f'T+{i + 1}': p for i, p in enumerate(predictions.values())}

        # 顯示模型驗證資訊
        if len(X_val) > 0:
            y_pred_val = models[0].predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            st.info(f"模型驗證 - RMSE: {rmse:.2f} (約 {rmse / last_close * 100:.1f}%)")
            feature_importance = models[0].feature_importances_
            top_features = sorted(zip(feats, feature_importance), key=lambda x: x[1], reverse=True)[:5]
            st.info(f"重要特徵: {', '.join([f'{feat}({imp:.3f})' for feat, imp in top_features])}")

        # --- 修正開始：增加回傳 df_clean ---
        return last_close, predictions, preds, df_clean
        # --- 修正結束 ---

    except Exception as e:
        st.error(f"預測過程發生錯誤: {str(e)}")
        # --- 修正開始：確保在錯誤時也回傳四個值 ---
        return None, None, None, None
        # --- 修正結束 ---


def get_trade_advice(last, preds):
    if not preds or len(preds) < 5:
        return "無法判斷"
    price_values = list(preds.values())
    avg_change = np.mean([p - last for p in price_values])
    change_percent = (avg_change / last) * 100
    if change_percent > 1.5:
        return f"強烈看漲 (預期上漲 {change_percent:.1f}%)"
    elif change_percent > 0.5:
        return f"看漲 (預期上漲 {change_percent:.1f}%)"
    elif change_percent < -1.5:
        return f"強烈看跌 (預期下跌 {abs(change_percent):.1f}%)"
    elif change_percent < -0.5:
        return f"看跌 (預期下跌 {abs(change_percent):.1f}%)"
    else:
        return f"盤整 (預期變動 {change_percent:.1f}%)"

# --- Streamlit UI ---
st.set_page_config(page_title="AI 股價預測系統", layout="wide")
st.title("📈 AI 智慧股價預測系統")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    code = st.text_input("請輸入股票代號（例如: 2330）", "2330")
with col2:
    mode = st.selectbox("預測模式", ["中期模式", "短期模式", "長期模式"])

mode_info = {
    "短期模式": ("使用 100 天歷史資料，高敏感度", 100, 0.008),
    "中期模式": ("使用 200 天歷史資料，平衡敏感度", 200, 0.005),
    "長期模式": ("使用 400 天歷史資料，低敏感度", 400, 0.002)
}
st.info(f"**{mode}**: {mode_info[mode][0]}")
days, decay_factor = mode_info[mode][1], mode_info[mode][2]

if st.button("🔮 開始預測", type="primary", use_container_width=True):
    full_code = code.strip()
    if not full_code.upper().endswith(".TW"):
        full_code = f"{full_code}.TW"
        
    with st.spinner("🚀 正在下載數據、訓練模型並進行預測..."):
        # --- 修正開始：接收第四個回傳值 df_clean ---
        last, forecast, preds, df_clean = predict_next_5(full_code, days, decay_factor)
        # --- 修正結束 ---

    if last is None:
        st.error("❌ 預測失敗，請檢查上方錯誤訊息或網路連線")
    else:
        st.success("✅ 預測完成！")

        try:
            ticker_info = yf.Ticker(full_code).info
            company_name = ticker_info.get('shortName') or ticker_info.get('longName') or "無法取得名稱"
        except Exception:
            company_name = "無法取得名稱"

        ch_name = stock_name_dict.get(full_code, "無中文名稱")
        st.header(f"{ch_name} ({company_name}) - {full_code}")

        main_col1, main_col2 = st.columns([1, 2])
        with main_col1:
            st.metric("最新收盤價", f"${last:.2f}")
            advice = get_trade_advice(last, preds)
            if "看漲" in advice:
                st.success(f"📈 **交易建議**: {advice}")
            elif "看跌" in advice:
                st.error(f"📉 **交易建議**: {advice}")
            else:
                st.warning(f"📊 **交易建議**: {advice}")

            st.markdown("### 📌 預測期間最佳買賣點")
            if forecast:
                min_date = min(forecast, key=forecast.get)
                min_price = forecast[min_date]
                max_date = max(forecast, key=forecast.get)
                max_price = forecast[max_date]
                st.write(f"🟢 **潛在買點**: {min_date} @ ${min_price:.2f}")
                st.write(f"🔴 **潛在賣點**: {max_date} @ ${max_price:.2f}")

        with main_col2:
            st.subheader("📅 未來 5 日預測")
            if forecast:
                forecast_df = pd.DataFrame(list(forecast.items()), columns=['日期', '預測股價'])
                forecast_df['漲跌'] = forecast_df['預測股價'] - last
                forecast_df['漲跌幅 (%)'] = (forecast_df['漲跌'] / last) * 100
                
                def color_change(val):
                    color = 'red' if val > 0 else 'green' if val < 0 else 'gray'
                    return f'color: {color}'
                
                st.dataframe(forecast_df.style.format({
                    '預測股價': '${:,.2f}',
                    '漲跌': '{:+.2f}',
                    '漲跌幅 (%)': '{:+.2f}%'
                }).apply(lambda x: x.map(color_change), subset=['漲跌', '漲跌幅 (%)']), use_container_width=True)

        st.subheader("📈 預測趨勢圖")
        if forecast and df_clean is not None and not df_clean.empty:
            # --- 修正開始：使用傳回來的 df_clean ---
            chart_data = pd.DataFrame({
                '日期': [df_clean.index[-1].date()] + list(forecast.keys()),
                '股價': [last] + list(forecast.values())
            })
            st.line_chart(chart_data.set_index('日期'))
            # --- 修正結束 ---

st.markdown("---")
st.caption("⚠️ 此預測基於歷史數據與 AI 模型，僅供學術研究與參考，不構成任何投資建議。投資有風險，請謹慎決策。")
