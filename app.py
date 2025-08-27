import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
import time

# 股票代號到中文名稱簡易對照字典，可自行擴充
stock_name_dict = {
    "2330.TW": "台灣積體電路製造股份有限公司",
    "2317.TW": "鴻海精密工業股份有限公司",
    "2412.TW": "中華電信股份有限公司",
}

def calculate_technical_indicators(df):
    """手動計算技術指標"""
    df['MA5'] = df['Close'].rolling(5, min_periods=1).mean()
    df['MA10'] = df['Close'].rolling(10, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_High'] = df['Close'].rolling(window=20).mean() + df['Close'].rolling(window=20).std() * 2
    df['BB_Low'] = df['Close'].rolling(window=20).mean() - df['Close'].rolling(window=20).std() * 2

    # ADX (Average Directional Index)
    df['TR'] = np.maximum(np.maximum(df['High'] - df['Low'], abs(df['High'] - df['Close'].shift(1))), abs(df['Low'] - df['Close'].shift(1)))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['PlusDM'] = (df['High'] - df['High'].shift(1)).where(lambda x: x > 0, 0)
    df['MinusDM'] = (df['Low'].shift(1) - df['Low']).where(lambda x: x > 0, 0)
    df['DI_Plus'] = 100 * (df['PlusDM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['DI_Minus'] = 100 * (df['MinusDM'].ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
    df['ADX'] = 100 * (abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])).ewm(alpha=1/14, adjust=False).mean()

    # Stochastic Oscillator
    df['14-low'] = df['Low'].rolling(14).min()
    df['14-high'] = df['High'].rolling(14).max()
    df['STOCH_K'] = 100 * ((df['Close'] - df['14-low']) / (df['14-high'] - df['14-low']))
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()

    # CCI (Commodity Channel Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma_tp = tp.rolling(window=20).mean()
    dev_tp = (tp - ma_tp).abs().rolling(window=20).mean()
    df['CCI'] = (tp - ma_tp) / (0.015 * dev_tp)

    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(window=10).mean()

    # 新增 ATR (Average True Range)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # 新增 ROC (Rate of Change)
    df['ROC'] = df['Close'].diff(periods=12) / df['Close'].shift(periods=12) * 100
    
    # === 新增三大法人相關指標（需外部數據支持） ===
    # 假設您已經有一個 DataFrame `institutional_data` 包含了三大法人的每日買賣超數據
    # 範例：institutional_data = pd.read_csv('path/to/your/institutional_data.csv', index_col='Date', parse_dates=True)
    # df = df.join(institutional_data, how='left') # 假設 `institutional_data` 的欄位名稱與您的一樣
    
    # 這裡我們手動創建一些空欄位，代表未來將整合進來的數據
    # 這些欄位將是模型預測的關鍵特徵
    df['institutional_net_buy_sell'] = np.nan # 單日買賣超金額
    df['institutional_5d_cum_net_buy_sell'] = np.nan # 5日累計買賣超
    df['institutional_20d_cum_net_buy_sell'] = np.nan # 20日累計買賣超
    df['institutional_10d_net_buy_sell_ma'] = np.nan # 10日買賣超移動平均
    
    return df

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
        start = end - pd.Timedela(days=days)
        max_retries = 3
        df, twii, sp = None, None, None

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

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = [col[0] for col in twii.columns]
        if isinstance(sp.columns, pd.MultiIndex):
            sp.columns = [col[0] for col in sp.columns]

        if not all(col in df.columns for col in ['Close', 'High', 'Low', 'Volume']):
            st.error("股票數據中缺少必要的欄位 (Close, High, Low, Volume)。")
            return None, None, None

        df['TWII_Close'] = twii['Close'].reindex(df.index, method='ffill').fillna(method='bfill')
        df['SP500_Close'] = sp['Close'].reindex(df.index, method='ffill').fillna(method='bfill')

        df = calculate_technical_indicators(df)
        
        # 新增三大法人買賣超欄位 (目前為空值，需手動匯入數據)
        df['institutional_net_buy_sell'] = np.nan # 假設資料為每日買賣超金額
        # 您可以在此處加入程式碼，從其他來源讀取並填入 df['institutional_net_buy_sell'] 的值
        # 範例：df = pd.read_csv('institutional_data.csv', index_col='Date', parse_dates=True)
        # df['institutional_net_buy_sell'] = institutional_data['Net_Buy_Sell']

        df['Prev_Close'] = df['Close'].shift(1)
        for i in range(1, 4):
            df[f'Prev_Close_Lag{i}'] = df['Close'].shift(i)

        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(10, min_periods=1).mean()
        else:
            df['Volume_MA'] = 0

        df['Volatility'] = df['Close'].rolling(10, min_periods=1).std()

        feats = [
            'Prev_Close', 'MA5', 'MA10', 'MA20', 'Volume_MA', 'RSI', 'MACD',
            'MACD_Signal', 'TWII_Close', 'SP500_Close', 'Volatility', 'BB_High',
            'BB_Low', 'ADX', 'STOCH_K', 'STOCH_D', 'CCI', 'OBV', 'OBV_MA', 'ATR', 'ROC',
            'institutional_net_buy_sell' # 新增的籌碼面特徵
        ] + [f'Prev_Close_Lag{i}' for i in range(1, 4)]
        
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

        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1

        X_normalized = (X - X_mean) / X_std
        weights = np.exp(-decay_factor * np.arange(len(X))[::-1])
        weights = weights / np.sum(weights)

        split_idx = int(len(X_normalized) * 0.8)
        X_train, X_val = X_normalized[:split_idx], X_normalized[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        train_weights = weights[:split_idx]

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
        max_deviation_pct = 0.10

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
            
            upper_limit = last_close * (1 + max_deviation_pct)
            lower_limit = last_close * (1 - max_deviation_pct)
            final_pred = min(max(final_pred, lower_limit), upper_limit)

            predictions[date] = float(final_pred)
            predicted_prices.append(final_pred)

            if i < 4:
                new_features = current_features[0].copy()
                prev_close_idx = feats.index('Prev_Close')
                new_features[prev_close_idx] = (final_pred - X_mean[prev_close_idx]) / X_std[prev_close_idx]

                for j in range(1, min(4, len(predicted_prices))):
                    if f'Prev_Close_Lag{j}' in feats:
                        lag_idx = feats.index(f'Prev_Close_Lag{j}')
                        if len(predicted_prices) > j:
                            lag_price = predicted_prices[-(j + 1)]
                            new_features[lag_idx] = (lag_price - X_mean[lag_idx]) / X_std[lag_idx]

                for ma in [5, 10]:
                    if f'MA{ma}' in feats and len(predicted_prices) >= ma + 1:
                        ma_idx = feats.index(f'MA{ma}')
                        recent_ma = np.mean(predicted_prices[-min(ma, len(predicted_prices)) - 1:-1])
                        new_features[ma_idx] = (recent_ma - X_mean[ma_idx]) / X_std[ma_idx]
                
                if 'Volatility' in feats and len(predicted_prices) >= 3:
                    volatility_idx = feats.index('Volatility')
                    recent_volatility = np.std(predicted_prices[-min(10, len(predicted_prices)):])
                    new_features[volatility_idx] = (recent_volatility - X_mean[volatility_idx]) / X_std[volatility_idx]
                
                # 在此處為新的特徵賦予預測值 (目前暫時設為0)
                institutional_idx = feats.index('institutional_net_buy_sell')
                new_features[institutional_idx] = 0

                current_features = new_features.reshape(1, -1)

        preds = {f'T+{i + 1}': pred for i, pred in enumerate(predictions.values())}

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
