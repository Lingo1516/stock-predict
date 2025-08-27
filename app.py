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
from collections import deque

# 股票代號到中文名稱簡易對照字典，可自行擴充
stock_name_dict = {
    "2330.TW": "台灣積體電路製造股份有限公司",
    "2317.TW": "鴻海精密工業股份有限公司",
    "2412.TW": "中華電信股份有限公司",
}

@st.cache_data
def predict_next_5(stock, days, decay_factor):
    try:
        end = pd.Timestamp(datetime.today().date())
        # 擴展歷史數據範圍至至少 5 年
        start = end - pd.Timedelta(days=max(days, 365 * 5)) # 確保至少5年數據
        max_retries = 5 # 增加重試次數
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
                time.sleep(3) # 增加等待時間

            if attempt == max_retries - 1:
                st.error(f"無法下載資料：{stock}，請檢查股票代號或網路連線。")
                return None, None, None

        # 更健壯的缺失值處理：線性插值
        if df is not None and not df.empty:
            df = df.interpolate(method='linear', limit_direction='both')
            df = df.fillna(method='bfill').fillna(method='ffill') # 再次填充可能存在的開頭或結尾NaN

        if df is None or len(df) < 100: # 提高數據量不足的閾值
            st.error(f"資料不足，僅有 {len(df) if df is not None else 0} 行數據。請確保股票代號正確且有足夠的歷史數據。")
            return None, None, None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = [col[0] for col in twii.columns]
        if isinstance(sp.columns, pd.MultiIndex):
            sp.columns = [col[0] for col in sp.columns]

        close = df["Close"].squeeze() if "Close" in df.columns else df.iloc[:, 3].squeeze()
        df["TWII_Close"] = twii["Close"].reindex(df.index, method="ffill").fillna(method="bfill")
        df["SP500_Close"] = sp["Close"].reindex(df.index, method="ffill").fillna(method="bfill")

        # 確保TWII_Close和SP500_Close沒有NaN
        if df["TWII_Close"].isnull().any() or df["SP500_Close"].isnull().any():
            st.error("市場指數數據缺失，無法進行預測。")
            return None, None, None

        df["MA5"] = close.rolling(5, min_periods=1).mean()
        df["MA10"] = close.rolling(10, min_periods=1).mean()
        df["MA20"] = close.rolling(20, min_periods=1).mean()
        df["MA60"] = close.rolling(60, min_periods=1).mean() # 新增MA60

        # 優化 ta 庫異常處理，確保手動計算邏輯一致
        try:
            df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        except Exception as e:
            st.warning(f"RSI計算失敗，嘗試手動計算: {e}")
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

        try:
            macd = ta.trend.MACD(close)
            df["MACD"] = macd.macd()
            df["MACD_Signal"] = macd.macd_signal()
            df["MACD_Hist"] = macd.macd_diff() # 新增MACD柱狀圖
        except Exception as e:
            st.warning(f"MACD計算失敗，嘗試手動計算: {e}")
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df["MACD"] = ema12 - ema26
            df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        bb_indicator = BollingerBands(close, window=20, window_dev=2)
        df["BB_High"] = bb_indicator.bollinger_hband()
        df["BB_Low"] = bb_indicator.bollinger_lband()
        df["BB_Mid"] = bb_indicator.bollinger_mavg() # 新增布林中軌
        df["BB_Width"] = bb_indicator.bollinger_wband() # 新增布林帶寬度

        adx_indicator = ADXIndicator(df["High"], df["Low"], close, window=14)
        df["ADX"] = adx_indicator.adx()
        df["DIP"] = adx_indicator.plus_di() # 新增+DI
        df["DIM"] = adx_indicator.minus_di() # 新增-DI

        # 新增KDJ指標
        # 計算RSV
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['RSV'] = (close - low_14) / (high_14 - low_14) * 100
        # 計算K值
        df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
        # 計算D值
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        # 計算J值
        df['J'] = 3 * df['K'] - 2 * df['D']

        df["Prev_Close"] = close.shift(1)
        for i in range(1, 6): # 增加滯後天數到5天
            df[f"Prev_Close_Lag{i}"] = close.shift(i)

        # 改進成交量缺失處理
        if "Volume" in df.columns and not df["Volume"].isnull().all():
            df["Volume_MA"] = df["Volume"].rolling(10, min_periods=1).mean()
            df["Volume_Change"] = df["Volume"].pct_change()
        else:
            st.warning("成交量數據缺失或為空，將使用0填充成交量相關特徵。")
            df["Volume"] = 0 # 確保Volume列存在，避免後續計算報錯
            df["Volume_MA"] = 0
            df["Volume_Change"] = 0

        df["Volatility"] = close.rolling(10, min_periods=1).std()
        df["Daily_Return"] = close.pct_change()

        # 引入基本面數據 (簡化示例，實際應用需更複雜的數據獲取和處理)
        try:
            ticker_info = yf.Ticker(stock).info
            df["PE_Ratio"] = ticker_info.get("trailingPE", 0) # 使用get避免KeyError，默認0
            df["Market_Cap"] = ticker_info.get("marketCap", 0)
            # 基本面數據通常是靜態的，需要填充到所有行
            df["PE_Ratio"] = df["PE_Ratio"].replace(0, np.nan).ffill().fillna(0)
            df["Market_Cap"] = df["Market_Cap"].replace(0, np.nan).ffill().fillna(0)
        except Exception as e:
            st.warning(f"無法獲取基本面數據: {e}")
            df["PE_Ratio"] = 0
            df["Market_Cap"] = 0

        feats = [
            "Prev_Close", "MA5", "MA10", "MA20", "MA60",
            "Volume_MA", "Volume_Change",
            "RSI", "MACD", "MACD_Signal", "MACD_Hist",
            "BB_High", "BB_Low", "BB_Mid", "BB_Width",
            "ADX", "DIP", "DIM",
            "K", "D", "J",
            "TWII_Close", "SP500_Close",
            "Volatility", "Daily_Return",
            "PE_Ratio", "Market_Cap"
        ] + [f"Prev_Close_Lag{i}" for i in range(1, 6)]

        # 檢查並移除缺失過多的特徵
        initial_len = len(df)
        df_cleaned_features = df[feats].dropna()
        if len(df_cleaned_features) < initial_len * 0.8: # 如果超過20%的數據因為NaN被移除，則考慮移除該特徵
            for col in feats:
                if df[col].isnull().sum() / initial_len > 0.2:
                    st.warning(f"特徵 '{col}' 缺失值過多，將從特徵列表中移除。")
                    feats.remove(col)

        missing_feats = [f for f in feats if f not in df.columns]
        if missing_feats:
            st.error(f"缺少特徵: {missing_feats}。請檢查數據完整性。")
            return None, None, None

        df_clean = df[feats + ["Close"]].dropna()
        if len(df_clean) < 50: # 提高清理後數據的最低要求
            st.error(f"清理後資料不足，僅有 {len(df_clean)} 行數據。請嘗試更長的歷史數據範圍或更換股票。")
            return None, None, None

        X = df_clean[feats].values
        y = df_clean["Close"].values

        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # 防止除以0

        X_normalized = (X - X_mean) / X_std
        weights = np.exp(-decay_factor * np.arange(len(X))[::-1])
        weights = weights / np.sum(weights)

        # 滾動時間窗口驗證 (簡化實現，實際應更複雜)
        # 這裡仍然使用簡單的80/20劃分，但可以考慮在後續階段實現更嚴格的回測
        split_idx = int(len(X_normalized) * 0.8)
        X_train, X_val = X_normalized[:split_idx], X_normalized[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        train_weights = weights[:split_idx]

        models = []

        # 隨機森林模型參數優化 (示例，實際應通過GridSearchCV等優化)
        rf_model = RandomForestRegressor(
            n_estimators=150, # 增加估計器數量
            max_depth=15,     # 增加最大深度
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train, sample_weight=train_weights)
        models.append(("RF1", rf_model))

        rf_model2 = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=123,
            n_jobs=-1
        )
        rf_model2.fit(X_train, y_train, sample_weight=train_weights)
        models.append(("RF2", rf_model2))

        # 可以考慮引入其他模型，例如XGBoost或LightGBM
        # from xgboost import XGBRegressor
        # xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        # xgb_model.fit(X_train, y_train, sample_weight=train_weights)
        # models.append(('XGB', xgb_model))

        last_features = X_normalized[-1:].copy()
        last_close = float(y[-1])
        predictions = {}
        future_dates = []
        current_date = end
        for i in range(5):
            current_date = current_date + pd.offsets.BDay(1)
            future_dates.append(current_date.date())

        current_features = last_features.copy()
        predicted_prices_history = deque([last_close], maxlen=60) # 使用deque維護歷史價格，用於MA和波動率計算

        # 動態調整最大偏離限制
        # 使用歷史日收益率的標準差來估計波動率
        daily_returns = df_clean["Daily_Return"].dropna()
        if len(daily_returns) > 30:
            historical_daily_volatility = daily_returns.std()
            # 將日波動率轉換為5日波動率的近似值 (sqrt(5) * daily_volatility)
            # 並乘以一個係數，例如3倍標準差作為合理波動範圍
            max_deviation_pct = historical_daily_volatility * np.sqrt(5) * 3
            max_deviation_pct = min(max_deviation_pct, 0.15) # 設定上限，防止過大
            st.info(f"動態最大偏離限制設定為: {max_deviation_pct:.2%}")
        else:
            max_deviation_pct = 0.10 # 數據不足時使用默認值

        for i, date in enumerate(future_dates):
            day_predictions = []
            for model_name, model in models:
                pred = model.predict(current_features)[0]
                # 隨機變異和波動率調整參數動態化
                # 隨機變異基於預測價格和歷史波動率
                variation_scale = pred * historical_daily_volatility * 0.5 # 調整係數
                variation = np.random.normal(0, variation_scale)
                day_predictions.append(pred + variation)

            # 優化集成策略：可以考慮基於模型在驗證集上的表現來調整權重
            # 這裡仍然使用固定權重，但可以作為未來改進點
            weights_ensemble = [0.5, 0.5] # 如果有兩個模型
            if len(models) == 3: # 如果有三個模型
                weights_ensemble = [0.4, 0.3, 0.3]
            ensemble_pred = np.average(day_predictions, weights=weights_ensemble[:len(day_predictions)])

            final_pred = ensemble_pred

            # 限制預測價格在合理範圍內
            upper_limit = predicted_prices_history[-1] * (1 + max_deviation_pct)
            lower_limit = predicted_prices_history[-1] * (1 - max_deviation_pct)
            final_pred = min(max(final_pred, lower_limit), upper_limit)

            predictions[date] = float(final_pred)
            predicted_prices_history.append(final_pred) # 將新的預測價格加入歷史記錄

            if i < 4: # 為下一次預測準備特徵
                new_features = current_features[0].copy()

                # 更新Prev_Close和Prev_Close_Lag
                for j in range(1, 6): # 最多滯後5天
                    if f"Prev_Close_Lag{j}" in feats:
                        lag_idx = feats.index(f"Prev_Close_Lag{j}")
                        if len(predicted_prices_history) > j:
                            lag_price = predicted_prices_history[-(j + 1)]
                            new_features[lag_idx] = (lag_price - X_mean[lag_idx]) / X_std[lag_idx]

                # 更新MA5, MA10, MA20, MA60
                for ma_window in [5, 10, 20, 60]:
                    ma_feat_name = f"MA{ma_window}"
                    if ma_feat_name in feats and len(predicted_prices_history) >= ma_window:
                        ma_idx = feats.index(ma_feat_name)
                        recent_ma = np.mean(list(predicted_prices_history)[-ma_window:])
                        new_features[ma_idx] = (recent_ma - X_mean[ma_idx]) / X_std[ma_idx]

                # 更新Volatility
                if "Volatility" in feats and len(predicted_prices_history) >= 10:
                    volatility_idx = feats.index("Volatility")
                    recent_volatility = np.std(list(predicted_prices_history)[-10:])
                    new_features[volatility_idx] = (recent_volatility - X_mean[volatility_idx]) / X_std[volatility_idx]

                # 更新Daily_Return
                if "Daily_Return" in feats and len(predicted_prices_history) >= 2:
                    daily_return_idx = feats.index("Daily_Return")
                    current_day_return = (predicted_prices_history[-1] - predicted_prices_history[-2]) / predicted_prices_history[-2]
                    new_features[daily_return_idx] = (current_day_return - X_mean[daily_return_idx]) / X_std[daily_return_idx]

                # 其他技術指標的更新會更複雜，這裡暫時保持不變或簡化處理
                # 實際應用中，這些指標也需要根據新的預測價格進行迭代計算

                current_features = new_features.reshape(1, -1)

        preds = {f"T+{i + 1}": pred for i, pred in enumerate(predictions.values())}

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
    if not preds:
        return "無法判斷"
    price_changes = [preds[f"T+{d}"] - last for d in range(1, 6)]
    avg_change = np.mean(price_changes)
    change_percent = (avg_change / last) * 100

    # 動態調整交易閾值 (示例，實際應更複雜)
    # 可以根據歷史波動率或市場情緒來調整
    buy_threshold_strong = 2.5 # 提高強烈買入閾值
    buy_threshold = 0.8      # 提高買入閾值
    sell_threshold_strong = -2.5 # 降低強烈賣出閾值
    sell_threshold = -0.8      # 降低賣出閾值

    if change_percent > buy_threshold_strong:
        return f"強烈買入 (預期上漲 {change_percent:.1f}%)"
    elif change_percent > buy_threshold:
        return f"買入 (預期上漲 {change_percent:.1f}%)"
    elif change_percent < sell_threshold_strong:
        return f"強烈賣出 (預期下跌 {abs(change_percent):.1f}%)"
    elif change_percent < sell_threshold:
        return f"賣出 (預期下跌 {abs(change_percent):.1f}%)"
    else:
        return f"持有 (預期變動 {change_percent:.1f}%)"


# Streamlit UI
st.set_page_config(layout="wide") # 設置頁面為寬模式
st.title("📈 5 日股價預測系統")
st.markdown("---")

col1, col2, col3 = st.columns([2, 1, 1]) # 增加一列用於輸入股票代號
with col1:
    code = st.text_input("請輸入股票代號 (例如: 2330.TW 或 AAPL)", "2330.TW")
with col2:
    mode = st.selectbox("預測模式", ["短期模式", "中期模式", "長期模式"])
with col3:
    st.markdown("<br>", unsafe_allow_html=True) # 為了對齊按鈕
    if st.button("🔮 開始預測", type="primary"):
        pass # 按鈕點擊後執行下面的邏輯

mode_info = {
    "短期模式": ("使用 1 年歷史資料，高敏感度", 365, 0.008),
    "中期模式": ("使用 3 年歷史資料，平衡敏感度", 365 * 3, 0.005),
    "長期模式": ("使用 5 年歷史資料，低敏感度", 365 * 5, 0.002)
}
st.info(f"**{mode}**: {mode_info[mode][0]}")
days, decay_factor = mode_info[mode][1], mode_info[mode][2]

if st.button("🔮 開始預測", type="primary", key="predict_button_bottom"):
    full_code = code.strip().upper()
    if not (".TW" in full_code or ".US" in full_code or ".HK" in full_code): # 簡單判斷市場
        st.warning("請輸入完整的股票代號，例如 2330.TW (台股) 或 AAPL (美股)。")
        full_code = f"{full_code}.TW" # 默認為台股

    with st.spinner("正在下載資料並進行預測..."):
        last, forecast, preds = predict_next_5(full_code, days, decay_factor)

    if last is None:
        st.error("❌ 預測失敗，請檢查股票代號或網路連線")
    else:
        st.success("✅ 預測完成！")

        # 顯示中英文股票名稱
        company_name = "無法取得名稱"
        ch_name = stock_name_dict.get(full_code, "無中文名稱")
        try:
            ticker_info = yf.Ticker(full_code).info
            company_name = ticker_info.get("shortName") or ticker_info.get("longName") or "無法取得名稱"
            if ch_name == "無中文名稱" and "zh-Hant" in ticker_info.get("summaryProfile", {}).get("language", ""):
                ch_name = ticker_info.get("longName", "無中文名稱") # 嘗試從yfinance獲取中文名稱
        except Exception:
            pass

        st.write(f"📌 股票名稱：**{ch_name} ({company_name})**")

        col_metric, col_forecast = st.columns([1, 2])
        with col_metric:
            st.metric("當前股價", f"${last:.2f}")
            advice = get_trade_advice(last, preds)
            if "買入" in advice:
                st.success(f"📈 **交易建議**: {advice}")
            elif "賣出" in advice:
                st.error(f"📉 **交易建議**: {advice}")
            else:
                st.warning(f"📊 **交易建議**: {advice}")

            # 顯示最佳買賣點
            if forecast:
                min_date = min(forecast, key=forecast.get)
                min_price = forecast[min_date]
                max_date = max(forecast, key=forecast.get)
                max_price = forecast[max_date]

                st.markdown("### 📌 預測期間最佳買賣點")
                st.write(f"最佳買點：**{min_date}**，預測價格：${min_price:.2f}")
                st.write(f"最佳賣點：**{max_date}**，預測價格：${max_price:.2f}")

        with col_forecast:
            st.subheader("📅 未來 5 日預測")
            for date, price in forecast.items():
                change = price - last
                change_pct = (change / last) * 100
                if change > 0:
                    st.write(f"**{date}**: ${price:.2f} (<span style='color:green'>+{change:.2f}, +{change_pct:.1f}%</span>)", unsafe_allow_html=True)
                else:
                    st.write(f"**{date}**: ${price:.2f} (<span style='color:red'>{change:.2f}, {change_pct:.1f}%</span>)", unsafe_allow_html=True)

            st.subheader("📈 預測趨勢")
            chart_data = pd.DataFrame({
                "日期": ["今日"] + list(forecast.keys()),
                "股價": [last] + list(forecast.values())
            })
            # 使用Plotly Express提供更豐富的交互圖表
            import plotly.express as px
            fig = px.line(chart_data, x="日期", y="股價", title="未來 5 日股價預測趨勢")
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("⚠️ 此預測僅供參考，投資有風險，請謹慎決策")


