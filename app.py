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
    try:
        end = pd.Timestamp(datetime.today().date())
        start = end - pd.Timedelta(days=days)

        # 下載資料並添加錯誤處理
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
                st.error(f"無法下載資料：{stock}")
                return None, None, None

        # 檢查資料是否充足
        if df is None or len(df) < 50:
            st.error(f"資料不足，僅有 {len(df) if df is not None else 0} 行數據")
            return None, None, None

        # 處理多重索引
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = [col[0] for col in twii.columns]
        if isinstance(sp.columns, pd.MultiIndex):
            sp.columns = [col[0] for col in sp.columns]

        # 確保收盤價是一維序列
        close = df['Close'].squeeze() if 'Close' in df.columns else df.iloc[:, 3].squeeze()
        
        # 填充外部指數資料
        df['TWII_Close'] = twii['Close'].reindex(df.index, method='ffill').fillna(method='bfill')
        df['SP500_Close'] = sp['Close'].reindex(df.index, method='ffill').fillna(method='bfill')

        # 計算技術指標
        df['MA10'] = close.rolling(10, min_periods=1).mean()
        df['MA20'] = close.rolling(20, min_periods=1).mean()
        df['MA5'] = close.rolling(5, min_periods=1).mean()
        
        # 計算 RSI
        try:
            df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        except:
            # 簡單的 RSI 計算作為後備
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

        # 計算 MACD
        try:
            macd = ta.trend.MACD(close)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
        except:
            # 簡單的 MACD 計算作為後備
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        # 添加滯後特徵
        df['Prev_Close'] = close.shift(1)
        for i in range(1, 4):  # 減少滯後特徵數量
            df[f'Prev_Close_Lag{i}'] = close.shift(i)

        # 添加成交量特徵
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(10, min_periods=1).mean()
        else:
            df['Volume_MA'] = 0

        # 添加波動率指標
        df['Volatility'] = close.rolling(10, min_periods=1).std()
        
        # 選擇特徵
        feats = ['Prev_Close', 'MA5', 'MA10', 'MA20', 'Volume_MA', 'RSI', 'MACD', 
                'MACD_Signal', 'TWII_Close', 'SP500_Close', 'Volatility'] + \
                [f'Prev_Close_Lag{i}' for i in range(1, 4)]
        
        # 檢查缺失的特徵
        missing_feats = [f for f in feats if f not in df.columns]
        if missing_feats:
            st.error(f"缺少特徵: {missing_feats}")
            return None, None, None

        # 移除有缺失值的行
        df_clean = df[feats + ['Close']].dropna()
        
        if len(df_clean) < 30:
            st.error(f"清理後資料不足，僅有 {len(df_clean)} 行數據")
            return None, None, None

        # 準備訓練數據
        X = df_clean[feats].values
        y = df_clean['Close'].values
        
        # 標準化特徵
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # 避免除以零
        X_normalized = (X - X_mean) / X_std

        # 計算時間權重
        weights = np.exp(-decay_factor * np.arange(len(X))[::-1])
        weights = weights / np.sum(weights)

        # 分割訓練和驗證集
        split_idx = int(len(X_normalized) * 0.8)
        X_train, X_val = X_normalized[:split_idx], X_normalized[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        train_weights = weights[:split_idx]

        # 訓練模型
        model = RandomForestRegressor(
            n_estimators=100,  # 減少樹的數量以提高速度
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train, sample_weight=train_weights)

        # 預測未來 5 天
        last_features = X_normalized[-1:].copy()
        last_close = float(y[-1])
        predictions = {}
        
        # 創建未來日期
        future_dates = []
        current_date = end
        for i in range(5):
            current_date = current_date + pd.offsets.BDay(1)
            future_dates.append(current_date.date())

        # 逐步預測
        current_features = last_features.copy()
        for i, date in enumerate(future_dates):
            pred = model.predict(current_features)[0]
            predictions[date] = float(pred)
            
            # 更新特徵用於下一次預測（簡化處理）
            if i < 4:  # 不是最後一次預測
                # 這裡可以更新一些特徵，如移動平均等
                pass

        # 計算預測字典
        preds = {f'T+{i+1}': pred for i, pred in enumerate(predictions.values())}

        # 驗證模型
        if len(X_val) > 0:
            y_pred_val = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            st.info(f"模型驗證 - RMSE: {rmse:.2f} (約 {rmse/last_close*100:.1f}%)")
        
        return last_close, predictions, preds

    except Exception as e:
        st.error(f"預測過程發生錯誤: {str(e)}")
        return None, None, None

def get_trade_advice(last, preds):
    """根據預測結果給出交易建議"""
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

# Streamlit 介面
st.title("📈 5 日股價預測系統")
st.markdown("---")

# 輸入區域
col1, col2 = st.columns([2, 1])
with col1:
    code = st.text_input("請輸入股票代號", "2330.TW", help="例如：2330.TW (台積電)、AAPL (蘋果)")

with col2:
    mode = st.selectbox("預測模式", ["中期模式", "短期模式", "長期模式"])

# 模式說明
mode_info = {
    "短期模式": ("使用 100 天歷史資料，高敏感度", 100, 0.008),
    "中期模式": ("使用 200 天歷史資料，平衡敏感度", 200, 0.005),
    "長期模式": ("使用 400 天歷史資料，低敏感度", 400, 0.002)
}

st.info(f"**{mode}**: {mode_info[mode][0]}")
days, decay_factor = mode_info[mode][1], mode_info[mode][2]

if st.button("🔮 開始預測", type="primary"):
    with st.spinner("正在下載資料並進行預測..."):
        last, forecast, preds = predict_next_5(code.strip().upper(), days, decay_factor)
    
    if last is None:
        st.error("❌ 預測失敗，請檢查股票代號或網路連線")
    else:
        # 顯示結果
        st.success("✅ 預測完成！")
        
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
        
        # 繪製趨勢圖
        st.subheader("📈 預測趨勢")
        chart_data = pd.DataFrame({
            '日期': ['今日'] + list(forecast.keys()),
            '股價': [last] + list(forecast.values())
        })
        st.line_chart(chart_data.set_index('日期'))

st.markdown("---")
st.caption("⚠️ 此預測僅供參考，投資有風險，請謹慎決策")
