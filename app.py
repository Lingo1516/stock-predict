import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import ta

# 快取股票清單
@st.cache_data(ttl=3600)  # 快取1小時
def get_taiwan_stocks():
    """獲取台灣股市所有股票清單"""
    try:
        # 上市股票
        url_listed = 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2'
        response = requests.get(url_listed, timeout=10)
        response.encoding = 'big5'
        
        stocks = {}
        lines = response.text.split('\n')
        
        for line in lines:
            if '股票' in line and 'ESVUFR' not in line:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    code_name = parts[0].strip()
                    if '　' in code_name:
                        code, name = code_name.split('　', 1)
                        code = code.strip()
                        name = name.strip()
                        if code.isdigit() and len(code) == 4:
                            stocks[code] = name
                            stocks[name] = code
        
        # 上櫃股票
        url_otc = 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=4'
        response = requests.get(url_otc, timeout=10)
        response.encoding = 'big5'
        
        lines = response.text.split('\n')
        for line in lines:
            if '股票' in line and 'ESVUFR' not in line:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    code_name = parts[0].strip()
                    if '　' in code_name:
                        code, name = code_name.split('　', 1)
                        code = code.strip()
                        name = name.strip()
                        if code.isdigit() and len(code) == 4:
                            stocks[code] = name
                            stocks[name] = code
        
        return stocks
        
    except Exception as e:
        st.warning(f"無法獲取完整股票清單: {e}")
        # 返回常見股票的備用清單
        return {
            '2330': '台積電', '台積電': '2330',
            '2317': '鴻海', '鴻海': '2317',
            '2454': '聯發科', '聯發科': '2454',
            '2881': '富邦金', '富邦金': '2881',
            '2412': '中華電', '中華電': '2412',
            '2303': '聯電', '聯電': '2303',
            '2002': '中鋼', '中鋼': '2002',
            '1301': '台塑', '台塑': '1301',
        }

def parse_stock_input(user_input, stock_dict):
    """
    解析用戶輸入，支援多種格式：
    - 中文名稱：台積電
    - 純數字代號：2330
    - 完整代號：2330.TW
    """
    user_input = user_input.strip()
    
    # 情況1: 直接輸入中文名稱
    if user_input in stock_dict:
        code = stock_dict[user_input]
        name = user_input if not code.isdigit() else stock_dict.get(code, '未知股票')
        return f"{code}.TW", name
    
    # 情況2: 純數字代號
    if user_input.isdigit() and len(user_input) == 4:
        code = user_input
        name = stock_dict.get(code, f"股票{code}")
        return f"{code}.TW", name
    
    # 情況3: 已經包含 .TW 的代號
    if user_input.endswith('.TW') or user_input.endswith('.TWO'):
        code = user_input.split('.')[0]
        if code.isdigit():
            name = stock_dict.get(code, f"股票{code}")
            return user_input, name
    
    # 情況4: 模糊搜尋中文名稱
    for name, code in stock_dict.items():
        if not code.isdigit() and user_input in name:
            return f"{stock_dict[name]}.TW", name
    
    # 情況5: 其他格式（如美股等）
    return user_input, user_input

@st.cache_data
def predict_next_5(stock_input, days, decay_factor):
    """股價預測主函數"""
    try:
        # 獲取股票清單
        stock_dict = get_taiwan_stocks()
        
        # 解析用戶輸入
        stock_code, stock_name = parse_stock_input(stock_input, stock_dict)
        
        # 設定時間範圍
        end = pd.Timestamp(datetime.today().date())
        start = end - pd.Timedelta(days=days)

        # 下載資料
        max_retries = 3
        df, twii, sp = None, None, None
        
        for attempt in range(max_retries):
            try:
                df = yf.download(stock_code, start=start, end=end + pd.Timedelta(days=1),
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
                st.error(f"無法下載資料：{stock_code}")
                return None, None, None, None

        # 檢查資料充足性
        if df is None or len(df) < 50:
            st.error(f"資料不足，僅有 {len(df) if df is not None else 0} 行數據")
            return None, None, None, None

        # 處理多重索引
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = [col[0] for col in twii.columns]
        if isinstance(sp.columns, pd.MultiIndex):
            sp.columns = [col[0] for col in sp.columns]

        # 獲取收盤價
        close = df['Close'].squeeze() if 'Close' in df.columns else df.iloc[:, 3].squeeze()
        
        # 填充外部指數資料
        df['TWII_Close'] = twii['Close'].reindex(df.index, method='ffill').fillna(method='bfill')
        df['SP500_Close'] = sp['Close'].reindex(df.index, method='ffill').fillna(method='bfill')

        # 計算技術指標
        df['MA5'] = close.rolling(5, min_periods=1).mean()
        df['MA10'] = close.rolling(10, min_periods=1).mean()
        df['MA20'] = close.rolling(20, min_periods=1).mean()
        
        # RSI
        try:
            df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        except:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        try:
            macd = ta.trend.MACD(close)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
        except:
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        # 添加滯後特徵
        df['Prev_Close'] = close.shift(1)
        for i in range(1, 4):
            df[f'Prev_Close_Lag{i}'] = close.shift(i)

        # 成交量和波動率
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(10, min_periods=1).mean()
        else:
            df['Volume_MA'] = 0
        
        df['Volatility'] = close.rolling(10, min_periods=1).std()

        # 定義特徵
        features = ['Prev_Close', 'MA5', 'MA10', 'MA20', 'Volume_MA', 'RSI', 'MACD',
                   'MACD_Signal', 'TWII_Close', 'SP500_Close', 'Volatility'] + \
                  [f'Prev_Close_Lag{i}' for i in range(1, 4)]

        # 清理資料
        df_clean = df[features + ['Close']].dropna()
        
        if len(df_clean) < 30:
            st.error(f"清理後資料不足，僅有 {len(df_clean)} 行")
            return None, None, None, None

        # 準備訓練資料
        X = df_clean[features].values
        y = df_clean['Close'].values
        
        # 標準化
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X_normalized = (X - X_mean) / X_std

        # 時間權重
        weights = np.exp(-decay_factor * np.arange(len(X))[::-1])
        weights = weights / np.sum(weights)

        # 訓練/驗證分割
        split_idx = int(len(X_normalized) * 0.8)
        X_train, X_val = X_normalized[:split_idx], X_normalized[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        train_weights = weights[:split_idx]

        # 訓練模型
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train, sample_weight=train_weights)

        # 預測未來5天
        last_close = float(y[-1])
        current_features = X_normalized[-1:].copy()
        predictions = {}
        predicted_prices = [last_close]
        
        # 生成未來交易日
        future_dates = []
        current_date = end
        for i in range(5):
            current_date = current_date + pd.offsets.BDay(1)
            future_dates.append(current_date.date())

        # 逐步預測
        for i, date in enumerate(future_dates):
            pred = model.predict(current_features)[0]
            
            # 加入合理的隨機變化
            volatility = np.std(y[-30:]) / np.mean(y[-30:])
            variation = np.random.normal(0, pred * volatility * 0.1)
            final_pred = pred + variation
            
            predictions[date] = float(final_pred)
            predicted_prices.append(final_pred)
            
            # 更新特徵
            if i < 4:
                new_features = current_features[0].copy()
                
                # 更新價格相關特徵
                prev_close_idx = features.index('Prev_Close')
                new_features[prev_close_idx] = (final_pred - X_mean[prev_close_idx]) / X_std[prev_close_idx]
                
                # 更新滯後特徵
                for j in range(1, min(4, len(predicted_prices))):
                    if f'Prev_Close_Lag{j}' in features:
                        lag_idx = features.index(f'Prev_Close_Lag{j}')
                        lag_price = predicted_prices[-(j+1)]
                        new_features[lag_idx] = (lag_price - X_mean[lag_idx]) / X_std[lag_idx]
                
                current_features = new_features.reshape(1, -1)

        # 計算預測字典
        preds = {f'T+{i+1}': pred for i, pred in enumerate(predictions.values())}
        
        return last_close, predictions, preds, stock_name

    except Exception as e:
        st.error(f"預測過程發生錯誤: {str(e)}")
        return None, None, None, None

def get_trade_advice(last, preds):
    """交易建議"""
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
st.title("📈 股價預測系統")
st.markdown("### 支援多種輸入格式：中文名稱、純數字代號、完整代號")

# 創建三欄佈局
col1, col2, col3 = st.columns([3, 2, 1])

with col1:
    stock_input = st.text_input(
        "🔍 輸入股票", 
        "台積電", 
        help="支援格式：\n• 中文名稱：台積電、鴻海\n• 純數字：2330、2317\n• 完整代號：2330.TW\n• 美股：AAPL、TSLA"
    )

with col2:
    mode = st.selectbox("📊 預測模式", ["中期模式", "短期模式", "長期模式"])

with col3:
    st.write("")  # 空白用於對齊
    st.write("")

# 模式說明
mode_info = {
    "短期模式": ("使用 100 天歷史資料，高敏感度", 100, 0.008),
    "中期模式": ("使用 200 天歷史資料，平衡敏感度", 200, 0.005),
    "長期模式": ("使用 400 天歷史資料，低敏感度", 400, 0.002)
}

st.info(f"**{mode}**: {mode_info[mode][0]}")
days, decay_factor = mode_info[mode][1], mode_info[mode][2]

# 即時顯示解析結果
if stock_input:
    stock_dict = get_taiwan_stocks()
    parsed_code, parsed_name = parse_stock_input(stock_input.strip(), stock_dict)
    st.success(f"🎯 識別股票：**{parsed_name}** ({parsed_code})")

st.markdown("---")

if st.button("🔮 開始預測", type="primary", use_container_width=True):
    with st.spinner("📥 正在下載資料並進行預測..."):
        last, forecast, preds, stock_name = predict_next_5(stock_input.strip(), days, decay_factor)
    
    if last is None:
        st.error("❌ 預測失敗，請檢查股票輸入或網路連線")
    else:
        st.success("✅ 預測完成！")
        
        # 結果展示
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📊 當前股價", f"${last:.2f}")
            st.metric("🏷️ 股票名稱", stock_name)
            
            advice = get_trade_advice(last, preds)
            if "強烈買入" in advice or "買入" in advice:
                st.success(f"📈 **交易建議**: {advice}")
            elif "強烈賣出" in advice or "賣出" in advice:
                st.error(f"📉 **交易建議**: {advice}")
            else:
                st.warning(f"📊 **交易建議**: {advice}")
        
        with col2:
            st.subheader("📅 未來 5 日預測")
            for date, price in forecast.items():
                change = price - last
                change_pct = (change / last) * 100
                
                color = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                sign = "+" if change > 0 else ""
                st.write(f"{color} **{date}**: ${price:.2f} ({sign}{change:.2f}, {sign}{change_pct:.1f}%)")
        
        # 趨勢圖
        st.subheader("📈 預測趨勢圖")
        chart_data = pd.DataFrame({
            '日期': ['今日'] + [str(d) for d in forecast.keys()],
            '股價': [last] + list(forecast.values())
        })
        st.line_chart(chart_data.set_index('日期'), use_container_width=True)
        
        # 額外統計資訊
        with st.expander("📊 詳細分析"):
            max_price = max(forecast.values())
            min_price = min(forecast.values())
            avg_price = np.mean(list(forecast.values()))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最高預測價", f"${max_price:.2f}")
            with col2:
                st.metric("最低預測價", f"${min_price:.2f}")
            with col3:
                st.metric("平均預測價", f"${avg_price:.2f}")

st.markdown("---")
st.caption("⚠️ 此預測僅供參考，投資有風險，請謹慎決策")

# 側邊欄顯示常用股票
with st.sidebar:
    st.header("🌟 常用股票")
    popular_stocks = [
        "台積電", "鴻海", "聯發科", "富邦金", "中華電",
        "聯電", "中鋼", "台塑", "南亞", "台化"
    ]
    
    for stock in popular_stocks:
        if st.button(stock, key=f"popular_{stock}"):
            st.session_state.stock_input = stock
            st.rerun()
