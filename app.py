# --------------------------------------------------------------------------
# 核心套件載入
# --------------------------------------------------------------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import datetime as dt # 使用 dt 別名避免衝突

# FinMind 用於抓取台灣股市的三大法人與融資融券資料
from FinMind.data import DataLoader


# --------------------------------------------------------------------------
# 初始設定與資料字典
# --------------------------------------------------------------------------

# 股票代號到中文名稱簡易對照字典，可自行擴充
stock_name_dict = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2412.TW": "中華電",
    "2454.TW": "聯發科",
    "2603.TW": "長榮",
    "2881.TW": "富邦金",
    "2882.TW": "國泰金",
}

# --------------------------------------------------------------------------
# 功能函式定義
# --------------------------------------------------------------------------

def calculate_technical_indicators(df, twii_close):
    """計算各種技術指標"""
    df['MA5'] = df['Close'].rolling(5, min_periods=1).mean()
    df['MA10'] = df['Close'].rolling(10, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()

    # RSI (相對強弱指數)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (指數平滑異同移動平均線)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 布林通道 (Bollinger Bands)
    bb_window = 20
    middle_band = df['Close'].rolling(window=bb_window).mean()
    std_dev = df['Close'].rolling(window=bb_window).std()
    df['BB_High'] = middle_band + (std_dev * 2)
    df['BB_Low'] = middle_band - (std_dev * 2)
    
    # ATR (平均真實波幅)
    df['TR'] = np.maximum.reduce([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ])
    df['ATR'] = df['TR'].ewm(span=14, adjust=False).mean()
    
    # OBV (能量潮指標)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # 將大盤指數加入
    df['TWII_Close'] = twii_close.reindex(df.index, method='ffill').fillna(method='bfill')
    
    return df

def generate_mock_institutional_data(df):
    """模擬生成三大法人買賣超數據 (僅作為模型特徵，非真實顯示數據)"""
    institutional_data = pd.DataFrame(index=df.index)
    np.random.seed(42)
    institutional_data['net_buy_sell'] = np.random.uniform(-5000, 5000, len(df))
    institutional_data['5d_cum'] = institutional_data['net_buy_sell'].rolling(window=5).sum()
    twii_change = df['TWII_Close'].pct_change()
    institutional_data['net_buy_sell'] = institutional_data['net_buy_sell'] * (1 + twii_change.fillna(0) * 5)
    return institutional_data

@st.cache_data(ttl=600) # 快取10分鐘，避免重複下載
def get_market_data(stock, start_date, end_date):
    """下載股票與指數資料"""
    df = yf.download(stock, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    twii = yf.download("^TWII", start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    sp = yf.download("^GSPC", start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)

    # 處理 yfinance 可能回傳 MultiIndex 欄位的問題，將其 "壓平"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if isinstance(twii.columns, pd.MultiIndex):
        twii.columns = twii.columns.droplevel(1)
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = sp.columns.droplevel(1)

    if df.empty or twii.empty or sp.empty:
        return None, None, None
        
    return df, twii, sp

@st.cache_data(ttl=1800) # 快取30分鐘
def predict_next_5(stock, days, decay_factor):
    """主預測函式"""
    try:
        end = pd.Timestamp(dt.date.today()) + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=days)
        
        df, twii, sp = get_market_data(stock, start, end)

        if df is None or len(df) < 50:
            st.error(f"資料不足 (僅 {len(df) if df is not None else 0} 筆)，無法進行有效預測。")
            return None, None

        df = calculate_technical_indicators(df, twii['Close'])
        df['SP500_Close'] = sp['Close'].reindex(df.index, method='ffill').fillna(method='bfill')
        
        institutional_data = generate_mock_institutional_data(df)
        df = df.join(institutional_data, how='left')

        df['Prev_Close'] = df['Close'].shift(1)
        df['Volume_MA'] = df['Volume'].rolling(10, min_periods=1).mean()
        df['Volatility'] = df['Close'].rolling(10, min_periods=1).std()

        feats = [
            'Prev_Close', 'MA5', 'MA10', 'MA20', 'Volume_MA', 'RSI', 'MACD',
            'MACD_Signal', 'TWII_Close', 'SP500_Close', 'Volatility', 'BB_High',
            'BB_Low', 'ATR', 'OBV', 'net_buy_sell', '5d_cum'
        ]
        
        df_clean = df[feats + ['Close']].fillna(method='ffill').fillna(0)
        
        X = df_clean[feats].values
        y = df_clean['Close'].values

        weights = np.exp(-decay_factor * np.arange(len(X))[::-1])
        weights /= np.sum(weights)

        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X, y, sample_weight=weights)

        last_features = df_clean[feats].iloc[-1:].values
        last_close = float(y[-1])
        
        predictions = {}
        current_features = last_features.copy()
        future_dates = pd.bdate_range(start=df.index[-1], periods=6)[1:]

        for i in range(5):
            pred = model.predict(current_features)[0]
            predictions[future_dates[i].date()] = float(pred)
            
            # 迭代更新下一天的特徵 (簡易版)
            current_features[0][feats.index('Prev_Close')] = pred
        
        return last_close, predictions

    except Exception as e:
        st.error(f"預測過程發生錯誤: {str(e)}")
        return None, None

@st.cache_data(ttl=3600) # 快取1小時
def get_institutional_data(stock_code):
    """使用 FinMind API 抓取最新的三大法人與融資融券資料"""
    try:
        api = DataLoader()
        today_str = dt.datetime.now().strftime("%Y-%m-%d")
        start_str = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y-%m-%d")
        stock_id = stock_code.replace(".TW", "")

        df_institutional = api.taiwan_stock_institutional_investors(
            stock_id=stock_id, start_date=start_str, end_date=today_str
        )
        df_margin = api.taiwan_stock_margin_purchase_short_sale(
            stock_id=stock_id, start_date=start_str, end_date=today_str
        )

        if df_institutional.empty: 
            return None, None
        
        latest_institutional = df_institutional.iloc[-1]
        latest_margin = df_margin.iloc[-1] if not df_margin.empty else None
        
        return latest_institutional, latest_margin
    except Exception as e:
        st.warning(f"抓取籌碼資料時發生錯誤: {e}。可能是API請求次數達到上限或FinMind服務暫時中斷。")
        return None, None

# --------------------------------------------------------------------------
# Streamlit App 介面佈局
# --------------------------------------------------------------------------

st.set_page_config(page_title="AI 股價預測系統", layout="wide")
st.title("📈 AI 智慧股價預測系統")
st.markdown("---")

# --- 輸入區域 ---
col1, col2 = st.columns([2, 1])
with col1:
    code = st.text_input("請輸入股票代號 (例如 2330)", "2330")
with col2:
    mode = st.selectbox("預測模式", ["中期模式 (推薦)", "短期模式", "長期模式"])

mode_info = {
    "短期模式": ("使用100天歷史資料，對短期波動較敏感", 100, 0.008),
    "中期模式 (推薦)": ("使用200天歷史資料，兼顧趨勢與近期變化", 200, 0.005),
    "長期模式": ("使用400天歷史資料，更側重長期趨勢", 400, 0.002)
}
st.info(f"**{mode}**: {mode_info[mode][0]}")
days, decay_factor = mode_info[mode][1], mode_info[mode][2]

if st.button("🔮 開始預測", type="primary", use_container_width=True):
    full_code = code.strip()
    if not full_code.upper().endswith(".TW"):
        full_code = f"{full_code}.TW"
    
    with st.spinner("🤖 AI 模型正在分析與運算..."):
        last_close, forecast = predict_next_5(full_code, days, decay_factor)

    if last_close is None:
        st.error("❌ 預測失敗，請檢查股票代號或網路連線。")
    else:
        st.success("✅ 預測完成！")
        
        company_name = stock_name_dict.get(full_code, "未知的公司")
        
        # --- 顯示預測結果 ---
        main_col1, main_col2 = st.columns([1, 1])
        with main_col1:
            st.header(f"{company_name} ({full_code})")
            st.metric("最新收盤價", f"${last_close:,.2f}")
            
            st.subheader("📅 未來 5 日股價預測")
            if forecast:
                forecast_df = pd.DataFrame(list(forecast.items()), columns=['日期', '預測股價'])
                forecast_df['漲跌'] = forecast_df['預測股價'] - last_close
                forecast_df['漲跌幅 (%)'] = (forecast_df['漲跌'] / last_close) * 100
                
                def color_change(val):
                    color = 'red' if val > 0 else 'green' if val < 0 else 'gray'
                    return f'color: {color}'
                
                st.dataframe(forecast_df.style.format({
                    '預測股價': '${:,.2f}',
                    '漲跌': '{:+.2f}',
                    '漲跌幅 (%)': '{:+.2f}%'
                }).apply(lambda x: x.map(color_change), subset=['漲跌', '漲跌幅 (%)']), use_container_width=True)
        
        with main_col2:
            st.header("📈 預測趨勢圖")
            if forecast:
                df_for_date, _, _ = get_market_data(full_code, dt.date.today() - dt.timedelta(days=10), dt.date.today() + dt.timedelta(days=1))
                if df_for_date is not None and not df_for_date.empty:
                    latest_date = pd.to_datetime(df_for_date.index[-1].date())
                    chart_data = pd.DataFrame({
                        '日期': [latest_date] + [pd.to_datetime(d) for d in forecast.keys()],
                        '股價': [last_close] + list(forecast.values())
                    })
                    st.line_chart(chart_data.set_index('日期'))
        
        # --- 顯示籌碼資訊 ---
        st.markdown("---")
        st.header("📊 最新籌碼分佈 (盤後資料)")

        latest_institutional, latest_margin = get_institutional_data(full_code)

        if latest_institutional is not None:
            try:
                data_date = latest_institutional['date']
                st.caption(f"資料日期：{data_date}")

                # --- 修正開始：使用 FinMind 最新的欄位名稱 ---
                foreign_net = latest_institutional['Foreign_Investor_Buy_Sell']
                trust_net = latest_institutional['Investment_Trust_Buy_Sell']
                dealer_net = latest_institutional['Dealer_Buy_Sell']
                # --- 修正結束 ---
                
                total_institutional = foreign_net + trust_net + dealer_net
                
                chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
                with chip_col1:
                    st.metric("外資買賣超 (股)", f"{foreign_net:,.0f}")
                with chip_col2:
                    st.metric("投信買賣超 (股)", f"{trust_net:,.0f}")
                with chip_col3:
                    st.metric("自營商買賣超 (股)", f"{dealer_net:,.0f}")
                
                if latest_margin is not None:
                    # --- 修正開始：使用 FinMind 最新的欄位名稱 ---
                    margin_balance = latest_margin['Margin_Purchase_Balance']
                    # --- 修正結束 ---
                    with chip_col4:
                        st.metric("融資餘額 (股)", f"{margin_balance:,.0f}")
                
                if total_institutional > 0:
                    st.success(f"📈 三大法人合計： **買超 {total_institutional:,.0f} 股**")
                elif total_institutional < 0:
                    st.error(f"📉 三大法人合計： **賣超 {abs(total_institutional):,.0f} 股**")
                else:
                    st.info(f"三大法人合計： **持平**")

            except KeyError as e:
                st.error(f"顯示籌碼時發生欄位錯誤：找不到欄位 {e}。這可能是因為 FinMind API 更新了欄位名稱。")
                st.info("以下是目前 API 回傳的所有可用欄位，請根據這些資訊更新程式碼：")
                st.json(latest_institutional.index.to_list()) # 直接列出所有可用的欄位名稱

        else:
            st.warning("今日盤後籌碼資料尚未公佈，或查無該股票籌碼資料。")
            
st.markdown("---")
st.caption("⚠️ 此預測基於歷史數據與 AI 模型，僅供學術研究與參考，不構成任何投資建議。投資有風險，請謹慎決策。")
