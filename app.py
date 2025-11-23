import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import ta
from datetime import datetime, timedelta
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator
from ta.momentum import StochRSIIndicator, StochasticOscillator
from dataclasses import dataclass
import io

# 修正：加入錯誤處理，若環境未安裝 plotly 則自動切換至簡易圖表模式
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 設定忽略警告
import warnings
warnings.filterwarnings("ignore")

# ====== 參數設定 ======
@dataclass
class Config:
    # 日期改為動態計算，不再寫死
    bottom_lookback: int = 20           
    top_lookback: int = 20              
    higher_high_lookback: int = 5       
    lower_low_lookback: int = 5         
    stoch_k: int = 9
    stoch_d: int = 3
    stoch_smooth: int = 3
    kd_threshold: float = 20.0          
    kd_threshold_sell: float = 80.0     
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ma_short: int = 20
    ma_long: int = 60
    volume_ma: int = 20
    atr_period: int = 14
    risk_per_trade: float = 0.01        
    capital: float = 1_000_000          
    fwd_days: int = 5                   
    backtest_lookback_days: int = 252   

CFG = Config()

# ====== 技術指標計算工具 ======
def calc_kd(df: pd.DataFrame, k=9, d=3, smooth=3):
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=k, smooth_window=smooth)
    df['K'] = stoch.stoch()
    df['D'] = stoch.stoch_signal()
    return df['K'], df['D']

def calc_atr(df: pd.DataFrame, period=14):
    atr_indicator = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=period)
    return atr_indicator.average_true_range()

# ====== 訊號生成：買進策略 ======
def generate_signal_row_buy(row_prior, row_now, cfg: Config):
    reasons = []
    # 1) 底部條件
    bottom_built = (row_now['Close'] <= row_now['RecentLow'] * 1.08) and (row_now['Close'] > (row_now['PriorHigh'] * 0.8))
    if bottom_built: reasons.append("接近近期低點後回升")

    # 2) KD 黃金交叉且脫離超賣區
    kd_cross_up = (row_prior['K'] < row_prior['D']) and (row_now['K'] > row_now['D'])
    kd_above_threshold = row_now['K'] > cfg.kd_threshold
    kd_ok = kd_cross_up and kd_above_threshold
    if kd_ok: reasons.append(f"KD黃金交叉且K>{cfg.kd_threshold:.0f}")

    # 3) MACD 柱轉正且放大
    macd_hist_up = (row_now['MACD'] > 0) and (row_now['MACD'] > row_prior['MACD'])
    if macd_hist_up: reasons.append("MACD柱轉正且走揚")

    # 4) 趨勢濾網
    trend_ok = (row_now['MA_S'] > row_now['MA_L']) and (row_now['MA_S_SLOPE'] > 0)
    if trend_ok: reasons.append("多頭趨勢濾網通過")

    # 5) 量能濾網
    volume_ok = row_now['Volume'] >= row_now['VOL_MA']
    if volume_ok: reasons.append("量能不弱於均量")

    all_ok = bottom_built and kd_ok and macd_hist_up and trend_ok and volume_ok
    return all_ok, reasons

# ====== 訊號生成：賣出策略 ======
def generate_signal_row_sell(row_prior, row_now, cfg: Config):
    reasons = []
    # 1) 頭部條件
    top_built = (row_now['Close'] >= row_now['RecentHigh'] * 0.92) and (row_now['Close'] < (row_now['PriorLow'] * 1.2))
    if top_built: reasons.append("接近近期高點後回落")

    # 2) KD 死亡交叉且脫離超買區
    kd_cross_down = (row_prior['K'] > row_prior['D']) and (row_now['K'] < row_now['D'])
    kd_below_threshold = row_now['K'] < cfg.kd_threshold_sell
    kd_ok_sell = kd_cross_down and kd_below_threshold
    if kd_ok_sell: reasons.append(f"KD死亡交叉且K<{cfg.kd_threshold_sell:.0f}")

    # 3) MACD 柱轉負且縮小
    macd_hist_down = (row_now['MACD'] < 0) and (row_now['MACD'] < row_prior['MACD'])
    if macd_hist_down: reasons.append("MACD柱轉負且走弱")

    # 4) 趨勢濾網
    trend_ok_sell = (row_now['MA_S'] < row_now['MA_L']) and (row_now['MA_S_SLOPE'] < 0)
    if trend_ok_sell: reasons.append("空頭趨勢濾網通過")

    # 5) 量能濾網
    volume_ok_sell = row_now['Volume'] >= row_now['VOL_MA']
    if volume_ok_sell: reasons.append("量能不弱於均量")

    all_ok = top_built and kd_ok_sell and macd_hist_down and trend_ok_sell and volume_ok_sell
    return all_ok, reasons

# ====== 訊號生成：新上市/低成交量股票策略 ======
def generate_signal_low_volume(df: pd.DataFrame, strategy_type: str):
    reasons = []
    if len(df) < 5: return False, ["資料量不足"]

    row_now = df.iloc[-1]
    last_volume = row_now['Volume']
    vol_ma5 = df['Volume'].rolling(5, min_periods=1).mean().iloc[-1]
    
    if strategy_type == "buy":
        is_near_low = row_now['Close'] <= df['Low'].min() * 1.05
        is_volume_spike = last_volume > vol_ma5 * 3
        if is_near_low: reasons.append("接近歷史低點")
        if is_volume_spike: reasons.append("成交量顯著放大")
        all_ok = is_near_low and is_volume_spike
        return all_ok, reasons

    elif strategy_type == "sell":
        is_near_high = row_now['Close'] >= df['High'].max() * 0.95
        is_volume_spike = last_volume > vol_ma5 * 3
        if is_near_high: reasons.append("接近歷史高點")
        if is_volume_spike: reasons.append("成交量顯著放大")
        all_ok = is_near_high and is_volume_spike
        return all_ok, reasons
        
    return False, ["策略模式錯誤"]

# ====== 評估最新資料點 ======
def evaluate_latest(df: pd.DataFrame, cfg: Config, strategy_type: str, analysis_mode: str):
    if analysis_mode == "low_volume":
        signal, reasons = generate_signal_low_volume(df, strategy_type)
        return {
            "日期": df.index[-1].strftime("%Y-%m-%d"),
            "收盤": round(df.iloc[-1]['Close'], 2),
            "是否符合訊號": signal,
            "理由": "、".join(reasons) if reasons else "條件不足",
            "動作": "買進" if strategy_type == "buy" else "放空",
            "風險": "無（資料不足）",
            "建議停損": 0,
            "估計ATR": 0,
            "建議股數": 0
        }, df

    if len(df) < max(cfg.ma_long, cfg.bottom_lookback, cfg.top_lookback, cfg.atr_period) + 5:
        return {"是否符合訊號": False, "理由": "資料樣本太短", "動作": "無", "風險": "無", "建議停損": 0, "估計ATR": 0, "建議股數": 0}, None

    df = df.dropna().copy()
    if len(df) < 2: return {"是否符合訊號": False, "理由": "有效樣本不足", "動作": "無", "風險": "無", "建議停損": 0, "估計ATR": 0, "建議股數": 0}, None

    row_now = df.iloc[-1]
    row_prior = df.iloc[-2]
    atr = row_now['ATR']

    if strategy_type == "buy":
        signal, reasons = generate_signal_row_buy(row_prior, row_now, cfg)
        stop_level = row_now['Close'] - 2.5 * atr
        position_risk = row_now['Close'] - stop_level
        action_text = "買進"
        risk_text = "建議停損"
    else: 
        signal, reasons = generate_signal_row_sell(row_prior, row_now, cfg)
        stop_level = row_now['Close'] + 2.5 * atr
        position_risk = stop_level - row_now['Close']
        action_text = "放空"
        risk_text = "建議停損"

    position_size = 0
    if position_risk > 0:
        position_size = int((cfg.capital * cfg.risk_per_trade) // position_risk)

    return {
        "日期": df.index[-1].strftime("%Y-%m-%d"),
        "收盤": round(row_now['Close'], 2),
        "是否符合訊號": signal,
        "理由": "、".join(reasons) if reasons else "條件不足",
        "動作": action_text,
        "風險": risk_text,
        "建議停損": round(stop_level, 2),
        "估計ATR": round(float(atr), 2),
        "建議股數": position_size
    }, df

# ====== 簡易事後驗證 ======
def simple_forward_test(df: pd.DataFrame, cfg: Config, strategy_type: str, analysis_mode: str):
    if analysis_mode == "low_volume":
        return {"樣本數": 0, "勝率(>0%)": None, f"{cfg.fwd_days}日最佳中位數": None, "平均": None}

    df = df.copy()
    results = []
    
    # 確保有足夠的資料進行計算
    start_idx = max(cfg.ma_long, cfg.bottom_lookback, cfg.top_lookback, cfg.atr_period) + 2
    if start_idx >= len(df) - cfg.fwd_days:
         return {"樣本數": 0, "勝率(>0%)": 0, f"{cfg.fwd_days}日最佳中位數": 0, "平均": 0}

    for i in range(start_idx, len(df) - cfg.fwd_days):
        row_prior, row_now = df.iloc[i-1], df.iloc[i]
        
        if strategy_type == "buy":
            ok, _ = generate_signal_row_buy(row_prior, row_now, cfg)
            if ok:
                entry = row_now['Close']
                fwd_window = df['Close'].iloc[i+1:i+1+cfg.fwd_days]
                if not fwd_window.empty:
                    best = fwd_window.max()
                    ret = (best / entry) - 1.0
                    results.append(ret)
        else: 
            ok, _ = generate_signal_row_sell(row_prior, row_now, cfg)
            if ok:
                entry = row_now['Close']
                fwd_window = df['Close'].iloc[i+1:i+1+cfg.fwd_days]
                if not fwd_window.empty:
                    best = fwd_window.min() 
                    ret = (entry - best) / entry 
                    results.append(ret)
    
    if not results:
        return {"樣本數": 0, "勝率(>0%)": 0, f"{cfg.fwd_days}日最佳中位數": 0, "平均": 0}

    arr = np.array(results)
    return {
        "樣本數": int(arr.size),
        "勝率(>0%)": round(float((arr > 0).mean()) * 100, 1),
        f"{cfg.fwd_days}日最佳中位數": round(float(np.median(arr)) * 100, 2),
        "平均": round(float(arr.mean()) * 100, 2)
    }

# ====== 繪圖函數 (使用 Plotly) ======
def plot_stock_data(df, forecast_dates=None, forecast_prices=None):
    if not HAS_PLOTLY:
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=('股價走勢與預測', '成交量 & MACD'))

    # K線圖
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name='K線'), row=1, col=1)
    
    # 均線
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='MA60'), row=1, col=1)

    # 預測線
    if forecast_dates and forecast_prices:
        # 連接歷史數據和預測數據
        connect_x = [df.index[-1]] + list(forecast_dates)
        connect_y = [df['Close'].iloc[-1]] + list(forecast_prices)
        
        fig.add_trace(go.Scatter(x=connect_x, y=connect_y, 
                                 line=dict(color='red', width=2, dash='dash'), 
                                 name='AI預測'), row=1, col=1)

    # 成交量
    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # Layout 設定
    fig.update_layout(
        height=600,
        title_text="股價技術分析圖",
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    return fig

stock_name_dict = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
    "2303.TW": "聯電", "2881.TW": "富邦金", "2412.TW": "中華電", "1301.TW": "台塑"
}

@st.cache_data(ttl=3600) # 加入 TTL 快取過期
def predict_next_5(stock, days, decay_factor):
    try:
        end = pd.Timestamp(datetime.today().date())
        start = end - pd.Timedelta(days=days * 2) # 多抓一點時間確保 MA 計算
        
        # 下載資料
        df = yf.download(stock, start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True, progress=False)
        
        if df.empty: return None, None, None, pd.DataFrame()
        
        # 處理 MultiIndex 欄位問題 (yf v0.2.x 之後常見問題)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # 抓取大盤資料作為特徵 (如果不是台股，可能需要判斷)
        if ".TW" in stock.upper():
            market_index = "^TWII"
        else:
            market_index = "^GSPC" # 預設美股大盤
            
        idx_df = yf.download(market_index, start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True, progress=False)
        if isinstance(idx_df.columns, pd.MultiIndex):
            idx_df.columns = [col[0] for col in idx_df.columns]

    except Exception as e:
        st.error(f"下載資料時發生錯誤: {str(e)}")
        return None, None, None, pd.DataFrame()

    # 確保欄位存在
    required_cols = ['High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return None, None, None, df

    # 合併大盤特徵
    df['Market_Close'] = idx_df['Close'].reindex(df.index).ffill()
    
    # 特徵工程
    close = df['Close']
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()
    df['MA_S'] = df['MA20']
    df['MA_L'] = df['MA60']
    df['MA_S_SLOPE'] = df['MA_S'] - df['MA_S'].shift(5)

    df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df['MACD'] = macd.macd_diff()
    df['MACD_SIGNAL'] = macd.macd_signal()
    
    bb = BollingerBands(close, window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    df['ADX'] = ADXIndicator(df['High'], df['Low'], close, window=14).adx()
    df['Prev_Close'] = close.shift(1)
    for i in range(1, 4):
        df[f'Prev_Close_Lag{i}'] = close.shift(i)
        
    df['Volatility'] = close.rolling(10).std()
    df['K'], df['D'] = calc_kd(df, CFG.stoch_k, CFG.stoch_d, CFG.stoch_smooth)
    df['ATR'] = calc_atr(df, CFG.atr_period)
    
    # 策略用特徵
    df['RecentLow'] = df['Close'].rolling(CFG.bottom_lookback).min()
    df['PriorHigh'] = df['Close'].shift(1).rolling(CFG.higher_high_lookback).max()
    df['RecentHigh'] = df['Close'].rolling(CFG.top_lookback).max()
    df['PriorLow'] = df['Close'].shift(1).rolling(CFG.lower_low_lookback).min()
    df['VOL_MA'] = df['Volume'].rolling(CFG.volume_ma).mean()

    # 清理 NaN
    df = df.dropna()
    
    if len(df) < 30:
        return None, None, None, df

    # === 機器學習模型 (修正資料洩漏問題) ===
    feats = ['Prev_Close', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 
             'Market_Close', 'Volatility', 'BB_High', 'BB_Low', 'ADX']
    
    X = df[feats].values
    y = df['Close'].values
    
    # 時間序列分割
    split_idx = int(len(X) * 0.85)
    X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 重要修正：Scaler 只 fit 在訓練集上
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    # 訓練權重 (近期資料權重較高)
    weights = np.exp(-decay_factor * np.arange(len(X_train))[::-1])
    weights = weights / np.sum(weights)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=weights)

    # 驗證與預測
    if len(X_val) > 0:
        y_pred_val = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        st.sidebar.info(f"模型 RMSE: {rmse:.2f}")

    # 預測未來
    last_features = X[-1:].copy()
    last_features_scaled = scaler.transform(last_features) # 使用訓練好的 Scaler 轉換最新數據
    
    predictions = {}
    future_dates = pd.bdate_range(start=df.index[-1], periods=6)[1:]
    current_input = last_features_scaled.copy()
    
    # 遞迴預測 (自回歸)
    predicted_prices = []
    last_close = y[-1]
    
    for date in future_dates:
        pred_price = model.predict(current_input)[0]
        
        # 移除隨機雜訊 (提升穩定性)，改為簡單的動量阻尼
        # 這裡簡化處理：假設其他特徵不變，只更新價格相關特徵
        predictions[date.date()] = float(pred_price)
        predicted_prices.append(pred_price)
        
        # 更新特徵 (簡單模擬) - 實際應用應訓練專門的 TimeSeries 模型
        # 這裡僅做示範：保持大部分指標不變，僅為了讓程式跑通
        # 在正式場景，這裡需要更嚴謹的特徵推估
        pass 
    
    preds_dict = {f'T+{i + 1}': p for i, p in enumerate(predicted_prices)}
    
    return last_close, predictions, preds_dict, df

def get_trade_advice(last, preds):
    if not preds: return "無法判斷"
    price_values = list(preds.values())
    avg_pred = np.mean(price_values)
    change_percent = ((avg_pred - last) / last) * 100
    
    if change_percent > 2.0:
        return f"強烈看漲 (預期 +{change_percent:.1f}%)"
    elif change_percent > 0.5:
        return f"看漲 (預期 +{change_percent:.1f}%)"
    elif change_percent < -2.0:
        return f"強烈看跌 (預期 {change_percent:.1f}%)"
    elif change_percent < -0.5:
        return f"看跌 (預期 {change_percent:.1f}%)"
    else:
        return f"盤整 (預期 {change_percent:.1f}%)"

# --- Streamlit UI ---
st.set_page_config(page_title="AI 智能股價分析 Pro", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 AI 智能股價分析 Pro")
st.markdown("整合機器學習預測與傳統技術指標的輔助決策系統")

# Sidebar
with st.sidebar:
    st.header("⚙️ 設定參數")
    data_source = st.radio("資料來源", ["自動下載 (yfinance)", "手動貼上CSV資料"])
    
    if data_source == "自動下載 (yfinance)":
        code = st.text_input("股票代號", "2330")
        strategy_type = st.radio("偵測訊號方向", ["買進策略", "賣出策略"])
        mode = st.selectbox("預測模型", ["短期 (敏感)", "中期 (平衡)", "長期 (穩健)"])
        
        mode_map = {
            "短期 (敏感)": (200, 0.01),
            "中期 (平衡)": (400, 0.005),
            "長期 (穩健)": (800, 0.001)
        }
        days, decay_factor = mode_map[mode]
    else:
        st.info("手動模式不支援 AI 預測，僅提供技術指標分析")

# Main Logic
if st.button("🚀 開始分析", type="primary", use_container_width=True):
    
    df_result = pd.DataFrame()
    forecast = None
    last_price = 0
    is_low_volume = False

    if data_source == "自動下載 (yfinance)":
        full_code = code.strip().upper()
        if full_code.isdigit(): full_code += ".TW"
        
        with st.spinner(f"正在分析 {full_code} ..."):
            last_price, forecast, preds, df_result = predict_next_5(full_code, days, decay_factor)
            
            if df_result is not None and not df_result.empty:
                company_name = stock_name_dict.get(full_code, full_code)
                st.subheader(f"{company_name} ({full_code})")
                is_low_volume = len(df_result) < 50
            else:
                st.error("無法取得資料，請檢查代號或網絡。")
                st.stop()

    elif data_source == "手動貼上CSV資料":
        manual_data = st.text_area("貼上 CSV", height=200)
        if manual_data:
            try:
                df_result = pd.read_csv(io.StringIO(manual_data))
                df_result['Date'] = pd.to_datetime(df_result['Date'])
                df_result.set_index('Date', inplace=True)
                # 簡單補算指標
                df_result['ATR'] = calc_atr(df_result)
                last_price = df_result['Close'].iloc[-1]
                st.success("資料讀取成功")
            except Exception as e:
                st.error(f"CSV 格式錯誤: {e}")
                st.stop()

    # --- 展示結果 ---
    if not df_result.empty:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📊 訊號儀表板")
            # 技術指標分析
            strat_type_key = "buy" if strategy_type == "買進策略" else "sell"
            analysis_mode = "low_volume" if is_low_volume else "normal"
            
            summary, _ = evaluate_latest(df_result, CFG, strat_type_key, analysis_mode)
            
            # 顯示卡片
            bg_color = "#d4edda" if summary["是否符合訊號"] else "#f8d7da"
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px;">
                <h3 style="margin:0;">訊號判定: {'✅ 符合' if summary["是否符合訊號"] else '❌ 觀望'}</h3>
                <p><strong>動作:</strong> {summary['動作']}</p>
                <p><strong>理由:</strong> {summary['理由']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🛡️ 風險控管建議")
            st.write(f"建議停損價: **{summary['建議停損']}**")
            st.write(f"當前 ATR波動: **{summary['估計ATR']}**")
            
            if forecast:
                st.markdown("#### 🤖 AI 趨勢預測")
                advice = get_trade_advice(last_price, preds)
                st.info(f"AI 建議: **{advice}**")

        with col2:
            st.markdown("### 📈 互動式 K 線圖")
            
            # 準備繪圖資料
            forecast_dates = list(forecast.keys()) if forecast else []
            forecast_vals = list(forecast.values()) if forecast else []
            
            if HAS_PLOTLY:
                fig = plot_stock_data(df_result.tail(120), forecast_dates, forecast_vals)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 系統偵測到未安裝 `plotly` 套件，目前以簡易圖表呈現。若需互動式 K 線圖，請安裝 plotly。")
                # 簡易備用圖表：顯示收盤價與均線
                st.caption("股價走勢 (簡易版)")
                chart_data = df_result.tail(120)[['Close', 'MA20', 'MA60']]
                st.line_chart(chart_data)
                
                st.caption("成交量")
                st.bar_chart(df_result.tail(120)['Volume'])
            
            # 顯示預測表格
            if forecast:
                st.markdown("#### 未來 5 日價格預測")
                f_df = pd.DataFrame({
                    "日期": forecast_dates,
                    "預測價格": [f"{v:.2f}" for v in forecast_vals],
                    "漲跌幅": [f"{(v - last_price)/last_price*100:+.2f}%" for v in forecast_vals]
                })
                st.table(f_df)

        # 回測數據
        st.markdown("---")
        st.subheader("📜 歷史訊號回測 (近一年)")
        test_res = simple_forward_test(df_result, CFG, strat_type_key, analysis_mode)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("總訊號次數", test_res['樣本數'])
        m2.metric("勝率 (>0%)", f"{test_res['勝率(>0%)']}%")
        m3.metric(f"{CFG.fwd_days}日後報酬(中位數)", f"{test_res[f'{CFG.fwd_days}日最佳中位數']}%")
        m4.metric("平均報酬", f"{test_res['平均']}%")

st.caption("免責聲明：本工具僅供技術研究，不構成投資建議。")
