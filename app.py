import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import ta
from datetime import datetime, timedelta, time
import pytz
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator
from ta.momentum import StochRSIIndicator, StochasticOscillator
from dataclasses import dataclass
import io
import sys

# 錯誤捕捉設定：偵測 Plotly 是否安裝
PLOTLY_ERROR = ""
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError as e:
    HAS_PLOTLY = False
    PLOTLY_ERROR = str(e)

import warnings
warnings.filterwarnings("ignore")

# ====== 參數設定 ======
@dataclass
class Config:
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

# ====== 擴充股票代碼對照表 ======
stock_name_dict = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
    "2303.TW": "聯電", "3711.TW": "日月光投控", "3034.TW": "聯詠", "2379.TW": "瑞昱",
    "3008.TW": "大立光", "2327.TW": "國巨", "2382.TW": "廣達", "3231.TW": "緯創",
    "2357.TW": "華碩", "2356.TW": "英業達", "2301.TW": "光寶科", "2412.TW": "中華電",
    "3045.TW": "台灣大", "4904.TW": "遠傳", "2345.TW": "智邦", "2368.TW": "金像電",
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2886.TW": "兆豐金",
    "2884.TW": "玉山金", "2892.TW": "第一金", "2885.TW": "元大金", "2880.TW": "華南金",
    "2883.TW": "開發金", "2890.TW": "永豐金",
    "2002.TW": "中鋼", "1301.TW": "台塑", "1303.TW": "南亞", "1326.TW": "台化",
    "6505.TW": "台塑化", "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海",
    "2618.TW": "長榮航", "2610.TW": "華航", "1101.TW": "台泥", "1102.TW": "亞泥",
    "1216.TW": "統一", "2912.TW": "統一超",
    "2376.TW": "技嘉", "2377.TW": "微星", "6669.TW": "緯穎", "3035.TW": "智原",
    "3443.TW": "創意", "3661.TW": "世芯-KY", "3017.TW": "奇鋐", "3324.TW": "雙鴻"
}

# ====== 核心功能：技術指標計算 ======
def add_technical_indicators(df: pd.DataFrame, cfg: Config):
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    
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
    
    df['ADX'] = ADXIndicator(high, low, close, window=14).adx()
    
    df['Prev_Close'] = close.shift(1)
    for i in range(1, 6): 
        df[f'Prev_Close_Lag{i}'] = close.shift(i)
        
    df['Volatility'] = close.rolling(10).std()
    
    stoch = StochasticOscillator(high=high, low=low, close=close, window=cfg.stoch_k, smooth_window=cfg.stoch_smooth)
    df['K'] = stoch.stoch()
    df['D'] = stoch.stoch_signal()
    
    atr_indicator = ta.volatility.AverageTrueRange(high, low, close, window=cfg.atr_period)
    df['ATR'] = atr_indicator.average_true_range()
    
    df['RecentLow'] = close.rolling(cfg.bottom_lookback).min()
    df['PriorHigh'] = close.shift(1).rolling(cfg.higher_high_lookback).max()
    df['RecentHigh'] = close.rolling(cfg.top_lookback).max()
    df['PriorLow'] = close.shift(1).rolling(cfg.lower_low_lookback).min()
    df['VOL_MA'] = df['Volume'].rolling(cfg.volume_ma).mean()
    
    return df

# ====== 輔助計算工具 ======
def calc_kd(df: pd.DataFrame, k=9, d=3, smooth=3):
    return df['K'], df['D']

def calc_atr(df: pd.DataFrame, period=14):
    return df['ATR']

# ====== 訊號生成邏輯 ======
def generate_signal_row_buy(row_prior, row_now, cfg: Config):
    reasons = []
    bottom_built = (row_now['Close'] <= row_now['RecentLow'] * 1.08) and (row_now['Close'] > (row_now['PriorHigh'] * 0.8))
    if bottom_built: reasons.append("接近近期低點後回升")
    kd_cross_up = (row_prior['K'] < row_prior['D']) and (row_now['K'] > row_now['D'])
    kd_above_threshold = row_now['K'] > cfg.kd_threshold
    kd_ok = kd_cross_up and kd_above_threshold
    if kd_ok: reasons.append(f"KD黃金交叉且K>{cfg.kd_threshold:.0f}")
    macd_hist_up = (row_now['MACD'] > 0) and (row_now['MACD'] > row_prior['MACD'])
    if macd_hist_up: reasons.append("MACD柱轉正且走揚")
    trend_ok = (row_now['MA_S'] > row_now['MA_L']) and (row_now['MA_S_SLOPE'] > 0)
    if trend_ok: reasons.append("多頭趨勢濾網通過")
    volume_ok = row_now['Volume'] >= row_now['VOL_MA']
    if volume_ok: reasons.append("量能不弱於均量")
    all_ok = bottom_built and kd_ok and macd_hist_up and trend_ok and volume_ok
    return all_ok, reasons

def generate_signal_row_sell(row_prior, row_now, cfg: Config):
    reasons = []
    top_built = (row_now['Close'] >= row_now['RecentHigh'] * 0.92) and (row_now['Close'] < (row_now['PriorLow'] * 1.2))
    if top_built: reasons.append("接近近期高點後回落")
    kd_cross_down = (row_prior['K'] > row_prior['D']) and (row_now['K'] < row_now['D'])
    kd_below_threshold = row_now['K'] < cfg.kd_threshold_sell
    kd_ok_sell = kd_cross_down and kd_below_threshold
    if kd_ok_sell: reasons.append(f"KD死亡交叉且K<{cfg.kd_threshold_sell:.0f}")
    macd_hist_down = (row_now['MACD'] < 0) and (row_now['MACD'] < row_prior['MACD'])
    if macd_hist_down: reasons.append("MACD柱轉負且走弱")
    trend_ok_sell = (row_now['MA_S'] < row_now['MA_L']) and (row_now['MA_S_SLOPE'] < 0)
    if trend_ok_sell: reasons.append("空頭趨勢濾網通過")
    volume_ok_sell = row_now['Volume'] >= row_now['VOL_MA']
    if volume_ok_sell: reasons.append("量能不弱於均量")
    all_ok = top_built and kd_ok_sell and macd_hist_down and trend_ok_sell and volume_ok_sell
    return all_ok, reasons

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
        action_text = "多方(買進)模式"
        risk_text = "建議停損"
    else: 
        signal, reasons = generate_signal_row_sell(row_prior, row_now, cfg)
        stop_level = row_now['Close'] + 2.5 * atr
        position_risk = stop_level - row_now['Close']
        action_text = "空方(放空)模式"
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

def simple_forward_test(df: pd.DataFrame, cfg: Config, strategy_type: str, analysis_mode: str):
    if analysis_mode == "low_volume":
        return {"樣本數": 0, "勝率(>0%)": None, f"{cfg.fwd_days}日最佳中位數": None, "平均": None}
    df = df.copy()
    results = []
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

def plot_stock_data(df, forecast_dates=None, forecast_prices=None):
    if not HAS_PLOTLY:
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=('股價走勢與預測', '成交量 & MACD'))

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name='K線'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='MA60'), row=1, col=1)

    if 'AI_Pred' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['AI_Pred'], 
                                 line=dict(color='purple', width=2, dash='dot'),
                                 name='AI 歷史軌跡 (穩定混合)'), row=1, col=1)

    if forecast_dates and forecast_prices:
        connect_x = [df.index[-1]] + list(forecast_dates)
        connect_y = [df['Close'].iloc[-1]] + list(forecast_prices)
        fig.add_trace(go.Scatter(x=connect_x, y=connect_y, 
                                 line=dict(color='red', width=3, dash='dash'), 
                                 name='AI 未來預測'), row=1, col=1)

    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    fig.update_layout(
        height=600,
        title_text="股價技術分析圖 (智慧權重穩定版)",
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    return fig

def plot_accuracy_chart(df):
    if not HAS_PLOTLY or 'AI_Pred' not in df.columns:
        return None
    
    df = df.copy()
    df['Error_Pct'] = ((df['AI_Pred'] - df['Close']) / df['Close']) * 100
    plot_df = df.tail(60)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df['Error_Pct'],
        mode='lines+markers',
        name='誤差趨勢線 (%)',
        line=dict(color='#FF4B4B', width=2),
        marker=dict(size=6, color='#FF4B4B'),
        hovertemplate='日期: %{x}<br>誤差: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_shape(type="line",
        x0=plot_df.index[0], y0=0, x1=plot_df.index[-1], y1=0,
        line=dict(color="white", width=1, dash="dash")
    )

    fig.add_hrect(
        y0=-1.5, y1=1.5,
        fillcolor="green", opacity=0.15,
        layer="below", line_width=0,
    )
    
    fig.add_annotation(
        x=plot_df.index[0], y=1.6,
        text="準確區間 (±1.5%)",
        showarrow=False,
        yshift=10,
        font=dict(color="lightgreen")
    )
    
    fig.update_layout(
        title="🎯 AI 預測誤差趨勢 (移除隨機雜訊後)",
        yaxis_title="誤差百分比 (%)",
        yaxis=dict(range=[-5, 5], showgrid=True, zeroline=False),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    return fig

@st.cache_data(ttl=3600)
def predict_next_5(stock, days, decay_factor):
    try:
        end = pd.Timestamp(datetime.today().date())
        start = end - pd.Timedelta(days=days * 2)
        
        df = yf.download(stock, start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True, progress=False)
        
        if df.empty: return None, None, None, pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        if ".TW" in stock.upper():
            market_index = "^IXIC" 
        else:
            market_index = "^GSPC"
            
        idx_df = yf.download(market_index, start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True, progress=False)
        if isinstance(idx_df.columns, pd.MultiIndex):
            idx_df.columns = [col[0] for col in idx_df.columns]

    except Exception as e:
        st.error(f"下載資料時發生錯誤: {str(e)}")
        return None, None, None, pd.DataFrame()

    required_cols = ['High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return None, None, None, df

    df['Market_Close'] = idx_df['Close'].reindex(df.index).ffill()
    
    df = add_technical_indicators(df, CFG)
    df = df.dropna()
    
    if len(df) < 30:
        return None, None, None, df

    feats = ['Prev_Close', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 
             'Market_Close', 'Volatility', 'BB_High', 'BB_Low', 'ADX']
    
    X = df[feats].values
    y = df['Close'].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)
    y_train = y
    
    weights = np.exp(-decay_factor * np.arange(len(X_train))[::-1])
    weights = weights / np.sum(weights)

    model_trend = LinearRegression()
    model_trend.fit(X_train, y_train, sample_weight=weights)
    trend_pred_train = model_trend.predict(X_train)
    y_train_resid = y_train - trend_pred_train

    np.random.seed(42)
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    model_rf.fit(X_train, y_train_resid, sample_weight=weights)

    ma20_vals = df['MA20'].values
    ma60_vals = df['MA60'].values
    adx_vals = df['ADX'].values
    
    all_inputs_scaled = scaler.transform(X)
    trend_all = model_trend.predict(all_inputs_scaled)
    resid_all = model_rf.predict(all_inputs_scaled)
    
    history_preds = []
    for i in range(len(X)):
        t_pred = trend_all[i]
        r_pred = resid_all[i]
        curr_adx = adx_vals[i]
        
        if curr_adx < 20:
            resid_weight = 1.2 
        elif curr_adx > 40:
            resid_weight = 0.5 
        else:
            resid_weight = 0.9
            
        history_preds.append(t_pred + r_pred * resid_weight)

    df['AI_Pred'] = history_preds

    simulation_df = df.tail(100).copy()
    future_dates = pd.bdate_range(start=df.index[-1], periods=6)[1:]
    
    predictions = {}
    predicted_prices = []
    last_close_real = y[-1]
    
    current_atr = simulation_df['ATR'].iloc[-1]

    for date in future_dates:
        last_row_feats = simulation_df[feats].iloc[-1:].values
        current_input_scaled = scaler.transform(last_row_feats)
        
        pred_trend = model_trend.predict(current_input_scaled)[0]
        pred_resid = model_rf.predict(current_input_scaled)[0]
        
        curr_adx = simulation_df['ADX'].iloc[-1]
        
        if curr_adx < 20:
            w_resid = 1.2
        elif curr_adx > 40:
            w_resid = 0.5
        else:
            w_resid = 0.9
            
        final_pred = pred_trend + (pred_resid * w_resid)
        
        curr_ma20 = simulation_df['MA20'].iloc[-1]
        curr_atr = simulation_df['ATR'].iloc[-1]
        
        upper_bound = curr_ma20 + 3 * curr_atr
        lower_bound = curr_ma20 - 3 * curr_atr
        
        if final_pred > upper_bound:
            final_pred = upper_bound
        elif final_pred < lower_bound:
            final_pred = lower_bound
            
        predictions[date.date()] = float(final_pred)
        predicted_prices.append(final_pred)
        
        sim_open = final_pred
        sim_high = final_pred + (curr_atr * 0.2)
        sim_low = final_pred - (curr_atr * 0.2)
        sim_vol = simulation_df['Volume'].mean()
        
        new_row = pd.DataFrame({
            'Open': [sim_open],
            'High': [sim_high],
            'Low': [sim_low],
            'Close': [final_pred],
            'Volume': [sim_vol],
            'Market_Close': [simulation_df['Market_Close'].iloc[-1]]
        }, index=[date])
        
        simulation_df = pd.concat([simulation_df, new_row])
        simulation_df = add_technical_indicators(simulation_df, CFG)
    
    preds_dict = {f'T+{i + 1}': p for i, p in enumerate(predicted_prices)}
    
    return last_close_real, predictions, preds_dict, df

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
    .suggestion-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 AI 智能股價分析 Pro")
st.markdown("整合機器學習預測與傳統技術指標的輔助決策系統")

# Session State for History
if 'recent_stocks' not in st.session_state:
    st.session_state.recent_stocks = []

# Sidebar Logic
with st.sidebar:
    st.header("⚙️ 設定參數")
    data_source = st.radio("資料來源", ["自動下載 (yfinance)", "手動貼上CSV資料"])
    
    if data_source == "自動下載 (yfinance)":
        # History Dropdown
        if st.session_state.recent_stocks:
            selected_history = st.selectbox(
                "📜 最近瀏覽紀錄", 
                ["請選擇..."] + st.session_state.recent_stocks
            )
            if selected_history != "請選擇...":
                default_code = selected_history.split(" ")[0].replace(".TW", "")
            else:
                default_code = "2330"
        else:
            default_code = "2330"

        code = st.text_input("股票代號", value=default_code)
        
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

if st.button("🚀 開始分析", type="primary", use_container_width=True):
    
    df_result = pd.DataFrame()
    forecast = None
    last_price = 0
    is_low_volume = False

    if data_source == "自動下載 (yfinance)":
        full_code = code.strip().upper()
        if full_code.isdigit(): full_code += ".TW"
        
        stock_name = stock_name_dict.get(full_code, "未知名稱")
        if stock_name == "未知名稱":
             try:
                 ticker = yf.Ticker(full_code)
                 pass
             except:
                 pass

        with st.spinner(f"正在分析 {stock_name} ({full_code}) ..."):
            last_price, forecast, preds, df_result = predict_next_5(full_code, days, decay_factor)
            
            if df_result is not None and not df_result.empty:
                history_item = f"{full_code} {stock_name}"
                if history_item not in st.session_state.recent_stocks:
                    st.session_state.recent_stocks.insert(0, history_item)
                    if len(st.session_state.recent_stocks) > 10:
                        st.session_state.recent_stocks.pop()
                
                st.subheader(f"{stock_name} ({full_code}) - 股價分析報告")
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
                df_result = add_technical_indicators(df_result, CFG)
                last_price = df_result['Close'].iloc[-1]
                st.success("資料讀取成功")
            except Exception as e:
                st.error(f"CSV 格式錯誤: {e}")
                st.stop()

    if not df_result.empty:
        # ====== 即時操盤建議專區 ======
        tz_tw = pytz.timezone('Asia/Taipei')
        now_tw = datetime.now(tz_tw)
        market_open_time = time(9, 0)
        market_close_time = time(13, 30)
        
        is_market_open = market_open_time <= now_tw.time() <= market_close_time
        # 如果是週末，也算收盤
        if now_tw.weekday() >= 5:
            is_market_open = False
            
        status_text = "🌞 開盤中 (即時數據)" if is_market_open else "🌙 已收盤 (使用昨收數據)"
        
        # 決定紅綠燈
        strat_type_key = "buy" if strategy_type == "買進策略" else "sell"
        analysis_mode = "low_volume" if is_low_volume else "normal"
        summary, _ = evaluate_latest(df_result, CFG, strat_type_key, analysis_mode)
        
        signal_color = "gray"
        signal_emoji = "🟡"
        signal_text = "觀望 (WAIT)"
        
        if summary["是否符合訊號"]:
            if summary["動作"].startswith("多方"):
                signal_color = "#d4edda" # Light Green
                signal_emoji = "🟢"
                signal_text = "買進訊號 (BUY)"
            else:
                signal_color = "#f8d7da" # Light Red
                signal_emoji = "🔴"
                signal_text = "放空訊號 (SELL)"
        else:
            signal_color = "#fff3cd" # Light Yellow
            signal_emoji = "🟡"
            signal_text = "觀望 / 空手 (WAIT)"

        st.markdown(f"""
        <div style="background-color: {signal_color}; padding: 20px; border-radius: 15px; text-align: center; border: 2px solid #ccc;">
            <h4 style="margin:0; color: #555;">{status_text} | 資料時間: {summary['日期']}</h4>
            <h1 style="font-size: 48px; margin: 10px 0;">{signal_emoji} {signal_text}</h1>
            <p style="font-size: 18px;"><b>檢測策略模式:</b> {summary['動作']} | <b>收盤價:</b> {summary['收盤']}</p>
        </div>
        """, unsafe_allow_html=True)
        # ============================

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📊 詳細訊號數據")
            
            st.write(f"**理由**: {summary['理由']}")
            st.markdown("#### 🛡️ 風險控管建議")
            st.write(f"建議停損價: **{summary['建議停損']}**")
            st.write(f"當前 ATR波動: **{summary['估計ATR']}**")
            
            if forecast:
                st.markdown("#### 🤖 AI 趨勢預測")
                advice = get_trade_advice(last_price, preds)
                st.info(f"AI 建議: **{advice}**")

        with col2:
            st.markdown("### 📈 互動式 K 線圖 (含 AI 歷史軌跡)")
            
            forecast_dates = list(forecast.keys()) if forecast else []
            forecast_vals = list(forecast.values()) if forecast else []
            
            if HAS_PLOTLY:
                fig = plot_stock_data(df_result.tail(120), forecast_dates, forecast_vals)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠️ 系統偵測到未安裝 `plotly` 套件。")
                st.caption("股價走勢 (簡易版)")
                chart_data = df_result.tail(120)[['Close', 'MA20', 'MA60']]
                st.line_chart(chart_data)
            
            if forecast:
                st.markdown("#### 未來 5 日價格預測")
                f_df = pd.DataFrame({
                    "日期": forecast_dates,
                    "預測價格": [f"{v:.2f}" for v in forecast_vals],
                    "漲跌幅": [f"{(v - last_price)/last_price*100:+.2f}%" for v in forecast_vals]
                })
                st.table(f_df)

        st.markdown("---")
        st.subheader("🎯 AI 準確度檢測 (歷史回測)")
        
        if 'AI_Pred' in df_result.columns:
            acc_fig = plot_accuracy_chart(df_result)
            if acc_fig:
                st.plotly_chart(acc_fig, use_container_width=True)
            
            recent_df = df_result.tail(30)
            mae = np.mean(np.abs(recent_df['AI_Pred'] - recent_df['Close']))
            mape = np.mean(np.abs((recent_df['AI_Pred'] - recent_df['Close']) / recent_df['Close'])) * 100
            
            col_acc1, col_acc2 = st.columns(2)
            col_acc1.metric("近30日平均誤差 (元)", f"${mae:.2f}")
            col_acc2.metric("近30日平均誤差率 (%)", f"{mape:.2f}%", help="數值越低越準，通常 <3% 為優秀")
        else:
            st.info("需等待 AI 運算完成後才能顯示準確度分析。")

        st.markdown("---")
        st.subheader("📜 歷史訊號回測 (近一年)")
        test_res = simple_forward_test(df_result, CFG, strat_type_key, analysis_mode)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("總訊號次數", test_res['樣本數'])
        m2.metric("勝率 (>0%)", f"{test_res['勝率(>0%)']}%")
        m3.metric(f"{CFG.fwd_days}日後報酬(中位數)", f"{test_res[f'{CFG.fwd_days}日最佳中位數']}%")
        m4.metric("平均報酬", f"{test_res['平均']}%")

st.caption("免責聲明：本工具僅供技術研究，不構成投資建議。")
