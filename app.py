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
from ta.momentum import StochRSIIndicator, StochasticOscillator
from dataclasses import dataclass

# 設定忽略警告，避免不必要的輸出
import warnings
warnings.filterwarnings("ignore")

# ====== 參數設定 ======
@dataclass
class Config:
    symbol: str = "2330.TW"
    start: str = "2025-01-01"
    end: str = "2025-08-28"
    # 底部/頭部判定
    bottom_lookback: int = 20           # 買進：近期低點回看天數
    top_lookback: int = 20              # 賣出：近期高點回看天數
    higher_high_lookback: int = 5       # 買進：近期前高回看天數
    lower_low_lookback: int = 5         # 賣出：近期前低回看天數
    # KD
    stoch_k: int = 9
    stoch_d: int = 3
    stoch_smooth: int = 3
    kd_threshold: float = 20.0          # 買進：脫離超賣區
    kd_threshold_sell: float = 80.0     # 賣出：脫離超買區
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # 趨勢/量能濾網
    ma_short: int = 20
    ma_long: int = 60
    volume_ma: int = 20
    # 風險控管
    atr_period: int = 14
    risk_per_trade: float = 0.01        # 每筆風險 1% 資金
    capital: float = 1_000_000          # 假設資金
    # 事後驗證
    fwd_days: int = 5                   # 訊號後觀察天數
    backtest_lookback_days: int = 252   # 回看一年做驗證

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
    """根據單日資料與配置生成買點訊號"""
    reasons = []

    # 1) 底部條件
    bottom_built = (row_now['Close'] <= row_now['RecentLow'] * 1.08) and (row_now['Close'] > (row_now['PriorHigh'] * 0.8))
    if bottom_built:
        reasons.append("接近近期低點後回升")

    # 2) KD 黃金交叉且脫離超賣區
    kd_cross_up = (row_prior['K'] < row_prior['D']) and (row_now['K'] > row_now['D'])
    kd_above_threshold = row_now['K'] > cfg.kd_threshold
    kd_ok = kd_cross_up and kd_above_threshold
    if kd_ok:
        reasons.append(f"KD黃金交叉且K>{cfg.kd_threshold:.0f}")

    # 3) MACD 柱轉正且放大
    macd_hist_up = (row_now['MACD'] > 0) and (row_now['MACD'] > row_prior['MACD'])
    if macd_hist_up:
        reasons.append("MACD柱轉正且走揚")

    # 4) 趨勢濾網
    trend_ok = (row_now['MA_S'] > row_now['MA_L']) and (row_now['MA_S_SLOPE'] > 0)
    if trend_ok:
        reasons.append("多頭趨勢濾網通過")

    # 5) 量能濾網
    volume_ok = row_now['Volume'] >= row_now['VOL_MA']
    if volume_ok:
        reasons.append("量能不弱於均量")

    all_ok = bottom_built and kd_ok and macd_hist_up and trend_ok and volume_ok
    return all_ok, reasons

# ====== 訊號生成：賣出策略 ======
def generate_signal_row_sell(row_prior, row_now, cfg: Config):
    """根據單日資料與配置生成賣點訊號"""
    reasons = []

    # 1) 頭部條件
    top_built = (row_now['Close'] >= row_now['RecentHigh'] * 0.92) and (row_now['Close'] < (row_now['PriorLow'] * 1.2))
    if top_built:
        reasons.append("接近近期高點後回落")

    # 2) KD 死亡交叉且脫離超買區
    kd_cross_down = (row_prior['K'] > row_prior['D']) and (row_now['K'] < row_now['D'])
    kd_below_threshold = row_now['K'] < cfg.kd_threshold_sell
    kd_ok_sell = kd_cross_down and kd_below_threshold
    if kd_ok_sell:
        reasons.append(f"KD死亡交叉且K<{cfg.kd_threshold_sell:.0f}")

    # 3) MACD 柱轉負且縮小
    macd_hist_down = (row_now['MACD'] < 0) and (row_now['MACD'] < row_prior['MACD'])
    if macd_hist_down:
        reasons.append("MACD柱轉負且走弱")

    # 4) 趨勢濾網
    trend_ok_sell = (row_now['MA_S'] < row_now['MA_L']) and (row_now['MA_S_SLOPE'] < 0)
    if trend_ok_sell:
        reasons.append("空頭趨勢濾網通過")

    # 5) 量能濾網
    volume_ok_sell = row_now['Volume'] >= row_now['VOL_MA']
    if volume_ok_sell:
        reasons.append("量能不弱於均量")

    all_ok = top_built and kd_ok_sell and macd_hist_down and trend_ok_sell and volume_ok_sell
    return all_ok, reasons

# ====== 訊號生成：新上市/低成交量股票策略 ======
def generate_signal_low_volume(df: pd.DataFrame, strategy_type: str):
    """
    針對資料不足的股票，使用簡單的價量關係進行買/賣點判斷。
    買進訊號：收盤價接近歷史新低 + 量能顯著放大
    賣出訊號：收盤價接近歷史新高 + 量能顯著放大
    """
    reasons = []
    
    # 確保有足夠的數據進行基本計算
    if len(df) < 5:
        return False, ["資料量不足，無法判斷"]

    row_now = df.iloc[-1]
    last_volume = row_now['Volume']
    # 計算近 5 日均量
    vol_ma5 = df['Volume'].rolling(5, min_periods=1).mean().iloc[-1]
    
    # 買進策略
    if strategy_type == "buy":
        # 條件：收盤價接近歷史低點 (近 1.05 倍歷史低點)
        is_near_low = row_now['Close'] <= df['Low'].min() * 1.05
        # 條件：當日成交量顯著放大 (超過近 5 日均量 3 倍)
        is_volume_spike = last_volume > vol_ma5 * 3
        
        if is_near_low:
            reasons.append("接近歷史低點")
        if is_volume_spike:
            reasons.append("成交量顯著放大")
            
        all_ok = is_near_low and is_volume_spike
        return all_ok, reasons

    # 賣出策略
    elif strategy_type == "sell":
        # 條件：收盤價接近歷史高點 (近 0.95 倍歷史高點)
        is_near_high = row_now['Close'] >= df['High'].max() * 0.95
        # 條件：當日成交量顯著放大 (超過近 5 日均量 3 倍)
        is_volume_spike = last_volume > vol_ma5 * 3
        
        if is_near_high:
            reasons.append("接近歷史高點")
        if is_volume_spike:
            reasons.append("成交量顯著放大")
        
        all_ok = is_near_high and is_volume_spike
        return all_ok, reasons
        
    return False, ["策略模式錯誤"]

# ====== 評估最新資料點 ======
def evaluate_latest(df: pd.DataFrame, cfg: Config, strategy_type: str, analysis_mode: str):
    """評估最新資料點是否符合訊號，並給出風險控管建議"""
    if analysis_mode == "low_volume":
        signal, reasons = generate_signal_low_volume(df, strategy_type)
        summary = {
            "日期": df.index[-1].strftime("%Y-%m-%d"),
            "收盤": round(df.iloc[-1]['Close'], 2),
            "是否符合訊號": signal,
            "理由": "、".join(reasons) if reasons else "條件不足",
            "動作": "買進" if strategy_type == "buy" else "放空",
            "風險": "無（資料不足）",
            "建議停損": "無（資料不足）",
            "估計ATR": "無（資料不足）",
            "建議股數": "無（資料不足）"
        }
        return summary, df

    # 原始策略需要足夠資料
    if len(df) < max(cfg.ma_long, cfg.bottom_lookback, cfg.top_lookback, cfg.atr_period) + 5:
        return {"是否符合訊號": False, "理由": "資料樣本太短，無法可靠判斷。"}, None

    df = df.dropna().copy()
    if len(df) < 2:
        return {"是否符合訊號": False, "理由": "有效樣本不足以產生訊號。"}, None

    row_now = df.iloc[-1]
    row_prior = df.iloc[-2]

    if strategy_type == "buy":
        signal, reasons = generate_signal_row_buy(row_prior, row_now, cfg)
        # 風險控管：買進停損
        atr = row_now['ATR']
        stop_level = row_now['Close'] - 2.5 * atr
        position_risk = row_now['Close'] - stop_level
        action_text = "買進"
        risk_text = "建議停損"
    else: # strategy_type == "sell"
        signal, reasons = generate_signal_row_sell(row_prior, row_now, cfg)
        # 風險控管：放空停損
        atr = row_now['ATR']
        stop_level = row_now['Close'] + 2.5 * atr
        position_risk = stop_level - row_now['Close']
        action_text = "放空"
        risk_text = "建議停損"

    position_size = 0
    if position_risk > 0:
        position_size = int((cfg.capital * cfg.risk_per_trade) // position_risk)

    summary = {
        "日期": df.index[-1].strftime("%Y-%m-%d"),
        "收盤": round(row_now['Close'], 2),
        "是否符合訊號": signal,
        "理由": "、".join(reasons) if reasons else "條件不足",
        "動作": action_text,
        "風險": risk_text,
        "建議停損": round(stop_level, 2),
        "估計ATR": round(float(atr), 2),
        "建議股數": position_size
    }
    return summary, df

# ====== 簡易事後驗證 ======
def simple_forward_test(df: pd.DataFrame, cfg: Config, strategy_type: str, analysis_mode: str):
    """簡易事後驗證：訊號後觀察最佳報酬"""
    # 新上市/低成交量模式沒有回測數據
    if analysis_mode == "low_volume":
        return {"樣本數": 0, "勝率(>0%)": None, f"{cfg.fwd_days}日最佳中位數": None, "平均": None}

    df = df.copy()
    results = []
    
    start_idx = max(cfg.ma_long, cfg.bottom_lookback, cfg.top_lookback, cfg.atr_period) + 2
    
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
        else: # strategy_type == "sell"
            ok, _ = generate_signal_row_sell(row_prior, row_now, cfg)
            if ok:
                entry = row_now['Close']
                fwd_window = df['Close'].iloc[i+1:i+1+cfg.fwd_days]
                if not fwd_window.empty:
                    best = fwd_window.min() # 賣出策略觀察最低點
                    ret = (entry - best) / entry # 計算放空報酬
                    results.append(ret)
    
    if not results:
        return {"樣本數": 0, "勝率(>0%)": None, f"{cfg.fwd_days}日最佳中位數": None, "平均": None}

    arr = np.array(results)
    return {
        "樣本數": int(arr.size),
        "勝率(>0%)": round(float((arr > 0).mean()) * 100, 1),
        f"{cfg.fwd_days}日最佳中位數": round(float(np.median(arr)) * 100, 2),
        "平均": round(float(arr.mean()) * 100, 2)
    }

# 股票代號到中文名稱簡易對照字典
stock_name_dict = {
    "2330.TW": "台灣積體電路製造股份有限公司",
    "2317.TW": "鴻海精密工業股份有限公司",
    "2412.TW": "中華電信股份有限公司",
}

@st.cache_data
def predict_next_5(stock, days, decay_factor):
    # 優化下載邏輯，確保在下載失敗時不會報錯
    try:
        end = pd.Timestamp(datetime.today().date())
        start = end - pd.Timedelta(days=days)
        df = yf.download(stock, start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True, progress=False)
        twii = yf.download("^TWII", start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True, progress=False)
        sp = yf.download("^GSPC", start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True, progress=False)
        
        if df.empty or twii.empty or sp.empty:
            # 如果下載失敗，直接返回空的資料
            return None, None, None, pd.DataFrame()
            
    except Exception as e:
        st.error(f"下載資料時發生錯誤: {str(e)}")
        return None, None, None, pd.DataFrame()

    # 處理欄位名稱
    for frame in [df, twii, sp]:
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [col[0] for col in frame.columns]

    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        st.error("下載的資料缺少 'High', 'Low', 或 'Close' 欄位。")
        return None, None, None, df

    close = df['Close']
    df['TWII_Close'] = twii['Close'].reindex(df.index, method='ffill').fillna(method='bfill')
    df['SP500_Close'] = sp['Close'].reindex(df.index, method='ffill').fillna(method='bfill')

    # 計算技術指標
    df['MA5'] = close.rolling(5, min_periods=1).mean()
    df['MA10'] = close.rolling(10, min_periods=1).mean()
    df['MA20'] = close.rolling(20, min_periods=1).mean()
    df['MA60'] = close.rolling(60, min_periods=1).mean()
    df['MA_S'] = df['MA20']
    df['MA_L'] = df['MA60']
    df['MA_S_SLOPE'] = df['MA_S'] - df['MA_S'].shift(5)

    df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df['MACD'] = macd.macd_diff()
    df['MACD_SIGNAL'] = macd.macd_signal()
    bb_indicator = BollingerBands(close, window=20, window_dev=2)
    df['BB_High'] = bb_indicator.bollinger_hband()
    df['BB_Low'] = bb_indicator.bollinger_lband()
    adx_indicator = ADXIndicator(df['High'], df['Low'], close, window=14)
    df['ADX'] = adx_indicator.adx()
    df['Prev_Close'] = close.shift(1)
    for i in range(1, 4):
        df[f'Prev_Close_Lag{i}'] = close.shift(i)
    df['Volume_MA'] = df['Volume'].rolling(10, min_periods=1).mean() if 'Volume' in df.columns else 0
    df['Volatility'] = close.rolling(10, min_periods=1).std()
    
    # 新增底部/頭部策略所需指標
    df['K'], df['D'] = calc_kd(df, CFG.stoch_k, CFG.stoch_d, CFG.stoch_smooth)
    df['ATR'] = calc_atr(df, CFG.atr_period)
    df['RecentLow'] = df['Close'].rolling(CFG.bottom_lookback, min_periods=1).min()
    df['PriorHigh'] = df['Close'].shift(1).rolling(CFG.higher_high_lookback, min_periods=1).max()
    df['RecentHigh'] = df['Close'].rolling(CFG.top_lookback, min_periods=1).max()
    df['PriorLow'] = df['Close'].shift(1).rolling(CFG.lower_low_lookback, min_periods=1).min()
    df['VOL_MA'] = df['Volume'].rolling(CFG.volume_ma, min_periods=1).mean()

    # 準備特徵
    feats = ['Prev_Close', 'MA5', 'MA10', 'MA20', 'Volume_MA', 'RSI', 'MACD',
             'MACD_SIGNAL', 'TWII_Close', 'SP500_Close', 'Volatility', 'BB_High',
             'BB_Low', 'ADX'] + [f'Prev_Close_Lag{i}' for i in range(1, 4)]
    
    df_clean = df[feats + ['Close']].dropna()
    
    # 如果數據量不足，返回完整數據，讓後續的 "新上市" 策略處理
    if len(df_clean) < 30:
        return None, None, None, df

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

    return last_close, predictions, preds, df


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
st.set_page_config(page_title="AI 智慧股價預測與買/賣點分析", layout="wide")
st.title("📈 AI 智慧股價預測與買/賣點分析")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    code = st.text_input("請輸入股票代號（例如: 2330）", "2330")
with col2:
    strategy_type = st.radio("分析模式", ["買進策略", "賣出策略"])

col3, col4 = st.columns([2,1])
with col3:
    mode = st.selectbox("預測模式", ["中期模式", "短期模式", "長期模式"])
mode_info = {
    "短期模式": ("使用 100 天歷史資料，高敏感度", 100, 0.008),
    "中期模式": ("使用 200 天歷史資料，平衡敏感度", 200, 0.005),
    "長期模式": ("使用 400 天歷史資料，低敏感度", 400, 0.002)
}
with col4:
    st.info(f"**{mode}**: {mode_info[mode][0]}")
days, decay_factor = mode_info[mode][1], mode_info[mode][2]

if st.button("🔮 開始分析", type="primary", use_container_width=True):
    full_code = code.strip()
    if not full_code.upper().endswith(".TW"):
        full_code = f"{full_code}.TW"
        
    with st.spinner("🚀 正在下載數據、訓練模型並進行分析..."):
        last, forecast, preds, df_with_indicators = predict_next_5(full_code, days, decay_factor)
        
    if df_with_indicators.empty:
        st.error(f"❌ 無法下載資料：{full_code}")
        st.warning("請檢查股票代號是否正確，或此股票目前無資料。")
    else:
        # Check if there is enough data for the full analysis
        is_low_volume_stock = len(df_with_indicators) < 50
        
        st.success("✅ 分析完成！")
        try:
            ticker_info = yf.Ticker(full_code).info
            company_name = ticker_info.get('shortName') or ticker_info.get('longName') or "無法取得名稱"
        except Exception:
            company_name = "無法取得名稱"

        ch_name = stock_name_dict.get(full_code, "無中文名稱")
        st.header(f"股票分析報告：{ch_name} ({company_name}) - {full_code}")

        # --- AI 預測結果區塊 ---
        st.subheader("🤖 AI 智慧預測")
        main_col1, main_col2 = st.columns([1, 2])
        if not is_low_volume_stock:
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
                st.subheader("📅 未來 5 日預
