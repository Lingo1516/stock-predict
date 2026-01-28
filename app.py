import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, time
import pytz
import io
import warnings

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

import ta
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator

warnings.filterwarnings("ignore")

# ====== Plotly optional ======
PLOTLY_ERROR = ""
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception as e:
    HAS_PLOTLY = False
    PLOTLY_ERROR = str(e)

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
    ma_short: int = 20
    ma_long: int = 60
    volume_ma: int = 20
    atr_period: int = 14
    risk_per_trade: float = 0.01
    capital: float = 1_000_000
    fwd_days: int = 5
    backtest_lookback_days: int = 252

CFG = Config()

# ====== 股票代碼對照表（可自行擴充） ======
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

# ====== 資料下載（處理 MultiIndex） ======
def safe_download(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def pick_market_index(stock_code: str):
    code = stock_code.upper()
    if code.endswith(".TW"):
        return ["^TWII", "0050.TW"]  # 台股：加權指數優先，ETF fallback
    return ["^GSPC"]  # 美股：S&P 500

# ====== 技術指標 ======
def add_technical_indicators(df: pd.DataFrame, cfg: Config):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # MA
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA_S"] = df["MA20"]
    df["MA_L"] = df["MA60"]
    df["MA_S_SLOPE"] = df["MA_S"] - df["MA_S"].shift(5)

    # RSI / MACD / BB / ADX
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    df["ADX"] = ADXIndicator(high, low, close, window=14).adx()

    # KD（Stochastic）
    stoch = StochasticOscillator(high=high, low=low, close=close, window=cfg.stoch_k, smooth_window=cfg.stoch_smooth)
    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()

    # ATR
    atr_indicator = ta.volatility.AverageTrueRange(high, low, close, window=cfg.atr_period)
    df["ATR"] = atr_indicator.average_true_range()

    # 簡單的底/頂參考 + 量均
    df["RecentLow"] = close.rolling(cfg.bottom_lookback).min()
    df["PriorHigh"] = close.shift(1).rolling(cfg.higher_high_lookback).max()
    df["RecentHigh"] = close.rolling(cfg.top_lookback).max()
    df["PriorLow"] = close.shift(1).rolling(cfg.lower_low_lookback).min()
    df["VOL_MA"] = df["Volume"].rolling(cfg.volume_ma).mean()

    return df

def add_return_features(df: pd.DataFrame):
    df = df.copy()
    df["Ret1"] = np.log(df["Close"]).diff()
    df["Ret5"] = np.log(df["Close"]).diff(5)
    df["Vol10"] = df["Ret1"].rolling(10).std() * np.sqrt(252)
    df["Vol20"] = df["Ret1"].rolling(20).std() * np.sqrt(252)
    df["VolChg"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    return df

# ====== 訊號生成（保留你原本邏輯，微調可讀性） ======
def generate_signal_row_buy(row_prior, row_now, cfg: Config):
    reasons = []
    bottom_built = (row_now["Close"] <= row_now["RecentLow"] * 1.08) and (row_now["Close"] > (row_now["PriorHigh"] * 0.8))
    if bottom_built: reasons.append("接近近期低點後回升")

    kd_cross_up = (row_prior["K"] < row_prior["D"]) and (row_now["K"] > row_now["D"])
    kd_above_threshold = row_now["K"] > cfg.kd_threshold
    kd_ok = kd_cross_up and kd_above_threshold
    if kd_ok: reasons.append(f"KD黃金交叉且K>{cfg.kd_threshold:.0f}")

    macd_up = (row_now["MACD"] > 0) and (row_now["MACD"] > row_prior["MACD"])
    if macd_up: reasons.append("MACD柱轉正且走揚")

    trend_ok = (row_now["MA_S"] > row_now["MA_L"]) and (row_now["MA_S_SLOPE"] > 0)
    if trend_ok: reasons.append("多頭趨勢濾網通過")

    volume_ok = row_now["Volume"] >= row_now["VOL_MA"]
    if volume_ok: reasons.append("量能不弱於均量")

    all_ok = bottom_built and kd_ok and macd_up and trend_ok and volume_ok
    return all_ok, reasons

def generate_signal_row_sell(row_prior, row_now, cfg: Config):
    reasons = []
    top_built = (row_now["Close"] >= row_now["RecentHigh"] * 0.92) and (row_now["Close"] < (row_now["PriorLow"] * 1.2))
    if top_built: reasons.append("接近近期高點後回落")

    kd_cross_down = (row_prior["K"] > row_prior["D"]) and (row_now["K"] < row_now["D"])
    kd_below_threshold = row_now["K"] < cfg.kd_threshold_sell
    kd_ok = kd_cross_down and kd_below_threshold
    if kd_ok: reasons.append(f"KD死亡交叉且K<{cfg.kd_threshold_sell:.0f}")

    macd_down = (row_now["MACD"] < 0) and (row_now["MACD"] < row_prior["MACD"])
    if macd_down: reasons.append("MACD柱轉負且走弱")

    trend_ok = (row_now["MA_S"] < row_now["MA_L"]) and (row_now["MA_S_SLOPE"] < 0)
    if trend_ok: reasons.append("空頭趨勢濾網通過")

    volume_ok = row_now["Volume"] >= row_now["VOL_MA"]
    if volume_ok: reasons.append("量能不弱於均量")

    all_ok = top_built and kd_ok and macd_down and trend_ok and volume_ok
    return all_ok, reasons

def evaluate_latest(df: pd.DataFrame, cfg: Config, strategy_type: str):
    # basic data length check
    need = max(cfg.ma_long, cfg.bottom_lookback, cfg.top_lookback, cfg.atr_period) + 5
    if len(df) < need:
        return {"是否符合訊號": False, "理由": "資料樣本太短", "動作": "無", "建議停損": 0, "估計ATR": 0, "建議股數": 0}

    df = df.dropna().copy()
    if len(df) < 2:
        return {"是否符合訊號": False, "理由": "有效樣本不足", "動作": "無", "建議停損": 0, "估計ATR": 0, "建議股數": 0}

    row_now = df.iloc[-1]
    row_prior = df.iloc[-2]
    atr = float(row_now["ATR"])

    if strategy_type == "buy":
        signal, reasons = generate_signal_row_buy(row_prior, row_now, cfg)
        stop_level = float(row_now["Close"] - 2.5 * atr)
        position_risk = float(row_now["Close"] - stop_level)
        action_text = "多方(買進)模式"
    else:
        signal, reasons = generate_signal_row_sell(row_prior, row_now, cfg)
        stop_level = float(row_now["Close"] + 2.5 * atr)
        position_risk = float(stop_level - row_now["Close"])
        action_text = "空方(放空)模式"

    position_size = 0
    if position_risk > 0:
        position_size = int((cfg.capital * cfg.risk_per_trade) // position_risk)

    return {
        "日期": df.index[-1].strftime("%Y-%m-%d"),
        "收盤": round(float(row_now["Close"]), 2),
        "是否符合訊號": bool(signal),
        "理由": "、".join(reasons) if reasons else "條件不足",
        "動作": action_text,
        "建議停損": round(float(stop_level), 2),
        "估計ATR": round(float(atr), 2),
        "建議股數": int(position_size)
    }

# ====== 時序交叉驗證 ======
def rolling_cv_metrics(X, y, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, mapes = [], []
    for tr, te in tscv.split(X):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        true = y[te]
        mae = mean_absolute_error(true, pred)
        mape = np.mean(np.abs((pred - true) / np.maximum(true, 1e-9))) * 100
        maes.append(mae)
        mapes.append(mape)
    return float(np.mean(maes)), float(np.mean(mapes))

# ====== 2026 升級版預測（點估計 + 區間） ======
@st.cache_data(ttl=3600)
def predict_next_5(stock: str, days: int, decay_factor: float):
    end = pd.Timestamp(datetime.today().date()) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days * 2)

    df = safe_download(stock, start, end)
    if df.empty:
        return None, None, None, pd.DataFrame(), {"cv_mae": None, "cv_mape": None, "resid_sigma": None}

    # Market index
    idx_df = pd.DataFrame()
    for idx in pick_market_index(stock):
        tmp = safe_download(idx, start, end)
        if not tmp.empty and "Close" in tmp.columns:
            idx_df = tmp
            break

    if idx_df.empty:
        df["Market_Close"] = np.nan
    else:
        df["Market_Close"] = idx_df["Close"].reindex(df.index).ffill()

    # indicators + returns
    df = add_technical_indicators(df, CFG)
    df = add_return_features(df)

    # relative strength
    df["Mkt_Ret1"] = np.log(df["Market_Close"]).diff()
    df["RelStrength1"] = df["Ret1"] - df["Mkt_Ret1"]

    df = df.dropna().copy()
    if len(df) < 80:
        return None, None, None, df, {"cv_mae": None, "cv_mape": None, "resid_sigma": None}

    feats = [
        "Ret1", "Ret5", "Vol10", "Vol20", "VolChg",
        "MA5", "MA10", "MA20", "MA60",
        "RSI", "MACD", "ADX",
        "BB_High", "BB_Low",
        "RelStrength1"
    ]

    X = df[feats].values
    y = df["Close"].values

    # recency weights
    w = np.exp(-decay_factor * np.arange(len(X))[::-1])
    w = w / np.sum(w)

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        random_state=42
    )
    model.fit(X, y, sample_weight=w)

    df["AI_Pred"] = model.predict(X)

    # residual sigma for interval (粗略，但比沒有好)
    resid = df["Close"].values - df["AI_Pred"].values
    resid_sigma = float(np.std(resid[-60:])) if len(resid) >= 60 else float(np.std(resid))

    # time-series CV (more honest)
    cv_mae, cv_mape = rolling_cv_metrics(
        X, y,
        HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42),
        n_splits=5
    )

    last_close = float(df["Close"].iloc[-1])
    future_dates = pd.bdate_range(start=df.index[-1], periods=6)[1:]

    preds = {}
    pred_prices = []
    pred_hi = []
    pred_lo = []

    # minimal extrapolation: use last feature row as anchor
    last_feat_row = df[feats].iloc[-1:].copy()
    last_atr = float(df["ATR"].iloc[-1])
    last_ma20 = float(df["MA20"].iloc[-1])

    for d in future_dates:
        x_last = last_feat_row.values
        p = float(model.predict(x_last)[0])

        # guard rails by MA20 +/- 3*ATR
        upper = last_ma20 + 3 * last_atr
        lower = last_ma20 - 3 * last_atr
        p = min(max(p, lower), upper)

        # interval: +/- 1.28 sigma (約 80% 近似區間) + 再加上 ATR 小幅保守
        hi = p + 1.28 * resid_sigma + 0.25 * last_atr
        lo = p - 1.28 * resid_sigma - 0.25 * last_atr

        preds[d.date()] = p
        pred_prices.append(p)
        pred_hi.append(hi)
        pred_lo.append(lo)

    preds_dict = {f"T+{i+1}": float(p) for i, p in enumerate(pred_prices)}

    extra = {"cv_mae": cv_mae, "cv_mape": cv_mape, "resid_sigma": resid_sigma,
             "pred_hi": pred_hi, "pred_lo": pred_lo, "future_dates": list(future_dates)}

    return last_close, preds, preds_dict, df, extra

def get_trade_advice(last, preds):
    if not preds:
        return "無法判斷"
    avg_pred = float(np.mean(list(preds.values())))
    change_percent = ((avg_pred - last) / last) * 100
    if change_percent > 2.0:
        return f"強烈看漲 (預期 +{change_percent:.1f}%)"
    elif change_percent > 0.5:
        return f"看漲 (預期 +{change_percent:.1f}%)"
    elif change_percent < -2.0:
        return f"強烈看跌 (預期 {change_percent:.1f}%)"
    elif change_percent < -0.5:
        return f"看跌 (預期 {change_percent:.1f}%)"
    return f"盤整 (預期 {change_percent:.1f}%)"

# ====== Plot ======
def plot_stock_data(df, extra=None):
    if not HAS_PLOTLY:
        return None

    df = df.copy()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.7, 0.3],
        subplot_titles=("股價走勢（含AI預測軌跡）", "成交量")
    )

    fig.add_trace(
        go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K線"),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA60"], name="MA60"), row=1, col=1)

    if "AI_Pred" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["AI_Pred"], name="AI 歷史預測", line=dict(dash="dot")), row=1, col=1)

    # future forecast line + interval
    if extra and extra.get("future_dates") and extra.get("pred_hi") and extra.get("pred_lo"):
        fd = extra["future_dates"]
        hi = extra["pred_hi"]
        lo = extra["pred_lo"]

        # point forecast (use preds from dict order)
        # make a simple line from last close to forecast points
        # (we don't inject them into df to avoid drift)
        # Build x/y
        # Here: use mean of hi/lo as point, or use separate preds is fine
        # We'll use middle: (hi+lo)/2 for display alignment
        mid = [(h + l) / 2 for h, l in zip(hi, lo)]

        connect_x = [df.index[-1]] + list(fd)
        connect_y = [float(df["Close"].iloc[-1])] + list(mid)

        fig.add_trace(go.Scatter(x=connect_x, y=connect_y, name="AI 未來預測", line=dict(dash="dash", width=3)), row=1, col=1)

        # interval band (only for future)
        fig.add_trace(go.Scatter(x=list(fd) + list(fd)[::-1],
                                 y=hi + lo[::-1],
                                 fill="toself",
                                 name="預測區間(約80%)",
                                 opacity=0.2,
                                 line=dict(width=0)),
                      row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)

    fig.update_layout(height=650, xaxis_rangeslider_visible=False, hovermode="x unified")
    return fig

# ====== UI ======
st.set_page_config(page_title="AI 智能股價分析 Pro (2026)", layout="wide", page_icon="📈")

st.markdown("""
<style>
.metric-card {background-color:#f0f2f6;border-radius:10px;padding:15px;margin:10px 0;}
.suggestion-box {padding:20px;border-radius:10px;text-align:center;margin-bottom:20px;}
</style>
""", unsafe_allow_html=True)

st.title("📈 AI 智能股價分析 Pro（2026 重寫版）")
st.markdown("整合機器學習預測、技術指標與時序交叉驗證（避免過度樂觀）。")

if "recent_stocks" not in st.session_state:
    st.session_state.recent_stocks = []

with st.sidebar:
    st.header("⚙️ 設定參數")
    data_source = st.radio("資料來源", ["自動下載 (yfinance)", "手動貼上CSV資料"])

    if data_source == "自動下載 (yfinance)":
        if st.session_state.recent_stocks:
            selected_history = st.selectbox("📜 最近瀏覽紀錄", ["請選擇..."] + st.session_state.recent_stocks)
            if selected_history != "請選擇...":
                default_code = selected_history.split(" ")[0].replace(".TW", "")
            else:
                default_code = "2330"
        else:
            default_code = "2330"

        code = st.text_input("股票代號（台股可輸入 2330）", value=default_code)

        strategy_type = st.radio("偵測訊號方向", ["買進策略", "賣出策略"])
        mode = st.selectbox("模型敏感度", ["短期 (敏感)", "中期 (平衡)", "長期 (穩健)"])

        mode_map = {
            "短期 (敏感)": (200, 0.012),
            "中期 (平衡)": (400, 0.006),
            "長期 (穩健)": (800, 0.002),
        }
        days, decay_factor = mode_map[mode]

        show_interval = st.checkbox("顯示預測區間（建議開）", value=True)
    else:
        st.info("手動模式：僅技術指標與訊號，不跑 AI 預測")
        show_interval = False

run_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)

if run_btn:
    df_result = pd.DataFrame()
    forecast = None
    preds = None
    last_price = None
    extra = {}

    # ---- data ----
    if data_source == "自動下載 (yfinance)":
        full_code = code.strip().upper()
        if full_code.isdigit():
            full_code += ".TW"

        stock_name = stock_name_dict.get(full_code, "未知名稱")

        with st.spinner(f"正在分析 {stock_name} ({full_code}) ..."):
            last_price, forecast, preds, df_result, extra = predict_next_5(full_code, days, decay_factor)

        if df_result is None or df_result.empty or last_price is None:
            st.error("無法取得資料，請檢查代號或網路連線。")
            st.stop()

        history_item = f"{full_code} {stock_name}"
        if history_item not in st.session_state.recent_stocks:
            st.session_state.recent_stocks.insert(0, history_item)
            if len(st.session_state.recent_stocks) > 10:
                st.session_state.recent_stocks.pop()

        st.subheader(f"{stock_name} ({full_code}) - 股價分析報告（資料時間：{df_result.index[-1].strftime('%Y-%m-%d')}）")

    else:
        manual_data = st.text_area("貼上 CSV（需含 Date, Open, High, Low, Close, Volume 欄位）", height=200)
        if manual_data:
            try:
                df_result = pd.read_csv(io.StringIO(manual_data))
                df_result["Date"] = pd.to_datetime(df_result["Date"])
                df_result.set_index("Date", inplace=True)
                df_result = add_technical_indicators(df_result, CFG).dropna()
                last_price = float(df_result["Close"].iloc[-1])
            except Exception as e:
                st.error(f"CSV 格式錯誤: {e}")
                st.stop()
        else:
            st.warning("請先貼上 CSV 資料。")
            st.stop()

    # ---- market status (display only) ----
    tz_tw = pytz.timezone("Asia/Taipei")
    now_tw = datetime.now(tz_tw)
    market_open_time = time(9, 0)
    market_close_time = time(13, 30)
    is_market_open = (now_tw.weekday() < 5) and (market_open_time <= now_tw.time() <= market_close_time)
    status_text = "🌞 開盤中（提示用，日線仍以收盤資料為主）" if is_market_open else "🌙 已收盤（使用最近交易日資料）"

    strat_key = "buy" if strategy_type == "買進策略" else "sell"
    summary = evaluate_latest(df_result, CFG, strat_key)

    # AI trend
    ai_trend_pct = 0.0
    if forecast:
        avg_pred = float(np.mean(list(forecast.values())))
        ai_trend_pct = ((avg_pred - last_price) / last_price) * 100

    # signal light
    if summary["是否符合訊號"]:
        if summary["動作"].startswith("多方"):
            signal_color, signal_emoji, signal_text = "#d4edda", "🟢", "買進訊號 (BUY)"
        else:
            signal_color, signal_emoji, signal_text = "#f8d7da", "🔴", "放空訊號 (SELL)"
        ai_hint = ""
    else:
        signal_color, signal_emoji = "#fff3cd", "🟡"
        trend_direction = "偏多" if ai_trend_pct > 0 else "偏空"
        signal_text = f"觀望 (WAIT) - 趨勢{trend_direction}"
        ai_hint = f" | <b>AI 趨勢:</b> {trend_direction} (預期 {ai_trend_pct:+.1f}%)，但技術面尚未確認" if forecast else ""

    st.markdown(f"""
    <div style="background-color:{signal_color};padding:18px;border-radius:14px;text-align:center;border:2px solid #ccc;color:#333;">
        <div style="color:#666;">{status_text} | 資料日期: {summary.get('日期','-')}</div>
        <div style="font-size:34px;margin:8px 0;">{signal_emoji} {signal_text}</div>
        <div style="font-size:16px;"><b>模式:</b> {summary.get('動作','-')} | <b>收盤:</b> {summary.get('收盤','-')}{ai_hint}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📌 詳細訊號與風控")
        st.metric("基準收盤價 (Last Close)", f"{last_price:.2f}")
        st.write(f"**技術面訊號**：{summary.get('理由','-')}")
        st.write(f"**建議停損**：{summary.get('建議停損','-')}")
        st.write(f"**ATR**：{summary.get('估計ATR','-')}")
        st.write(f"**建議股數（依資金風險）**：{summary.get('建議股數','-')}")

        if forecast:
            st.markdown("### 🤖 AI 趨勢建議")
            st.info(f"AI 建議：**{get_trade_advice(last_price, forecast)}**")

        if extra and extra.get("cv_mape") is not None:
            st.markdown("### ✅ 時序交叉驗證（更可信）")
            st.write(f"Rolling CV MAE：**{extra['cv_mae']:.2f}**")
            st.write(f"Rolling CV MAPE：**{extra['cv_mape']:.2f}%**")

    with col2:
        st.markdown("### 📈 圖表")
        plot_df = df_result.tail(160).copy()

        if HAS_PLOTLY:
            fig = plot_stock_data(plot_df, extra if (show_interval and data_source == "自動下載 (yfinance)") else None)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ 未安裝 plotly，改用簡易線圖。")
            st.line_chart(plot_df[["Close", "MA20", "MA60"]].dropna())

        if forecast:
            st.markdown("### 🔮 未來 5 日預測（點估計）")
            f_dates = list(forecast.keys())
            f_vals = list(forecast.values())

            f_df = pd.DataFrame({
                "日期": [str(d) for d in f_dates],
                "預測價": [f"{v:.2f}" for v in f_vals],
                "漲跌幅": [f"{(v - last_price) / last_price * 100:+.2f}%" for v in f_vals],
            })

            if show_interval and extra and extra.get("pred_hi"):
                f_df["區間下界(約80%)"] = [f"{v:.2f}" for v in extra["pred_lo"]]
                f_df["區間上界(約80%)"] = [f"{v:.2f}" for v in extra["pred_hi"]]

            st.table(f_df)

    st.markdown("---")
    st.caption("免責聲明：本工具僅供技術研究與學習，不構成任何投資建議。")
