import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
import warnings
from dataclasses import dataclass
from datetime import datetime, time

warnings.filterwarnings("ignore")

# =========================
# Optional: Plotly
# =========================
PLOTLY_ERROR = ""
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception as e:
    HAS_PLOTLY = False
    PLOTLY_ERROR = str(e)

# =========================
# Optional: TW market calendar
# =========================
HAS_TW_CAL = False
try:
    import pandas_market_calendars as mcal
    HAS_TW_CAL = True
except Exception:
    HAS_TW_CAL = False

# =========================
# ML
# =========================
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

import ta

# =========================
# Config
# =========================
@dataclass
class Config:
    forecast_days: int = 10
    min_train_rows: int = 240

    # 預測護欄：用真實歷史 MA20/ATR 限制預測範圍，避免爆走
    atr_period: int = 14
    guard_atr_mult: float = 3.0

    # Ensemble model settings
    rf_estimators: int = 300
    rf_max_depth: int = 10
    hgb_max_iter: int = 500

    # Interval (about 80%)
    interval_z: float = 1.28

    # Scenario simulation
    sim_paths: int = 200
    sim_noise_mult: float = 1.0
    mean_revert_strength: float = 0.25

    # Turning rules (RSI + Bollinger)
    rsi_hi: float = 70.0
    rsi_lo: float = 30.0
    bb_window: int = 20
    bb_std: float = 2.0

CFG = Config()

# =========================
# Utils
# =========================
def safe_download(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def market_status(code: str) -> str:
    is_tw = code.upper().endswith(".TW")
    if is_tw and HAS_TW_CAL:
        try:
            cal = mcal.get_calendar("XTAI")
            now = pd.Timestamp.now(tz="Asia/Taipei")
            sched = cal.schedule(start_date=now.date(), end_date=now.date())
            if sched.empty:
                return "非交易日"
            open_t = sched.iloc[0]["market_open"].tz_convert("Asia/Taipei")
            close_t = sched.iloc[0]["market_close"].tz_convert("Asia/Taipei")
            if open_t <= now <= close_t:
                return "盤中"
            return "已收盤"
        except Exception:
            pass

    now = datetime.now()
    if now.weekday() >= 5:
        return "非交易日(推測)"
    if is_tw:
        if time(9, 0) <= now.time() <= time(13, 30):
            return "盤中(推測)"
        return "已收盤(推測)"
    return "日線資料(不判斷盤中)"

# =========================
# Feature Engineering (recursive-friendly)
# =========================
FEATURES = [
    "Ret1", "Ret5",
    "MA5", "MA10", "MA20", "MA60",
    "RSI",
    "Vol20",
    "VolChg"
]

def add_guard_indicators_real(df_raw: pd.DataFrame) -> pd.DataFrame:
    """只在真實歷史資料上算 MA20/ATR（護欄用）"""
    df = df_raw.copy()
    close = df["Close"].astype(float)
    df["MA20"] = close.rolling(20).mean()
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], close, window=CFG.atr_period)
    df["ATR"] = atr.average_true_range()
    return df.dropna().copy()

def add_model_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """模型特徵：只用 Close/Volume 可推進特徵（避免 High/Low 自我餵食）"""
    df = df_raw.copy()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    df["Ret1"] = np.log(close).diff()
    df["Ret5"] = np.log(close).diff(5)

    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()

    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    r1 = np.log(close).diff()
    df["Vol20"] = r1.rolling(20).std() * np.sqrt(252)

    df["VolChg"] = vol.pct_change().replace([np.inf, -np.inf], np.nan)

    return df.dropna().copy()

def compute_next_feature_row(close_hist: list[float], vol_hist: list[float]) -> np.ndarray:
    s = pd.Series(close_hist, dtype="float64")
    v = pd.Series(vol_hist, dtype="float64")

    ret1 = float(np.log(s.iloc[-1]) - np.log(s.iloc[-2])) if len(s) >= 2 else 0.0
    ret5 = float(np.log(s.iloc[-1]) - np.log(s.iloc[-6])) if len(s) >= 6 else 0.0

    ma5 = float(s.iloc[-5:].mean()) if len(s) >= 5 else float(s.mean())
    ma10 = float(s.iloc[-10:].mean()) if len(s) >= 10 else float(s.mean())
    ma20 = float(s.iloc[-20:].mean()) if len(s) >= 20 else float(s.mean())
    ma60 = float(s.iloc[-60:].mean()) if len(s) >= 60 else float(s.mean())

    if len(s) >= 15:
        rsi = float(ta.momentum.RSIIndicator(s, window=14).rsi().iloc[-1])
    else:
        rsi = 50.0

    r1 = np.log(s).diff().dropna()
    if len(r1) >= 20:
        vol20 = float(r1.iloc[-20:].std() * np.sqrt(252))
    elif len(r1) >= 2:
        vol20 = float(r1.std() * np.sqrt(252))
    else:
        vol20 = 0.0

    if len(v) >= 2 and v.iloc[-2] != 0:
        volchg = float(v.iloc[-1] / v.iloc[-2] - 1.0)
    else:
        volchg = 0.0

    return np.array([[ret1, ret5, ma5, ma10, ma20, ma60, rsi, vol20, volchg]], dtype="float64")

# =========================
# Ensemble + CV weighting
# =========================
def train_ensemble_with_cv(X: np.ndarray, y: np.ndarray, seed: int = 42):
    models = {
        "HGB": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=CFG.hgb_max_iter,
            random_state=seed
        ),
        "RF": RandomForestRegressor(
            n_estimators=CFG.rf_estimators,
            max_depth=CFG.rf_max_depth,
            min_samples_split=6,
            random_state=seed,
            n_jobs=-1
        )
    }

    tscv = TimeSeriesSplit(n_splits=5)
    cv_mae = {}

    for name, model in models.items():
        fold_mae = []
        for tr, te in tscv.split(X):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            fold_mae.append(mean_absolute_error(y[te], pred))
        cv_mae[name] = float(np.mean(fold_mae))

    inv = {k: 1.0 / max(v, 1e-9) for k, v in cv_mae.items()}
    s = sum(inv.values())
    weights = {k: inv[k] / s for k in inv}

    trained = {}
    for name, model in models.items():
        model.fit(X, y)
        trained[name] = model

    return trained, weights, cv_mae

def ensemble_predict(models: dict, weights: dict, X: np.ndarray) -> np.ndarray:
    pred = None
    for name, model in models.items():
        p = model.predict(X)
        w = weights.get(name, 0.0)
        pred = p * w if pred is None else pred + p * w
    return pred

def estimate_sigma(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    resid = y_true - y_pred
    if resid.size >= 80:
        resid = resid[-80:]
    return float(np.std(resid))

# =========================
# 10-day recursive forecast
# =========================
def forecast_recursive(models, weights, df_feat: pd.DataFrame, df_raw: pd.DataFrame, df_guard: pd.DataFrame):
    future_dates = pd.bdate_range(start=df_feat.index[-1], periods=CFG.forecast_days + 1)[1:]

    close_hist = df_feat["Close"].astype(float).tolist()
    vol_hist = df_raw.loc[df_feat.index, "Volume"].astype(float).tolist()
    future_vol = float(np.mean(vol_hist[-20:])) if len(vol_hist) >= 20 else float(np.mean(vol_hist))

    last_close = float(df_feat["Close"].iloc[-1])

    # guard rails
    if not df_guard.empty:
        last_ma20 = float(df_guard["MA20"].iloc[-1])
        last_atr = float(df_guard["ATR"].iloc[-1])
    else:
        last_ma20 = float(df_feat["MA20"].iloc[-1])
        last_atr = 0.0

    upper = last_ma20 + CFG.guard_atr_mult * last_atr if last_atr > 0 else np.inf
    lower = last_ma20 - CFG.guard_atr_mult * last_atr if last_atr > 0 else -np.inf

    preds = []
    for _ in range(CFG.forecast_days):
        x_next = compute_next_feature_row(close_hist, vol_hist)
        p = float(ensemble_predict(models, weights, x_next)[0])
        p = min(max(p, lower), upper)
        preds.append(p)
        close_hist.append(p)
        vol_hist.append(future_vol)

    return future_dates, last_close, preds

# =========================
# RSI + Bollinger helpers
# =========================
def compute_rsi_bbands(close_series: pd.Series):
    rsi = ta.momentum.RSIIndicator(close_series, window=14).rsi()
    bb = ta.volatility.BollingerBands(close_series, window=CFG.bb_window, window_dev=CFG.bb_std)
    bb_h = bb.bollinger_hband()
    bb_l = bb.bollinger_lband()
    return rsi, bb_h, bb_l

# =========================
# Scenario simulation + turning stats (safe concat, no np.r_ for 2D)
# =========================
def simulate_paths_and_turning(df_raw: pd.DataFrame, future_dates, base_preds, sigma: float):
    T = int(len(base_preds))
    n = int(CFG.sim_paths)

    last_close = float(df_raw["Close"].iloc[-1])
    hist_close = df_raw["Close"].astype(float)

    if T <= 0:
        turn_df = pd.DataFrame(columns=["日期", "由漲轉跌機率(%)", "由跌轉漲機率(%)", "可能高點(%)_RSI+BB", "可能低點(%)_RSI+BB"])
        summary = {"連漲10天機率(%)": 0.0, "連跌10天機率(%)": 0.0}
        return np.zeros((n, 0)), np.zeros((n, 0), dtype=int), turn_df, summary

    ma20_target = float(hist_close.tail(20).mean()) if len(hist_close) >= 20 else float(hist_close.mean())

    base = np.array(base_preds, dtype=float)
    # 這裡是 1D safe：np.r_ 用在 1D 沒問題
    base_ret = np.diff(np.log(np.r_[last_close, base]))  # length T

    rng = np.random.default_rng(42)
    paths = np.zeros((n, T), dtype=float)

    for i in range(n):
        c = last_close
        for t in range(T):
            noise = rng.normal(0.0, sigma / max(c, 1e-9)) * float(CFG.sim_noise_mult)
            mr = -float(CFG.mean_revert_strength) * ((c - ma20_target) / max(ma20_target, 1e-9)) / max(T, 1)
            r = float(base_ret[t]) + float(mr) + float(noise)
            c = c * np.exp(r)
            paths[i, t] = c

    # prev shape: (n, T)
    if T == 1:
        prev = np.full((n, 1), last_close, dtype=float)
    else:
        prev = np.concatenate([np.full((n, 1), last_close, dtype=float), paths[:, :-1]], axis=1)

    diff = paths - prev
    sign = np.where(diff >= 0, 1, -1).astype(int)

    p_all_up = float(np.mean(np.all(sign == 1, axis=1)) * 100.0)
    p_all_dn = float(np.mean(np.all(sign == -1, axis=1)) * 100.0)

    up_to_dn = np.zeros(T, dtype=float)
    dn_to_up = np.zeros(T, dtype=float)
    for t in range(1, T):
        up_to_dn[t] = float(np.mean((sign[:, t-1] == 1) & (sign[:, t] == -1)) * 100.0)
        dn_to_up[t] = float(np.mean((sign[:, t-1] == -1) & (sign[:, t] == 1)) * 100.0)

    top_prob = np.zeros(T, dtype=float)
    bot_prob = np.zeros(T, dtype=float)

    hist_tail = hist_close.tail(120).copy()

    for t in range(T - 1):  # needs next day
        top_hits = 0
        bot_hits = 0
        for i in range(n):
            sim_close = pd.concat(
                [hist_tail, pd.Series(paths[i, :t+1], index=future_dates[:t+1])],
                axis=0
            )
            rsi, bb_h, bb_l = compute_rsi_bbands(sim_close)

            c_t = float(sim_close.iloc[-1])
            rsi_t = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
            bh_t = float(bb_h.iloc[-1]) if not np.isnan(bb_h.iloc[-1]) else np.inf
            bl_t = float(bb_l.iloc[-1]) if not np.isnan(bb_l.iloc[-1]) else -np.inf

            overbought = (rsi_t >= CFG.rsi_hi) or (c_t >= bh_t)
            oversold = (rsi_t <= CFG.rsi_lo) or (c_t <= bl_t)

            next_down = paths[i, t+1] < paths[i, t]
            next_up = paths[i, t+1] > paths[i, t]

            if overbought and next_down:
                top_hits += 1
            if oversold and next_up:
                bot_hits += 1

        top_prob[t] = top_hits / n * 100.0
        bot_prob[t] = bot_hits / n * 100.0

    turn_df = pd.DataFrame({
        "日期": [d.date() for d in future_dates],
        "由漲轉跌機率(%)": np.round(up_to_dn, 1),
        "由跌轉漲機率(%)": np.round(dn_to_up, 1),
        "可能高點(%)_RSI+BB": np.round(top_prob, 1),
        "可能低點(%)_RSI+BB": np.round(bot_prob, 1),
    })

    summary = {
        "連漲10天機率(%)": round(p_all_up, 2),
        "連跌10天機率(%)": round(p_all_dn, 2),
    }

    return paths, sign, turn_df, summary

# =========================
# Decision Summary Engine (你要的：人話結論 + 哪天買賣 + 強度 + 買多少)
# =========================
def _clip_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))

def generate_trade_summary(
    turn_df: pd.DataFrame,
    result_df: pd.DataFrame,
    last_close: float,
    atr: float,
    capital: float,
    risk_pct: float
):
    """
    依 turn_df + 預測表 + ATR 產生：
    - 結論（偏多/偏空/盤整）
    - Buy/Sell Strength（0~100）
    - 最佳買點日 / 最佳賣點日
    - 建議股數（依風險與停損）
    """
    # ---- safety ----
    if turn_df is None or turn_df.empty or result_df is None or result_df.empty or atr <= 0:
        return {
            "bias": "資料不足",
            "buy_strength": 0.0,
            "sell_strength": 0.0,
            "best_buy_day": "N/A",
            "best_sell_day": "N/A",
            "shares": 0,
            "stop_price": 0.0,
            "summary_text": "資料不足，無法產生交易結論。"
        }

    # 預期報酬（以每一天預測價相對 last_close）
    pred_ret_pct = (result_df["預測價"].astype(float) / float(last_close) - 1.0) * 100.0

    # Buy / Sell 分數（你要的質性+量化）
    buy_scores = (
        0.45 * turn_df["由跌轉漲機率(%)"].astype(float) +
        0.35 * turn_df["可能低點(%)_RSI+BB"].astype(float) +
        0.20 * pred_ret_pct.clip(lower=0.0)
    )

    sell_scores = (
        0.45 * turn_df["由漲轉跌機率(%)"].astype(float) +
        0.35 * turn_df["可能高點(%)_RSI+BB"].astype(float) +
        0.20 * (-pred_ret_pct).clip(lower=0.0)
    )

    buy_strength = _clip_0_100(float(buy_scores.max()))
    sell_strength = _clip_0_100(float(sell_scores.max()))

    # 偏多/偏空/盤整
    if buy_strength > sell_strength + 15:
        bias = "偏多"
    elif sell_strength > buy_strength + 15:
        bias = "偏空"
    else:
        bias = "盤整"

    best_buy_idx = int(buy_scores.idxmax())
    best_sell_idx = int(sell_scores.idxmax())

    best_buy_day = str(result_df.loc[best_buy_idx, "日期"])
    best_sell_day = str(result_df.loc[best_sell_idx, "日期"])

    # 強度文字
    def strength_label(x: float, side: str) -> str:
        if x >= 75:
            return f"強烈{side}"
        if x >= 60:
            return f"明確{side}"
        if x >= 40:
            return f"偏向{side}"
        return "觀望"

    buy_label = strength_label(buy_strength, "買進")
    sell_label = strength_label(sell_strength, "賣出")

    # 部位計算：風險金額 / 每股風險
    risk_amount = float(capital) * float(risk_pct)
    stop_price = float(last_close) - 2.5 * float(atr)
    risk_per_share = max(float(last_close) - float(stop_price), 1e-6)
    base_shares = int(risk_amount // risk_per_share)

    # 用買進強度調倉位（你要的「趕快」強度對應到買多少）
    if buy_strength >= 75:
        shares = int(base_shares * 1.3)
    elif buy_strength >= 60:
        shares = int(base_shares * 1.0)
    elif buy_strength >= 40:
        shares = int(base_shares * 0.5)
    else:
        shares = 0

    # 取出買點原因數字（最重要三個）
    row_buy = turn_df.loc[best_buy_idx]
    row_sell = turn_df.loc[best_sell_idx]

    buy_dn2up = float(row_buy["由跌轉漲機率(%)"])
    buy_bottom = float(row_buy["可能低點(%)_RSI+BB"])
    sell_up2dn = float(row_sell["由漲轉跌機率(%)"])
    sell_top = float(row_sell["可能高點(%)_RSI+BB"])

    summary_text = (
        f"【整體判斷】{bias}\n"
        f"Buy Strength：{buy_strength:.0f}/100（{buy_label}）｜Sell Strength：{sell_strength:.0f}/100（{sell_label}）\n\n"
        f"【最佳買點】{best_buy_day}\n"
        f"- 由跌轉漲機率：{buy_dn2up:.1f}%\n"
        f"- 可能低點(RSI+布林)：{buy_bottom:.1f}%\n\n"
        f"【最佳賣點/風險日】{best_sell_day}\n"
        f"- 由漲轉跌機率：{sell_up2dn:.1f}%\n"
        f"- 可能高點(RSI+布林)：{sell_top:.1f}%\n\n"
        f"【建議部位（依資金/風險/停損自動計算）】\n"
        f"- 資金：{capital:,.0f}｜單筆風險：{risk_pct*100:.0f}%（{risk_amount:,.0f}）\n"
        f"- 建議買進：{shares:,} 股\n"
        f"- 建議停損價：約 {stop_price:.2f}（= 現價 - 2.5×ATR）\n\n"
        f"【底線】跌破停損價 → 本次判斷失效，必須出場"
    )

    return {
        "bias": bias,
        "buy_strength": buy_strength,
        "sell_strength": sell_strength,
        "best_buy_day": best_buy_day,
        "best_sell_day": best_sell_day,
        "shares": shares,
        "stop_price": stop_price,
        "summary_text": summary_text
    }

# =========================
# Plot helpers
# =========================
def pick_top_days(df: pd.DataFrame, col: str, topk: int = 3):
    tmp = df.sort_values(col, ascending=False).head(topk)
    return tmp[["日期", col]]

def plot_history_and_pred(df_raw: pd.DataFrame, future_dates, preds):
    hist = df_raw[["Close"]].tail(120).copy()
    fut = pd.DataFrame({"Close": preds}, index=future_dates)
    return pd.concat([hist, fut], axis=0)

def plot_k_with_forecast(df_raw: pd.DataFrame, future_dates, preds, lo=None, hi=None):
    if not HAS_PLOTLY:
        return None
    dfp = df_raw.tail(160).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28],
                        subplot_titles=("K線 + 10日預測", "成交量"))

    fig.add_trace(go.Candlestick(
        x=dfp.index, open=dfp["Open"], high=dfp["High"], low=dfp["Low"], close=dfp["Close"], name="K線"
    ), row=1, col=1)

    connect_x = [dfp.index[-1]] + list(future_dates)
    connect_y = [float(dfp["Close"].iloc[-1])] + list(preds)
    fig.add_trace(go.Scatter(x=connect_x, y=connect_y, name="預測", line=dict(dash="dash", width=3)), row=1, col=1)

    if lo is not None and hi is not None:
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=list(hi) + list(lo)[::-1],
            fill="toself", opacity=0.18,
            line=dict(width=0),
            name="約80%區間"
        ), row=1, col=1)

    fig.add_trace(go.Bar(x=dfp.index, y=dfp["Volume"], name="Volume"), row=2, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, hovermode="x unified")
    return fig

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="AI 10日趨勢判定（含買賣結論）", layout="wide", page_icon="📈")
st.title("📈 AI 10日趨勢判定（含買賣結論、強度、買多少、停損）")
st.caption("這版會直接給你幾句結論：哪天買、哪天賣、強度多大、建議買多少（依資金100萬/風險10%計算）。")

with st.sidebar:
    st.header("⚙️ 設定")
    data_source = st.radio("資料來源", ["自動下載 (yfinance)", "手動貼上CSV資料"])

    show_interval = st.checkbox("顯示預測區間（約80%）", value=True)
    show_plotly = st.checkbox("使用 Plotly K 線圖（需 plotly）", value=True)

    st.divider()
    st.subheader("💰 風控設定")
    capital = st.number_input("資金", min_value=0.0, value=1_000_000.0, step=50_000.0)
    risk_pct = st.slider("單筆風險 (%)", 0.1, 20.0, 10.0, 0.1) / 100.0

    st.divider()
    st.subheader("🧪 模擬設定")
    CFG.sim_paths = st.slider("多情境路徑數", 50, 400, CFG.sim_paths, 50)
    CFG.mean_revert_strength = st.slider("均值回歸強度", 0.0, 0.8, CFG.mean_revert_strength, 0.05)
    CFG.sim_noise_mult = st.slider("模擬噪音倍率", 0.3, 2.0, CFG.sim_noise_mult, 0.1)

    st.divider()
    if data_source == "自動下載 (yfinance)":
        code = st.text_input("股票代號（台股輸入 2330、美股 AAPL）", "2330").strip()
        if code.isdigit():
            code = code + ".TW"
        code = code.upper()
        lookback_days = st.selectbox("訓練資料量（越多越穩、越慢）", [600, 900, 1200, 1600, 2000], index=2)

        st.write(f"市場狀態：{market_status(code)}（提示用；預測基於日線）")
        st.write(f"預測交易日數：{CFG.forecast_days}")
    else:
        st.info("CSV 需含 Date, Open, High, Low, Close, Volume 欄位。")

run_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)

if run_btn:
    # ===== load data =====
    if data_source == "手動貼上CSV資料":
        manual = st.text_area("貼上 CSV（需含 Date, Open, High, Low, Close, Volume）", height=240)
        if not manual.strip():
            st.error("請先貼上 CSV。")
            st.stop()
        try:
            df_raw = pd.read_csv(io.StringIO(manual))
            df_raw["Date"] = pd.to_datetime(df_raw["Date"])
            df_raw = df_raw.set_index("Date").sort_index()
        except Exception as e:
            st.error(f"CSV 解析失敗：{e}")
            st.stop()
    else:
        end = pd.Timestamp(datetime.today().date()) + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=int(lookback_days))
        with st.spinner("下載資料中..."):
            df_raw = safe_download(code, start, end)
        if df_raw.empty:
            st.error("資料下載失敗，請檢查代號或網路。")
            st.stop()

    # ===== features =====
    df_guard = add_guard_indicators_real(df_raw)
    df_feat = add_model_features(df_raw)

    if len(df_feat) < CFG.min_train_rows:
        st.error("有效樣本不足：請調高訓練資料量或換標的。")
        st.stop()

    # ===== train =====
    with st.spinner("訓練 Ensemble + 計算殘差波動中..."):
        X = df_feat[FEATURES].values
        y = df_feat["Close"].values
        models, weights, cv_mae = train_ensemble_with_cv(X, y)
        y_pred = ensemble_predict(models, weights, X)
        sigma = estimate_sigma(y, y_pred)

    # ===== forecast =====
    with st.spinner("進行 10 日遞迴預測中..."):
        future_dates, last_close, base_preds = forecast_recursive(models, weights, df_feat, df_raw, df_guard)

    base_hi = [p + CFG.interval_z * sigma for p in base_preds]
    base_lo = [p - CFG.interval_z * sigma for p in base_preds]

    result_df = pd.DataFrame({
        "日期": [d.date() for d in future_dates],
        "預測價": np.round(base_preds, 2),
        "漲跌幅(相對昨收)": [f"{(p - last_close) / last_close * 100:+.2f}%" for p in base_preds],
        "區間下界(約80%)": np.round(base_lo, 2),
        "區間上界(約80%)": np.round(base_hi, 2),
    })

    # ===== simulate + turning =====
    with st.spinner("多情境路徑模擬 + 轉折機率統計中..."):
        paths, sign, turn_df, summary_prob = simulate_paths_and_turning(df_raw, future_dates, base_preds, sigma)

    # ===== summary engine (你要的結論) =====
    atr_val = float(df_guard["ATR"].iloc[-1]) if (df_guard is not None and not df_guard.empty and "ATR" in df_guard.columns) else 0.0
    decision = generate_trade_summary(
        turn_df=turn_df,
        result_df=result_df,
        last_close=float(last_close),
        atr=float(atr_val),
        capital=float(capital),
        risk_pct=float(risk_pct)
    )

    # ===== output =====
    st.subheader("🧾 交易決策摘要（你要的結論就在這裡）")
    st.success(decision["summary_text"])

    cA, cB, cC = st.columns(3)
    cA.metric("連漲10天機率", f"{summary_prob.get('連漲10天機率(%)', 0.0):.2f}%")
    cB.metric("連跌10天機率", f"{summary_prob.get('連跌10天機率(%)', 0.0):.2f}%")
    cC.metric("ATR（停損用）", f"{atr_val:.4f}" if atr_val > 0 else "N/A")

    st.subheader("📌 模型摘要（參考用）")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最後收盤價", f"{last_close:.2f}")
    c2.metric("CV MAE (HGB)", f"{cv_mae.get('HGB', np.nan):.3f}")
    c3.metric("CV MAE (RF)", f"{cv_mae.get('RF', np.nan):.3f}")
    c4.metric("sigma（殘差波動）", f"{sigma:.4f}")

    st.write(f"權重：HGB **{weights.get('HGB', 0):.2f}**｜RF **{weights.get('RF', 0):.2f}**")

    st.subheader("🔮 10 個交易日預測（基準路徑）")
    st.dataframe(
        result_df if show_interval else result_df.drop(columns=["區間下界(約80%)", "區間上界(約80%)"]),
        use_container_width=True
    )

    st.subheader("📊 轉折機率明細（10天逐日）")
    st.dataframe(turn_df, use_container_width=True)

    st.markdown("**方向轉折 Top 3（由漲轉跌 / 由跌轉漲）**")
    t1, t2 = st.columns(2)
    with t1:
        st.write("由漲轉跌機率最高日：")
        st.table(pick_top_days(turn_df, "由漲轉跌機率(%)", 3))
    with t2:
        st.write("由跌轉漲機率最高日：")
        st.table(pick_top_days(turn_df, "由跌轉漲機率(%)", 3))

    st.markdown("**技術面轉折 Top 3（RSI + 布林帶）**")
    b1, b2 = st.columns(2)
    with b1:
        st.write("可能高點：")
        st.table(pick_top_days(turn_df, "可能高點(%)_RSI+BB", 3))
    with b2:
        st.write("可能低點：")
        st.table(pick_top_days(turn_df, "可能低點(%)_RSI+BB", 3))

    st.subheader("📈 走勢（歷史 + 預測）")
    merged = plot_history_and_pred(df_raw, future_dates, base_preds)
    st.line_chart(merged)

    if show_plotly and HAS_PLOTLY:
        fig = plot_k_with_forecast(
            df_raw, future_dates, base_preds,
            base_lo if show_interval else None,
            base_hi if show_interval else None
        )
        st.plotly_chart(fig, use_container_width=True)
    elif show_plotly and not HAS_PLOTLY:
        st.warning(f"未安裝 plotly：{PLOTLY_ERROR}")

st.caption("⚠️ 免責聲明：本工具僅供研究與學習，不構成任何投資建議。")
