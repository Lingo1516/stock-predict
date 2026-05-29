import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import ta
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import t as t_dist

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

warnings.filterwarnings("ignore")

TZ_TW = pytz.timezone("Asia/Taipei")
FORECAST_DAYS_DEFAULT = 10
SIM_PATHS_DEFAULT = 600

# ─────────────────────────────────────────────
# 資料下載（自動偵測上市 .TW / 上櫃 .TWO）
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def download_data(code: str, days: int = 1200) -> tuple[pd.DataFrame, str]:
    end   = datetime.now(TZ_TW).date() + timedelta(days=1)
    start = end - timedelta(days=days)
    base  = code.replace(".TW", "").replace(".TWO", "")
    codes_to_try = [base + ".TW", base + ".TWO"] if base.isdigit() else [code]
    for c in codes_to_try:
        for _ in range(3):
            try:
                df = yf.download(c, start=start, end=end,
                                 auto_adjust=True, progress=False, timeout=15)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] for col in df.columns]
                    return df.dropna().copy(), c
            except Exception:
                continue
    return pd.DataFrame(), code


@st.cache_data(ttl=3600)
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    vol   = df["Volume"].astype(float)

    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()

    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd_ind          = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd_ind.macd()
    df["MACD_signal"] = macd_ind.macd_signal()
    df["MACD_hist"]   = macd_ind.macd_diff()

    stoch    = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
    df["K"]  = stoch.stoch()
    df["D"]  = stoch.stoch_signal()

    bb             = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_mid"]   = bb.bollinger_mavg()
    df["BB_pct"]   = bb.bollinger_pband()

    df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()

    df["RET"]     = np.log(close).diff()
    df["SIGMA20"] = df["RET"].rolling(20).std()

    return df.dropna().copy()


# ─────────────────────────────────────────────
# 日期工具
# ─────────────────────────────────────────────
def next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(d).tz_localize(None)
    while True:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            return d

def future_dates(df: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    last_hist = pd.Timestamp(df.index[-1]).tz_localize(None)
    today     = pd.Timestamp(datetime.now(TZ_TW).date())
    base      = max(last_hist, today)
    return pd.bdate_range(start=next_business_day(base), periods=horizon)


# ─────────────────────────────────────────────
# 取得 GARCH 波動率序列（若無 arch 套件則退回 SIGMA20）
# ─────────────────────────────────────────────
def get_sigma_forecast(df: pd.DataFrame, T: int) -> np.ndarray:
    if HAS_ARCH:
        try:
            ret_pct = df["RET"].dropna() * 100
            res     = arch_model(ret_pct, vol='Garch', p=1, q=1, dist='t').fit(disp='off')
            fc      = res.forecast(horizon=T)
            sigma_t = np.sqrt(fc.variance.values[-1]) / 100
            return np.clip(sigma_t, 1e-4, None)
        except Exception:
            pass
    base_sigma = float(df["SIGMA20"].iloc[-1])
    if not np.isfinite(base_sigma) or base_sigma <= 0:
        base_sigma = max(float(df["RET"].tail(60).std()), 0.01)
    return np.full(T, base_sigma)


# ─────────────────────────────────────────────
# 蒙地卡羅模擬
#   ✅ Student-t 肥尾噪音
#   ✅ GARCH(1,1) 動態波動率（可選）
#   ✅ EWMA drift（近期加權）
#   ✅ 動態偏移縮放（相對 sigma）
#   ✅ OBV 斜率偏移
#   ✅ Antithetic Variates 降低模擬誤差
# ─────────────────────────────────────────────
def simulate_paths(df, fdates, n_paths, mean_revert, noise_mult, atr_mult) -> np.ndarray:
    close  = df["Close"].astype(float)
    ret    = df["RET"].astype(float)
    last_p = float(close.iloc[-1])
    ma20   = float(df["MA20"].iloc[-1])
    T      = len(fdates)

    # ── EWMA drift（近期加權更敏感）──
    drift  = float(ret.ewm(span=10, adjust=False).mean().iloc[-1])

    # ── 動態波動率（GARCH 或 rolling std）──
    sigma_t = get_sigma_forecast(df, T) * noise_mult

    # ── 動態偏移單位 = 基準 sigma 的 15% ──
    sigma_base = float(df["SIGMA20"].iloc[-1])
    bias_unit  = sigma_base * 0.15

    macd_val  = float(df["MACD"].iloc[-1])
    macd_sig  = float(df["MACD_signal"].iloc[-1])
    bb_pct    = float(df["BB_pct"].iloc[-1])
    k_val     = float(df["K"].iloc[-1])

    macd_bias = bias_unit  if macd_val > macd_sig  else -bias_unit
    bb_bias   = -bias_unit if bb_pct > 0.85        else (bias_unit if bb_pct < 0.15 else 0.0)
    kd_bias   = bias_unit  if k_val < 20           else (-bias_unit if k_val > 80   else 0.0)

    # ── OBV 斜率偏移（5日）──
    obv_slope = float(df["OBV"].tail(5).diff().mean())
    obv_norm  = obv_slope / (abs(obv_slope) + 1e-9)
    obv_bias  = obv_norm * bias_unit * 0.5

    adj_drift = drift + macd_bias + bb_bias + kd_bias + obv_bias

    # ── 擬合 t 分布自由度 ──
    ret_clean = ret.dropna().values
    try:
        df_t, _, scale_t = t_dist.fit(ret_clean, floc=0)
        df_t = max(df_t, 2.5)   # 最低自由度 2.5，確保有限方差
    except Exception:
        df_t, scale_t = 5.0, sigma_base

    # ── Antithetic Variates：前半路徑 + 鏡像後半 ──
    half = n_paths // 2
    rng  = np.random.default_rng(42)

    paths  = np.zeros((n_paths, T))
    prices = np.full(n_paths, last_p)

    for t in range(T):
        raw_half = t_dist.rvs(df=df_t, loc=0, scale=scale_t * sigma_t[t],
                              size=half, random_state=rng.integers(1e9))
        eps   = np.concatenate([raw_half, -raw_half])
        mr    = -mean_revert * ((prices - ma20) / max(ma20, 1e-9)) / max(T, 1)
        prices = prices * np.exp(adj_drift + mr + eps)
        paths[:, t] = prices

    return paths


# ─────────────────────────────────────────────
# 找轉彎點
# ─────────────────────────────────────────────
def find_turning_points(med: np.ndarray):
    s    = pd.Series(med)
    d    = s.diff().fillna(0)
    sign = np.sign(d.values)
    sign[np.abs(d.values) < (np.nanstd(d.values) * 0.05 + 1e-12)] = 0
    valleys, peaks = [], []
    for t in range(1, len(sign) - 1):
        if sign[t] < 0 and sign[t + 1] > 0: valleys.append(t)
        if sign[t] > 0 and sign[t + 1] < 0: peaks.append(t)
    valley_idx = int(s.iloc[valleys].idxmin()) if valleys else None
    peak_idx   = int(s.iloc[peaks].idxmax())   if peaks   else None
    trend      = float(s.iloc[-1] - s.iloc[0])
    std_s      = float(np.nanstd(s.values))
    trend_text = ""
    if valley_idx is None and peak_idx is None:
        if abs(trend) < max(1e-9, std_s * 0.2):
            trend_text = "這段看起來大多是『來回晃』，沒有很明顯的低點或高點。"
        elif trend > 0:
            trend_text = "這段看起來是『慢慢往上』，沒有明顯轉彎低點。"
        else:
            trend_text = "這段看起來是『慢慢往下』，沒有明顯轉彎高點。"
    return valley_idx, peak_idx, trend_text


# ─────────────────────────────────────────────
# 產生報告（加入 5%/95% 極端情境）
# ─────────────────────────────────────────────
def make_report(df, fdates, paths, capital, risk_pct, atr_mult):
    last_close = float(df["Close"].iloc[-1])
    atr        = float(df["ATR"].iloc[-1])
    rsi        = float(df["RSI"].iloc[-1])
    macd_val   = float(df["MACD"].iloc[-1])
    macd_sig   = float(df["MACD_signal"].iloc[-1])
    k_val      = float(df["K"].iloc[-1])
    d_val      = float(df["D"].iloc[-1])
    bb_pct     = float(df["BB_pct"].iloc[-1])
    bb_upper   = float(df["BB_upper"].iloc[-1])
    bb_lower   = float(df["BB_lower"].iloc[-1])

    med  = np.median(paths, axis=0)
    p5   = np.percentile(paths,  5, axis=0)
    p20  = np.percentile(paths, 20, axis=0)
    p80  = np.percentile(paths, 80, axis=0)
    p95  = np.percentile(paths, 95, axis=0)

    prev       = np.concatenate([np.full((paths.shape[0], 1), last_close), paths[:, :-1]], axis=1)
    up_prob    = (paths > prev).mean(axis=0) * 100.0

    stop_price = last_close - atr_mult * atr
    hit_stop_p = (paths <= stop_price).mean(axis=0) * 100.0

    valley_idx, peak_idx, trend_text = find_turning_points(med)
    buy_day    = fdates[valley_idx].date() if valley_idx is not None else None
    sell_day   = fdates[peak_idx].date()   if peak_idx   is not None else None
    buy_action = "建議**分批小量買**（不要一次全買）" if buy_day  else "沒有明顯低點，建議**觀望或少量分批**"
    sell_action= "建議**先賣一部分收錢**"             if sell_day else "沒有明顯高點，用**停損線保護**就好"

    first_hit = np.full(paths.shape[0], -1, dtype=int)
    for i in range(paths.shape[0]):
        hits = np.where(paths[i] <= stop_price)[0]
        if hits.size > 0:
            first_hit[i] = int(hits[0])
    hit_any_prob = (first_hit >= 0).mean() * 100.0
    if hit_any_prob >= 5:
        mode_idx  = int(pd.Series(first_hit[first_hit >= 0]).mode().iloc[0])
        stop_text = f"最可能在 **{fdates[mode_idx].date()}** 碰到停損（機率 {hit_any_prob:.1f}%）"
    else:
        stop_text = f"碰到停損機率低（約 {hit_any_prob:.1f}%）"

    risk_money     = capital * risk_pct
    per_share_risk = max(last_close - stop_price, 1e-6)
    shares         = int(risk_money // per_share_risk)

    # ── 多空訊號評分（加權：MACD 權重 x2）──
    score   = 0
    signals = []
    if rsi < 35:
        score += 1; signals.append("🟢 RSI 超賣（有機會反彈）")
    elif rsi > 65:
        score -= 1; signals.append("🔴 RSI 超買（小心回跌）")
    else:
        signals.append("🟡 RSI 中性")

    # MACD 權重 x2
    if macd_val > macd_sig:
        score += 2; signals.append("🟢 MACD 黃金交叉（偏多，權重x2）")
    else:
        score -= 2; signals.append("🔴 MACD 死亡交叉（偏空，權重x2）")

    if k_val < 20:
        score += 1; signals.append("🟢 KD 超賣（K<20，偏多）")
    elif k_val > 80:
        score -= 1; signals.append("🔴 KD 超買（K>80，偏空）")
    else:
        signals.append("🟡 KD 中性")

    if bb_pct < 0.2:
        score += 1; signals.append("🟢 布林通道下緣（偏多）")
    elif bb_pct > 0.8:
        score -= 1; signals.append("🔴 布林通道上緣（偏空）")
    else:
        signals.append("🟡 布林通道中間")

    # 評分範圍 -5 到 +5（MACD x2）
    if score >= 2:
        mood_emoji  = "🟢"; action_line = f"多指標偏多，**最多買 {shares:,} 股**（仍要設停損）"
    elif score <= -2:
        mood_emoji  = "🔴"; action_line = "多指標偏空，**建議觀望或不買**"
    elif 45 <= rsi <= 55:
        mood_emoji  = "😐"; action_line = "方向不明，**建議先不要買**"
        shares      = 0
    else:
        mood_emoji  = "🟡"; action_line = f"訊號混雜，**最多買 {shares:,} 股**（謹慎操作）"

    mood = (
        "多指標同時偏多，反彈機率較高。" if score >= 2 else
        "多指標同時偏空，下跌風險較高。" if score <= -2 else
        "指標訊號混雜，方向不明，來回晃的機率高。"
    )
    extra = f"\n💡 補充：{trend_text}" if trend_text else ""

    sd = {
        "mood": mood, "mood_emoji": mood_emoji, "action_line": action_line,
        "buy_day": buy_day, "buy_action": buy_action,
        "sell_day": sell_day, "sell_action": sell_action,
        "stop_price": stop_price, "stop_text": stop_text, "extra": extra,
        "rsi": rsi, "last_close": last_close, "atr": atr,
        "macd_val": macd_val, "macd_sig": macd_sig,
        "k_val": k_val, "d_val": d_val,
        "bb_pct": bb_pct, "bb_upper": bb_upper, "bb_lower": bb_lower,
        "signals": signals, "score": score,
    }
    table = pd.DataFrame({
        "日期":              [d.date() for d in fdates],
        "可能價格(中間值)":  np.round(med, 2),
        "極端最低(5%)":      np.round(p5,  2),
        "可能範圍_低(20%)":  np.round(p20, 2),
        "可能範圍_高(80%)":  np.round(p80, 2),
        "極端最高(95%)":     np.round(p95, 2),
        "上漲機率(%)":       np.round(up_prob, 1),
        "碰到停損機率(%)":   np.round(hit_stop_p, 1),
    })
    return sd, table, stop_price, med, p20, p80, p5, p95


# ─────────────────────────────────────────────
# 歷史走勢圖
# ─────────────────────────────────────────────
def build_main_chart(df, fdates, med, p20, p80, p5, p95, stop_price):
    hist       = df.tail(80).copy()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    fdates_ts  = [pd.Timestamp(d) for d in fdates]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.2, 0.2, 0.15],
        vertical_spacing=0.04,
        subplot_titles=("📈 股價 + 布林通道 + 預測（含5%/95%極端區間）", "MACD", "KD 指標", "成交量")
    )

    # ── Row 1：股價 + 布林 + MA + 預測（雙層區間）──
    fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_upper"],
        line=dict(color="rgba(150,150,255,0.4)", width=1, dash="dot"),
        name="布林上軌", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_lower"],
        line=dict(color="rgba(150,150,255,0.4)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(150,150,255,0.06)",
        name="布林下軌", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"],
        name="歷史收盤", line=dict(color="#4A90D9", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA20"],
        name="MA20", line=dict(color="#FFA500", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA60"],
        name="MA60", line=dict(color="#9B59B6", width=1.2, dash="dot")), row=1, col=1)

    # 外層 5%~95% 極端區間（淺色）
    fig.add_trace(go.Scatter(
        x=fdates_ts + fdates_ts[::-1],
        y=list(p95) + list(p5[::-1]),
        fill="toself", fillcolor="rgba(100,200,100,0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        name="極端區間(5~95%)", hoverinfo="skip"), row=1, col=1)

    # 內層 20%~80% 主要區間（深色）
    fig.add_trace(go.Scatter(
        x=fdates_ts + fdates_ts[::-1],
        y=list(p80) + list(p20[::-1]),
        fill="toself", fillcolor="rgba(100,200,100,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="主要區間(20~80%)", hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Scatter(x=fdates_ts, y=med,
        name="預測中間值", line=dict(color="#27AE60", width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=fdates_ts, y=[stop_price]*len(fdates_ts),
        name="停損線", line=dict(color="#E74C3C", width=1.5, dash="longdash")), row=1, col=1)

    # ── Row 2：MACD ──
    colors_hist = ["#27AE60" if v >= 0 else "#E74C3C" for v in hist["MACD_hist"]]
    fig.add_trace(go.Bar(x=hist.index, y=hist["MACD_hist"],
        marker_color=colors_hist, name="MACD柱", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD"],
        line=dict(color="#3498DB", width=1.5), name="MACD快線"), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD_signal"],
        line=dict(color="#E74C3C", width=1.5), name="MACD慢線"), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", row=2, col=1)

    # ── Row 3：KD ──
    fig.add_trace(go.Scatter(x=hist.index, y=hist["K"],
        line=dict(color="#F39C12", width=1.5), name="K值"), row=3, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["D"],
        line=dict(color="#9B59B6", width=1.5), name="D值"), row=3, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="rgba(231,76,60,0.4)",  row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="rgba(39,174,96,0.4)",  row=3, col=1)

    # ── Row 4：成交量 ──
    vol_colors = ["#27AE60" if c >= o else "#E74C3C"
                  for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"],
        marker_color=vol_colors, name="成交量"), row=4, col=1)

    fig.update_layout(
        height=800, template="plotly_dark",
        legend=dict(orientation="h", y=-0.06),
        margin=dict(l=40, r=20, t=45, b=20),
        hovermode="x unified"
    )
    return fig


# ─────────────────────────────────────────────
# 每日預測視覺化圖（含 5%/95%）
# ─────────────────────────────────────────────
def build_forecast_chart(table):
    dates     = [str(d) for d in table["日期"].values]
    med_vals  = table["可能價格(中間值)"].values
    p5_vals   = table["極端最低(5%)"].values
    p20_vals  = table["可能範圍_低(20%)"].values
    p80_vals  = table["可能範圍_高(80%)"].values
    p95_vals  = table["極端最高(95%)"].values
    up_vals   = table["上漲機率(%)"].values
    stop_vals = table["碰到停損機率(%)"].values

    bar_colors  = ["#27AE60" if v >= 55 else "#F39C12" if v >= 45 else "#E74C3C" for v in up_vals]
    stop_colors = ["#E74C3C" if v >= 10 else "#F39C12" if v >= 3  else "#27AE60" for v in stop_vals]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.06,
        subplot_titles=("💰 預測價格區間（含極端5%/95%）", "⬆️ 明天漲機率 (%)", "🛑 碰到停損機率 (%)")
    )

    # 外層極端區間
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1], y=list(p95_vals) + list(p5_vals[::-1]),
        fill="toself", fillcolor="rgba(100,180,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="極端區間(5~95%)", hoverinfo="skip"), row=1, col=1)

    # 內層主要區間
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1], y=list(p80_vals) + list(p20_vals[::-1]),
        fill="toself", fillcolor="rgba(100,180,255,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="主要區間(20~80%)", hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=p95_vals, mode="lines",
        line=dict(color="rgba(100,180,255,0.35)", width=1, dash="dot"),
        name="極端最高(95%)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=p80_vals, mode="lines",
        line=dict(color="rgba(100,180,255,0.6)", width=1, dash="dot"),
        name="最好情況(80%)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=p20_vals, mode="lines",
        line=dict(color="rgba(255,120,120,0.6)", width=1, dash="dot"),
        name="最壞情況(20%)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=p5_vals, mode="lines",
        line=dict(color="rgba(255,80,80,0.35)", width=1, dash="dot"),
        name="極端最低(5%)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=med_vals, mode="lines+markers",
        line=dict(color="#F1C40F", width=2.5),
        marker=dict(size=7, color="#F1C40F"),
        name="預測中間值",
        hovertemplate="<b>%{x}</b><br>預測價：%{y:,.2f}<extra></extra>"), row=1, col=1)

    fig.add_trace(go.Bar(x=dates, y=up_vals, marker_color=bar_colors,
        name="上漲機率",
        text=[f"{v:.1f}%" for v in up_vals], textposition="outside",
        hovertemplate="<b>%{x}</b><br>上漲機率：%{y:.1f}%<extra></extra>"), row=2, col=1)
    fig.add_hline(y=55, line_dash="dash", line_color="rgba(39,174,96,0.5)",  row=2, col=1)
    fig.add_hline(y=45, line_dash="dash", line_color="rgba(231,76,60,0.5)",  row=2, col=1)

    fig.add_trace(go.Bar(x=dates, y=stop_vals, marker_color=stop_colors,
        name="碰停損機率",
        text=[f"{v:.1f}%" for v in stop_vals], textposition="outside",
        hovertemplate="<b>%{x}</b><br>碰停損機率：%{y:.1f}%<extra></extra>"), row=3, col=1)
    fig.add_hline(y=10, line_dash="dash", line_color="rgba(231,76,60,0.5)",  row=3, col=1)
    fig.add_hline(y=3,  line_dash="dash", line_color="rgba(243,156,18,0.5)", row=3, col=1)

    fig.update_layout(
        height=700, template="plotly_dark",
        legend=dict(orientation="h", y=-0.06, x=0),
        margin=dict(l=40, r=20, t=45, b=20),
        hovermode="x unified", bargap=0.35,
    )
    return fig


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="📘 股票助手（精準版）", layout="wide", page_icon="📘")
st.title("📘 股票助手｜低點 / 高點預測 + 多指標風控")
st.caption("👆 看三件事：**要不要買** → **低點日怎麼買** → **高點日怎麼賣**")

if HAS_ARCH:
    st.info("⚡ 已啟用 GARCH(1,1) 動態波動率模型（更精準）")
else:
    st.warning("⚠️ 未安裝 `arch` 套件，使用 rolling std 替代。安裝：`pip install arch`")

with st.sidebar:
    st.header("⚙️ 設定")
    code_raw   = st.text_input("股票代號（台股輸入數字即可）", "2330").strip()
    code_input = code_raw.replace(".TW", "").replace(".TWO", "").upper()
    st.divider()
    capital  = st.number_input("資金（元）", min_value=0.0, value=200_000.0, step=10_000.0)
    risk_pct = st.slider("最多可以賠幾 %", 1, 20, 10) / 100.0
    st.divider()
    forecast_days = st.slider("看幾個交易日", 5, 20, FORECAST_DAYS_DEFAULT)
    sim_paths_n   = st.slider("模擬條數（越多越穩）", 200, 1200, SIM_PATHS_DEFAULT, 100)
    mean_revert   = st.slider("均值回歸強度", 0.0, 0.6, 0.25, 0.05)
    noise_mult    = st.slider("波動倍數", 0.5, 2.0, 1.0, 0.1)
    atr_mult      = st.slider("停損 ATR 倍數", 1.5, 3.5, 2.5, 0.5,
                               help="停損線 = 現價 - N × ATR；越大停損空間越寬")

col_run, col_reset = st.columns([3, 1])
with col_run:
    run_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)
with col_reset:
    if st.button("🔄 清除快取", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if run_btn:
    with st.spinner("📡 抓取資料中（自動偵測上市/上櫃）..."):
        df_raw, used_code = download_data(code_input)
    if df_raw.empty:
        st.error("❌ 抓不到資料，請確認代號是否正確。")
        st.stop()

    market_label = "上櫃（OTC）" if used_code.endswith(".TWO") else "上市（TWSE）"
    st.success(f"✅ 成功抓到資料：**{used_code}**（{market_label}）")

    df = add_indicators(df_raw)
    if len(df) < 80:
        st.error("❌ 資料不足 80 個交易日，無法計算。")
        st.stop()

    fdates = future_dates(df, horizon=forecast_days)

    with st.spinner("🎲 蒙地卡羅模擬中（t分布 + GARCH + Antithetic Variates）..."):
        paths = simulate_paths(df, fdates, sim_paths_n, mean_revert, noise_mult, atr_mult)

    sd, table, stop_price, med, p20, p80, p5, p95 = make_report(
        df, fdates, paths, capital, risk_pct, atr_mult)

    # ── 指標卡片（8格）──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("現價",    f"{sd['last_close']:.2f}")
    c2.metric("RSI(14)", f"{sd['rsi']:.1f}",
              delta="超買⚠️" if sd['rsi'] > 65 else ("超賣💡" if sd['rsi'] < 35 else "中性"))
    c3.metric("MACD",    f"{sd['macd_val']:.3f}",
              delta="黃金交叉🟢" if sd['macd_val'] > sd['macd_sig'] else "死亡交叉🔴")
    c4.metric("ATR(14)", f"{sd['atr']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("停損線", f"{stop_price:.2f}",
              delta=f"-{(sd['last_close'] - stop_price) / sd['last_close'] * 100:.1f}%",
              delta_color="inverse")
    c6.metric("K值", f"{sd['k_val']:.1f}",
              delta="超賣💡" if sd['k_val'] < 20 else ("超買⚠️" if sd['k_val'] > 80 else "中性"))
    c7.metric("D值", f"{sd['d_val']:.1f}")
    c8.metric("布林位置", f"{sd['bb_pct']*100:.0f}%",
              delta="靠上軌⚠️" if sd['bb_pct'] > 0.8 else ("靠下軌💡" if sd['bb_pct'] < 0.2 else "中間"))

    st.divider()

    # ── 多空訊號評分（MACD 加權 x2，範圍 -5 到 +5）──
    st.subheader(f"{sd['mood_emoji']} 多指標綜合訊號（評分：{sd['score']:+d} / -5 到 +5，MACD 加權）")
    st.info(f"{sd['mood']}\n\n**操作建議：** {sd['action_line']}")
    sig_cols = st.columns(len(sd['signals']))
    for col, sig in zip(sig_cols, sd['signals']):
        col.markdown(f"**{sig}**")

    st.divider()

    # ── 低點/高點/停損 ──
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("🟢 低點日（買進參考）")
        if sd['buy_day']:
            st.success(f"📅 **{sd['buy_day']}**\n\n{sd['buy_action']}")
        else:
            st.warning(sd['buy_action'])
    with col_r:
        st.subheader("🔴 高點日（賣出參考）")
        if sd['sell_day']:
            st.error(f"📅 **{sd['sell_day']}**\n\n{sd['sell_action']}")
        else:
            st.warning(sd['sell_action'])

    st.subheader("🛡️ 停損資訊")
    st.warning(f"停損價：**{stop_price:.2f}**　｜　{sd['stop_text']}{sd['extra']}")

    st.divider()

    # ── 歷史走勢圖 ──
    st.subheader("📈 歷史走勢（布林通道 / MACD / KD / 成交量）+ 雙層預測區間")
    st.plotly_chart(
        build_main_chart(df, fdates, med, p20, p80, p5, p95, stop_price),
        use_container_width=True)

    st.divider()

    # ── 每日預測視覺化 ──
    st.subheader("📊 每日預測（視覺化，含極端 5% / 95%）")
    st.plotly_chart(build_forecast_chart(table), use_container_width=True)

    leg1, leg2 = st.columns(2)
    with leg1:
        st.markdown("**⬆️ 上漲機率圖例**")
        st.markdown("🟢 **≥55%** 偏漲　🟡 **45~55%** 持平　🔴 **<45%** 偏跌")
    with leg2:
        st.markdown("**🛑 碰停損機率圖例**")
        st.markdown("✅ **<3%** 安全　⚠️ **3~10%** 偏高　🚨 **≥10%** 危險")

    # ── 詳細預測數據表 ──
    st.divider()
    st.subheader("📋 每日預測數據（含極端情境）")
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.caption("⚠️ 免責聲明：此工具僅供輔助思考，不構成投資建議，請自行評估風險。")
