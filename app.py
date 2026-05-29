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

warnings.filterwarnings("ignore")

TZ_TW = pytz.timezone("Asia/Taipei")
FORECAST_DAYS_DEFAULT = 10
SIM_PATHS_DEFAULT = 800

# ─────────────────────────────────────────────
# 資料下載
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

    df["MA20"]  = close.rolling(20).mean()
    df["MA60"]  = close.rolling(60).mean()
    df["RSI"]   = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd_ind          = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd_ind.macd()
    df["MACD_signal"] = macd_ind.macd_signal()
    df["MACD_hist"]   = macd_ind.macd_diff()

    stoch   = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()

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
# 純 NumPy GARCH(1,1)
# ─────────────────────────────────────────────
def get_sigma_forecast(df: pd.DataFrame, T: int) -> np.ndarray:
    ret   = df["RET"].dropna().values
    n     = len(ret)
    omega, alpha, beta = 1e-6, 0.10, 0.85
    h = np.full(n, np.var(ret))
    for i in range(1, n):
        h[i] = omega + alpha * ret[i-1]**2 + beta * h[i-1]
    long_run = omega / max(1 - alpha - beta, 1e-6)
    sigma_t  = np.zeros(T)
    h_curr   = h[-1]
    for t in range(T):
        h_curr     = omega + (alpha + beta) * h_curr
        h_curr     = h_curr * 0.7 + long_run * 0.3
        sigma_t[t] = np.sqrt(max(h_curr, 1e-8))
    return np.clip(sigma_t, 1e-4, None)


# ─────────────────────────────────────────────
# 蒙地卡羅模擬（修正版）
# 修正點：
#   1. 移除 Antithetic Variates（對稱導致 up_prob=50% 計算錯誤）
#   2. 隨機種子改為 None（每次真實隨機）
#   3. sigma 下限保護，確保路徑有足夠波動
#   4. prices.copy() 確保陣列不互相污染
# ─────────────────────────────────────────────
def simulate_paths(df, n_paths: int, T: int, mean_revert: float, noise_mult: float) -> np.ndarray:
    close  = df["Close"].astype(float)
    ret    = df["RET"].astype(float)
    last_p = float(close.iloc[-1])
    ma20   = float(df["MA20"].iloc[-1])

    # EWMA drift
    drift = float(ret.ewm(span=10, adjust=False).mean().iloc[-1])

    # sigma：GARCH 與 rolling std 取最大值
    sigma_roll = float(df["SIGMA20"].iloc[-1])
    if not np.isfinite(sigma_roll) or sigma_roll <= 0:
        sigma_roll = max(float(ret.tail(60).std()), 0.008)
    sigma_garch = get_sigma_forecast(df, T)
    sigma_t = np.maximum(sigma_garch, sigma_roll) * noise_mult

    # 動態偏移（相對 sigma 縮放）
    bias_unit = sigma_roll * 0.20
    macd_val  = float(df["MACD"].iloc[-1])
    macd_sig  = float(df["MACD_signal"].iloc[-1])
    bb_pct    = float(df["BB_pct"].iloc[-1])
    k_val     = float(df["K"].iloc[-1])
    macd_bias = bias_unit  if macd_val > macd_sig else -bias_unit
    bb_bias   = -bias_unit if bb_pct > 0.85       else (bias_unit if bb_pct < 0.15 else 0.0)
    kd_bias   = bias_unit  if k_val < 20          else (-bias_unit if k_val > 80   else 0.0)
    obv_slope = float(df["OBV"].tail(5).diff().mean())
    obv_norm  = obv_slope / (abs(obv_slope) + 1e-9)
    obv_bias  = obv_norm * bias_unit * 0.5
    adj_drift = drift + macd_bias + bb_bias + kd_bias + obv_bias

    # Student-t，確保 scale 不為零
    ret_clean = ret.dropna().values
    try:
        df_t, _, scale_t = t_dist.fit(ret_clean, floc=0)
        df_t    = float(np.clip(df_t, 2.5, 30.0))
        scale_t = max(float(scale_t), sigma_roll * 0.8)
    except Exception:
        df_t, scale_t = 5.0, sigma_roll

    # 純隨機路徑（不使用 Antithetic）
    rng    = np.random.default_rng()          # 每次真實隨機
    paths  = np.zeros((n_paths, T))
    prices = np.full(n_paths, last_p, dtype=float)

    for t in range(T):
        s   = float(sigma_t[t])
        eps = t_dist.rvs(df=df_t, loc=0, scale=scale_t * s,
                         size=n_paths,
                         random_state=rng.integers(int(1e9)))
        mr     = -mean_revert * ((prices - ma20) / max(ma20, 1e-9)) / max(T, 1)
        prices = prices * np.exp(adj_drift + mr + eps)
        prices = np.clip(prices, last_p * 0.4, last_p * 2.5)
        paths[:, t] = prices.copy()

    return paths


# ─────────────────────────────────────────────
# 次日 + 後三日核心計算（修正 up_prob）
# ─────────────────────────────────────────────
def make_short_forecast(last_close: float, fdates, paths: np.ndarray) -> list:
    results = []
    for i in range(min(4, paths.shape[1])):
        day_paths = paths[:, i]
        base      = last_close if i == 0 else paths[:, i-1]   # 逐路徑比較

        med     = float(np.median(day_paths))
        p10     = float(np.percentile(day_paths, 10))
        p90     = float(np.percentile(day_paths, 90))
        up_prob = float(np.mean(day_paths > base) * 100)

        ref     = last_close if i == 0 else float(np.median(paths[:, i-1]))
        chg_pct = (med - ref) / ref * 100

        if up_prob >= 55:
            direction, color = "⬆️ 偏漲", "#2ECC71"
        elif up_prob <= 45:
            direction, color = "⬇️ 偏跌", "#E74C3C"
        else:
            direction, color = "↔️ 盤整", "#F39C12"

        label = "次日" if i == 0 else f"第{i+1}日"
        results.append({
            "label": label, "date": fdates[i].date(),
            "med": med, "p10": p10, "p90": p90,
            "up_prob": up_prob, "chg_pct": chg_pct,
            "direction": direction, "color": color,
        })
    return results


# ─────────────────────────────────────────────
# 圖一：最直觀卡片式（水平進度條）
# ─────────────────────────────────────────────
def build_prob_chart(sf: list, last_close: float):
    """4 欄水平進度條：長條長度 = 上漲機率，顏色直接判斷方向"""
    fig = make_subplots(
        rows=4, cols=1,
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )

    for i, r in enumerate(sf):
        prob  = r["up_prob"]
        color = r["color"]
        down  = 100 - prob

        # 上漲段
        fig.add_trace(go.Bar(
            x=[prob], y=[r["label"]],
            orientation="h",
            marker=dict(color=color, line=dict(color="white", width=0)),
            showlegend=False,
            hovertemplate=f"<b>{r['label']} {r['date']}</b><br>上漲機率：{prob:.1f}%<extra></extra>",
            width=0.6,
        ), row=i+1, col=1)

        # 下跌段（灰色補足 100%）
        fig.add_trace(go.Bar(
            x=[down], y=[r["label"]],
            orientation="h",
            marker=dict(color="rgba(255,255,255,0.08)", line=dict(color="white", width=0)),
            showlegend=False,
            hoverinfo="skip",
            width=0.6,
        ), row=i+1, col=1)

        # 50% 基準線
        axis_key = "xaxis" if i == 0 else f"xaxis{i+1}"
        fig.update_layout(**{
            axis_key: dict(
                range=[0, 110],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            )
        })

        # 機率數字（條內）
        fig.add_annotation(
            xref=axis_key, yref=f"y{i+1}" if i > 0 else "y",
            x=prob / 2, y=r["label"],
            text=f"<b>{prob:.1f}%</b>",
            showarrow=False,
            font=dict(color="white", size=16),
        )

        # 右側：方向 + 預測價 + 漲跌幅 + 區間
        fig.add_annotation(
            xref=axis_key, yref=f"y{i+1}" if i > 0 else "y",
            x=102, y=r["label"],
            text=(
                f"<b>{r['direction']}</b>　"
                f"預測價 <b style='color:{color}'>{r['med']:.2f}</b> "
                f"（{r['chg_pct']:+.2f}%）　"
                f"區間 {r['p10']:.2f} ~ {r['p90']:.2f}"
            ),
            showarrow=False,
            xanchor="left",
            font=dict(color="white", size=14),
        )

        # 50% 分隔線
        fig.add_vline(x=50,
                      line_dash="dash",
                      line_color="rgba(255,255,255,0.35)",
                      row=i+1, col=1)

    fig.update_layout(
        barmode="stack",
        height=340,
        template="plotly_dark",
        title=dict(
            text="📊 次日 + 後三日｜上漲機率進度條（綠=偏漲 / 橙=盤整 / 紅=偏跌，虛線=50%基準）",
            font=dict(size=15),
        ),
        margin=dict(l=60, r=20, t=60, b=10),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────
# 圖二：趨勢折線（從現價出發）
# ─────────────────────────────────────────────
def build_trend_chart(sf: list, last_close: float):
    x = ["現在"] + [f"{r['label']}\n{r['date']}" for r in sf]
    y_med = [last_close] + [r["med"]  for r in sf]
    y_p10 = [last_close] + [r["p10"]  for r in sf]
    y_p90 = [last_close] + [r["p90"]  for r in sf]
    colors = ["white"] + [r["color"]  for r in sf]
    sizes  = [10]      + [16]         * len(sf)

    fig = go.Figure()

    # 信賴帶
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=y_p90 + y_p10[::-1],
        fill="toself", fillcolor="rgba(100,180,255,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="10%~90% 區間", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=x, y=y_p90, mode="lines",
        line=dict(color="rgba(100,180,255,0.4)", width=1, dash="dot"),
        name="90% 上限"))
    fig.add_trace(go.Scatter(x=x, y=y_p10, mode="lines",
        line=dict(color="rgba(255,100,100,0.4)", width=1, dash="dot"),
        name="10% 下限"))

    # 中間值折線 + 數字標籤
    fig.add_trace(go.Scatter(
        x=x, y=y_med,
        mode="lines+markers+text",
        line=dict(color="#F1C40F", width=3),
        marker=dict(color=colors, size=sizes, line=dict(color="white", width=2)),
        text=[f"  {v:.2f}" for v in y_med],
        textposition="top right",
        textfont=dict(size=13, color="white"),
        name="預測中間值",
        hovertemplate="<b>%{x}</b><br>預測股價：%{y:,.2f}<extra></extra>",
    ))

    # 每個預測點下方標漲跌幅
    for r in sf:
        fig.add_annotation(
            x=f"{r['label']}\n{r['date']}", y=r["p10"],
            text=f"<b>{r['chg_pct']:+.2f}%</b>",
            showarrow=False, yshift=-20,
            font=dict(color=r["color"], size=14),
        )

    fig.update_layout(
        height=400, template="plotly_dark",
        title=dict(
            text="📉 次日 + 後三日趨勢折線（黃點=預測中間價 / 色帶=10%~90%合理區間 / 下方=漲跌幅）",
            font=dict(size=14)
        ),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=50, r=20, t=60, b=30),
        hovermode="x unified",
        yaxis=dict(title="股價"),
    )
    return fig


# ─────────────────────────────────────────────
# 圖三：歷史走勢主圖
# ─────────────────────────────────────────────
def build_main_chart(df, fdates, paths, stop_price):
    med = np.median(paths, axis=0)
    p20 = np.percentile(paths, 20, axis=0)
    p80 = np.percentile(paths, 80, axis=0)
    p5  = np.percentile(paths,  5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    hist       = df.tail(80).copy()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    fdates_ts  = [pd.Timestamp(d) for d in fdates]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.20, 0.20, 0.15],
        vertical_spacing=0.04,
        subplot_titles=("📈 股價 + 布林通道 + 雙層預測區間", "MACD", "KD 指標", "成交量")
    )

    fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_upper"],
        line=dict(color="rgba(150,150,255,0.4)", width=1, dash="dot"), name="布林上軌"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_lower"],
        line=dict(color="rgba(150,150,255,0.4)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(150,150,255,0.06)", name="布林下軌"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"],
        name="歷史收盤", line=dict(color="#4A90D9", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA20"],
        name="MA20", line=dict(color="#FFA500", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA60"],
        name="MA60", line=dict(color="#9B59B6", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fdates_ts + fdates_ts[::-1], y=list(p95) + list(p5[::-1]),
        fill="toself", fillcolor="rgba(100,200,100,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="極端區間(5~95%)", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fdates_ts + fdates_ts[::-1], y=list(p80) + list(p20[::-1]),
        fill="toself", fillcolor="rgba(100,200,100,0.20)",
        line=dict(color="rgba(0,0,0,0)"), name="主要區間(20~80%)", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fdates_ts, y=med,
        name="預測中間值", line=dict(color="#27AE60", width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=fdates_ts, y=[stop_price]*len(fdates_ts),
        name="停損線", line=dict(color="#E74C3C", width=1.5, dash="longdash")), row=1, col=1)

    colors_h = ["#27AE60" if v >= 0 else "#E74C3C" for v in hist["MACD_hist"]]
    fig.add_trace(go.Bar(x=hist.index, y=hist["MACD_hist"],
        marker_color=colors_h, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD"],
        line=dict(color="#3498DB", width=1.5), name="MACD快線"), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD_signal"],
        line=dict(color="#E74C3C", width=1.5), name="MACD慢線"), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", row=2, col=1)

    fig.add_trace(go.Scatter(x=hist.index, y=hist["K"],
        line=dict(color="#F39C12", width=1.5), name="K值"), row=3, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["D"],
        line=dict(color="#9B59B6", width=1.5), name="D值"), row=3, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="rgba(231,76,60,0.4)",  row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="rgba(39,174,96,0.4)",  row=3, col=1)

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
# 多空訊號評分
# ─────────────────────────────────────────────
def calc_signals(df):
    rsi      = float(df["RSI"].iloc[-1])
    macd_val = float(df["MACD"].iloc[-1])
    macd_sig = float(df["MACD_signal"].iloc[-1])
    k_val    = float(df["K"].iloc[-1])
    bb_pct   = float(df["BB_pct"].iloc[-1])

    score, signals = 0, []
    if rsi < 35:    score += 1; signals.append("🟢 RSI 超賣")
    elif rsi > 65:  score -= 1; signals.append("🔴 RSI 超買")
    else:                        signals.append("🟡 RSI 中性")

    if macd_val > macd_sig: score += 2; signals.append("🟢 MACD 黃金交叉（x2）")
    else:                   score -= 2; signals.append("🔴 MACD 死亡交叉（x2）")

    if k_val < 20:    score += 1; signals.append("🟢 KD 超賣")
    elif k_val > 80:  score -= 1; signals.append("🔴 KD 超買")
    else:                          signals.append("🟡 KD 中性")

    if bb_pct < 0.2:   score += 1; signals.append("🟢 布林下緣")
    elif bb_pct > 0.8: score -= 1; signals.append("🔴 布林上緣")
    else:                           signals.append("🟡 布林中間")

    return score, signals, rsi, macd_val, macd_sig, k_val, bb_pct


# ─────────────────────────────────────────────
# 完整預測數據表
# ─────────────────────────────────────────────
def make_full_table(last_close, fdates, paths, stop_price):
    med = np.median(paths, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    p5  = np.percentile(paths,  5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    up_prob    = np.zeros(paths.shape[1])
    up_prob[0] = np.mean(paths[:, 0] > last_close) * 100
    for i in range(1, paths.shape[1]):
        up_prob[i] = np.mean(paths[:, i] > paths[:, i-1]) * 100

    hit_stop = np.mean(paths <= stop_price, axis=0) * 100
    prev_med = np.concatenate([[last_close], med[:-1]])
    chg_pct  = (med - prev_med) / prev_med * 100

    directions = []
    for p in up_prob:
        if p >= 55:   directions.append("⬆️ 偏漲")
        elif p <= 45: directions.append("⬇️ 偏跌")
        else:         directions.append("↔️ 盤整")

    return pd.DataFrame({
        "日期":           [d.date() for d in fdates],
        "方向":           directions,
        "預測股價":       np.round(med, 2),
        "漲跌幅(%)":      np.round(chg_pct, 2),
        "低(10%)":        np.round(p10, 2),
        "高(90%)":        np.round(p90, 2),
        "極端低(5%)":     np.round(p5,  2),
        "極端高(95%)":    np.round(p95, 2),
        "上漲機率(%)":    np.round(up_prob, 1),
        "碰停損機率(%)":  np.round(hit_stop, 1),
    })


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="📘 股票助手", layout="wide", page_icon="📘")
st.title("📘 股票助手｜次日 + 後三日明確預測")
st.caption("🎯 三步驟：**① 次日漲跌機率** → **② 後三日趨勢折線** → **③ 技術面確認**")

with st.sidebar:
    st.header("⚙️ 設定")
    code_raw      = st.text_input("股票代號（台股輸入數字）", "2330").strip()
    code_input    = code_raw.replace(".TW","").replace(".TWO","").upper()
    st.divider()
    capital       = st.number_input("資金（元）", min_value=0.0, value=200_000.0, step=10_000.0)
    risk_pct      = st.slider("最多可以賠幾 %", 1, 20, 10) / 100.0
    st.divider()
    forecast_days = st.slider("預測交易日數", 5, 20, FORECAST_DAYS_DEFAULT)
    sim_paths_n   = st.slider("模擬條數", 200, 1200, SIM_PATHS_DEFAULT, 100)
    mean_revert   = st.slider("均值回歸強度", 0.0, 0.6, 0.25, 0.05)
    noise_mult    = st.slider("波動倍數", 0.8, 3.0, 1.5, 0.1,
                               help="建議 1.2~1.8，確保路徑有足夠波動")
    atr_mult      = st.slider("停損 ATR 倍數", 1.5, 3.5, 2.5, 0.5)

col_run, col_reset = st.columns([3, 1])
with col_run:
    run_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)
with col_reset:
    if st.button("🔄 清除快取", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if run_btn:
    with st.spinner("📡 抓取資料中..."):
        df_raw, used_code = download_data(code_input)
    if df_raw.empty:
        st.error("❌ 抓不到資料，請確認代號。")
        st.stop()

    market_label = "上櫃（OTC）" if used_code.endswith(".TWO") else "上市（TWSE）"
    st.success(f"✅ **{used_code}**（{market_label}）")

    df = add_indicators(df_raw)
    if len(df) < 80:
        st.error("❌ 資料不足 80 交易日。")
        st.stop()

    fdates     = future_dates(df, horizon=forecast_days)
    last_close = float(df["Close"].iloc[-1])
    atr        = float(df["ATR"].iloc[-1])
    stop_price = last_close - atr_mult * atr

    with st.spinner("🎲 蒙地卡羅模擬中..."):
        paths = simulate_paths(df, sim_paths_n, len(fdates), mean_revert, noise_mult)

    # Debug 確認（可上線後刪除）
    debug_up = float(np.mean(paths[:, 0] > last_close) * 100)
    debug_sig = float(df["SIGMA20"].iloc[-1])
    st.caption(f"🔍 模擬確認：次日上漲機率 = **{debug_up:.1f}%**，sigma = {debug_sig:.5f}")

    sf = make_short_forecast(last_close, fdates, paths)

    # ══════════════════════════════════════════
    # 區塊一：次日明確結論（最顯眼）
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("🎯 次日明確結論")
    t0 = sf[0]
    arrow = "🟢⬆️" if t0["up_prob"] >= 55 else ("🔴⬇️" if t0["up_prob"] <= 45 else "🟡↔️")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("現價",          f"{last_close:.2f}")
    c2.metric("次日預測價",    f"{t0['med']:.2f}", delta=f"{t0['chg_pct']:+.2f}%")
    c3.metric("上漲機率",      f"{t0['up_prob']:.1f}%",
              delta="偏漲✅" if t0["up_prob"] >= 55 else ("偏跌❌" if t0["up_prob"] <= 45 else "盤整⚠️"))
    c4.metric("低點參考(10%)", f"{t0['p10']:.2f}")
    c5.metric("高點參考(90%)", f"{t0['p90']:.2f}")

    if t0["up_prob"] >= 55:
        st.success(f"**{arrow} 次日偏漲**｜預測股價 **{t0['med']:.2f}**（上漲 {t0['chg_pct']:+.2f}%），合理落點 **{t0['p10']:.2f} ~ {t0['p90']:.2f}**")
    elif t0["up_prob"] <= 45:
        st.error(f"**{arrow} 次日偏跌**｜預測股價 **{t0['med']:.2f}**（下跌 {t0['chg_pct']:+.2f}%），合理落點 **{t0['p10']:.2f} ~ {t0['p90']:.2f}**")
    else:
        st.warning(f"**{arrow} 次日盤整**｜預測股價 **{t0['med']:.2f}**（{t0['chg_pct']:+.2f}%），合理落點 **{t0['p10']:.2f} ~ {t0['p90']:.2f}**")

    # ══════════════════════════════════════════
    # 區塊二：漲跌機率進度條
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("📊 次日 + 後三日｜上漲機率進度條")
    st.plotly_chart(build_prob_chart(sf, last_close), use_container_width=True)
    st.caption("長條越長 = 上漲機率越高｜🟢 ≥55% 偏漲　🟡 45~55% 盤整　🔴 ≤45% 偏跌｜虛線 = 50% 中性基準")

    # ══════════════════════════════════════════
    # 區塊三：趨勢折線
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("📉 後三日趨勢折線")
    st.plotly_chart(build_trend_chart(sf, last_close), use_container_width=True)

    # ══════════════════════════════════════════
    # 區塊四：後三日逐日結論卡片
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("📋 後三日逐日結論")
    cols = st.columns(3)
    for col, r in zip(cols, sf[1:]):
        emoji = "🟢" if r["up_prob"] >= 55 else ("🔴" if r["up_prob"] <= 45 else "🟡")
        col.markdown(f"""
**{emoji} {r['label']}　{r['date']}**

| 項目 | 數值 |
|---|---|
| 預測股價 | **{r['med']:.2f}** |
| 漲跌幅   | **{r['chg_pct']:+.2f}%** |
| 上漲機率 | **{r['up_prob']:.1f}%** |
| 合理低點 | {r['p10']:.2f} |
| 合理高點 | {r['p90']:.2f} |
| 方向判斷 | **{r['direction']}** |
        """)

    # 整體趨勢總結
    n_up   = sum(1 for r in sf if r["up_prob"] >= 55)
    n_down = sum(1 for r in sf if r["up_prob"] <= 45)
    total_chg = (sf[-1]["med"] - last_close) / last_close * 100
    st.markdown("---")
    if n_up >= 3:
        st.success(f"📈 **4日整體偏多**（{n_up}/4日偏漲）｜4日累計預估漲幅 **{total_chg:+.2f}%**　{last_close:.2f} → {sf[-1]['med']:.2f}")
    elif n_down >= 3:
        st.error(f"📉 **4日整體偏空**（{n_down}/4日偏跌）｜4日累計預估跌幅 **{total_chg:+.2f}%**　{last_close:.2f} → {sf[-1]['med']:.2f}")
    else:
        st.warning(f"↔️ **4日整體震盪**｜4日累計預估變化 **{total_chg:+.2f}%**　{last_close:.2f} → {sf[-1]['med']:.2f}")

    # ══════════════════════════════════════════
    # 區塊五：技術指標 + 操作建議
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔬 技術指標現況 + 操作建議")
    score, signals, rsi, macd_val, macd_sig, k_val, bb_pct = calc_signals(df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("現價",    f"{last_close:.2f}")
    c2.metric("RSI(14)", f"{rsi:.1f}",
              delta="超買⚠️" if rsi > 65 else ("超賣💡" if rsi < 35 else "中性"))
    c3.metric("MACD",    f"{macd_val:.3f}",
              delta="黃金交叉🟢" if macd_val > macd_sig else "死亡交叉🔴")
    c4.metric("K值",     f"{k_val:.1f}",
              delta="超賣💡" if k_val < 20 else ("超買⚠️" if k_val > 80 else "中性"))
    c5.metric("ATR(14)", f"{atr:.2f}")
    c6.metric("停損線",  f"{stop_price:.2f}",
              delta=f"-{(last_close - stop_price)/last_close*100:.1f}%",
              delta_color="inverse")

    mood_emoji = "🟢" if score >= 2 else ("🔴" if score <= -2 else "🟡")
    st.subheader(f"{mood_emoji} 綜合評分：{score:+d}（-5 到 +5）")
    sig_cols = st.columns(len(signals))
    for col, sig in zip(sig_cols, signals):
        col.markdown(f"**{sig}**")

    risk_money     = capital * risk_pct
    per_share_risk = max(last_close - stop_price, 1e-6)
    shares         = int(risk_money // per_share_risk)
    if score >= 2:
        st.success(f"✅ 多指標偏多 → **最多可買 {shares:,} 股**，停損設 **{stop_price:.2f}**（現價 -{(last_close-stop_price)/last_close*100:.1f}%）")
    elif score <= -2:
        st.error("❌ 多指標偏空 → **建議觀望，不要進場**")
    else:
        st.warning(f"⚠️ 訊號混雜 → **謹慎，最多 {shares:,} 股**，嚴守停損 **{stop_price:.2f}**")

    # ══════════════════════════════════════════
    # 區塊六：歷史走勢主圖
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("📈 歷史走勢 + 完整預測區間")
    st.plotly_chart(build_main_chart(df, fdates, paths, stop_price), use_container_width=True)

    # ══════════════════════════════════════════
    # 區塊七：完整預測數據表
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("📋 完整預測數據表")
    full_table = make_full_table(last_close, fdates, paths, stop_price)

    def color_dir(val):
        if "偏漲" in str(val): return "color:#2ECC71;font-weight:bold"
        if "偏跌" in str(val): return "color:#E74C3C;font-weight:bold"
        return "color:#F39C12;font-weight:bold"

    def color_prob(val):
        try:
            v = float(val)
            if v >= 55: return "background-color:rgba(46,204,113,0.2)"
            if v <= 45: return "background-color:rgba(231,76,60,0.2)"
        except: pass
        return ""

    styled = (full_table.style
              .applymap(color_dir,  subset=["方向"])
              .applymap(color_prob, subset=["上漲機率(%)"])
              .format({
                  "預測股價":       "{:.2f}",
                  "漲跌幅(%)":      "{:+.2f}%",
                  "低(10%)":        "{:.2f}",
                  "高(90%)":        "{:.2f}",
                  "極端低(5%)":     "{:.2f}",
                  "極端高(95%)":    "{:.2f}",
                  "上漲機率(%)":    "{:.1f}%",
                  "碰停損機率(%)":  "{:.1f}%",
              }))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("**圖例：** 🟢 ≥55% 偏漲　🟡 45~55% 盤整　🔴 ≤45% 偏跌")
    st.caption("⚠️ 免責聲明：本工具僅供輔助思考，不構成投資建議，請自行評估風險。")
