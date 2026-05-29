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
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
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
# 純 NumPy GARCH(1,1) 動態波動率
# ─────────────────────────────────────────────
def get_sigma_forecast(df: pd.DataFrame, T: int) -> np.ndarray:
    ret   = df["RET"].dropna().values
    n     = len(ret)
    omega, alpha, beta = 1e-6, 0.10, 0.85

    h = np.full(n, np.var(ret))
    for i in range(1, n):
        h[i] = omega + alpha * ret[i - 1] ** 2 + beta * h[i - 1]

    long_run = omega / max(1 - alpha - beta, 1e-6)
    sigma_t  = np.zeros(T)
    h_curr   = h[-1]
    for t in range(T):
        h_curr     = omega + (alpha + beta) * h_curr
        h_curr     = h_curr * 0.7 + long_run * 0.3
        sigma_t[t] = np.sqrt(max(h_curr, 1e-8))

    return np.clip(sigma_t, 1e-4, None)


# ─────────────────────────────────────────────
# 蒙地卡羅模擬
# ─────────────────────────────────────────────
def simulate_paths(df, fdates, n_paths, mean_revert, noise_mult, atr_mult) -> np.ndarray:
    close  = df["Close"].astype(float)
    ret    = df["RET"].astype(float)
    last_p = float(close.iloc[-1])
    ma20   = float(df["MA20"].iloc[-1])
    T      = len(fdates)

    drift      = float(ret.ewm(span=10, adjust=False).mean().iloc[-1])
    sigma_t    = get_sigma_forecast(df, T) * noise_mult
    sigma_base = float(df["SIGMA20"].iloc[-1])
    if not np.isfinite(sigma_base) or sigma_base <= 0:
        sigma_base = max(float(ret.tail(60).std()), 0.01)

    bias_unit = sigma_base * 0.15
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

    ret_clean = ret.dropna().values
    try:
        df_t, _, scale_t = t_dist.fit(ret_clean, floc=0)
        df_t = max(df_t, 2.5)
    except Exception:
        df_t, scale_t = 5.0, sigma_base

    half   = n_paths // 2
    rng    = np.random.default_rng(42)
    paths  = np.zeros((n_paths, T))
    prices = np.full(n_paths, last_p)

    for t in range(T):
        raw_half = t_dist.rvs(df=df_t, loc=0, scale=scale_t * sigma_t[t],
                              size=half, random_state=rng.integers(int(1e9)))
        eps    = np.concatenate([raw_half, -raw_half])
        mr     = -mean_revert * ((prices - ma20) / max(ma20, 1e-9)) / max(T, 1)
        prices = prices * np.exp(adj_drift + mr + eps)
        paths[:, t] = prices

    return paths


# ─────────────────────────────────────────────
# 次日 + 後三日核心預測
# ─────────────────────────────────────────────
def make_short_forecast(df, fdates, paths):
    last_close = float(df["Close"].iloc[-1])
    atr        = float(df["ATR"].iloc[-1])

    results = []
    prev_price = last_close
    for i in range(min(4, len(fdates))):   # 次日 + 後三日 共4筆
        day_paths  = paths[:, i]
        med        = float(np.median(day_paths))
        p10        = float(np.percentile(day_paths, 10))
        p90        = float(np.percentile(day_paths, 90))
        up_prob    = float((day_paths > prev_price).mean() * 100)
        chg        = med - prev_price
        chg_pct    = chg / prev_price * 100

        # 明確方向判斷
        if up_prob >= 60:
            direction = "⬆️ 偏漲"
            color     = "green"
        elif up_prob <= 40:
            direction = "⬇️ 偏跌"
            color     = "red"
        else:
            direction = "↔️ 盤整"
            color     = "orange"

        label = "次日" if i == 0 else f"第{i+1}日"
        results.append({
            "label":     label,
            "date":      fdates[i].date(),
            "med":       med,
            "p10":       p10,
            "p90":       p90,
            "up_prob":   up_prob,
            "chg":       chg,
            "chg_pct":   chg_pct,
            "direction": direction,
            "color":     color,
        })
        prev_price = med   # 後三日以中間值為基準遞推

    return results


# ─────────────────────────────────────────────
# 次日大圖（儀表板風格）
# ─────────────────────────────────────────────
def build_tomorrow_gauge(sf: list, last_close: float):
    """4格儀表板：次日 + 後三日，每格顯示漲跌機率 gauge + 價格"""
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[f"{r['label']}｜{r['date']}" for r in sf],
        specs=[[{"type": "indicator"}] * 4]
    )

    for i, r in enumerate(sf):
        # Gauge：漲跌機率（50% = 中性基準）
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=r["up_prob"],
            number={"suffix": "%", "font": {"size": 28}},
            delta={"reference": 50, "valueformat": ".1f",
                   "increasing": {"color": "#27AE60"},
                   "decreasing": {"color": "#E74C3C"}},
            title={"text": f"{r['direction']}<br><span style='font-size:13px'>預測：{r['med']:.2f}（{r['chg_pct']:+.2f}%）</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": "#27AE60" if r["up_prob"] >= 60
                                  else ("#E74C3C" if r["up_prob"] <= 40 else "#F39C12")},
                "steps": [
                    {"range": [0,  40], "color": "rgba(231,76,60,0.15)"},
                    {"range": [40, 60], "color": "rgba(243,156,18,0.15)"},
                    {"range": [60,100], "color": "rgba(39,174,96,0.15)"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
            }
        ), row=1, col=i + 1)

    fig.update_layout(
        height=280, template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=10),
        title_text="🎯 次日 + 後三日漲跌機率儀表板（50% 以上偏漲，以下偏跌）",
        title_font_size=15,
    )
    return fig


# ─────────────────────────────────────────────
# 次日 + 後三日價格區間橫條圖
# ─────────────────────────────────────────────
def build_short_range_chart(sf: list, last_close: float):
    labels    = [f"{r['label']}<br>{r['date']}" for r in sf]
    med_vals  = [r["med"]  for r in sf]
    p10_vals  = [r["p10"]  for r in sf]
    p90_vals  = [r["p90"]  for r in sf]
    up_probs  = [r["up_prob"] for r in sf]
    colors    = ["#27AE60" if p >= 60 else ("#E74C3C" if p <= 40 else "#F39C12") for p in up_probs]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=("📊 預測價格區間（10%低 / 中間值 / 90%高）", "📈 上漲機率 (%)"),
    )

    # 左：價格區間（error bar 橫式）
    fig.add_trace(go.Scatter(
        x=med_vals, y=labels,
        mode="markers",
        marker=dict(color=colors, size=14, symbol="diamond"),
        error_x=dict(
            type="data",
            symmetric=False,
            array=[h - m for h, m in zip(p90_vals, med_vals)],
            arrayminus=[m - l for m, l in zip(med_vals, p10_vals)],
            color="rgba(255,255,255,0.5)",
            thickness=3, width=6,
        ),
        text=[f"中間值：{m:.2f}<br>低：{l:.2f}　高：{h:.2f}<br>漲跌：{r['chg_pct']:+.2f}%"
              for m, l, h, r in zip(med_vals, p10_vals, p90_vals, sf)],
        hoverinfo="text",
        name="預測價格區間",
    ), row=1, col=1)

    # 昨收參考線
    fig.add_vline(x=last_close, line_dash="dash",
                  line_color="rgba(255,255,255,0.4)",
                  annotation_text=f"現價 {last_close:.2f}",
                  annotation_font_color="white", row=1, col=1)

    # 右：漲跌機率橫條
    fig.add_trace(go.Bar(
        x=up_probs, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in up_probs],
        textposition="outside",
        name="上漲機率",
    ), row=1, col=2)
    fig.add_vline(x=50, line_dash="dash",
                  line_color="rgba(255,255,255,0.4)",
                  annotation_text="50%", row=1, col=2)
    fig.add_vline(x=60, line_dash="dot",
                  line_color="rgba(39,174,96,0.5)",
                  annotation_text="60%", row=1, col=2)
    fig.add_vline(x=40, line_dash="dot",
                  line_color="rgba(231,76,60,0.5)",
                  annotation_text="40%", row=1, col=2)

    fig.update_layout(
        height=320, template="plotly_dark",
        showlegend=False,
        margin=dict(l=20, r=40, t=45, b=10),
        hovermode="y unified",
        xaxis2=dict(range=[0, 110]),
    )
    return fig


# ─────────────────────────────────────────────
# 後三日趨勢折線（帶信賴帶）
# ─────────────────────────────────────────────
def build_trend_line_chart(sf: list, last_close: float, fdates):
    # 包含現價作為起點
    all_dates = ["現在"] + [f"{r['label']}\n{r['date']}" for r in sf]
    all_med   = [last_close] + [r["med"]  for r in sf]
    all_p10   = [last_close] + [r["p10"]  for r in sf]
    all_p90   = [last_close] + [r["p90"]  for r in sf]
    colors    = ["white"] + ["#27AE60" if r["up_prob"] >= 60
                              else ("#E74C3C" if r["up_prob"] <= 40 else "#F39C12")
                              for r in sf]
    sizes     = [10] + [14] * len(sf)

    fig = go.Figure()

    # 信賴帶
    fig.add_trace(go.Scatter(
        x=all_dates + all_dates[::-1],
        y=all_p90 + all_p10[::-1],
        fill="toself", fillcolor="rgba(100,180,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="10%~90% 區間", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=all_dates, y=all_p90, mode="lines",
        line=dict(color="rgba(100,180,255,0.35)", width=1, dash="dot"),
        name="90% 高", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=all_dates, y=all_p10, mode="lines",
        line=dict(color="rgba(255,100,100,0.35)", width=1, dash="dot"),
        name="10% 低", hoverinfo="skip"))

    # 中間值折線
    fig.add_trace(go.Scatter(
        x=all_dates, y=all_med,
        mode="lines+markers+text",
        line=dict(color="#F1C40F", width=2.5),
        marker=dict(color=colors, size=sizes,
                    line=dict(color="white", width=1.5)),
        text=[f"{v:.2f}" for v in all_med],
        textposition="top center",
        textfont=dict(size=12, color="white"),
        name="預測中間值",
        hovertemplate="<b>%{x}</b><br>預測：%{y:,.2f}<extra></extra>",
    ))

    # 漲跌標註箭頭
    for i, r in enumerate(sf):
        pct = r["chg_pct"]
        fig.add_annotation(
            x=f"{r['label']}\n{r['date']}", y=r["med"],
            text=f"{pct:+.2f}%",
            showarrow=False, yshift=-22,
            font=dict(color="#27AE60" if pct > 0 else "#E74C3C", size=12),
        )

    fig.update_layout(
        height=360, template="plotly_dark",
        title="📉 次日 + 後三日趨勢折線（含漲跌幅標註）",
        title_font_size=14,
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=40, r=20, t=50, b=20),
        hovermode="x unified",
        yaxis=dict(title="股價"),
    )
    return fig


# ─────────────────────────────────────────────
# 完整預測表（全部 N 日）
# ─────────────────────────────────────────────
def make_full_table(df, fdates, paths, stop_price):
    last_close = float(df["Close"].iloc[-1])
    med  = np.median(paths, axis=0)
    p10  = np.percentile(paths, 10, axis=0)
    p90  = np.percentile(paths, 90, axis=0)
    p5   = np.percentile(paths,  5, axis=0)
    p95  = np.percentile(paths, 95, axis=0)
    prev = np.concatenate([np.full((paths.shape[0], 1), last_close), paths[:, :-1]], axis=1)
    up_prob    = (paths > prev).mean(axis=0) * 100.0
    hit_stop_p = (paths <= stop_price).mean(axis=0) * 100.0

    directions = []
    for p in up_prob:
        if p >= 60:   directions.append("⬆️ 偏漲")
        elif p <= 40: directions.append("⬇️ 偏跌")
        else:         directions.append("↔️ 盤整")

    prev_med = np.concatenate([[last_close], med[:-1]])
    chg_pct  = (med - prev_med) / prev_med * 100

    return pd.DataFrame({
        "日期":            [d.date() for d in fdates],
        "方向":            directions,
        "預測價(中間值)":  np.round(med, 2),
        "漲跌幅(%)":       np.round(chg_pct, 2),
        "低估(10%)":       np.round(p10,  2),
        "高估(90%)":       np.round(p90,  2),
        "極端低(5%)":      np.round(p5,   2),
        "極端高(95%)":     np.round(p95,  2),
        "上漲機率(%)":     np.round(up_prob, 1),
        "碰停損機率(%)":   np.round(hit_stop_p, 1),
    })


# ─────────────────────────────────────────────
# 歷史走勢主圖
# ─────────────────────────────────────────────
def build_main_chart(df, fdates, paths, stop_price):
    med  = np.median(paths, axis=0)
    p20  = np.percentile(paths, 20, axis=0)
    p80  = np.percentile(paths, 80, axis=0)
    p5   = np.percentile(paths,  5, axis=0)
    p95  = np.percentile(paths, 95, axis=0)

    hist       = df.tail(80).copy()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    fdates_ts  = [pd.Timestamp(d) for d in fdates]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.2, 0.2, 0.15],
        vertical_spacing=0.04,
        subplot_titles=("📈 股價 + 布林通道 + 雙層預測區間", "MACD", "KD 指標", "成交量")
    )

    fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_upper"],
        line=dict(color="rgba(150,150,255,0.4)", width=1, dash="dot"),
        name="布林上軌"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_lower"],
        line=dict(color="rgba(150,150,255,0.4)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(150,150,255,0.06)",
        name="布林下軌"), row=1, col=1)
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
    fig.add_trace(go.Scatter(x=fdates_ts, y=[stop_price] * len(fdates_ts),
        name="停損線", line=dict(color="#E74C3C", width=1.5, dash="longdash")), row=1, col=1)

    colors_hist = ["#27AE60" if v >= 0 else "#E74C3C" for v in hist["MACD_hist"]]
    fig.add_trace(go.Bar(x=hist.index, y=hist["MACD_hist"],
        marker_color=colors_hist, name="MACD柱", showlegend=False), row=2, col=1)
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
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="📘 股票助手", layout="wide", page_icon="📘")
st.title("📘 股票助手｜次日 + 後三日明確預測")
st.caption("🎯 核心：**次日漲跌機率 + 目標價格** → **後三日趨勢** → **技術面風控**")
st.info("⚡ 純 NumPy GARCH(1,1) + Student-t + Antithetic Variates（無需額外套件）")

with st.sidebar:
    st.header("⚙️ 設定")
    code_raw   = st.text_input("股票代號（台股輸入數字即可）", "2330").strip()
    code_input = code_raw.replace(".TW", "").replace(".TWO", "").upper()
    st.divider()
    capital  = st.number_input("資金（元）", min_value=0.0, value=200_000.0, step=10_000.0)
    risk_pct = st.slider("最多可以賠幾 %", 1, 20, 10) / 100.0
    st.divider()
    forecast_days = st.slider("預測交易日數", 5, 20, FORECAST_DAYS_DEFAULT)
    sim_paths_n   = st.slider("模擬條數（越多越穩）", 200, 1200, SIM_PATHS_DEFAULT, 100)
    mean_revert   = st.slider("均值回歸強度", 0.0, 0.6, 0.25, 0.05)
    noise_mult    = st.slider("波動倍數", 0.5, 2.0, 1.0, 0.1)
    atr_mult      = st.slider("停損 ATR 倍數", 1.5, 3.5, 2.5, 0.5,
                               help="停損線 = 現價 - N × ATR")

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
        paths = simulate_paths(df, fdates, sim_paths_n, mean_revert, noise_mult, atr_mult)

    sf = make_short_forecast(df, fdates, paths)

    # ══════════════════════════════════════════
    # 區塊一：次日重點預測（最顯眼）
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("🎯 次日明確預測")

    t0 = sf[0]
    arrow = "🟢⬆️" if t0["up_prob"] >= 60 else ("🔴⬇️" if t0["up_prob"] <= 40 else "🟡↔️")

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("現價",          f"{last_close:.2f}")
    col_b.metric("次日預測價",    f"{t0['med']:.2f}",
                 delta=f"{t0['chg_pct']:+.2f}%")
    col_c.metric("上漲機率",      f"{t0['up_prob']:.1f}%",
                 delta=f"{'偏漲✅' if t0['up_prob']>=60 else ('偏跌❌' if t0['up_prob']<=40 else '盤整⚠️')}")
    col_d.metric("低估區間(10%)", f"{t0['p10']:.2f}")
    col_e.metric("高估區間(90%)", f"{t0['p90']:.2f}")

    # 明確結論框
    if t0["up_prob"] >= 60:
        st.success(f"**{arrow} 次日偏向上漲**｜預測價 **{t0['med']:.2f}**，漲幅約 **{t0['chg_pct']:+.2f}%**，合理區間 {t0['p10']:.2f} ~ {t0['p90']:.2f}")
    elif t0["up_prob"] <= 40:
        st.error(f"**{arrow} 次日偏向下跌**｜預測價 **{t0['med']:.2f}**，跌幅約 **{t0['chg_pct']:+.2f}%**，合理區間 {t0['p10']:.2f} ~ {t0['p90']:.2f}")
    else:
        st.warning(f"**{arrow} 次日方向不明（盤整）**｜預測價 **{t0['med']:.2f}**，漲跌約 **{t0['chg_pct']:+.2f}%**，合理區間 {t0['p10']:.2f} ~ {t0['p90']:.2f}")

    # ══════════════════════════════════════════
    # 區塊二：儀表板（4格 gauge）
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("📊 次日 + 後三日漲跌機率儀表板")
    st.plotly_chart(build_tomorrow_gauge(sf, last_close), use_container_width=True)

    # ══════════════════════════════════════════
    # 區塊三：價格區間橫條 + 漲跌機率
    # ══════════════════════════════════════════
    st.plotly_chart(build_short_range_chart(sf, last_close), use_container_width=True)

    # ══════════════════════════════════════════
    # 區塊四：趨勢折線（含漲跌幅標註）
    # ══════════════════════════════════════════
    st.plotly_chart(build_trend_line_chart(sf, last_close, fdates), use_container_width=True)

    # ══════════════════════════════════════════
    # 區塊五：後三日文字總結
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("📋 後三日趨勢總結")
    cols = st.columns(3)
    for i, (col, r) in enumerate(zip(cols, sf[1:])):
        emoji = "🟢" if r["up_prob"] >= 60 else ("🔴" if r["up_prob"] <= 40 else "🟡")
        col.markdown(f"""
**{emoji} {r['label']}｜{r['date']}**
- 預測價：**{r['med']:.2f}**（{r['chg_pct']:+.2f}%）
- 上漲機率：**{r['up_prob']:.1f}%**
- 區間：{r['p10']:.2f} ~ {r['p90']:.2f}
- 方向：**{r['direction']}**
        """)

    # 四日整體趨勢判斷
    overall_up   = sum(1 for r in sf if r["up_prob"] >= 60)
    overall_down = sum(1 for r in sf if r["up_prob"] <= 40)
    total_chg    = sf[-1]["med"] - last_close
    total_chg_pct= total_chg / last_close * 100

    st.markdown("---")
    if overall_up >= 3:
        st.success(f"📈 **整體趨勢偏多**｜4日中有 {overall_up} 日偏漲，預估累計漲幅 **{total_chg_pct:+.2f}%**（{last_close:.2f} → {sf[-1]['med']:.2f}）")
    elif overall_down >= 3:
        st.error(f"📉 **整體趨勢偏空**｜4日中有 {overall_down} 日偏跌，預估累計跌幅 **{total_chg_pct:+.2f}%**（{last_close:.2f} → {sf[-1]['med']:.2f}）")
    else:
        st.warning(f"↔️ **整體趨勢震盪**｜多空訊號混雜，預估累計變化 **{total_chg_pct:+.2f}%**（{last_close:.2f} → {sf[-1]['med']:.2f}）")

    # ══════════════════════════════════════════
    # 區塊六：技術指標卡片
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔬 技術指標現況")
    score, signals, rsi, macd_val, macd_sig, k_val, bb_pct = calc_signals(df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("現價",      f"{last_close:.2f}")
    c2.metric("RSI(14)",   f"{rsi:.1f}",
              delta="超買⚠️" if rsi > 65 else ("超賣💡" if rsi < 35 else "中性"))
    c3.metric("MACD",      f"{macd_val:.3f}",
              delta="黃金交叉🟢" if macd_val > macd_sig else "死亡交叉🔴")
    c4.metric("K值",       f"{k_val:.1f}",
              delta="超賣💡" if k_val < 20 else ("超買⚠️" if k_val > 80 else "中性"))
    c5.metric("ATR(14)",   f"{atr:.2f}")
    c6.metric("停損線",    f"{stop_price:.2f}",
              delta=f"-{(last_close - stop_price)/last_close*100:.1f}%",
              delta_color="inverse")

    mood_emoji = "🟢" if score >= 2 else ("🔴" if score <= -2 else "🟡")
    st.subheader(f"{mood_emoji} 綜合評分：{score:+d} / -5 到 +5（MACD 加權 x2）")
    sig_cols = st.columns(len(signals))
    for col, sig in zip(sig_cols, signals):
        col.markdown(f"**{sig}**")

    # 操作建議
    risk_money     = capital * risk_pct
    per_share_risk = max(last_close - stop_price, 1e-6)
    shares         = int(risk_money // per_share_risk)
    if score >= 2:
        st.success(f"✅ 多指標偏多 → **最多可買 {shares:,} 股**，停損設 **{stop_price:.2f}**")
    elif score <= -2:
        st.error("❌ 多指標偏空 → **建議不買或觀望**")
    else:
        st.warning(f"⚠️ 訊號混雜 → **謹慎操作，最多 {shares:,} 股**，嚴守停損 **{stop_price:.2f}**")

    # ══════════════════════════════════════════
    # 區塊七：歷史走勢主圖
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("📈 歷史走勢 + 完整預測區間")
    st.plotly_chart(build_main_chart(df, fdates, paths, stop_price), use_container_width=True)

    # ══════════════════════════════════════════
    # 區塊八：完整預測數據表
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("📋 完整預測數據表")
    full_table = make_full_table(df, fdates, paths, stop_price)

    def highlight_direction(val):
        if "偏漲" in str(val):  return "color: #27AE60; font-weight: bold"
        if "偏跌" in str(val):  return "color: #E74C3C; font-weight: bold"
        return "color: #F39C12; font-weight: bold"

    def highlight_prob(val):
        try:
            v = float(val)
            if v >= 60:  return "background-color: rgba(39,174,96,0.25)"
            if v <= 40:  return "background-color: rgba(231,76,60,0.25)"
        except: pass
        return ""

    styled = (full_table.style
              .applymap(highlight_direction, subset=["方向"])
              .applymap(highlight_prob,      subset=["上漲機率(%)"])
              .format({
                  "預測價(中間值)": "{:.2f}",
                  "漲跌幅(%)":      "{:+.2f}%",
                  "低估(10%)":      "{:.2f}",
                  "高估(90%)":      "{:.2f}",
                  "極端低(5%)":     "{:.2f}",
                  "極端高(95%)":    "{:.2f}",
                  "上漲機率(%)":    "{:.1f}%",
                  "碰停損機率(%)":  "{:.1f}%",
              }))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("**圖例：** 🟢 ≥60% 偏漲　🟡 40~60% 盤整　🔴 ≤40% 偏跌")
    st.caption("⚠️ 免責聲明：此工具僅供輔助思考，不構成投資建議，請自行評估風險。")
