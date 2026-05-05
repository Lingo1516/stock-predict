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

warnings.filterwarnings("ignore")

TZ_TW = pytz.timezone("Asia/Taipei")
FORECAST_DAYS_DEFAULT = 10
SIM_PATHS_DEFAULT = 600

# ─────────────────────────────────────────────
# 資料下載 & 指標計算（加快取）
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def download_data(code: str, days: int = 1200) -> pd.DataFrame:
    end = datetime.now(TZ_TW).date() + timedelta(days=1)
    start = end - timedelta(days=days)
    try:
        df = yf.download(code, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        st.error(f"下載錯誤：{e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.dropna().copy()


@st.cache_data(ttl=3600)
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    df["MA20"]   = close.rolling(20).mean()
    df["MA60"]   = close.rolling(60).mean()  # 新增 MA60
    df["RSI"]    = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["ATR"]    = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["RET"]    = np.log(close).diff()
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
    today = pd.Timestamp(datetime.now(TZ_TW).date())
    base = max(last_hist, today)
    return pd.bdate_range(start=next_business_day(base), periods=horizon)


# ─────────────────────────────────────────────
# 向量化蒙地卡羅模擬（速度優化）
# ─────────────────────────────────────────────
def simulate_paths(df, fdates, n_paths, mean_revert, noise_mult) -> np.ndarray:
    close     = df["Close"].astype(float)
    ret       = df["RET"].astype(float)
    last_p    = float(close.iloc[-1])
    drift     = float(ret.tail(10).mean())
    sigma     = float(df["SIGMA20"].iloc[-1])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(float(ret.tail(60).std()), 0.01)
    ma20 = float(df["MA20"].iloc[-1])

    T   = len(fdates)
    rng = np.random.default_rng(42)

    # 向量化：一次產生所有隨機數
    eps = rng.normal(0, sigma * noise_mult, size=(n_paths, T))
    paths = np.zeros((n_paths, T))
    prices = np.full(n_paths, last_p)

    for t in range(T):
        mr = -mean_revert * ((prices - ma20) / max(ma20, 1e-9)) / max(T, 1)
        r  = drift + mr + eps[:, t]
        prices = prices * np.exp(r)
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
        if sign[t] < 0 and sign[t + 1] > 0:
            valleys.append(t)
        if sign[t] > 0 and sign[t + 1] < 0:
            peaks.append(t)

    valley_idx = int(s.iloc[valleys].idxmin()) if valleys else None
    peak_idx   = int(s.iloc[peaks].idxmax())   if peaks   else None

    trend = float(s.iloc[-1] - s.iloc[0])
    if valley_idx is None and peak_idx is None:
        std_s = float(np.nanstd(s.values))
        if abs(trend) < max(1e-9, std_s * 0.2):
            trend_text = "這段看起來大多是『來回晃』，沒有很明顯的低點或高點。"
        elif trend > 0:
            trend_text = "這段看起來是『慢慢往上』，沒有明顯轉彎低點。"
        else:
            trend_text = "這段看起來是『慢慢往下』，沒有明顯轉彎高點。"
    else:
        trend_text = ""

    return valley_idx, peak_idx, trend_text


# ─────────────────────────────────────────────
# 產生報告 & 表格
# ─────────────────────────────────────────────
def make_report(df, fdates, paths, capital, risk_pct):
    last_close = float(df["Close"].iloc[-1])
    atr  = float(df["ATR"].iloc[-1])
    rsi  = float(df["RSI"].iloc[-1])

    med  = np.median(paths, axis=0)
    p20  = np.percentile(paths, 20, axis=0)
    p80  = np.percentile(paths, 80, axis=0)

    prev    = np.concatenate([np.full((paths.shape[0], 1), last_close), paths[:, :-1]], axis=1)
    up_prob = (paths > prev).mean(axis=0) * 100.0

    stop_price    = last_close - 2.5 * atr
    hit_stop_prob = (paths <= stop_price).mean(axis=0) * 100.0

    valley_idx, peak_idx, trend_text = find_turning_points(med)

    buy_day     = fdates[valley_idx].date() if valley_idx is not None else None
    sell_day    = fdates[peak_idx].date()   if peak_idx   is not None else None

    buy_action  = "建議**分批小量買**（不要一次全買）" if buy_day  else "沒有明顯低點，建議**觀望或少量分批**"
    sell_action = "建議**先賣一部分收錢**"             if sell_day else "沒有明顯高點，用**停損線保護**就好"

    # 停損命中時機
    first_hit = np.full(paths.shape[0], -1, dtype=int)
    for i in range(paths.shape[0]):
        hits = np.where(paths[i] <= stop_price)[0]
        if hits.size > 0:
            first_hit[i] = int(hits[0])
    hit_any_prob = (first_hit >= 0).mean() * 100.0
    if hit_any_prob >= 5:
        mode_idx = int(pd.Series(first_hit[first_hit >= 0]).mode().iloc[0])
        stop_text = f"最可能在 **{fdates[mode_idx].date()}** 碰到停損（機率 {hit_any_prob:.1f}%）"
    else:
        stop_text = f"碰到停損機率低（約 {hit_any_prob:.1f}%）"

    # 建議股數
    risk_money    = capital * risk_pct
    per_share_risk = max(last_close - stop_price, 1e-6)
    shares         = int(risk_money // per_share_risk)

    if 45 <= rsi <= 55:
        shares_suggest = 0
        action_line    = "現在方向不明，**建議先不要買**"
        mood_emoji     = "😐"
    elif rsi < 35:
        shares_suggest = shares
        action_line    = f"超賣區，**最多買 {shares:,} 股**（控制風險用）"
        mood_emoji     = "🟢"
    elif rsi > 65:
        shares_suggest = shares
        action_line    = f"超買區，**最多買 {shares:,} 股**，留意停損"
        mood_emoji     = "🔴"
    else:
        shares_suggest = shares
        action_line    = f"中性偏向，**最多買 {shares:,} 股**"
        mood_emoji     = "🟡"

    if rsi < 35:
        mood = "最近跌得比較多，有機會反彈，但也可能繼續晃。"
    elif rsi > 65:
        mood = "最近漲得比較多，要小心突然回頭跌。"
    else:
        mood = "最近不上不下，常常來回晃。"

    extra = f"\n💡 補充：{trend_text}" if trend_text else ""

    summary_data = {
        "mood": mood, "mood_emoji": mood_emoji,
        "action_line": action_line,
        "buy_day": buy_day, "buy_action": buy_action,
        "sell_day": sell_day, "sell_action": sell_action,
        "stop_price": stop_price, "stop_text": stop_text,
        "extra": extra, "rsi": rsi, "last_close": last_close,
        "atr": atr, "shares_suggest": shares_suggest,
    }

    table = pd.DataFrame({
        "日期":             [d.date() for d in fdates],
        "可能價格(中間值)": np.round(med, 2),
        "可能範圍_低(20%)": np.round(p20, 2),
        "可能範圍_高(80%)": np.round(p80, 2),
        "上漲機率(%)":      np.round(up_prob, 1),
        "碰到停損機率(%)":  np.round(hit_stop_prob, 1),
    })

    return summary_data, table, stop_price, med, p20, p80


# ─────────────────────────────────────────────
# Plotly 互動圖（優化重點）
# ─────────────────────────────────────────────
def build_chart(df, fdates, med, p20, p80, stop_price):
    hist = df[["Close", "MA20", "MA60"]].tail(80).copy()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.05,
        subplot_titles=("股價走勢 + 未來預測", "RSI")
    )

    # 歷史收盤
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        name="歷史收盤", line=dict(color="#4A90D9", width=2)
    ), row=1, col=1)

    # MA20 / MA60
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["MA20"],
        name="MA20", line=dict(color="#FFA500", width=1.2, dash="dot")
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["MA60"],
        name="MA60", line=dict(color="#9B59B6", width=1.2, dash="dot")
    ), row=1, col=1)

    # 未來區間（填色）
    fdates_ts = [pd.Timestamp(d) for d in fdates]
    fig.add_trace(go.Scatter(
        x=fdates_ts + fdates_ts[::-1],
        y=list(p80) + list(p20[::-1]),
        fill="toself", fillcolor="rgba(100,200,100,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="預測區間(20%-80%)", hoverinfo="skip"
    ), row=1, col=1)

    # 中間值
    fig.add_trace(go.Scatter(
        x=fdates_ts, y=med,
        name="預測中間值", line=dict(color="#27AE60", width=2, dash="dash")
    ), row=1, col=1)

    # 停損線
    fig.add_trace(go.Scatter(
        x=fdates_ts, y=[stop_price] * len(fdates_ts),
        name=f"停損線 {stop_price:.2f}",
        line=dict(color="#E74C3C", width=1.5, dash="longdash")
    ), row=1, col=1)

    # RSI
    rsi_series = df["RSI"].tail(80)
    rsi_index  = pd.to_datetime(rsi_series.index).tz_localize(None)
    fig.add_trace(go.Scatter(
        x=rsi_index, y=rsi_series,
        name="RSI(14)", line=dict(color="#E67E22", width=1.5)
    ), row=2, col=1)
    for level, color in [(70, "rgba(231,76,60,0.3)"), (30, "rgba(39,174,96,0.3)")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, row=2, col=1)

    fig.update_layout(
        height=580,
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=40, r=20, t=40, b=20),
        hovermode="x unified"
    )
    return fig


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="📘 股票助手（優化版）",
    layout="wide",
    page_icon="📘"
)

st.title("📘 股票助手｜低點 / 高點預測 + 風控建議")
st.caption("👆 看三件事：**要不要買** → **低點日怎麼買** → **高點日怎麼賣**")

# Sidebar
with st.sidebar:
    st.header("⚙️ 設定")
    code_input = st.text_input("股票代號（台股輸入 2330）", "2330").strip()
    if code_input.isdigit():
        code_input += ".TW"
    code_input = code_input.upper()

    st.divider()
    capital  = st.number_input("資金（元）", min_value=0.0, value=200_000.0, step=10_000.0)
    risk_pct = st.slider("最多可以賠幾 %", 1, 20, 10) / 100.0

    st.divider()
    forecast_days = st.slider("看幾個交易日", 5, 20, FORECAST_DAYS_DEFAULT)
    sim_paths     = st.slider("模擬條數（越多越穩）", 200, 1200, SIM_PATHS_DEFAULT, 100)
    mean_revert   = st.slider("均值回歸強度", 0.0, 0.6, 0.25, 0.05)
    noise_mult    = st.slider("波動倍數", 0.5, 2.0, 1.0, 0.1)

run_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)

if run_btn:
    # 1. 下載
    with st.spinner("📡 抓取資料中..."):
        df_raw = download_data(code_input)
    if df_raw.empty:
        st.error("❌ 抓不到資料，請確認代號或網路連線。")
        st.stop()

    # 2. 指標
    df = add_indicators(df_raw)
    if len(df) < 80:
        st.error("❌ 資料不足 80 個交易日，無法計算。")
        st.stop()

    # 3. 模擬
    fdates = future_dates(df, horizon=forecast_days)
    with st.spinner("🎲 蒙地卡羅模擬中..."):
        paths = simulate_paths(df, fdates, sim_paths, mean_revert, noise_mult)

    # 4. 報告
    sd, table, stop_price, med, p20, p80 = make_report(df, fdates, paths, capital, risk_pct)

    # ── 指標卡片（最上層）──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("現價",         f"{sd['last_close']:.2f}")
    c2.metric("RSI(14)",      f"{sd['rsi']:.1f}",
              delta="超買⚠️" if sd['rsi'] > 65 else ("超賣💡" if sd['rsi'] < 35 else "中性"))
    c3.metric("ATR(14)",      f"{sd['atr']:.2f}")
    c4.metric("停損線",       f"{stop_price:.2f}",
              delta=f"-{(sd['last_close'] - stop_price) / sd['last_close'] * 100:.1f}%",
              delta_color="inverse")

    st.divider()

    # ── 結論區塊 ──
    st.subheader(f"{sd['mood_emoji']} 一句話結論")
    st.info(f"{sd['mood']}\n\n**操作建議：** {sd['action_line']}")

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

    # ── Plotly 圖 ──
    st.subheader("📈 互動圖表（可縮放 / 懸停查看數字）")
    fig = build_chart(df, fdates, med, p20, p80, stop_price)
    st.plotly_chart(fig, use_container_width=True)

    # ── 每日數字表 ──
    st.subheader("📊 每日預測數字")
    st.dataframe(
        table.style.background_gradient(subset=["上漲機率(%)"], cmap="RdYlGn")
                   .background_gradient(subset=["碰到停損機率(%)"], cmap="RdYlGn_r"),
        use_container_width=True
    )

    st.caption("⚠️ 免責聲明：此工具僅供輔助思考，不構成投資建議，請自行評估風險。")
