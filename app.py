import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import ta
import warnings

warnings.filterwarnings("ignore")

TZ_TW = pytz.timezone("Asia/Taipei")

# ============ 預設值 ============
FORECAST_DAYS_DEFAULT = 10
SIM_PATHS_DEFAULT = 600

# ============ 下載資料 ============
@st.cache_data(ttl=3600)
def download_data(code: str, days: int = 1200) -> pd.DataFrame:
    end = datetime.now(TZ_TW).date() + timedelta(days=1)
    start = end - timedelta(days=days)
    df = yf.download(code, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna().copy()
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    df["MA20"] = close.rolling(20).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["ATR"] = atr.average_true_range()

    df["RET"] = np.log(close).diff()
    df["SIGMA20"] = df["RET"].rolling(20).std()

    return df.dropna().copy()

# ============ 日期：從下一個交易日開始 ============
def next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(d).tz_localize(None)
    while True:
        d = d + pd.Timedelta(days=1)
        if d.weekday() < 5:
            return d

def future_dates_from_now_or_last(df: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    last_hist = pd.Timestamp(df.index[-1]).tz_localize(None)
    today = pd.Timestamp(datetime.now(TZ_TW).date())
    base = max(last_hist, today)
    start = next_business_day(base)
    return pd.bdate_range(start=start, periods=horizon)

# ============ 模擬未來很多種可能 ============
def simulate_future_paths(
    df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    n_paths: int,
    mean_revert_strength: float,
    noise_mult: float
) -> np.ndarray:
    close = df["Close"].astype(float)
    last_close = float(close.iloc[-1])

    ret = df["RET"].astype(float)
    drift = float(ret.tail(10).mean())

    sigma = float(df["SIGMA20"].iloc[-1])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(ret.tail(60).std())
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.01

    ma20 = float(df["MA20"].iloc[-1])

    T = len(future_dates)
    rng = np.random.default_rng(42)
    paths = np.zeros((n_paths, T), dtype=float)

    for i in range(n_paths):
        p = last_close
        for t in range(T):
            mr = -mean_revert_strength * ((p - ma20) / max(ma20, 1e-9)) / max(T, 1)
            eps = rng.normal(0.0, sigma) * noise_mult
            r = drift + mr + eps
            p = p * np.exp(r)
            paths[i, t] = p

    return paths

# ============ 找「轉彎點」：不是找第一天/最後一天 ============
def find_turning_points(med: np.ndarray):
    """
    回傳：
    - valley_idx: 轉彎向上（像低點）的日子 index；如果沒有則 None
    - peak_idx: 轉彎向下（像高點）的日子 index；如果沒有則 None
    - trend_text: 如果沒有轉彎點，就說這段偏上/偏下/偏平
    """
    s = pd.Series(med)
    d = s.diff().fillna(0)

    # 判斷每天是上/下/平
    sign = np.sign(d.values)
    # 把很小的當作 0
    sign[np.abs(d.values) < (np.nanstd(d.values) * 0.05 + 1e-12)] = 0

    valleys = []
    peaks = []

    # 找：前一天在跌、下一天在漲 -> 轉彎向上（低點樣）
    # 找：前一天在漲、下一天在跌 -> 轉彎向下（高點樣）
    for t in range(1, len(sign) - 1):
        if sign[t] < 0 and sign[t + 1] > 0:
            valleys.append(t)
        if sign[t] > 0 and sign[t + 1] < 0:
            peaks.append(t)

    valley_idx = None
    peak_idx = None

    if valleys:
        # 在所有谷底候選裡，挑價格最低的那個（才像低點）
        valley_idx = int(s.iloc[valleys].idxmin())

    if peaks:
        # 在所有高點候選裡，挑價格最高的那個（才像高點）
        peak_idx = int(s.iloc[peaks].idxmax())

    # 如果沒有轉彎點，就判斷整段趨勢
    trend = float(s.iloc[-1] - s.iloc[0])
    if valley_idx is None and peak_idx is None:
        if abs(trend) < max(1e-9, float(np.nanstd(s.values)) * 0.2):
            trend_text = "這 10 天看起來大多是『來回晃』，沒有很明顯的低點或高點。"
        elif trend > 0:
            trend_text = "這 10 天看起來是『慢慢往上』，沒有明顯的轉彎低點。"
        else:
            trend_text = "這 10 天看起來是『慢慢往下』，沒有明顯的轉彎高點。"
    else:
        trend_text = ""

    return valley_idx, peak_idx, trend_text

# ============ 產生「小學生可讀」報告 ============
def make_kid_report(df, future_dates, paths, capital, risk_pct):
    last_close = float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])

    med = np.median(paths, axis=0)
    p20 = np.percentile(paths, 20, axis=0)
    p80 = np.percentile(paths, 80, axis=0)

    prev = np.concatenate([np.full((paths.shape[0], 1), last_close), paths[:, :-1]], axis=1)
    up_prob = (paths > prev).mean(axis=0) * 100.0

    stop_price = last_close - 2.5 * atr
    hit_stop_prob = (paths <= stop_price).mean(axis=0) * 100.0

    valley_idx, peak_idx, trend_text = find_turning_points(med)

    # 低點/高點日 + 要做什麼（白話）
    if valley_idx is not None:
        buy_day = future_dates[valley_idx].date()
        buy_action = f"這天比較像『跌完開始回來』：如果你要買，建議 **分批小量買**（不要一次全買）。"
    else:
        buy_day = "沒有明顯低點"
        buy_action = "這 10 天沒有看到明顯『先跌後彈』的轉彎點，所以 **不要硬抓低點**，比較安全是觀望或少量分批。"

    if peak_idx is not None:
        sell_day = future_dates[peak_idx].date()
        sell_action = f"這天比較像『漲完開始回頭』：如果你已經有買，建議 **先賣一部分**（先收錢）。"
    else:
        sell_day = "沒有明顯高點"
        sell_action = "這 10 天沒有看到明顯『先漲後跌』的轉彎點，所以 **不要硬猜高點**，用停損線保護自己就好。"

    # 停損最可能何時碰到
    first_hit = np.full(paths.shape[0], -1, dtype=int)
    for i in range(paths.shape[0]):
        hits = np.where(paths[i] <= stop_price)[0]
        if hits.size > 0:
            first_hit[i] = int(hits[0])

    hit_any_prob = (first_hit >= 0).mean() * 100.0
    if hit_any_prob >= 5:
        mode_idx = int(pd.Series(first_hit[first_hit >= 0]).mode().iloc[0])
        likely_hit_day = future_dates[mode_idx].date()
        likely_hit_text = f"如果真的會跌破停損，最常發生在 **{likely_hit_day}** 左右（大約 {hit_any_prob:.1f}% 的機率會碰到停損）。"
    else:
        likely_hit_text = f"以目前模擬來看，碰到停損的機率不高（大約 {hit_any_prob:.1f}%）。"

    # 建議買多少（簡單、保命）
    risk_money = capital * risk_pct
    per_share_risk = max(last_close - stop_price, 1e-6)
    shares = int(risk_money // per_share_risk)

    # 方向不清楚就不買（你之前想要的「不要亂買」）
    if 40 <= rsi <= 60:
        shares_suggest = 0
        action_line = "現在看不太出方向，**先不要買**（比較安全）。"
    else:
        shares_suggest = shares
        action_line = f"如果你要買，建議最多買 **{shares_suggest:,} 股**（就算做錯也比較不會傷太重）。"

    # 一句話總結
    if rsi < 35:
        mood = "最近跌得比較多，有機會反彈，但也可能還會晃。"
    elif rsi > 65:
        mood = "最近漲得比較多，要小心突然回頭跌。"
    else:
        mood = "最近不上不下，常常就是來回晃。"

    extra = f"\n【補充】{trend_text}" if trend_text else ""

    summary = f"""
【一句話結論】
{mood}
{action_line}

【低點日（告訴你要做什麼）】
低點/反彈起點：**{buy_day}**
{buy_action}

【高點日（告訴你要做什麼）】
高點/要小心：**{sell_day}**
{sell_action}

【你的保命線（停損）】
停損價：**{stop_price:.2f}**
{likely_hit_text}
{extra}
""".strip()

    table = pd.DataFrame({
        "日期": [d.date() for d in future_dates],
        "可能價格(中間值)": np.round(med, 2),
        "可能範圍_低(20%)": np.round(p20, 2),
        "可能範圍_高(80%)": np.round(p80, 2),
        "上漲機率(%)": np.round(up_prob, 1),
        "碰到停損機率(%)": np.round(hit_stop_prob, 1),
    })

    return summary, table, stop_price

# ============ Streamlit UI ============
st.set_page_config(page_title="小學生版：低點/高點不是固定兩天（已修好）", layout="wide", page_icon="📘")
st.title("📘 小學生版股票助手（已修好：不會永遠第一天低點、最後一天高點）")

st.markdown("""
你只要看三件事：

1) **現在要不要買**  
2) **低點日告訴你怎麼買（分批/觀望）**  
3) **高點日告訴你怎麼賣（先收錢/不要追）**

我會用很簡單的話講清楚。
""")

with st.sidebar:
    st.header("設定")

    code = st.text_input("股票代號（台股輸入 2330）", "2330").strip()
    if code.isdigit():
        code = code + ".TW"
    code = code.upper()

    st.divider()
    capital = st.number_input("資金", min_value=0.0, value=200_000.0, step=10_000.0)
    risk_pct = st.slider("最多可以賠幾 %（保命用）", 1, 20, 10) / 100.0

    st.divider()
    forecast_days = st.slider("看幾個交易日", 5, 20, FORECAST_DAYS_DEFAULT)
    sim_paths = st.slider("模擬幾條可能（越多越穩）", 200, 1200, SIM_PATHS_DEFAULT, 100)
    mean_revert = st.slider("避免一直漲/跌（越高越不會單邊）", 0.0, 0.6, 0.25, 0.05)
    noise_mult = st.slider("價格亂跳程度（越高越刺激）", 0.5, 2.0, 1.0, 0.1)

run_btn = st.button("🚀 開始幫我分析", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("抓資料中..."):
        df = download_data(code)

    if df.empty:
        st.error("抓不到資料，請檢查代號或網路。")
        st.stop()

    df = add_indicators(df)
    if len(df) < 80:
        st.error("資料太少，沒辦法算（至少要多一點交易日資料）。")
        st.stop()

    future_dates = future_dates_from_now_or_last(df, horizon=forecast_days)

    with st.spinner("用很多種可能性去模擬未來中..."):
        paths = simulate_future_paths(
            df, future_dates, n_paths=sim_paths,
            mean_revert_strength=mean_revert,
            noise_mult=noise_mult
        )

    summary, table, stop_price = make_kid_report(df, future_dates, paths, capital, risk_pct)

    st.subheader("🧠 結論（先看這裡就好）")
    st.success(summary)

    st.subheader("📊 每一天的數字（讓你知道不是亂講）")
    st.dataframe(table, use_container_width=True)

    st.subheader("📈 圖（看趨勢用）")
    hist = df[["Close"]].tail(80).copy()
    med = pd.Series(np.median(paths, axis=0), index=future_dates, name="未來可能(中間值)")
    p20 = pd.Series(np.percentile(paths, 20, axis=0), index=future_dates, name="可能偏低(20%)")
    p80 = pd.Series(np.percentile(paths, 80, axis=0), index=future_dates, name="可能偏高(80%)")
    stop_line = pd.Series([stop_price] * len(future_dates), index=future_dates, name="停損線")

    st.line_chart(pd.concat([
        hist["Close"].rename("歷史收盤"),
        med, p20, p80, stop_line
    ], axis=1))

    st.caption("⚠️ 免責聲明：這只是輔助思考，不是保證賺錢。重點是幫你少做錯事。")
