import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz
import ta
import warnings

warnings.filterwarnings("ignore")

# =========================
# 小學生模式設定（你可在側欄調）
# =========================
FORECAST_DAYS_DEFAULT = 10
SIM_PATHS_DEFAULT = 600  # 模擬路徑數越多越穩，但越慢

TZ_TW = pytz.timezone("Asia/Taipei")


# =========================
# 資料下載 + 指標
# =========================
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

    # 報酬波動（用來估計「明天可能亂跳多少」）
    df["RET"] = np.log(close).diff()
    df["SIGMA20"] = df["RET"].rolling(20).std()

    return df.dropna().copy()


# =========================
# 交易日日期：一定從「下一個交易日」開始
# =========================
def next_business_day(start_dt: pd.Timestamp) -> pd.Timestamp:
    # 找下一個工作日（不含當天）
    d = start_dt
    while True:
        d = d + pd.Timedelta(days=1)
        if d.weekday() < 5:
            return d

def future_dates_from_now_or_last(df: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    # 取「資料最後一天」以及「今天」的較大者，再往後找下一個交易日
    last_hist = pd.Timestamp(df.index[-1]).tz_localize(None)
    today = pd.Timestamp(datetime.now(TZ_TW).date())
    base = max(last_hist, today)
    start = next_business_day(base)
    return pd.bdate_range(start=start, periods=horizon)


# =========================
# 10 天模擬（核心）
# - 不用硬AI名詞
# - 用「最近的平均趨勢 + 波動 + 回到MA20」的方式
# =========================
def simulate_future_paths(df: pd.DataFrame, future_dates: pd.DatetimeIndex, n_paths: int,
                          mean_revert_strength: float = 0.25, noise_mult: float = 1.0):
    """
    回傳：
    paths: (n_paths, T) 的模擬未來價格
    """
    close = df["Close"].astype(float)
    last_close = float(close.iloc[-1])

    # 趨勢：最近 10 天平均 log return
    ret = df["RET"].astype(float)
    drift = float(ret.tail(10).mean())

    # 波動：最近 20 天 sigma（如果太小就用較長的）
    sigma = float(df["SIGMA20"].iloc[-1])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(ret.tail(60).std())
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.01

    # 回到 MA20 的目標
    ma20 = float(df["MA20"].iloc[-1])

    T = len(future_dates)
    rng = np.random.default_rng(42)
    paths = np.zeros((n_paths, T), dtype=float)

    for i in range(n_paths):
        p = last_close
        for t in range(T):
            # 均值回歸：如果價格偏離 MA20，就會被拉回來一點點（避免一直漲或一直跌）
            mr = -mean_revert_strength * ((p - ma20) / max(ma20, 1e-9)) / max(T, 1)

            # 隨機波動：模擬明天可能亂跳的程度
            eps = rng.normal(0.0, sigma) * noise_mult

            r = drift + mr + eps
            p = p * np.exp(r)
            paths[i, t] = p

    return paths


# =========================
# 把模擬結果變成「小學生也懂」的結論與表格
# =========================
def make_kid_report(df: pd.DataFrame, future_dates: pd.DatetimeIndex, paths: np.ndarray,
                    capital: float, risk_pct: float):
    last_close = float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])

    T = paths.shape[1]

    # 每一天的代表值：中位數/區間
    med = np.median(paths, axis=0)
    p20 = np.percentile(paths, 20, axis=0)
    p80 = np.percentile(paths, 80, axis=0)

    # 每一天「上漲機率」：比前一天高的機率
    prev = np.concatenate([np.full((paths.shape[0], 1), last_close), paths[:, :-1]], axis=1)
    up_prob = (paths > prev).mean(axis=0) * 100.0

    # 停損線：現價 - 2.5*ATR
    stop_price = last_close - 2.5 * atr

    # 每一天「跌到停損」的機率（當天價格 <= 停損線）
    hit_stop_prob = (paths <= stop_price).mean(axis=0) * 100.0

    # 估計「最可能反彈日」：用中位數最低的那天（代表最像低點）
    buy_idx = int(np.argmin(med))
    buy_day = future_dates[buy_idx].date()

    # 估計「最要小心日」：用中位數最高的那天（代表最像高點/過熱）
    sell_idx = int(np.argmax(med))
    sell_day = future_dates[sell_idx].date()

    # 估計「停損最可能在哪天碰到」
    # 對每條路徑找第一次跌破停損的日子（如果沒跌破就記為 -1）
    first_hit = np.full(paths.shape[0], -1, dtype=int)
    for i in range(paths.shape[0]):
        hits = np.where(paths[i] <= stop_price)[0]
        if hits.size > 0:
            first_hit[i] = int(hits[0])

    hit_any_prob = (first_hit >= 0).mean() * 100.0
    if hit_any_prob >= 5:
        # 找最常出現的那一天
        mode_idx = int(pd.Series(first_hit[first_hit >= 0]).mode().iloc[0])
        likely_hit_day = future_dates[mode_idx].date()
        likely_hit_text = f"如果真的會跌破停損，最常發生在 **{likely_hit_day}** 左右（機率約 {hit_any_prob:.1f}% 會碰到停損）。"
    else:
        likely_hit_text = f"以目前模擬來看，**碰到停損的機率不高**（大約 {hit_any_prob:.1f}%）。"

    # 部位：用你設定的「最多能賠多少」
    risk_money = capital * risk_pct
    per_share_risk = max(last_close - stop_price, 1e-6)
    shares = int(risk_money // per_share_risk)

    # 但如果現在不適合買，就會建議 0
    # 規則（超白話）：RSI 太中間 = 看不懂方向 → 先不要買
    if 40 <= rsi <= 60:
        shares_suggest = 0
        action_line = "現在方向不清楚，**先不要買**（比較安全）。"
    else:
        shares_suggest = shares
        action_line = f"如果你要買，建議最多買 **{shares_suggest:,} 股**（這樣就算輸也比較不會傷太重）。"

    # 白話總結（最上面那段）
    if rsi < 35:
        mood = "最近跌得比較多，有機會反彈，但也可能再晃一下。"
    elif rsi > 65:
        mood = "最近漲得比較多，要小心突然回頭跌。"
    else:
        mood = "最近不上不下，常常就是來回晃。"

    summary = f"""
【一句話結論】
{mood}
{action_line}

【比較值得注意的日子】
- 比較可能出現「低點/反彈起點」：**{buy_day}**
- 比較可能出現「高點/要小心回頭」：**{sell_day}**

【你的保命線（停損）】
- 停損價：**{stop_price:.2f}**
{likely_hit_text}

【提醒】
這個工具不是神預測，它的工作是：用比較保守的方法，告訴你「哪天比較像低點、哪天比較像高點、哪條線一定要跑」。
""".strip()

    # 表格（讓你看到數字概念）
    table = pd.DataFrame({
        "日期": [d.date() for d in future_dates],
        "比較可能的價格(中間值)": np.round(med, 2),
        "可能範圍(20%~80%)_低": np.round(p20, 2),
        "可能範圍(20%~80%)_高": np.round(p80, 2),
        "上漲機率(%)": np.round(up_prob, 1),
        "跌到停損機率(%)": np.round(hit_stop_prob, 1),
    })

    return summary, table, stop_price


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="小學生版：哪天比較像低點/高點 + 可能跌到停損哪天", layout="wide", page_icon="📘")
st.title("📘 小學生版股票助手（會講人話）")

st.markdown("""
你只要看三件事：

1) **現在要不要買**（我會用一句話講清楚）  
2) **哪一天比較像低點 / 哪一天比較像高點**  
3) **跌到哪裡一定要跑 + 大概哪一天比較可能碰到**

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
    forecast_days = st.slider("預測幾個交易日", 5, 20, FORECAST_DAYS_DEFAULT)
    sim_paths = st.slider("模擬幾條路徑（越多越穩）", 200, 1200, SIM_PATHS_DEFAULT, 100)
    mean_revert = st.slider("不要一直漲/跌的力度（越高越不會單邊）", 0.0, 0.6, 0.25, 0.05)
    noise_mult = st.slider("價格亂跳程度（越高越刺激）", 0.5, 2.0, 1.0, 0.1)

    st.caption("提示：你之前遇到『永遠同兩天』，就是因為沒用模擬、只用單一路徑。這版已修好。")

run_btn = st.button("🚀 開始幫我分析", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("抓資料中..."):
        df = download_data(code)

    if df.empty:
        st.error("抓不到資料，請檢查代號或網路。")
        st.stop()

    df = add_indicators(df)
    if len(df) < 80:
        st.error("資料太少，無法分析（至少要多一點交易日資料）。")
        st.stop()

    # 未來日期（一定從下一個交易日開始）
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

    st.subheader("📊 10 天每一天的『可能價格範圍』與『機率』")
    st.dataframe(table, use_container_width=True)

    st.subheader("📈 圖（看趨勢用）")
    # 畫出：歷史 close + 未來中位數 + 區間
    hist = df[["Close"]].tail(80).copy()
    med = pd.Series(np.median(paths, axis=0), index=future_dates, name="Close")
    p20 = pd.Series(np.percentile(paths, 20, axis=0), index=future_dates, name="P20")
    p80 = pd.Series(np.percentile(paths, 80, axis=0), index=future_dates, name="P80")
    stop_line = pd.Series([stop_price] * len(future_dates), index=future_dates, name="Stop")

    chart_df = pd.concat([hist["Close"], med, p20, p80, stop_line], axis=0).to_frame(name="Price")
    # 用 Streamlit 內建 line_chart（簡單不會壞）
    st.line_chart(pd.concat([
        hist["Close"].rename("歷史收盤"),
        med.rename("未來可能(中間值)"),
        p20.rename("可能偏低(20%)"),
        p80.rename("可能偏高(80%)"),
        stop_line.rename("停損線")
    ], axis=1))

    st.caption("⚠️ 免責聲明：這只是輔助思考，不是保證賺錢。重點是幫你少做錯事。")
