import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import ta

# =========================
# 基本設定（你不用改）
# =========================
FORECAST_DAYS = 10
CAPITAL_DEFAULT = 200_000      # 20 萬
RISK_PCT_DEFAULT = 0.10        # 10%

# =========================
# 小工具
# =========================
def download_data(code, days=300):
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(code, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def add_indicators(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["MA20"] = close.rolling(20).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_H"] = bb.bollinger_hband()
    df["BB_L"] = bb.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["ATR"] = atr.average_true_range()

    return df.dropna()

# =========================
# 超白話判斷邏輯
# =========================
def simple_forecast(df):
    """
    非 AI，只是用最近趨勢慢慢往前推
    （避免亂飆、連續 10 天同方向）
    """
    last_close = df["Close"].iloc[-1]
    trend = df["Close"].pct_change().rolling(5).mean().iloc[-1]

    preds = []
    price = last_close
    for i in range(FORECAST_DAYS):
        # 越往後越保守
        price = price * (1 + trend * 0.6)
        preds.append(price)

    dates = pd.bdate_range(start=df.index[-1], periods=FORECAST_DAYS + 1)[1:]
    return dates, preds

def make_kid_summary(df, future_dates, preds, capital, risk_pct):
    last_close = df["Close"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    rsi = df["RSI"].iloc[-1]

    # 判斷整體狀況
    if rsi < 35:
        status = "有點跌多了，可能會彈"
    elif rsi > 65:
        status = "漲得有點多，要小心"
    else:
        status = "不上不下，方向不明"

    # 找比較值得注意的日子
    diffs = np.diff([last_close] + preds)
    best_buy_day = future_dates[np.argmin(diffs)]
    best_sell_day = future_dates[np.argmax(diffs)]

    # 是否值得買（超簡單）
    want_buy = (rsi < 40)

    # 買多少（風控）
    risk_money = capital * risk_pct
    stop_price = last_close - 2.5 * atr
    per_share_risk = last_close - stop_price

    if per_share_risk <= 0 or not want_buy:
        shares = 0
    else:
        shares = int(risk_money // per_share_risk)

    # 白話總結
    if shares == 0:
        action = "現在先不要買"
    else:
        action = f"如果要買，最多買 {shares} 股"

    summary = f"""
【一句話結論】
{status}，所以建議：{action}

【比較值得注意的日子】
比較可能反彈的日子：{best_buy_day.date()}
比較要小心的日子：{best_sell_day.date()}

【很重要的保命線】
如果你真的有買，
跌到 {stop_price:.2f} 以下，一定要賣掉，不要撐。

（這不是猜，是保護你用的）
"""

    return summary.strip()

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="小學生版股票助手", layout="wide")
st.title("📘 小學生也看得懂的股票助手")

st.markdown("""
這個工具不講專業術語，只做三件事：

1️⃣ 現在要不要買  
2️⃣ 哪一天比較值得注意  
3️⃣ 跌到哪裡一定要跑  

看完就能關掉，不用研究。
""")

with st.sidebar:
    st.header("設定")
    code = st.text_input("股票代號（台股輸入 2330）", "2330").strip()
    if code.isdigit():
        code = code + ".TW"

    capital = st.number_input("你的資金", value=CAPITAL_DEFAULT, step=10_000)
    risk_pct = st.slider("最多可以賠幾 %（保命用）", 1, 20, int(RISK_PCT_DEFAULT*100)) / 100

if st.button("開始幫我想", use_container_width=True):
    df = download_data(code)
    if df.empty:
        st.error("抓不到資料，請檢查代號")
        st.stop()

    df = add_indicators(df)
    dates, preds = simple_forecast(df)

    summary = make_kid_summary(df, dates, preds, capital, risk_pct)

    st.subheader("🧠 結論（直接看這裡就好）")
    st.success(summary)

    st.subheader("📈 最近走勢（參考用）")
    chart_df = df[["Close"]].tail(60)
    future_df = pd.DataFrame({"Close": preds}, index=dates)
    st.line_chart(pd.concat([chart_df, future_df]))

st.caption("⚠️ 這只是輔助思考，不是保證賺錢。重點是幫你少做錯事。")
