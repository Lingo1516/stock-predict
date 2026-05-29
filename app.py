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

# 台灣股市顏色：漲=紅 跌=綠
RED   = "#E74C3C"
GREEN = "#2ECC71"
GOLD  = "#F39C12"

# ══════════════════════════════════════════════════════
# 1. 資料下載
# ══════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def download_data(code: str) -> tuple[pd.DataFrame, str]:
    end   = datetime.now(TZ_TW).date() + timedelta(days=1)
    start = end - timedelta(days=1200)
    base  = code.replace(".TW","").replace(".TWO","")
    tries = [base+".TW", base+".TWO"] if base.isdigit() else [code]
    for c in tries:
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


# ══════════════════════════════════════════════════════
# 2. 技術指標
# ══════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    vol   = df["Volume"].astype(float)

    df["MA20"]  = close.rolling(20).mean()
    df["MA60"]  = close.rolling(60).mean()
    df["RSI"]   = ta.momentum.RSIIndicator(close, 14).rsi()

    m = ta.trend.MACD(close, 26, 12, 9)
    df["MACD"]      = m.macd()
    df["MACD_sig"]  = m.macd_signal()
    df["MACD_hist"] = m.macd_diff()

    s = ta.momentum.StochasticOscillator(high, low, close, 9, 3)
    df["K"] = s.stoch()
    df["D"] = s.stoch_signal()

    bb = ta.volatility.BollingerBands(close, 20, 2)
    df["BB_up"]  = bb.bollinger_hband()
    df["BB_lo"]  = bb.bollinger_lband()
    df["BB_mid"] = bb.bollinger_mavg()
    df["BB_pct"] = bb.bollinger_pband()

    df["ATR"]     = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
    df["OBV"]     = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["RET"]     = np.log(close).diff()
    df["SIGMA20"] = df["RET"].rolling(20).std()
    return df.dropna().copy()


# ══════════════════════════════════════════════════════
# 3. 未來交易日
# ══════════════════════════════════════════════════════
def future_dates(df: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    last  = pd.Timestamp(df.index[-1]).tz_localize(None)
    today = pd.Timestamp(datetime.now(TZ_TW).date())
    base  = max(last, today)
    start = base + pd.Timedelta(days=1)
    while start.weekday() >= 5:
        start += pd.Timedelta(days=1)
    return pd.bdate_range(start=start, periods=horizon)


# ══════════════════════════════════════════════════════
# 4. GARCH(1,1) 波動率
# ══════════════════════════════════════════════════════
def garch_sigma(df: pd.DataFrame, T: int) -> np.ndarray:
    ret = df["RET"].dropna().values
    n   = len(ret)
    omega, alpha, beta = 1e-6, 0.10, 0.85
    h = np.full(n, np.var(ret))
    for i in range(1, n):
        h[i] = omega + alpha*ret[i-1]**2 + beta*h[i-1]
    lr = omega / max(1-alpha-beta, 1e-6)
    sc = np.zeros(T)
    hc = h[-1]
    for t in range(T):
        hc = omega + (alpha+beta)*hc
        hc = hc*0.7 + lr*0.3
        sc[t] = np.sqrt(max(hc, 1e-8))
    return np.clip(sc, 1e-4, None)


# ══════════════════════════════════════════════════════
# 5. 蒙地卡羅模擬
# ══════════════════════════════════════════════════════
def simulate(df, n_paths: int, T: int,
             mean_revert: float, noise_mult: float) -> np.ndarray:
    ret    = df["RET"].astype(float)
    last_p = float(df["Close"].iloc[-1])
    ma20   = float(df["MA20"].iloc[-1])

    d10   = float(ret.ewm(span=10, adjust=False).mean().iloc[-1])
    d60   = float(ret.ewm(span=60, adjust=False).mean().iloc[-1])
    drift = (d10 + d60) / 2.0

    sr = float(df["SIGMA20"].iloc[-1])
    if not np.isfinite(sr) or sr <= 0:
        sr = max(float(ret.tail(60).std()), 0.008)
    sigma_t = np.maximum(garch_sigma(df, T), sr) * noise_mult

    bu = sr * 0.05
    mv = float(df["MACD"].iloc[-1])
    ms = float(df["MACD_sig"].iloc[-1])
    bp = float(df["BB_pct"].iloc[-1])
    kv = float(df["K"].iloc[-1])
    ob = float(df["OBV"].tail(5).diff().mean())
    on = ob / (abs(ob)+1e-9)

    bias = (bu if mv>ms else -bu) \
         + (-bu if bp>0.85 else (bu if bp<0.15 else 0.0)) \
         + (bu if kv<20 else (-bu if kv>80 else 0.0)) \
         + on*bu*0.5

    adj = float(np.clip(drift + bias, -sr*0.5, sr*0.5))

    rc = ret.dropna().values
    try:
        dft, _, sct = t_dist.fit(rc, floc=0)
        dft = float(np.clip(dft, 2.5, 30.0))
        sct = max(float(sct), sr*0.8)
    except Exception:
        dft, sct = 5.0, sr

    rng    = np.random.default_rng()
    paths  = np.zeros((n_paths, T))
    prices = np.full(n_paths, last_p, dtype=float)

    for t in range(T):
        eps    = t_dist.rvs(df=dft, loc=0, scale=sct*float(sigma_t[t]),
                            size=n_paths, random_state=rng.integers(int(1e9)))
        mr     = -mean_revert * ((prices-ma20)/max(ma20,1e-9)) / max(T,1)
        prices = prices * np.exp(adj + mr + eps)
        prices = np.clip(prices, last_p*0.4, last_p*2.5)
        paths[:,t] = prices.copy()

    return paths


# ══════════════════════════════════════════════════════
# 6. 次日 + 後三日預測（漲紅跌綠）
# ══════════════════════════════════════════════════════
def short_forecast(last_close: float, fdates, paths: np.ndarray) -> list:
    out = []
    for i in range(min(4, paths.shape[1])):
        dp   = paths[:,i]
        base = last_close if i==0 else paths[:,i-1]
        med  = float(np.median(dp))
        p10  = float(np.percentile(dp, 10))
        p90  = float(np.percentile(dp, 90))
        prob = float(np.mean(dp > base) * 100)
        ref  = last_close if i==0 else float(np.median(paths[:,i-1]))
        chg  = (med - ref) / ref * 100

        if prob >= 55:   direc, col = "⬆️ 偏漲", RED    # 漲 = 紅
        elif prob <= 45: direc, col = "⬇️ 偏跌", GREEN  # 跌 = 綠
        else:            direc, col = "↔️ 盤整", GOLD

        out.append({
            "label": "次日" if i==0 else f"第{i+1}日",
            "date": fdates[i].date(),
            "med": med, "p10": p10, "p90": p90,
            "prob": prob, "chg": chg,
            "direc": direc, "col": col,
        })
    return out


# ══════════════════════════════════════════════════════
# 7. 圖一：上漲機率進度條
# ══════════════════════════════════════════════════════
def chart_prob(sf: list) -> go.Figure:
    fig = go.Figure()
    ys  = [3, 2, 1, 0]
    bh  = 0.45

    for i, r in enumerate(sf):
        y = ys[i]
        p = r["prob"]
        c = r["col"]

        fig.add_shape(type="rect", x0=0, x1=100,
                      y0=y-bh/2, y1=y+bh/2,
                      fillcolor="rgba(255,255,255,0.07)",
                      line=dict(color="rgba(0,0,0,0)"))
        fig.add_shape(type="rect", x0=0, x1=max(p, 0.5),
                      y0=y-bh/2, y1=y+bh/2,
                      fillcolor=c,
                      line=dict(color="rgba(255,255,255,0.2)", width=1))
        fig.add_shape(type="line", x0=50, x1=50,
                      y0=y-bh/2-0.05, y1=y+bh/2+0.05,
                      line=dict(color="white", dash="dash", width=1.5))

        fig.add_annotation(x=max(p/2, 6), y=y,
                           text=f"<b>{p:.1f}%</b>",
                           showarrow=False,
                           font=dict(color="white", size=20),
                           xanchor="center", yanchor="middle")
        fig.add_annotation(x=-1, y=y,
                           text=f"<b>{r['label']}</b><br>"
                                f"<span style='font-size:11px;color:#bbb'>{r['date']}</span>",
                           showarrow=False,
                           font=dict(color="white", size=13),
                           xanchor="right", yanchor="middle")
        fig.add_annotation(x=101, y=y,
                           text=(f"<b style='color:{c}'>{r['direc']}</b>　"
                                 f"預測 <b style='color:{c}'>{r['med']:.2f}</b>"
                                 f"（{r['chg']:+.2f}%）　"
                                 f"<span style='color:#aaa'>區間 {r['p10']:.2f}~{r['p90']:.2f}</span>"),
                           showarrow=False,
                           font=dict(color="white", size=13),
                           xanchor="left", yanchor="middle")

    # 參考線（漲紅跌綠）
    for xv, lbl, cl in [(45,"45%偏跌", GREEN),(55,"55%偏漲", RED)]:
        fig.add_shape(type="line", x0=xv, x1=xv, y0=-0.55, y1=3.55,
                      line=dict(color=cl, dash="dot", width=1))
        fig.add_annotation(x=xv, y=3.65,
                           text=f"<span style='color:{cl};font-size:11px'>{lbl}</span>",
                           showarrow=False, xanchor="center", yanchor="bottom")
    fig.add_annotation(x=50, y=3.65,
                       text="<span style='color:#888;font-size:11px'>50%基準</span>",
                       showarrow=False, xanchor="center", yanchor="bottom")

    fig.update_layout(
        height=300, template="plotly_dark",
        title=dict(text="📊 次日 + 後三日｜上漲機率進度條（台灣：漲紅跌綠）",
                   font=dict(size=15)),
        xaxis=dict(range=[-18,175], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.65,4.1], showticklabels=False, showgrid=False, zeroline=False),
        margin=dict(l=70, r=20, t=55, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════
# 8. 圖二：趨勢折線
# ══════════════════════════════════════════════════════
def chart_trend(sf: list, last_close: float) -> go.Figure:
    x    = ["現在"] + [f"{r['label']}\n{r['date']}" for r in sf]
    ymed = [last_close] + [r["med"]  for r in sf]
    yp10 = [last_close] + [r["p10"]  for r in sf]
    yp90 = [last_close] + [r["p90"]  for r in sf]
    cols = ["white"]    + [r["col"]  for r in sf]
    szs  = [10]         + [16]*len(sf)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x+x[::-1], y=yp90+yp10[::-1],
        fill="toself", fillcolor="rgba(200,150,150,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="10%~90%區間", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=yp90, mode="lines",
        line=dict(color=f"rgba(231,76,60,0.4)", width=1, dash="dot"), name="90%上限（偏漲）"))
    fig.add_trace(go.Scatter(x=x, y=yp10, mode="lines",
        line=dict(color=f"rgba(46,204,113,0.4)", width=1, dash="dot"), name="10%下限（偏跌）"))
    fig.add_trace(go.Scatter(x=x, y=ymed,
        mode="lines+markers+text",
        line=dict(color="#F1C40F", width=3),
        marker=dict(color=cols, size=szs, line=dict(color="white", width=2)),
        text=[f"  {v:.2f}" for v in ymed],
        textposition="top right", textfont=dict(size=13, color="white"),
        name="預測中間值",
        hovertemplate="<b>%{x}</b><br>預測：%{y:,.2f}<extra></extra>"))
    for r in sf:
        fig.add_annotation(x=f"{r['label']}\n{r['date']}", y=r["p10"],
                           text=f"<b>{r['chg']:+.2f}%</b>",
                           showarrow=False, yshift=-22,
                           font=dict(color=r["col"], size=14))
    fig.update_layout(
        height=380, template="plotly_dark",
        title=dict(text="📉 次日+後三日趨勢（黃點=預測中間價　帶狀=10%~90%　下方=漲跌幅）",
                   font=dict(size=14)),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=50, r=20, t=55, b=30),
        hovermode="x unified", yaxis=dict(title="股價"))
    return fig


# ══════════════════════════════════════════════════════
# 9. 圖三：歷史走勢主圖（漲紅跌綠）
# ══════════════════════════════════════════════════════
def chart_main(df, fdates, paths, stop_price) -> go.Figure:
    med = np.median(paths, axis=0)
    p20 = np.percentile(paths, 20, axis=0)
    p80 = np.percentile(paths, 80, axis=0)
    p5  = np.percentile(paths,  5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    hist = df.tail(80).copy()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    fts = [pd.Timestamp(d) for d in fdates]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.45,0.20,0.20,0.15], vertical_spacing=0.04,
        subplot_titles=("📈 股價+布林+預測區間","MACD","KD","成交量"))

    fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_up"],
        line=dict(color="rgba(150,150,255,0.4)",width=1,dash="dot"), name="布林上軌"), row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_lo"],
        line=dict(color="rgba(150,150,255,0.4)",width=1,dash="dot"),
        fill="tonexty", fillcolor="rgba(150,150,255,0.06)", name="布林下軌"), row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"],
        name="收盤", line=dict(color="#4A90D9",width=2)), row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA20"],
        name="MA20", line=dict(color="#FFA500",width=1.2,dash="dot")), row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA60"],
        name="MA60", line=dict(color="#9B59B6",width=1.2,dash="dot")), row=1,col=1)
    fig.add_trace(go.Scatter(x=fts+fts[::-1], y=list(p95)+list(p5[::-1]),
        fill="toself", fillcolor="rgba(231,76,60,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="極端區間", hoverinfo="skip"), row=1,col=1)
    fig.add_trace(go.Scatter(x=fts+fts[::-1], y=list(p80)+list(p20[::-1]),
        fill="toself", fillcolor="rgba(231,76,60,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="主要區間", hoverinfo="skip"), row=1,col=1)
    fig.add_trace(go.Scatter(x=fts, y=med,
        name="預測中間值", line=dict(color=RED,width=2,dash="dash")), row=1,col=1)
    fig.add_trace(go.Scatter(x=fts, y=[stop_price]*len(fts),
        name="停損線", line=dict(color=GREEN,width=1.5,dash="longdash")), row=1,col=1)

    # MACD 柱：漲紅跌綠
    ch = [RED if v>=0 else GREEN for v in hist["MACD_hist"]]
    fig.add_trace(go.Bar(x=hist.index, y=hist["MACD_hist"],
        marker_color=ch, showlegend=False), row=2,col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD"],
        line=dict(color="#3498DB",width=1.5), name="MACD"), row=2,col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD_sig"],
        line=dict(color=GOLD,width=1.5), name="Signal"), row=2,col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", row=2,col=1)

    fig.add_trace(go.Scatter(x=hist.index, y=hist["K"],
        line=dict(color=RED,width=1.5), name="K"), row=3,col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["D"],
        line=dict(color="#9B59B6",width=1.5), name="D"), row=3,col=1)
    fig.add_hline(y=80, line_dash="dash", line_color=f"rgba(231,76,60,0.4)", row=3,col=1)
    fig.add_hline(y=20, line_dash="dash", line_color=f"rgba(46,204,113,0.4)", row=3,col=1)

    # 成交量：漲紅跌綠
    vc = [RED if c>=o else GREEN
          for c,o in zip(hist["Close"],hist["Open"])]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"],
        marker_color=vc, name="成交量"), row=4,col=1)

    fig.update_layout(height=800, template="plotly_dark",
        legend=dict(orientation="h", y=-0.06),
        margin=dict(l=40,r=20,t=45,b=20), hovermode="x unified")
    return fig


# ══════════════════════════════════════════════════════
# 10. 技術面評分（漲紅🔴 跌綠🟢）
# ══════════════════════════════════════════════════════
def tech_score(df):
    rsi = float(df["RSI"].iloc[-1])
    mv  = float(df["MACD"].iloc[-1])
    ms  = float(df["MACD_sig"].iloc[-1])
    kv  = float(df["K"].iloc[-1])
    bp  = float(df["BB_pct"].iloc[-1])
    sc, sg = 0, []

    if rsi<35:    sc+=1; sg.append("🔴 RSI超賣（偏漲）")
    elif rsi>65:  sc-=1; sg.append("🟢 RSI超買（偏跌）")
    else:                sg.append("🟡 RSI中性")

    if mv>ms: sc+=2; sg.append("🔴 MACD黃金交叉（偏漲×2）")
    else:     sc-=2; sg.append("🟢 MACD死亡交叉（偏跌×2）")

    if kv<20:    sc+=1; sg.append("🔴 KD超賣（偏漲）")
    elif kv>80:  sc-=1; sg.append("🟢 KD超買（偏跌）")
    else:               sg.append("🟡 KD中性")

    if bp<0.2:   sc+=1; sg.append("🔴 布林下緣（偏漲）")
    elif bp>0.8: sc-=1; sg.append("🟢 布林上緣（偏跌）")
    else:               sg.append("🟡 布林中間")

    return sc, sg, rsi, mv, ms, kv, bp


# ══════════════════════════════════════════════════════
# 11. 完整預測表
# ══════════════════════════════════════════════════════
def full_table(last_close, fdates, paths, stop_price) -> pd.DataFrame:
    med  = np.median(paths, axis=0)
    p10  = np.percentile(paths, 10, axis=0)
    p90  = np.percentile(paths, 90, axis=0)
    p5   = np.percentile(paths,  5, axis=0)
    p95  = np.percentile(paths, 95, axis=0)
    up   = np.zeros(paths.shape[1])
    up[0]= np.mean(paths[:,0] > last_close) * 100
    for i in range(1, paths.shape[1]):
        up[i] = np.mean(paths[:,i] > paths[:,i-1]) * 100
    hit  = np.mean(paths <= stop_price, axis=0) * 100
    prev = np.concatenate([[last_close], med[:-1]])
    chg  = (med - prev) / prev * 100
    dirs = ["⬆️偏漲" if p>=55 else ("⬇️偏跌" if p<=45 else "↔️盤整") for p in up]
    return pd.DataFrame({
        "日期":        [d.date() for d in fdates],
        "方向":        dirs,
        "預測股價":    np.round(med,2),
        "漲跌幅(%)":   np.round(chg,2),
        "低(10%)":     np.round(p10,2),
        "高(90%)":     np.round(p90,2),
        "極端低(5%)":  np.round(p5, 2),
        "極端高(95%)": np.round(p95,2),
        "上漲機率(%)": np.round(up, 1),
        "碰停損(%)":   np.round(hit,1),
    })


# ══════════════════════════════════════════════════════
# 12. Streamlit 主程式
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="📘 股票助手", layout="wide", page_icon="📘")
st.title("📘 股票助手｜次日 + 後三日明確預測")
st.caption("🎯 ① 次日漲跌機率　② 後三日趨勢　③ 技術面確認　｜　🔴漲 🟢跌（台灣慣例）")

with st.sidebar:
    st.header("⚙️ 設定")
    raw    = st.text_input("股票代號（台股輸入數字）", "2330").strip()
    code   = raw.replace(".TW","").replace(".TWO","").upper()
    st.divider()
    cap    = st.number_input("資金（元）", min_value=0.0, value=200_000.0, step=10_000.0)
    riskp  = st.slider("最多可以賠幾 %", 1, 20, 10) / 100.0
    st.divider()
    fdays  = st.slider("預測交易日數", 5, 20, 10)
    npaths = st.slider("模擬條數", 200, 1200, 800, 100)
    mr     = st.slider("均值回歸強度", 0.0, 0.6, 0.25, 0.05)
    nm     = st.slider("波動倍數", 0.8, 3.0, 1.5, 0.1)
    atrm   = st.slider("停損 ATR 倍數", 1.5, 3.5, 2.5, 0.5)

c1, c2 = st.columns([3,1])
with c1: run = st.button("🚀 開始分析", type="primary", use_container_width=True)
with c2:
    if st.button("🔄 清除快取", use_container_width=True):
        st.cache_data.clear(); st.rerun()

if run:
    with st.spinner("📡 抓取資料..."):
        df_raw, ucode = download_data(code)
    if df_raw.empty:
        st.error("❌ 抓不到資料，請確認代號。"); st.stop()

    ml = "上櫃（OTC）" if ucode.endswith(".TWO") else "上市（TWSE）"
    st.success(f"✅ **{ucode}**（{ml}）")

    df = add_indicators(df_raw)
    if len(df) < 80:
        st.error("❌ 資料不足 80 日。"); st.stop()

    fdates = future_dates(df, fdays)
    lc     = float(df["Close"].iloc[-1])
    atr    = float(df["ATR"].iloc[-1])
    stop   = lc - atrm * atr

    with st.spinner("🎲 模擬中..."):
        paths = simulate(df, npaths, len(fdates), mr, nm)

    dbg_prob  = float(np.mean(paths[:,0] > lc) * 100)
    dbg_sigma = float(df["SIGMA20"].iloc[-1])
    dbg_drift = float((df["RET"].ewm(span=10,adjust=False).mean().iloc[-1] +
                       df["RET"].ewm(span=60,adjust=False).mean().iloc[-1]) / 2)
    st.caption(f"🔍 確認：次日上漲機率 **{dbg_prob:.1f}%**　sigma={dbg_sigma:.5f}　drift={dbg_drift:.6f}")

    sf = short_forecast(lc, fdates, paths)

    # ── 區塊一：次日結論 ──
    st.markdown("---")
    st.subheader("🎯 次日明確結論")
    t0 = sf[0]
    a  = "🔴⬆️" if t0["prob"]>=55 else ("🟢⬇️" if t0["prob"]<=45 else "🟡↔️")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("現價",       f"{lc:.2f}")
    c2.metric("次日預測價", f"{t0['med']:.2f}", delta=f"{t0['chg']:+.2f}%")
    c3.metric("上漲機率",   f"{t0['prob']:.1f}%",
              delta="偏漲🔴" if t0["prob"]>=55 else ("偏跌🟢" if t0["prob"]<=45 else "盤整🟡"))
    c4.metric("低點(10%)",  f"{t0['p10']:.2f}")
    c5.metric("高點(90%)",  f"{t0['p90']:.2f}")

    if t0["prob"]>=55:
        st.error(  f"**{a} 次日偏漲**｜預測 **{t0['med']:.2f}**（{t0['chg']:+.2f}%）　落點 **{t0['p10']:.2f}~{t0['p90']:.2f}**")
    elif t0["prob"]<=45:
        st.success(f"**{a} 次日偏跌**｜預測 **{t0['med']:.2f}**（{t0['chg']:+.2f}%）　落點 **{t0['p10']:.2f}~{t0['p90']:.2f}**")
    else:
        st.warning(f"**{a} 次日盤整**｜預測 **{t0['med']:.2f}**（{t0['chg']:+.2f}%）　落點 **{t0['p10']:.2f}~{t0['p90']:.2f}**")

    # ── 區塊二：進度條 ──
    st.markdown("---")
    st.subheader("📊 次日 + 後三日｜上漲機率進度條")
    st.plotly_chart(chart_prob(sf), use_container_width=True)
    st.caption("長條越長=上漲機率越高　🔴≥55%偏漲　🟡45~55%盤整　🟢≤45%偏跌　虛線=50%基準")

    # ── 區塊三：趨勢折線 ──
    st.markdown("---")
    st.subheader("📉 後三日趨勢折線")
    st.plotly_chart(chart_trend(sf, lc), use_container_width=True)

    # ── 區塊四：逐日結論 ──
    st.markdown("---")
    st.subheader("📋 後三日逐日結論")
    cols = st.columns(3)
    for col, r in zip(cols, sf[1:]):
        e = "🔴" if r["prob"]>=55 else ("🟢" if r["prob"]<=45 else "🟡")
        col.markdown(f"""
**{e} {r['label']}　{r['date']}**

| 項目 | 數值 |
|---|---|
| 預測股價 | **{r['med']:.2f}** |
| 漲跌幅 | **{r['chg']:+.2f}%** |
| 上漲機率 | **{r['prob']:.1f}%** |
| 合理低點 | {r['p10']:.2f} |
| 合理高點 | {r['p90']:.2f} |
| 方向 | **{r['direc']}** |
        """)

    nu = sum(1 for r in sf if r["prob"]>=55)
    nd = sum(1 for r in sf if r["prob"]<=45)
    tc = (sf[-1]["med"] - lc) / lc * 100
    st.markdown("---")
    if nu>=3:   st.error(  f"📈 **4日整體偏多**（{nu}/4偏漲）　累計 **{tc:+.2f}%**　{lc:.2f}→{sf[-1]['med']:.2f}")
    elif nd>=3: st.success(f"📉 **4日整體偏空**（{nd}/4偏跌）　累計 **{tc:+.2f}%**　{lc:.2f}→{sf[-1]['med']:.2f}")
    else:       st.warning(f"↔️ **4日震盪**　累計 **{tc:+.2f}%**　{lc:.2f}→{sf[-1]['med']:.2f}")

    # ── 區塊五：技術指標 ──
    st.markdown("---")
    st.subheader("🔬 技術指標 + 操作建議")
    sc, sg, rsi, mv, ms, kv, bp = tech_score(df)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("現價",    f"{lc:.2f}")
    c2.metric("RSI(14)", f"{rsi:.1f}",
              delta="超買🟢偏跌" if rsi>65 else ("超賣🔴偏漲" if rsi<35 else "中性"))
    c3.metric("MACD",    f"{mv:.3f}",
              delta="黃金交叉🔴偏漲" if mv>ms else "死亡交叉🟢偏跌")
    c4.metric("K值",     f"{kv:.1f}",
              delta="超賣🔴偏漲" if kv<20 else ("超買🟢偏跌" if kv>80 else "中性"))
    c5.metric("ATR(14)", f"{atr:.2f}")
    c6.metric("停損線",  f"{stop:.2f}",
              delta=f"-{(lc-stop)/lc*100:.1f}%", delta_color="inverse")

    me = "🔴" if sc>=2 else ("🟢" if sc<=-2 else "🟡")
    st.subheader(f"{me} 綜合評分：{sc:+d}（-5 到 +5）")
    scols = st.columns(len(sg))
    for col, s in zip(scols, sg):
        col.markdown(f"**{s}**")

    rm  = cap * riskp
    psr = max(lc - stop, 1e-6)
    shs = int(rm // psr)
    if sc>=2:    st.error(  f"🔴 偏多 → 最多買 **{shs:,} 股**，停損 **{stop:.2f}**（-{(lc-stop)/lc*100:.1f}%）")
    elif sc<=-2: st.success("🟢 偏空 → 建議觀望，不要進場")
    else:        st.warning(f"🟡 混雜 → 最多 **{shs:,} 股**，嚴守停損 **{stop:.2f}**")

    # ── 區塊六：歷史走勢 ──
    st.markdown("---")
    st.subheader("📈 歷史走勢 + 完整預測區間")
    st.plotly_chart(chart_main(df, fdates, paths, stop), use_container_width=True)

    # ── 區塊七：完整數據表 ──
    st.markdown("---")
    st.subheader("📋 完整預測數據表")
    ft = full_table(lc, fdates, paths, stop)

    def cd(v):
        if "偏漲" in str(v): return f"color:{RED};font-weight:bold"
        if "偏跌" in str(v): return f"color:{GREEN};font-weight:bold"
        return f"color:{GOLD};font-weight:bold"

    def cp(v):
        try:
            f = float(v)
            if f >= 55: return f"background-color:rgba(231,76,60,0.2)"
            if f <= 45: return f"background-color:rgba(46,204,113,0.2)"
        except Exception:
            pass
        return ""

    try:
        styled = ft.style.map(cd, subset=["方向"]).map(cp, subset=["上漲機率(%)"])
    except AttributeError:
        styled = ft.style.applymap(cd, subset=["方向"]).applymap(cp, subset=["上漲機率(%)"])

    styled = styled.format({
        "預測股價":    "{:.2f}",
        "漲跌幅(%)":   "{:+.2f}%",
        "低(10%)":     "{:.2f}",
        "高(90%)":     "{:.2f}",
        "極端低(5%)":  "{:.2f}",
        "極端高(95%)": "{:.2f}",
        "上漲機率(%)": "{:.1f}%",
        "碰停損(%)":   "{:.1f}%",
    })

    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown("🔴 ≥55% 偏漲　🟡 45~55% 盤整　🟢 ≤45% 偏跌")
    st.caption("⚠️ 本工具僅供參考，不構成投資建議，請自行評估風險。")
