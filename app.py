
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import warnings
import ta
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
TZ_TW = pytz.timezone("Asia/Taipei")

RED = "#E74C3C"
GREEN = "#2ECC71"
GOLD = "#F39C12"

FEATURE_NAMES = {
    "RET1":"1日報酬","RET2":"2日報酬","RET5":"5日報酬","RET10":"10日報酬","RET20":"20日報酬",
    "DEV_MA5":"乖離MA5","DEV_MA10":"乖離MA10","DEV_MA20":"乖離MA20","DEV_MA60":"乖離MA60","DEV_MA120":"乖離MA120",
    "MA5_SLOPE":"MA5斜率","MA20_SLOPE":"MA20斜率",
    "RSI":"RSI(14)","MACD_N":"MACD/ATR","MACD_HIST_N":"MACD柱/ATR",
    "K":"KD-K","D":"KD-D","BB_pct":"布林位置","ATR_PCT":"ATR波動率",
    "ADX":"ADX趨勢強度","CCI":"CCI","MFI":"MFI資金流","CMF":"CMF資金流",
    "VOL_RATIO5":"成交量/5日均量","VOL_RATIO20":"成交量/20日均量","OBV_SLOPE5":"OBV 5日斜率",
    "GAP":"今日跳空","RANGE_PCT":"當日振幅","CLOSE_POS":"收盤在日內區間位置",
    "DIST_H20":"距20日高點","DIST_H60":"距60日高點",
    "MKT_RET1":"大盤1日","MKT_RET5":"大盤5日","MKT_RET20":"大盤20日",
    "MKT_DEV20":"大盤乖離MA20","MKT_DEV60":"大盤乖離MA60","MKT_VOL20":"大盤20日波動",
    "RS1":"相對大盤1日","RS5":"相對大盤5日","RS20":"相對大盤20日",
    "US_NAS1":"前一晚NASDAQ","US_SOX1":"前一晚費半","US_VIX1":"前一晚VIX變化",
    "US_TNX1":"前一晚美債殖利率變化","US_TWD1":"前一晚美元/台幣變化",
    "EXT":"EXT短線過熱",
    "DOW_SIN":"星期週期(sin)","DOW_COS":"星期週期(cos)",
    "Q_SIN":"季別週期(sin)","Q_COS":"季別週期(cos)"
}
FEATURES = list(FEATURE_NAMES.keys())

@st.cache_data(ttl=1800, show_spinner=False)
def download_daily(code: str, years: int = 8):
    end = datetime.now(TZ_TW).date() + timedelta(days=1)
    start = end - timedelta(days=365*years + 45)
    base = code.replace(".TW","").replace(".TWO","").strip().upper()
    tries = [base+".TW", base+".TWO"] if base.isdigit() else [code]
    for ticker in tries:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                             progress=False, timeout=20, threads=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [x[0] for x in df.columns]
                df = df[["Open","High","Low","Close","Volume"]].dropna().copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df, ticker
        except Exception:
            pass
    return pd.DataFrame(), code

@st.cache_data(ttl=1800, show_spinner=False)
def download_symbol(symbol: str, years: int = 8):
    end = datetime.now(TZ_TW).date() + timedelta(days=1)
    start = end - timedelta(days=365*years + 45)
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True,
                         progress=False, timeout=20, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [x[0] for x in df.columns]
        cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        df = df[cols].dropna().copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def download_intraday(symbol: str):
    try:
        df = yf.download(symbol, period="5d", interval="5m", auto_adjust=True,
                         progress=False, timeout=15, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [x[0] for x in df.columns]
        return df[["Open","High","Low","Close","Volume"]].dropna().copy()
    except Exception:
        return pd.DataFrame()

def strict_prior_close_feature(tw_index, ext_df, name):
    out = pd.Series(0.0, index=tw_index, name=name)
    if ext_df is None or ext_df.empty or "Close" not in ext_df:
        return out
    s = ext_df["Close"].astype(float).pct_change().dropna().rename("v").reset_index()
    s.columns = ["ext_date","v"]
    left = pd.DataFrame({"tw_date":pd.DatetimeIndex(tw_index)})
    merged = pd.merge_asof(
        left.sort_values("tw_date"), s.sort_values("ext_date"),
        left_on="tw_date", right_on="ext_date",
        direction="backward", allow_exact_matches=False
    )
    return pd.Series(merged["v"].fillna(0.0).values, index=tw_index, name=name)

def rolling_z(s, win=252):
    mu = s.rolling(win, min_periods=80).mean()
    sd = s.rolling(win, min_periods=80).std().replace(0,np.nan)
    return ((s-mu)/sd).clip(-5,5)

def build_features(stock, market, nasdaq, sox, vix, tnx, usdtwd,
                   trade_target_pct=0.01, harvest_target_pct=0.03, harvest_stop_pct=0.025):
    df = stock.copy()
    o,h,l,c,v = [df[x].astype(float) for x in ["Open","High","Low","Close","Volume"]]

    for n in [1,2,5,10,20]:
        df[f"RET{n}"] = c.pct_change(n)
    for n in [5,10,20,60,120]:
        df[f"MA{n}"] = c.rolling(n).mean()
        df[f"DEV_MA{n}"] = c/df[f"MA{n}"] - 1

    df["MA5_SLOPE"] = df["MA5"].pct_change(3)
    df["MA20_SLOPE"] = df["MA20"].pct_change(5)

    df["RSI"] = ta.momentum.RSIIndicator(c,14).rsi()/100
    macd = ta.trend.MACD(c,26,12,9)
    df["MACD"] = macd.macd()
    df["MACD_SIG"] = macd.macd_signal()
    df["MACD_HIST"] = macd.macd_diff()

    atr = ta.volatility.AverageTrueRange(h,l,c,14).average_true_range()
    df["ATR"] = atr
    df["ATR_PCT"] = atr/c
    df["MACD_N"] = df["MACD"]/atr.replace(0,np.nan)
    df["MACD_HIST_N"] = df["MACD_HIST"]/atr.replace(0,np.nan)

    sto = ta.momentum.StochasticOscillator(h,l,c,9,3)
    df["K"] = sto.stoch()/100
    df["D"] = sto.stoch_signal()/100
    bb = ta.volatility.BollingerBands(c,20,2)
    df["BB_pct"] = bb.bollinger_pband()

    df["ADX"] = ta.trend.ADXIndicator(h,l,c,14).adx()/100
    df["CCI"] = ta.trend.CCIIndicator(h,l,c,20).cci()/200
    df["MFI"] = ta.volume.MFIIndicator(h,l,c,v,14).money_flow_index()/100
    df["CMF"] = ta.volume.ChaikinMoneyFlowIndicator(h,l,c,v,20).chaikin_money_flow()

    obv = ta.volume.OnBalanceVolumeIndicator(c,v).on_balance_volume()
    df["OBV_SLOPE5"] = obv.diff(5)/v.rolling(20).mean().replace(0,np.nan)
    df["VOL_RATIO5"] = v/v.rolling(5).mean()
    df["VOL_RATIO20"] = v/v.rolling(20).mean()

    pc = c.shift(1)
    df["GAP"] = o/pc - 1
    df["RANGE_PCT"] = (h-l)/pc.replace(0,np.nan)
    df["CLOSE_POS"] = (c-l)/(h-l).replace(0,np.nan)
    df["DIST_H20"] = c/h.rolling(20).max() - 1
    df["DIST_H60"] = c/h.rolling(60).max() - 1

    if market is not None and not market.empty:
        m = market.copy()
        mc = m["Close"].astype(float)
        m["MKT_RET1"] = mc.pct_change()
        m["MKT_RET5"] = mc.pct_change(5)
        m["MKT_RET20"] = mc.pct_change(20)
        m["MKT_DEV20"] = mc/mc.rolling(20).mean()-1
        m["MKT_DEV60"] = mc/mc.rolling(60).mean()-1
        m["MKT_VOL20"] = np.log(mc).diff().rolling(20).std()
        df = df.join(m[["MKT_RET1","MKT_RET5","MKT_RET20","MKT_DEV20","MKT_DEV60","MKT_VOL20"]],how="left").ffill()
    else:
        for k in ["MKT_RET1","MKT_RET5","MKT_RET20","MKT_DEV20","MKT_DEV60","MKT_VOL20"]:
            df[k]=0.0

    df["RS1"] = df["RET1"]-df["MKT_RET1"]
    df["RS5"] = df["RET5"]-df["MKT_RET5"]
    df["RS20"] = df["RET20"]-df["MKT_RET20"]

    df["US_NAS1"] = strict_prior_close_feature(df.index,nasdaq,"US_NAS1")
    df["US_SOX1"] = strict_prior_close_feature(df.index,sox,"US_SOX1")
    df["US_VIX1"] = strict_prior_close_feature(df.index,vix,"US_VIX1")
    df["US_TNX1"] = strict_prior_close_feature(df.index,tnx,"US_TNX1")
    df["US_TWD1"] = strict_prior_close_feature(df.index,usdtwd,"US_TWD1")

    z_r5 = rolling_z(df["RET5"])
    z_ma5 = rolling_z(df["DEV_MA5"])
    z_vol = rolling_z(df["VOL_RATIO20"])
    z_gap = rolling_z(df["GAP"])
    df["EXT"] = (np.maximum(z_r5,0)+np.maximum(z_ma5,0)+0.5*np.maximum(z_vol,0)+0.5*np.maximum(z_gap,0))/3

    dow = pd.Series(df.index.dayofweek,index=df.index)
    q = pd.Series(df.index.quarter,index=df.index)
    df["DOW_SIN"] = np.sin(2*np.pi*dow/5)
    df["DOW_COS"] = np.cos(2*np.pi*dow/5)
    df["Q_SIN"] = np.sin(2*np.pi*(q-1)/4)
    df["Q_COS"] = np.cos(2*np.pi*(q-1)/4)

    # Strategy-aligned targets
    entry = o.shift(-1)
    df["RET_T1_OC"] = c.shift(-1)/entry - 1
    df["RET_T5"] = c.shift(-5)/entry - 1

    highs = pd.concat([h.shift(-i) for i in range(1,6)],axis=1)
    lows  = pd.concat([l.shift(-i) for i in range(1,6)],axis=1)
    df["MFE5"] = highs.max(axis=1)/entry - 1
    df["MAE5"] = lows.min(axis=1)/entry - 1

    df["Y1"] = np.where(df["RET_T1_OC"].notna(),(df["RET_T1_OC"]>0.003).astype(int),np.nan)
    df["Y5"] = np.where(df["RET_T5"].notna(),(df["RET_T5"]>trade_target_pct).astype(int),np.nan)
    df["YH"] = np.where(
        df["MFE5"].notna() & df["MAE5"].notna(),
        ((df["MFE5"]>=harvest_target_pct)&(df["MAE5"]>-harvest_stop_pct)).astype(int),np.nan
    )

    return df.replace([np.inf,-np.inf],np.nan)

def walk_forward_model(df,target,ret_col,threshold=0.60,min_train=504,test_block=63,C=0.25):
    data = df[FEATURES+[target,ret_col]].dropna().copy()
    if len(data) < min_train + test_block*2:
        return None
    if len(data)>1800:
        data = data.tail(1800)

    X = data[FEATURES]
    y = data[target].astype(int)
    rr = data[ret_col].astype(float)

    probs = pd.Series(np.nan,index=data.index)
    for start in range(min_train,len(data),test_block):
        end = min(start+test_block,len(data))
        model = Pipeline([
            ("scaler",StandardScaler()),
            ("lr",LogisticRegression(C=C,class_weight="balanced",max_iter=1500,solver="lbfgs"))
        ])
        model.fit(X.iloc[:start],y.iloc[:start])
        probs.iloc[start:end] = model.predict_proba(X.iloc[start:end])[:,1]

    mask = probs.notna()
    yy,pp,rets = y.loc[mask],probs.loc[mask],rr.loc[mask]
    if len(yy)<120:
        return None

    pred = (pp>=0.5).astype(int)
    acc = float(accuracy_score(yy,pred))
    try:
        auc = float(roc_auc_score(yy,pp))
    except Exception:
        auc = np.nan
    brier = float(brier_score_loss(yy,pp))
    base = float(yy.mean())
    baseline_brier = float(np.mean((yy-base)**2))

    hi = (pp>=threshold)|(pp<=1-threshold)
    if hi.sum()>=20:
        hi_acc = float(accuracy_score(yy.loc[hi],(pp.loc[hi]>=0.5).astype(int)))
        hi_cov = float(hi.mean())
    else:
        hi_acc = np.nan
        hi_cov = float(hi.mean())

    long_mask = pp>=threshold
    tc = int(long_mask.sum())
    if tc>0:
        tr = rets.loc[long_mask]
        win_rate = float((tr>0).mean())
        avg_ret = float(tr.mean())
        med_ret = float(tr.median())
        gp = float(tr[tr>0].sum())
        gl = float(-tr[tr<0].sum())
        pf = gp/gl if gl>1e-12 else np.inf
    else:
        win_rate=avg_ret=med_ret=pf=np.nan

    oof = pd.DataFrame({"prob":pp,"actual":yy,"ret":rets})

    final_model = Pipeline([
        ("scaler",StandardScaler()),
        ("lr",LogisticRegression(C=C,class_weight="balanced",max_iter=1500,solver="lbfgs"))
    ])
    final_model.fit(X,y)
    current = df[FEATURES].dropna().iloc[[-1]]
    p_now = float(final_model.predict_proba(current)[0,1])

    scaler = final_model.named_steps["scaler"]
    lr = final_model.named_steps["lr"]
    z = scaler.transform(current)[0]
    contrib = pd.Series(lr.coef_[0]*z,index=FEATURES).sort_values()

    return {
        "prob":p_now,"acc":acc,"auc":auc,"brier":brier,"baseline_brier":baseline_brier,
        "base_rate":base,"hi_acc":hi_acc,"hi_cov":hi_cov,"trade_count":tc,
        "win_rate":win_rate,"avg_ret":avg_ret,"med_ret":med_ret,"profit_factor":pf,
        "oof":oof,"contrib":contrib,"n_oos":int(mask.sum()),"n_total":int(len(data))
    }

def model_grade(res):
    if res is None:
        return "資料不足","🔴"
    auc = res["auc"]
    brier_ok = res["brier"] < res["baseline_brier"]
    pf = res["profit_factor"]
    tc = res["trade_count"]
    if np.isfinite(auc) and auc>=0.56 and brier_ok and tc>=30 and np.isfinite(pf) and pf>=1.15:
        return "有統計優勢","🟢"
    if np.isfinite(auc) and auc>=0.52 and brier_ok and tc>=20 and np.isfinite(pf) and pf>=1.0:
        return "有限優勢","🟡"
    return "目前不可靠","🔴"

def calibration_table(oof):
    bins=[-0.001,0.40,0.50,0.60,0.70,1.001]
    labels=["<40%","40~50%","50~60%","60~70%",">=70%"]
    x=oof.copy()
    x["區間"]=pd.cut(x["prob"],bins=bins,labels=labels)
    g=x.groupby("區間",observed=False).agg(
        樣本數=("actual","size"),
        平均預測=("prob","mean"),
        實際成功率=("actual","mean"),
        平均報酬=("ret","mean")
    )
    return g.reset_index()

def market_regime(row):
    vals=[row.get("MKT_RET1",0),row.get("MKT_RET5",0),row.get("MKT_RET20",0),
          row.get("MKT_DEV20",0),row.get("MKT_DEV60",0)]
    score=sum(1 if float(v)>0 else -1 for v in vals)
    score += 1 if float(row.get("US_NAS1",0))>0 else -1
    score += 1 if float(row.get("US_SOX1",0))>0 else -1
    if float(row.get("US_VIX1",0))>0.05:
        score-=1
    if score>=4:
        return "RISK_ON",score
    if score<=-4:
        return "RISK_OFF",score
    return "NEUTRAL",score

def intraday_gate(symbol,benchmark,prev_close):
    s=download_intraday(symbol)
    m=download_intraday(benchmark)
    if s.empty:
        return {"state":"N/A","note":"抓不到5分K，不能假裝有盤中Gate"}
    try:
        if s.index.tz is not None:
            s.index=s.index.tz_convert(TZ_TW)
        if not m.empty and m.index.tz is not None:
            m.index=m.index.tz_convert(TZ_TW)
    except Exception:
        pass
    d=s.index[-1].date()
    day=s[s.index.date==d].between_time("09:00","13:30").copy()
    if len(day)<4:
        return {"state":"N/A","note":"今日5分K不足"}

    cur=float(day["Close"].iloc[-1])
    op=float(day["Open"].iloc[0])
    gap=op/prev_close-1
    vol=float(day["Volume"].sum())
    if vol>0:
        tp=(day["High"]+day["Low"]+day["Close"])/3
        vwap=float((tp*day["Volume"]).sum()/vol)
    else:
        vwap=float(day["Close"].mean())

    first30=day.between_time("09:00","09:30")
    ret30=float(first30["Close"].iloc[-1]/first30["Open"].iloc[0]-1) if len(first30)>=2 else float(cur/op-1)

    rs30=np.nan
    if not m.empty:
        md=m[m.index.date==d].between_time("09:00","09:30")
        if len(md)>=2:
            mret=float(md["Close"].iloc[-1]/md["Open"].iloc[0]-1)
            rs30=ret30-mret

    if cur>=vwap and ret30>=-0.002 and (not np.isfinite(rs30) or rs30>0) and gap<0.035:
        state="PASS"
    elif (cur<vwap and ret30<-0.008) or (gap>0.035 and ret30<0):
        state="FAIL"
    else:
        state="WAIT"

    return {"state":state,"current":cur,"vwap":vwap,"gap":gap,"ret30":ret30,"rs30":rs30,
            "note":"Entry Gate 是固定規則，不是歷史勝率。"}

def scenario_paths(df,horizon=5,n_paths=4000,seed=42):
    from scipy.stats import t as tdist
    ret=np.log(df["Close"]).diff().dropna()
    last=float(df["Close"].iloc[-1])
    mu=float(ret.ewm(span=20,adjust=False).mean().iloc[-1])
    mu=float(np.clip(mu,-0.004,0.004))
    sigma20=ret.rolling(20).std()
    sigma=float(sigma20.dropna().iloc[-1])
    sigma_lr=float(sigma20.dropna().tail(252).median())

    # 修正舊版：只對「標準化殘差」擬合 t-distribution
    z=(ret/sigma20).replace([np.inf,-np.inf],np.nan).dropna().clip(-8,8)
    try:
        df_t,_,scale_t=tdist.fit(z.values,floc=0)
        df_t=float(np.clip(df_t,3,30))
        scale_t=float(np.clip(scale_t,0.4,2.0))
    except Exception:
        df_t,scale_t=6.0,1.0

    rng=np.random.default_rng(seed)
    paths=np.zeros((n_paths,horizon))
    p=np.full(n_paths,last,dtype=float)
    sig=sigma
    for i in range(horizon):
        sig=0.85*sig+0.15*sigma_lr
        shock=tdist.rvs(df=df_t,loc=0,scale=scale_t,size=n_paths,random_state=rng)
        step=np.clip(mu+sig*shock,-0.15,0.15)
        p*=np.exp(step)
        paths[:,i]=p
    return paths

def future_sessions(last_date,n=5):
    try:
        import exchange_calendars as xcals
        cal=xcals.get_calendar("XTAI")
        start=pd.Timestamp(last_date)+pd.Timedelta(days=1)
        end=start+pd.Timedelta(days=30)
        s=cal.sessions_in_range(start,end)
        s=pd.DatetimeIndex([pd.Timestamp(x).tz_localize(None) for x in s])
        return s[:n]
    except Exception:
        return pd.bdate_range(start=pd.Timestamp(last_date)+pd.Timedelta(days=1),periods=n)

def price_chart(df,paths,fdates):
    hist=df.tail(100)
    med=np.median(paths,axis=0)
    p10=np.percentile(paths,10,axis=0)
    p90=np.percentile(paths,90,axis=0)
    fig=go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index,open=hist["Open"],high=hist["High"],low=hist["Low"],close=hist["Close"],
        increasing_line_color=RED,decreasing_line_color=GREEN,name="歷史K線"
    ))
    fig.add_trace(go.Scatter(x=fdates,y=med,mode="lines+markers",
                             line=dict(color=GOLD,width=3,dash="dash"),name="5日情境中位"))
    fig.add_trace(go.Scatter(
        x=list(fdates)+list(fdates[::-1]),y=list(p90)+list(p10[::-1]),
        fill="toself",fillcolor="rgba(241,196,15,0.12)",
        line=dict(color="rgba(0,0,0,0)"),name="10~90%情境區間",hoverinfo="skip"
    ))
    fig.update_layout(template="plotly_dark",height=520,xaxis_rangeslider_visible=False)
    return fig

st.set_page_config(page_title="股票助手 V3 驗證版",layout="wide",page_icon="📘")
st.title("📘 股票助手 V3｜先回測，再預測")
st.caption("如果歷史樣本外驗證不夠好，程式必須顯示『目前不可靠』，而不是硬給漂亮機率。")

with st.sidebar:
    st.header("⚙️ 自用設定")
    raw=st.text_input("股票代號","2330").strip()
    code=raw.replace(".TW","").replace(".TWO","").upper()

    st.markdown("### 交易成功定義")
    target5=st.slider("T+5 最低目標報酬",0.0,5.0,1.0,0.25)/100
    harvest_target=st.slider("5日內收割目標(MFE)",1.0,10.0,3.0,0.5)/100
    harvest_stop=st.slider("Harvest允許最大不利(MAE)",1.0,8.0,2.5,0.5)/100

    st.markdown("### 訊號門檻")
    signal_th=st.slider("高信心機率門檻",0.55,0.75,0.60,0.01)

    st.markdown("### 風險")
    capital=st.number_input("資金（元）",0.0,10_000_000.0,200_000.0,10_000.0)
    risk_pct=st.slider("單筆風險上限",0.5,5.0,1.5,0.5)/100
    atr_mult=st.slider("ATR停損倍數",1.0,3.5,2.0,0.25)

    npaths=st.slider("價格情境模擬",1000,8000,4000,500)
    use_intraday=st.checkbox("盤中時啟用 Entry Gate",True)

run=st.button("🚀 回測＋分析",type="primary",use_container_width=True)

if run:
    with st.spinner("抓歷史資料..."):
        stock,symbol=download_daily(code)
        if stock.empty:
            st.error("抓不到股票資料")
            st.stop()

        benchmark="^TWOII" if symbol.endswith(".TWO") else "^TWII"
        market=download_symbol(benchmark)
        nasdaq=download_symbol("^IXIC")
        sox=download_symbol("^SOX")
        vix=download_symbol("^VIX")
        tnx=download_symbol("^TNX")
        usdtwd=download_symbol("TWD=X")

        df=build_features(
            stock,market,nasdaq,sox,vix,tnx,usdtwd,
            trade_target_pct=target5,
            harvest_target_pct=harvest_target,
            harvest_stop_pct=harvest_stop
        )

    valid=df[FEATURES].dropna()
    if len(valid)<700:
        st.error(f"歷史有效樣本只有 {len(valid)} 天，不足以做嚴格回測。")
        st.stop()

    last_date=valid.index[-1]
    last=df.loc[last_date]
    last_close=float(last["Close"])

    with st.spinner("Expanding Walk-Forward 回測中..."):
        r1=walk_forward_model(df,"Y1","RET_T1_OC",threshold=signal_th)
        r5=walk_forward_model(df,"Y5","RET_T5",threshold=signal_th)
        rh=walk_forward_model(df,"YH","MFE5",threshold=signal_th)

    if r1 is None or r5 is None or rh is None:
        st.error("樣本外回測不足")
        st.stop()

    g1,e1=model_grade(r1)
    g5,e5=model_grade(r5)
    gh,eh=model_grade(rh)
    regime,regime_score=market_regime(last)
    ext=float(last["EXT"])

    st.success(f"✅ {symbol}｜資料截止 {last_date.date()}｜基準 {benchmark}")

    st.subheader("① 現在到底能不能信？")
    c1,c2,c3=st.columns(3)
    c1.metric("次日模型",f"{e1} {g1}",delta=f"現在 {r1['prob']*100:.1f}%")
    c2.metric("T+5模型",f"{e5} {g5}",delta=f"現在 {r5['prob']*100:.1f}%")
    c3.metric("5日收割機會",f"{eh} {gh}",delta=f"現在 {rh['prob']*100:.1f}%")

    st.info("『有統計優勢』只代表放回以前沒看過的資料時有可重現優勢，不代表保證會漲。")

    st.subheader("② 歷史樣本外成績｜最重要")
    score=pd.DataFrame([
        ["次日T+1",r1["n_oos"],r1["acc"],r1["auc"],r1["brier"],r1["baseline_brier"],r1["hi_acc"],r1["hi_cov"],r1["trade_count"],r1["win_rate"],r1["avg_ret"],r1["profit_factor"]],
        ["T+5",r5["n_oos"],r5["acc"],r5["auc"],r5["brier"],r5["baseline_brier"],r5["hi_acc"],r5["hi_cov"],r5["trade_count"],r5["win_rate"],r5["avg_ret"],r5["profit_factor"]],
        ["Harvest",rh["n_oos"],rh["acc"],rh["auc"],rh["brier"],rh["baseline_brier"],rh["hi_acc"],rh["hi_cov"],rh["trade_count"],rh["win_rate"],rh["avg_ret"],rh["profit_factor"]],
    ],columns=["模型","OOS樣本","Accuracy","AUC","Brier","基準Brier","高信心命中率","Coverage","多方交易數","多方勝率","平均報酬","Profit Factor"])
    st.dataframe(score,use_container_width=True,hide_index=True)

    with st.expander("這些數字是什麼？"):
        st.markdown("""
**Accuracy**：50%分界後，方向判對比例。不能單獨看。  
**AUC**：0.50≈亂猜；0.52~0.55很弱；0.56以上才開始值得研究。  
**Brier**：機率誤差，越低越好；必須低於「基準Brier」才算機率有幫助。  
**高信心命中率**：只看 >=高信心門檻 或 <=對稱門檻的樣本。  
**Coverage**：高信心訊號占全部日期比例。  
**多方勝率**：只有真的達到做多門檻時，實際報酬 >0 的比例。  
**Profit Factor**：所有獲利總和 ÷ 所有虧損絕對值；>1 才有正向歷史優勢。  
**OOS**：Out-of-sample，測試那一段資料在模型訓練時沒有看過。
        """)

    st.subheader("③ 機率有沒有說實話？")
    cal=calibration_table(r5["oof"])
    st.dataframe(cal,use_container_width=True,hide_index=True)
    st.caption("如果60~70%區間歷史實際只有48%，那顯示65%也不應該相信。")

    st.subheader("④ 最近20筆真正的歷史盲測")
    recent=r5["oof"].tail(20).copy()
    recent["預測T+5成功率"]=recent["prob"].map(lambda x:f"{x*100:.1f}%")
    recent["實際達標"]=recent["actual"].map(lambda x:"✅" if x==1 else "❌")
    recent["實際T+5報酬"]=recent["ret"].map(lambda x:f"{x*100:+.2f}%")
    st.dataframe(recent[["預測T+5成功率","實際達標","實際T+5報酬"]],use_container_width=True)

    st.subheader("⑤ 今天為什麼得到這個結果？")
    contrib=r5["contrib"]
    neg=contrib.head(7)
    pos=contrib.tail(7).sort_values(ascending=False)
    pc,nc=st.columns(2)
    with pc:
        st.markdown("**主要加分**")
        for k,v in pos.items():
            st.write(f"🟢 {FEATURE_NAMES.get(k,k)}：{v:+.3f}")
    with nc:
        st.markdown("**主要扣分**")
        for k,v in neg.items():
            st.write(f"🔴 {FEATURE_NAMES.get(k,k)}：{v:+.3f}")

    current=pd.DataFrame([
        ["Market Regime",regime,f"score {regime_score:+d}"],
        ["EXT",f"{ext:.2f}","高值=近期過熱，不等於基本面變差"],
        ["RSI",f"{float(last['RSI'])*100:.1f}","短線動能"],
        ["ATR%",f"{float(last['ATR_PCT'])*100:.2f}%","每日典型波動"],
        ["相對大盤5日",f"{float(last['RS5'])*100:+.2f}%","正值=近5日比大盤強"],
        ["距20日高點",f"{float(last['DIST_H20'])*100:+.2f}%","越接近0=越接近20日高點"],
        ["量比20日",f"{float(last['VOL_RATIO20']):.2f}x","大於1=量高於20日均量"],
        ["前一晚NASDAQ",f"{float(last['US_NAS1'])*100:+.2f}%","只用台股T0前已知資料"],
        ["前一晚費半",f"{float(last['US_SOX1'])*100:+.2f}%","半導體風格參考"],
    ],columns=["項目","目前數值","中文解釋"])
    st.dataframe(current,use_container_width=True,hide_index=True)

    st.subheader("⑥ 盤中 Entry Gate")
    gate={"state":"N/A","note":"未啟用"}
    if use_intraday:
        gate=intraday_gate(symbol,benchmark,last_close)
    if gate["state"]=="N/A":
        st.warning(f"Entry Gate：N/A｜{gate['note']}")
    else:
        rs="N/A" if not np.isfinite(gate["rs30"]) else f"{gate['rs30']*100:+.2f}%"
        st.write(f"**{gate['state']}**｜現價 {gate['current']:.2f}｜VWAP {gate['vwap']:.2f}｜Gap {gate['gap']*100:+.2f}%｜前30分 {gate['ret30']*100:+.2f}%｜RS30 {rs}")
        st.caption(gate["note"])

    st.subheader("⑦ 最後結論")
    if g5=="目前不可靠":
        action="不使用這個模型下單"
        why="T+5歷史樣本外回測沒有證明可靠優勢。"
    elif regime=="RISK_OFF":
        if gate["state"]=="PASS" and r5["prob"]>=signal_th and ext<0.8:
            action="僅小量候選"
            why="Risk-Off，只保留逆勢強勢股。"
        else:
            action="WAIT"
            why="市場環境偏弱。"
    else:
        if r5["prob"]>=signal_th and ext<0.8 and gate["state"] in ["PASS","N/A"]:
            action="可列入進場候選"
            why="回測、5D訊號與過熱條件較完整。"
        elif r5["prob"]>=0.55:
            action="WAIT"
            why="方向偏多，但條件還不完整。"
        else:
            action="不進"
            why="目前沒有足夠5D優勢。"
    st.markdown(f"### {action}")
    st.write(why)
    st.write(f"T+5達標機率 **{r5['prob']*100:.1f}%**｜Harvest **{rh['prob']*100:.1f}%**｜次日 **{r1['prob']*100:.1f}%**")

    st.subheader("⑧ 5日價格情境｜不是勝率")
    fdates=future_sessions(last_date,5)
    seed=abs(hash((symbol,str(last_date.date()))))%(2**32-1)
    paths=scenario_paths(df.loc[:last_date],5,npaths,seed)
    med=np.median(paths,axis=0)
    p10=np.percentile(paths,10,axis=0)
    p90=np.percentile(paths,90,axis=0)
    scen=pd.DataFrame({"日期":[d.date() for d in fdates],"情境中位價":np.round(med,2),"10%低區":np.round(p10,2),"90%高區":np.round(p90,2)})
    st.dataframe(scen,use_container_width=True,hide_index=True)
    st.plotly_chart(price_chart(df.loc[:last_date],paths,fdates),use_container_width=True)
    st.caption("Monte Carlo只回答『可能落在哪裡』，不再把路徑比例直接稱為真實勝率。")

    st.subheader("⑨ 風險與部位")
    atr=float(last["ATR"])
    stop=last_close-atr_mult*atr
    risk_budget=capital*risk_pct
    per_share=max(last_close-stop,1e-6)
    qty_risk=int(risk_budget//per_share)
    qty_cash=int(capital//last_close)
    qty=max(0,min(qty_risk,qty_cash))
    c1,c2,c3,c4=st.columns(4)
    c1.metric("最新收盤",f"{last_close:.2f}")
    c2.metric("ATR停損參考",f"{stop:.2f}")
    c3.metric("單筆風險預算",f"{risk_budget:,.0f}")
    c4.metric("最多股數",f"{qty:,}")

    st.warning("『精準』= 不偷看未來、回測可重現、機率有校準、不可靠時會拒絕給強訊號；不是保證每次都對。")
