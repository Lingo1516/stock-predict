
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
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")
TZ_TW = pytz.timezone("Asia/Taipei")

RED = "#E74C3C"      # 台股：漲
GREEN = "#2ECC71"    # 台股：跌
GOLD = "#F39C12"

# =========================================================
# 0. 模型欄位
# =========================================================
FEATURES = [
    "RET1","RET2","RET5","RET10","RET20",
    "DEV_MA5","DEV_MA10","DEV_MA20","DEV_MA60","DEV_MA120",
    "MA5_SLOPE","MA20_SLOPE",
    "RSI","MACD_N","MACD_HIST_N","K","D","BB_pct",
    "ATR_PCT","ADX","CCI","MFI","CMF",
    "VOL_RATIO5","VOL_RATIO20","OBV_SLOPE5",
    "GAP","RANGE_PCT","CLOSE_POS","DIST_H20","DIST_H60",
    "MKT_RET1","MKT_RET5","MKT_RET20","MKT_DEV20","MKT_DEV60","MKT_VOL20",
    "RS1","RS5","RS20",
    "US_NAS1","US_SOX1","US_VIX1","US_TNX1","US_TWD1",
    "EXT","DOW_SIN","DOW_COS","Q_SIN","Q_COS"
]

# =========================================================
# 1. 下載
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def download_daily(code: str, years: int = 8):
    end = datetime.now(TZ_TW).date() + timedelta(days=1)
    start = end - timedelta(days=365*years + 45)

    base = code.replace(".TW","").replace(".TWO","").strip().upper()
    tries = [base+".TW", base+".TWO"] if base.isdigit() else [code]

    for ticker in tries:
        try:
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False,
                timeout=20, threads=False
            )
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
        df = yf.download(
            symbol, start=start, end=end,
            auto_adjust=True, progress=False,
            timeout=20, threads=False
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [x[0] for x in df.columns]
        cols = [x for x in ["Open","High","Low","Close","Volume"] if x in df.columns]
        df = df[cols].dropna().copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=180, show_spinner=False)
def download_intraday(symbol: str):
    try:
        df = yf.download(
            symbol, period="5d", interval="5m",
            auto_adjust=True, progress=False,
            timeout=15, threads=False
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [x[0] for x in df.columns]
        return df[["Open","High","Low","Close","Volume"]].dropna().copy()
    except Exception:
        return pd.DataFrame()


# =========================================================
# 2. 防止偷看未來
# =========================================================
def strict_prior_return(tw_index, ext_df):
    if ext_df is None or ext_df.empty or "Close" not in ext_df:
        return pd.Series(0.0, index=tw_index)

    x = ext_df["Close"].astype(float).pct_change().dropna().rename("v").reset_index()
    x.columns = ["ext_date","v"]

    left = pd.DataFrame({"tw_date":pd.DatetimeIndex(tw_index)})
    merged = pd.merge_asof(
        left.sort_values("tw_date"),
        x.sort_values("ext_date"),
        left_on="tw_date",
        right_on="ext_date",
        direction="backward",
        allow_exact_matches=False
    )
    return pd.Series(merged["v"].fillna(0).values, index=tw_index)


def rolling_z(s, win=252):
    mu = s.rolling(win, min_periods=80).mean()
    sd = s.rolling(win, min_periods=80).std().replace(0,np.nan)
    return ((s-mu)/sd).clip(-5,5)


# =========================================================
# 3. 特徵 + T+1 / T+5 / T+10 真實交易目標
# =========================================================
def build_features(stock, market, nasdaq, sox, vix, tnx, usdtwd):
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

    # 台灣大盤
    if market is not None and not market.empty:
        m = market.copy()
        mc = m["Close"].astype(float)
        m["MKT_RET1"] = mc.pct_change()
        m["MKT_RET5"] = mc.pct_change(5)
        m["MKT_RET20"] = mc.pct_change(20)
        m["MKT_DEV20"] = mc/mc.rolling(20).mean()-1
        m["MKT_DEV60"] = mc/mc.rolling(60).mean()-1
        m["MKT_VOL20"] = np.log(mc).diff().rolling(20).std()

        df = df.join(
            m[["MKT_RET1","MKT_RET5","MKT_RET20",
               "MKT_DEV20","MKT_DEV60","MKT_VOL20"]],
            how="left"
        ).ffill()
    else:
        for k in ["MKT_RET1","MKT_RET5","MKT_RET20","MKT_DEV20","MKT_DEV60","MKT_VOL20"]:
            df[k]=0.0

    df["RS1"] = df["RET1"]-df["MKT_RET1"]
    df["RS5"] = df["RET5"]-df["MKT_RET5"]
    df["RS20"] = df["RET20"]-df["MKT_RET20"]

    # 海外：只能用台股當天開盤前已經知道的資料
    df["US_NAS1"] = strict_prior_return(df.index,nasdaq)
    df["US_SOX1"] = strict_prior_return(df.index,sox)
    df["US_VIX1"] = strict_prior_return(df.index,vix)
    df["US_TNX1"] = strict_prior_return(df.index,tnx)
    df["US_TWD1"] = strict_prior_return(df.index,usdtwd)

    # EXT：過熱
    df["EXT"] = (
        np.maximum(rolling_z(df["RET5"]),0)
        + np.maximum(rolling_z(df["DEV_MA5"]),0)
        + 0.5*np.maximum(rolling_z(df["VOL_RATIO20"]),0)
        + 0.5*np.maximum(rolling_z(df["GAP"]),0)
    )/3

    dow = pd.Series(df.index.dayofweek,index=df.index)
    q = pd.Series(df.index.quarter,index=df.index)
    df["DOW_SIN"] = np.sin(2*np.pi*dow/5)
    df["DOW_COS"] = np.cos(2*np.pi*dow/5)
    df["Q_SIN"] = np.sin(2*np.pi*(q-1)/4)
    df["Q_COS"] = np.cos(2*np.pi*(q-1)/4)

    # 真正交易目標：
    # T0 收盤做決策 -> T+1 開盤作為代理進場
    entry = o.shift(-1)

    df["R1"] = c.shift(-1)/entry - 1
    df["R5"] = c.shift(-5)/entry - 1
    df["R10"] = c.shift(-10)/entry - 1

    # 「可交易成功」而不是只漲 0.01%
    df["Y1"] = np.where(df["R1"].notna(),(df["R1"]>0.003).astype(int),np.nan)
    df["Y5"] = np.where(df["R5"].notna(),(df["R5"]>0.010).astype(int),np.nan)
    df["Y10"] = np.where(df["R10"].notna(),(df["R10"]>0.020).astype(int),np.nan)

    return df.replace([np.inf,-np.inf],np.nan)


# =========================================================
# 4. 背景自動回測 + 自動選參數
#    不顯示給使用者，只拿來修正今日機率
# =========================================================
def candidate_model(C):
    return Pipeline([
        ("scaler",StandardScaler()),
        ("lr",LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1500,
            solver="lbfgs"
        ))
    ])


def background_select_and_predict(df, target, ret_col):
    data = df[FEATURES+[target,ret_col]].dropna().copy()

    if len(data)>1800:
        data=data.tail(1800)

    if len(data)<900:
        return None

    X=data[FEATURES]
    y=data[target].astype(int)
    rr=data[ret_col].astype(float)

    # 前70%：調整公式；後30%：真正檢查修正後公式
    split=int(len(data)*0.70)
    split=max(split,600)

    dev_X,dev_y=X.iloc[:split],y.iloc[:split]
    val_X,val_y=X.iloc[split:],y.iloc[split:]
    val_r=rr.iloc[split:]

    Cs=[0.03,0.07,0.15,0.30,0.60,1.00]

    best=None
    best_score=-1e9

    # 在 dev 裡自己做 expanding OOS，選較穩的正則化
    for C in Cs:
        probs=[]
        acts=[]
        start=350
        block=63

        for s in range(start,len(dev_X),block):
            e=min(s+block,len(dev_X))
            model=candidate_model(C)
            model.fit(dev_X.iloc[:s],dev_y.iloc[:s])
            p=model.predict_proba(dev_X.iloc[s:e])[:,1]
            probs.extend(p)
            acts.extend(dev_y.iloc[s:e].values)

        if len(probs)<100:
            continue

        pp=np.array(probs)
        aa=np.array(acts)

        try:
            auc=roc_auc_score(aa,pp)
        except Exception:
            auc=0.5

        brier=brier_score_loss(aa,pp)
        base=float(np.mean(aa))
        base_brier=float(np.mean((aa-base)**2))

        score=(auc-0.5)*4 + max(base_brier-brier,0)*2

        if score>best_score:
            best_score=score
            best=C

    if best is None:
        return None

    # 先在 dev 全部 fit，得到真正沒看過的 validation 預測
    model=candidate_model(best)
    model.fit(dev_X,dev_y)
    raw_val=model.predict_proba(val_X)[:,1]

    # calibration 只用 dev 的 OOF
    dev_oof=np.full(len(dev_X),np.nan)
    start=350
    block=63
    for s in range(start,len(dev_X),block):
        e=min(s+block,len(dev_X))
        m=candidate_model(best)
        m.fit(dev_X.iloc[:s],dev_y.iloc[:s])
        dev_oof[s:e]=m.predict_proba(dev_X.iloc[s:e])[:,1]

    cmask=np.isfinite(dev_oof)
    calibrator=None
    if cmask.sum()>=100:
        try:
            calibrator=IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(dev_oof[cmask],dev_y.iloc[cmask].values)
            val_p=calibrator.predict(raw_val)
        except Exception:
            val_p=raw_val
            calibrator=None
    else:
        val_p=raw_val

    # validation 真實表現只用來決定「今天訊號縮多少」
    try:
        auc=roc_auc_score(val_y,val_p)
    except Exception:
        auc=0.5

    brier=brier_score_loss(val_y,val_p)
    base=float(val_y.mean())
    base_brier=float(np.mean((val_y-base)**2))

    long=val_p>=0.60
    if long.sum()>=10:
        tr=val_r.iloc[np.where(long)[0]]
        gp=float(tr[tr>0].sum())
        gl=float(-tr[tr<0].sum())
        pf=gp/gl if gl>1e-12 else 2.0
        avg_ret=float(tr.mean())
    else:
        pf=0.0
        avg_ret=-1.0

    # 背景修正係數：
    # 歷史真的有效 -> 保留今天訊號
    # 歷史普通 -> 壓低
    # 歷史差 -> 幾乎縮回50%
    if auc>=0.57 and brier<base_brier and pf>=1.15 and avg_ret>0:
        edge=1.00
    elif auc>=0.54 and brier<=base_brier*1.02 and pf>=1.00 and avg_ret>=0:
        edge=0.75
    elif auc>=0.51 and brier<=base_brier*1.05:
        edge=0.45
    else:
        edge=0.15

    # 最終用所有歷史資料重新訓練今天
    final=candidate_model(best)
    final.fit(X,y)

    cur=df[FEATURES].dropna().iloc[[-1]]
    raw=float(final.predict_proba(cur)[0,1])

    # 用全部歷史 OOF 重新校準今日概率
    all_oof=np.full(len(X),np.nan)
    start=500
    block=63
    for s in range(start,len(X),block):
        e=min(s+block,len(X))
        m=candidate_model(best)
        m.fit(X.iloc[:s],y.iloc[:s])
        all_oof[s:e]=m.predict_proba(X.iloc[s:e])[:,1]

    mask=np.isfinite(all_oof)
    if mask.sum()>=150:
        try:
            cal=IsotonicRegression(out_of_bounds="clip")
            cal.fit(all_oof[mask],y.iloc[mask].values)
            raw=float(cal.predict([raw])[0])
        except Exception:
            pass

    # 核心：回測差的模型，自動把過度自信壓回50%
    adjusted=0.5 + (raw-0.5)*edge

    return {
        "p":float(np.clip(adjusted,0.05,0.95)),
        "edge":edge,
        "C":best
    }


# =========================================================
# 5. 市場環境
# =========================================================
def regime(row):
    score=0

    for k in ["MKT_RET1","MKT_RET5","MKT_RET20","MKT_DEV20","MKT_DEV60"]:
        score += 1 if float(row.get(k,0))>0 else -1

    score += 1 if float(row.get("US_NAS1",0))>0 else -1
    score += 1 if float(row.get("US_SOX1",0))>0 else -1

    if float(row.get("US_VIX1",0))>0.05:
        score-=1

    if score>=4:
        return "偏多"
    if score<=-4:
        return "偏空"
    return "震盪"


# =========================================================
# 6. 盤中即時價 + Entry Gate
# =========================================================
def intraday_status(symbol,benchmark,last_close):
    s=download_intraday(symbol)
    m=download_intraday(benchmark)

    if s.empty:
        return None

    try:
        if s.index.tz is not None:
            s.index=s.index.tz_convert(TZ_TW)
        if not m.empty and m.index.tz is not None:
            m.index=m.index.tz_convert(TZ_TW)
    except Exception:
        pass

    d=s.index[-1].date()
    day=s[s.index.date==d].between_time("09:00","13:30").copy()

    if len(day)<2:
        return None

    cur=float(day["Close"].iloc[-1])
    op=float(day["Open"].iloc[0])
    gap=op/last_close-1

    vol=float(day["Volume"].sum())
    if vol>0:
        tp=(day["High"]+day["Low"]+day["Close"])/3
        vwap=float((tp*day["Volume"]).sum()/vol)
    else:
        vwap=float(day["Close"].mean())

    f30=day.between_time("09:00","09:30")
    if len(f30)>=2:
        ret30=float(f30["Close"].iloc[-1]/f30["Open"].iloc[0]-1)
    else:
        ret30=float(cur/op-1)

    rs30=np.nan
    if m is not None and not m.empty:
        md=m[m.index.date==d].between_time("09:00","09:30")
        if len(md)>=2:
            mr=float(md["Close"].iloc[-1]/md["Open"].iloc[0]-1)
            rs30=ret30-mr

    if cur>=vwap and ret30>=-0.002 and (not np.isfinite(rs30) or rs30>0) and gap<0.035:
        gate="PASS"
    elif (cur<vwap and ret30<-0.008) or (gap>0.035 and ret30<0):
        gate="FAIL"
    else:
        gate="WAIT"

    return {
        "price":cur,
        "vwap":vwap,
        "gate":gate,
        "gap":gap,
        "ret30":ret30,
        "rs30":rs30
    }


# =========================================================
# 7. 價格情境（只算區間）
# =========================================================
def scenario(df,horizon=10,n=5000,seed=42):
    ret=np.log(df["Close"]).diff().dropna()
    last=float(df["Close"].iloc[-1])

    sigma20=ret.rolling(20).std()
    sigma=float(sigma20.dropna().iloc[-1])
    sigma_lr=float(sigma20.dropna().tail(252).median())

    # 過去500日標準化殘差，直接bootstrap，不再硬假設分布
    z=(ret/sigma20).replace([np.inf,-np.inf],np.nan).dropna().tail(500)
    if len(z)<100:
        z=pd.Series(np.random.default_rng(seed).normal(size=500))

    mu=float(ret.ewm(span=20,adjust=False).mean().iloc[-1])
    mu=float(np.clip(mu,-0.004,0.004))

    rng=np.random.default_rng(seed)
    p=np.full(n,last,dtype=float)
    paths=np.zeros((n,horizon))
    sig=sigma

    zarr=z.values

    for i in range(horizon):
        sig=0.85*sig+0.15*sigma_lr
        shock=rng.choice(zarr,size=n,replace=True)
        step=np.clip(mu+sig*shock,-0.15,0.15)
        p*=np.exp(step)
        paths[:,i]=p

    return paths


def future_sessions(last_date,n=10):
    try:
        import exchange_calendars as xcals
        cal=xcals.get_calendar("XTAI")
        start=pd.Timestamp(last_date)+pd.Timedelta(days=1)
        end=start+pd.Timedelta(days=45)
        s=cal.sessions_in_range(start,end)
        s=pd.DatetimeIndex([pd.Timestamp(x).tz_localize(None) for x in s])
        return s[:n]
    except Exception:
        return pd.bdate_range(
            start=pd.Timestamp(last_date)+pd.Timedelta(days=1),
            periods=n
        )


# =========================================================
# 8. 中文解釋
# =========================================================
def direction_text(p):
    if p>=0.60:
        return "偏多"
    if p<=0.40:
        return "偏空"
    return "震盪"


def make_reason(row,p1,p5,p10,gate,market):
    reasons=[]

    rs5=float(row["RS5"])
    ext=float(row["EXT"])
    vr=float(row["VOL_RATIO20"])
    rsi=float(row["RSI"])*100

    if market=="偏多":
        reasons.append("大盤環境偏多")
    elif market=="偏空":
        reasons.append("大盤環境偏弱")

    if rs5>0.02:
        reasons.append("近5日明顯強於大盤")
    elif rs5<-0.02:
        reasons.append("近5日弱於大盤")

    if ext>=1.0:
        reasons.append("短線過熱，不宜追價")
    elif ext<0.35:
        reasons.append("短線過熱程度低")

    if vr>1.3:
        reasons.append("成交量高於20日均量")

    if rsi>70:
        reasons.append("RSI偏高，追價風險增加")
    elif rsi<35:
        reasons.append("RSI偏低，可能接近超賣")

    if gate=="PASS":
        reasons.append("盤中站穩VWAP且相對強")
    elif gate=="FAIL":
        reasons.append("盤中結構偏弱")

    return reasons[:5]


# =========================================================
# 9. 介面
# =========================================================
st.set_page_config(page_title="股票助手 V4",layout="wide",page_icon="📘")
st.title("📘 股票助手 V4｜直接告訴我：明天、5天、10天")
st.caption("背景會自己回測與修正公式；畫面只顯示交易需要的答案。")

with st.sidebar:
    st.header("股票")
    raw=st.text_input("股票代號","2330").strip()
    code=raw.replace(".TW","").replace(".TWO","").upper()

    capital=st.number_input(
        "預計投入資金",
        min_value=0.0,
        value=200000.0,
        step=10000.0
    )

    st.caption("回測、參數選擇、機率校準全部在背景自動完成。")

run=st.button("🚀 分析",type="primary",use_container_width=True)

if run:
    with st.spinner("抓資料、背景回測並修正模型..."):
        stock,symbol=download_daily(code)

        if stock.empty:
            st.error("抓不到資料，請確認股票代號。")
            st.stop()

        benchmark="^TWOII" if symbol.endswith(".TWO") else "^TWII"

        market=download_symbol(benchmark)
        nasdaq=download_symbol("^IXIC")
        sox=download_symbol("^SOX")
        vix=download_symbol("^VIX")
        tnx=download_symbol("^TNX")
        usdtwd=download_symbol("TWD=X")

        df=build_features(
            stock,market,nasdaq,sox,vix,tnx,usdtwd
        )

        m1=background_select_and_predict(df,"Y1","R1")
        m5=background_select_and_predict(df,"Y5","R5")
        m10=background_select_and_predict(df,"Y10","R10")

    if m1 is None or m5 is None or m10 is None:
        st.error("歷史資料不足，無法完成自動回測。")
        st.stop()

    valid=df[FEATURES].dropna()
    last_date=valid.index[-1]
    row=df.loc[last_date]
    close=float(row["Close"])
    atr=float(row["ATR"])

    market_state=regime(row)

    intra=intraday_status(symbol,benchmark,close)
    if intra is not None:
        current=float(intra["price"])
        gate=intra["gate"]
        vwap=float(intra["vwap"])
    else:
        current=close
        gate="N/A"
        vwap=np.nan

    p1=m1["p"]
    p5=m5["p"]
    p10=m10["p"]

    ext=float(row["EXT"])

    # -----------------------------------------------------
    # 10. 次日直接答案
    # -----------------------------------------------------
    buy_score=0

    if p1>=0.56: buy_score+=1
    if p5>=0.58: buy_score+=1
    if p10>=0.56: buy_score+=1

    if market_state=="偏多": buy_score+=1
    if market_state=="偏空": buy_score-=1

    if ext>=1.0: buy_score-=1

    if gate=="PASS": buy_score+=1
    elif gate=="FAIL": buy_score-=2

    if buy_score>=4:
        tomorrow_action="🟢 可買"
    elif buy_score>=2:
        tomorrow_action="🟡 等拉回／小量"
    else:
        tomorrow_action="🔴 先不買"

    # -----------------------------------------------------
    # 11. 5/10日情境價格
    # -----------------------------------------------------
    seed=abs(hash((symbol,str(last_date.date()))))%(2**32-1)
    paths=scenario(df.loc[:last_date],10,5000,seed)

    med=np.median(paths,axis=0)
    lo=np.percentile(paths,20,axis=0)
    hi=np.percentile(paths,80,axis=0)

    d5_med=float(med[4])
    d5_lo=float(lo[4])
    d5_hi=float(hi[4])

    d10_med=float(med[9])
    d10_lo=float(lo[9])
    d10_hi=float(hi[9])

    # -----------------------------------------------------
    # 12. 進場區 / 防守
    # -----------------------------------------------------
    atr_pct=atr/max(close,1e-9)

    entry_low=max(
        current-0.30*atr,
        min(float(row["MA5"]),float(row["MA10"]),current)
    )
    entry_high=current+0.10*atr

    stop=min(
        current-1.5*atr,
        float(row["MA20"])-0.5*atr
    )

    if stop<=0:
        stop=current-1.5*atr

    qty=int(capital//current) if current>0 else 0

    # -----------------------------------------------------
    # 13. 第一頁：只看答案
    # -----------------------------------------------------
    st.success(
        f"✅ {symbol}｜資料截止 {last_date.date()}｜"
        f"目前參考價 {current:.2f}"
    )

    st.subheader("🎯 結論")

    c1,c2,c3=st.columns(3)

    with c1:
        st.metric(
            "次日",
            tomorrow_action,
            delta=f"模型方向 {direction_text(p1)}"
        )
        st.write(f"次日偏多機率：**{p1*100:.1f}%**")

    with c2:
        st.metric(
            "後5天",
            direction_text(p5),
            delta=f"{d5_med/current-1:+.2%}"
        )
        st.write(
            f"5日中位價 **{d5_med:.2f}**  "
            f"｜主要區間 **{d5_lo:.2f}～{d5_hi:.2f}**"
        )

    with c3:
        st.metric(
            "後10天",
            direction_text(p10),
            delta=f"{d10_med/current-1:+.2%}"
        )
        st.write(
            f"10日中位價 **{d10_med:.2f}**  "
            f"｜主要區間 **{d10_lo:.2f}～{d10_hi:.2f}**"
        )

    st.markdown("---")

    # -----------------------------------------------------
    # 14. 操作
    # -----------------------------------------------------
    st.subheader("💰 操作")

    a1,a2,a3,a4=st.columns(4)

    a1.metric(
        "建議進場區",
        f"{entry_low:.2f}～{entry_high:.2f}"
    )

    a2.metric(
        "防守價",
        f"{stop:.2f}",
        delta=f"{(stop/current-1)*100:.1f}%"
    )

    a3.metric(
        "盤中 Gate",
        gate
    )

    a4.metric(
        "現金最多可買",
        f"{qty:,} 股"
    )

    if gate!="N/A":
        st.caption(
            f"盤中 VWAP：約 {vwap:.2f}。"
            "PASS=價格與相對強弱結構較適合進場；"
            "WAIT=先不要追；FAIL=今天不進。"
        )

    # -----------------------------------------------------
    # 15. 為什麼
    # -----------------------------------------------------
    st.subheader("🔍 為什麼？")

    reasons=make_reason(
        row,p1,p5,p10,gate,market_state
    )

    for x in reasons:
        st.write("• "+x)

    # -----------------------------------------------------
    # 16. 看得懂的數據
    # -----------------------------------------------------
    st.subheader("📊 目前重要數據")

    data=pd.DataFrame([
        ["目前參考價",f"{current:.2f}","現在用來判斷進場的位置"],
        ["MA5",f"{float(row['MA5']):.2f}","5日平均成本"],
        ["MA10",f"{float(row['MA10']):.2f}","10日平均成本"],
        ["MA20",f"{float(row['MA20']):.2f}","短中期趨勢線"],
        ["RSI",f"{float(row['RSI'])*100:.1f}",">70偏熱；<30偏超賣"],
        ["ATR",f"{atr:.2f}","這檔股票每天正常波動大約多大"],
        ["EXT過熱",f"{ext:.2f}","越高代表近期漲太快、越不適合追"],
        ["近5日相對大盤",f"{float(row['RS5'])*100:+.2f}%","正值=比大盤強"],
        ["成交量/20日均量",f"{float(row['VOL_RATIO20']):.2f}倍",">1代表量比平常大"],
        ["距20日高點",f"{float(row['DIST_H20'])*100:+.2f}%","越接近0代表接近短期高點"],
        ["市場環境",market_state,"大盤與海外環境綜合結果"],
        ["前一晚NASDAQ",f"{float(row['US_NAS1'])*100:+.2f}%","台股開盤前已知道的美股資訊"],
        ["前一晚費半",f"{float(row['US_SOX1'])*100:+.2f}%","電子/半導體的重要外部參考"],
    ],columns=["項目","數值","意思"])

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # 17. 5D / 10D 簡單圖
    # -----------------------------------------------------
    st.subheader("📈 未來10個交易日情境")

    fdates=future_sessions(last_date,10)

    fig=go.Figure()

    fig.add_trace(go.Scatter(
        x=fdates,
        y=med,
        mode="lines+markers",
        line=dict(color=GOLD,width=3),
        name="情境中位價"
    ))

    fig.add_trace(go.Scatter(
        x=list(fdates)+list(fdates[::-1]),
        y=list(hi)+list(lo[::-1]),
        fill="toself",
        fillcolor="rgba(241,196,15,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="20%～80%主要區間"
    ))

    fig.add_hline(
        y=current,
        line_dash="dash",
        line_color="white",
        annotation_text="目前價格"
    )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        yaxis_title="價格",
        hovermode="x unified",
        margin=dict(l=30,r=20,t=30,b=30)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "背景回測只負責自動修正模型；"
        "你實際要看的就是：次日能不能買、5天方向、10天方向、進場區與防守價。"
    )
