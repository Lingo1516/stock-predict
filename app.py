import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import ta
from scipy.stats import t as t_dist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import plotly.graph_objects as go

TZ = pytz.timezone('Asia/Taipei')
RED, GREEN, GOLD = '#E74C3C', '#2ECC71', '#F39C12'

FEATURES = [
    'RET1','RET2','RET5','RET10','RET20',
    'DEV_MA5','DEV_MA10','DEV_MA20','DEV_MA60','MA5_SLOPE','MA20_SLOPE',
    'RSI','MACD_N','MACD_HIST_N','K','D','BB_pct','ATR_PCT',
    'VOL_RATIO5','VOL_RATIO20','OBV_SLOPE5','GAP','RANGE_PCT','CLOSE_POS',
    'MKT_RET1','MKT_RET5','MKT_RET20','MKT_DEV20','MKT_DEV60','MKT_VOL20',
    'RS1','RS5','RS20','EXT','DOW_SIN','DOW_COS','Q_SIN','Q_COS'
]

@st.cache_data(ttl=1800, show_spinner=False)
def download_daily(code: str, years: int = 7):
    end = datetime.now(TZ).date() + timedelta(days=1)
    start = end - timedelta(days=365 * years + 30)
    base = code.replace('.TW','').replace('.TWO','').strip().upper()
    tries = [base+'.TW', base+'.TWO'] if base.isdigit() else [code]
    for ticker in tries:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                             progress=False, timeout=20, threads=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [x[0] for x in df.columns]
                df = df[['Open','High','Low','Close','Volume']].dropna().copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df, ticker
        except Exception:
            pass
    return pd.DataFrame(), code

@st.cache_data(ttl=1800, show_spinner=False)
def download_index(symbol='^TWII', years: int = 7):
    end = datetime.now(TZ).date() + timedelta(days=1)
    start = end - timedelta(days=365 * years + 30)
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True,
                         progress=False, timeout=20, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [x[0] for x in df.columns]
        df = df[['Open','High','Low','Close','Volume']].dropna().copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

def rz(s, win=252):
    mu = s.rolling(win, min_periods=60).mean()
    sd = s.rolling(win, min_periods=60).std().replace(0, np.nan)
    return ((s-mu)/sd).clip(-5,5)

def build_features(stock, market):
    df = stock.copy()
    c,o,h,l,v = [df[x].astype(float) for x in ['Close','Open','High','Low','Volume']]
    for n in [1,2,5,10,20]: df[f'RET{n}'] = c.pct_change(n)
    for n in [5,10,20,60]:
        df[f'MA{n}'] = c.rolling(n).mean()
        df[f'DEV_MA{n}'] = c/df[f'MA{n}']-1
    df['MA5_SLOPE'] = df['MA5'].pct_change(3)
    df['MA20_SLOPE'] = df['MA20'].pct_change(5)
    df['RSI'] = ta.momentum.RSIIndicator(c,14).rsi()/100
    m = ta.trend.MACD(c,26,12,9)
    atr = ta.volatility.AverageTrueRange(h,l,c,14).average_true_range()
    df['ATR'], df['ATR_PCT'] = atr, atr/c
    df['MACD_N'] = m.macd()/atr.replace(0,np.nan)
    df['MACD_HIST_N'] = m.macd_diff()/atr.replace(0,np.nan)
    s = ta.momentum.StochasticOscillator(h,l,c,9,3)
    df['K'], df['D'] = s.stoch()/100, s.stoch_signal()/100
    bb = ta.volatility.BollingerBands(c,20,2)
    df['BB_pct'] = bb.bollinger_pband()
    obv = ta.volume.OnBalanceVolumeIndicator(c,v).on_balance_volume()
    df['OBV_SLOPE5'] = obv.diff(5)/v.rolling(20).mean().replace(0,np.nan)
    df['VOL_RATIO5'] = v/v.rolling(5).mean()
    df['VOL_RATIO20'] = v/v.rolling(20).mean()
    prev = c.shift(1)
    df['GAP'] = o/prev-1
    df['RANGE_PCT'] = (h-l)/prev
    df['CLOSE_POS'] = (c-l)/(h-l).replace(0,np.nan)

    if market is not None and not market.empty:
        mc = market['Close'].astype(float)
        mf = pd.DataFrame(index=market.index)
        mf['MKT_RET1'] = mc.pct_change(1)
        mf['MKT_RET5'] = mc.pct_change(5)
        mf['MKT_RET20'] = mc.pct_change(20)
        mf['MKT_DEV20'] = mc/mc.rolling(20).mean()-1
        mf['MKT_DEV60'] = mc/mc.rolling(60).mean()-1
        mf['MKT_VOL20'] = np.log(mc).diff().rolling(20).std()
        df = df.join(mf, how='left').ffill()
    else:
        for x in ['MKT_RET1','MKT_RET5','MKT_RET20','MKT_DEV20','MKT_DEV60','MKT_VOL20']:
            df[x] = 0.0

    df['RS1'] = df['RET1']-df['MKT_RET1']
    df['RS5'] = df['RET5']-df['MKT_RET5']
    df['RS20'] = df['RET20']-df['MKT_RET20']
    df['EXT'] = (np.maximum(rz(df['RET5']),0) + np.maximum(rz(df['DEV_MA5']),0)
                 + 0.5*np.maximum(rz(df['VOL_RATIO20']),0) + 0.5*np.maximum(rz(df['GAP']),0))/3
    dow = pd.Series(df.index.dayofweek, index=df.index)
    q = pd.Series(df.index.quarter, index=df.index)
    df['DOW_SIN'], df['DOW_COS'] = np.sin(2*np.pi*dow/5), np.cos(2*np.pi*dow/5)
    df['Q_SIN'], df['Q_COS'] = np.sin(2*np.pi*(q-1)/4), np.cos(2*np.pi*(q-1)/4)

    df['FWD_RET1'] = c.shift(-1)/c-1
    df['FWD_RET5'] = c.shift(-5)/c-1
    # 避免把極小漲幅當成可交易成功：次日 >0.3%，5日 >0.6%
    df['Y1'] = np.where(df['FWD_RET1'].notna(), (df['FWD_RET1']>0.003).astype(int), np.nan)
    df['Y5'] = np.where(df['FWD_RET5'].notna(), (df['FWD_RET5']>0.006).astype(int), np.nan)
    return df.replace([np.inf,-np.inf],np.nan)

def fit_time_model(df, target):
    train = df[FEATURES+[target]].dropna().copy()
    if len(train)>1600: train = train.tail(1600)
    if len(train)<350: return None
    X, y = train[FEATURES], train[target].astype(int)
    pipe = Pipeline([
        ('z', StandardScaler()),
        ('lr', LogisticRegression(C=0.35, class_weight='balanced', max_iter=1200))
    ])
    tscv = TimeSeriesSplit(n_splits=5)
    oof = pd.Series(index=train.index, dtype=float)
    for tr, va in tscv.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof.iloc[va] = pipe.predict_proba(X.iloc[va])[:,1]
    mask = oof.notna(); yy = y.loc[mask]; pp = oof.loc[mask]
    pred = (pp>=0.5).astype(int)
    acc = accuracy_score(yy,pred)
    brier = brier_score_loss(yy,pp)
    try: auc = roc_auc_score(yy,pp)
    except Exception: auc = np.nan
    hi = (pp>=0.58)|(pp<=0.42)
    hi_acc = accuracy_score(yy.loc[hi], (pp.loc[hi]>=0.5).astype(int)) if hi.sum()>=25 else np.nan
    coverage = float(hi.mean())
    pipe.fit(X,y)
    cur = df[FEATURES].dropna().iloc[[-1]]
    prob = float(pipe.predict_proba(cur)[:,1][0])
    return dict(prob=prob, acc=float(acc), auc=float(auc), brier=float(brier),
                hi_acc=float(hi_acc) if np.isfinite(hi_acc) else np.nan,
                coverage=coverage, n=len(train), base=float(y.mean()))

def market_regime(row):
    vals = [row['MKT_RET1'],row['MKT_RET5'],row['MKT_RET20'],row['MKT_DEV20'],row['MKT_DEV60']]
    score = sum(1 if float(v)>0 else -1 for v in vals)
    return ('RISK_ON' if score>=3 else 'RISK_OFF' if score<=-3 else 'NEUTRAL'), score

def ext_label(x):
    return '高' if x>=1.0 else '中' if x>=0.45 else '低'

def scenario_paths(df, horizon=5, n_paths=3000, seed=42):
    ret = np.log(df['Close']).diff().dropna()
    last = float(df['Close'].iloc[-1])
    mu = float(np.clip(ret.ewm(span=20,adjust=False).mean().iloc[-1], -0.004,0.004))
    sig_series = ret.rolling(20).std().dropna()
    sig = float(sig_series.iloc[-1]); sig_lr = float(sig_series.tail(252).median())
    z = (ret/ret.rolling(20).std()).replace([np.inf,-np.inf],np.nan).dropna().clip(-8,8)
    try:
        dft,_,sct = t_dist.fit(z.values, floc=0)
        dft, sct = float(np.clip(dft,3,30)), float(np.clip(sct,0.4,2.0))
    except Exception:
        dft,sct = 6.0,1.0
    rng = np.random.default_rng(seed)
    prices = np.full(n_paths,last); paths = np.zeros((n_paths,horizon))
    for t in range(horizon):
        sig = 0.85*sig + 0.15*sig_lr
        shock = t_dist.rvs(df=dft,loc=0,scale=sct,size=n_paths,random_state=rng)
        step = np.clip(mu + sig*shock, -0.15,0.15)
        prices *= np.exp(step); paths[:,t] = prices
    return paths

def future_dates(last_date, n=5):
    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar('XTAI')
        start = pd.Timestamp(last_date)+pd.Timedelta(days=1)
        ss = cal.sessions_in_range(start, start+pd.Timedelta(days=30))
        return pd.DatetimeIndex([pd.Timestamp(x).tz_localize(None) for x in ss[:n]])
    except Exception:
        return pd.bdate_range(pd.Timestamp(last_date)+pd.Timedelta(days=1), periods=n)

def chart(df, paths, fdates):
    hist = df.tail(100)
    med = np.median(paths,axis=0); p10=np.percentile(paths,10,axis=0); p90=np.percentile(paths,90,axis=0)
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=hist.index,open=hist['Open'],high=hist['High'],low=hist['Low'],close=hist['Close'],
                                 increasing_line_color=RED,decreasing_line_color=GREEN,name='歷史'))
    fig.add_trace(go.Scatter(x=fdates,y=med,mode='lines+markers',line=dict(color=GOLD,dash='dash'),name='情境中位'))
    fig.add_trace(go.Scatter(x=list(fdates)+list(fdates[::-1]),y=list(p90)+list(p10[::-1]),fill='toself',
                             fillcolor='rgba(241,196,15,.12)',line=dict(color='rgba(0,0,0,0)'),name='10~90%區間'))
    fig.update_layout(template='plotly_dark',height=520,xaxis_rangeslider_visible=False)
    return fig

st.set_page_config(page_title='股票助手 V2',layout='wide',page_icon='📘')
st.title('📘 股票助手 V2｜時間序列驗證 + 1D / 5D')
st.caption('主訊號改為樣本外時間序列模型；Monte Carlo 只做價格區間，不再把模擬比例當成勝率。')

with st.sidebar:
    raw = st.text_input('股票代號','2330').strip()
    code = raw.replace('.TW','').replace('.TWO','').upper()
    capital = st.number_input('資金（元）',min_value=0.0,value=200000.0,step=10000.0)
    risk_pct = st.slider('單筆最大風險（%）',0.5,5.0,1.5,0.5)/100
    atr_mult = st.slider('ATR停損倍數',1.0,3.0,2.0,0.25)
    npaths = st.slider('情境模擬條數',1000,8000,3000,500)

if st.button('🚀 開始分析',type='primary',use_container_width=True):
    stock,symbol = download_daily(code)
    if stock.empty:
        st.error('抓不到股票資料'); st.stop()
    benchmark = '^TWOII' if symbol.endswith('.TWO') else '^TWII'
    market = download_index(benchmark)
    if market.empty and benchmark!='^TWII':
        benchmark='^TWII'; market=download_index(benchmark)
    df = build_features(stock,market)
    if len(df.dropna(subset=FEATURES))<350:
        st.error('歷史樣本不足'); st.stop()
    last = df.dropna(subset=FEATURES).iloc[-1]
    last_date = df.dropna(subset=FEATURES).index[-1]
    lc = float(last['Close']); atr=float(last['ATR'])
    m1,m5 = fit_time_model(df,'Y1'), fit_time_model(df,'Y5')
    if m1 is None or m5 is None:
        st.error('模型樣本不足'); st.stop()
    regime,score = market_regime(last)
    ext=float(last['EXT'])
    seed=abs(hash((symbol,str(last_date.date()))))%(2**32-1)
    paths=scenario_paths(df.loc[:last_date],5,npaths,seed)
    fdates=future_dates(last_date,5)
    med5=float(np.median(paths[:,-1])); p10=float(np.percentile(paths[:,-1],10)); p90=float(np.percentile(paths[:,-1],90))

    st.success(f'✅ {symbol}｜截止 {last_date.date()}｜市場基準 {benchmark}')
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric('最新收盤',f'{lc:.2f}')
    c2.metric('次日可交易上漲機率',f"{m1['prob']*100:.1f}%")
    c3.metric('5D可交易上漲機率',f"{m5['prob']*100:.1f}%")
    c4.metric('Market Regime',regime,delta=f'score {score:+d}')
    c5.metric('EXT過熱',ext_label(ext),delta=f'{ext:.2f}')

    st.subheader('🧪 樣本外驗證')
    q1,q2=st.columns(2)
    hi1 = f"{m1['hi_acc']:.3f}" if np.isfinite(m1['hi_acc']) else 'N/A'
    q1.write(f"次日：N={m1['n']}｜Acc={m1['acc']:.3f}｜AUC={m1['auc']:.3f}｜Brier={m1['brier']:.3f}｜高信心命中={hi1}")
    q2.write(f"5D：N={m5['n']}｜Acc={m5['acc']:.3f}｜AUC={m5['auc']:.3f}｜Brier={m5['brier']:.3f}｜Coverage={m5['coverage']:.1%}")

    weak = (np.isfinite(m1['auc']) and m1['auc']<0.52) or (np.isfinite(m5['auc']) and m5['auc']<0.52)
    if weak: st.warning('⚠️ 這檔目前樣本外辨識力偏弱，不要把機率當成可靠優勢。')

    if weak:
        action='觀察'
    elif regime=='RISK_OFF' and m5['prob']<0.62:
        action='WAIT'
    elif m5['prob']>=0.60 and ext<0.8:
        action='候選'
    elif m5['prob']>=0.55:
        action='WAIT'
    else:
        action='不進'
    st.subheader(f'🎯 綜合結論：{action}')
    st.write(f'5D情境中位 {med5:.2f}｜10~90%區間 {p10:.2f}~{p90:.2f}')
    st.caption('情境區間不是勝率；勝率看上面的樣本外模型。')

    stop=lc-atr_mult*atr
    risk_budget=capital*risk_pct
    qty_risk=int(risk_budget//max(lc-stop,1e-6))
    qty_cap=int(capital//lc)
    qty=max(0,min(qty_risk,qty_cap))
    st.subheader('🛡️ 風險控制')
    st.write(f'ATR停損參考：{stop:.2f}｜風險預算：{risk_budget:,.0f}｜最多股數：{qty:,}')

    st.plotly_chart(chart(df.loc[:last_date],paths,fdates),use_container_width=True)
    st.caption('⚠️ 工具提供機率與風險判讀，不保證報酬。')
