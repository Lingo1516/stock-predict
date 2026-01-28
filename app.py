import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
import warnings
from dataclasses import dataclass
from datetime import datetime, time

warnings.filterwarnings("ignore")

# =========================
# Optional: Plotly
# =========================
PLOTLY_ERROR = ""
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception as e:
    HAS_PLOTLY = False
    PLOTLY_ERROR = str(e)

# =========================
# Optional: TW market calendar
# =========================
HAS_TW_CAL = False
try:
    import pandas_market_calendars as mcal
    HAS_TW_CAL = True
except Exception:
    HAS_TW_CAL = False

# =========================
# Dependencies
# =========================
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

import ta


# =========================
# Config
# =========================
@dataclass
class Config:
    # 訓練資料長度（天）：越長越穩但越慢
    lookback_days: int = 1200
    # 預測交易日數（你要的 10 天）
    forecast_days: int = 10
    # 訓練最小樣本
    min_train_rows: int = 200

    # 遞迴預測護欄（避免爆走）
    atr_period: int = 14
    guard_atr_mult: float = 3.0  # MA20 +/- 3*ATR

    # 模型設定
    rf_estimators: int = 300
    rf_max_depth: int = 10
    hgb_max_iter: int = 450

    # 區間：~80%（常態近似 1.28 sigma）
    interval_z: float = 1.28

CFG = Config()


# =========================
# Helpers
# =========================
def safe_download(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def market_status(code: str) -> str:
    """台股優先用交易日曆，沒有就 fallback 推測。"""
    is_tw = code.upper().endswith(".TW")
    if is_tw and HAS_TW_CAL:
        try:
            cal = mcal.get_calendar("XTAI")
            now = pd.Timestamp.now(tz="Asia/Taipei")
            sched = cal.schedule(start_date=now.date(), end_date=now.date())
            if sched.empty:
                return "非交易日"
            open_t = sched.iloc[0]["market_open"].tz_convert("Asia/Taipei")
            close_t = sched.iloc[0]["market_close"].tz_convert("Asia/Taipei")
            if open_t <= now <= close_t:
                return "盤中"
            return "已收盤"
        except Exception:
            pass

    # fallback（簡化推測）
    now = datetime.now()
    if now.weekday() >= 5:
        return "非交易日(推測)"
    if is_tw:
        if time(9, 0) <= now.time() <= time(13, 30):
            return "盤中(推測)"
        return "已收盤(推測)"
    return "日線資料(不判斷盤中)"


# =========================
# Feature Engineering (只用 Close/Volume 可推進，避免 High/Low 自我餵食漂移)
# =========================
FEATURES = [
    "Ret1", "Ret5",
    "MA5", "MA10", "MA20", "MA60",
    "RSI",
    "Vol20",
    "VolChg"
]

def add_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """只在「真實歷史資料」上計算 ATR / MA20（護欄用）"""
    df = df.copy()
    close = df["Close"]
    df["MA20"] = close.rolling(20).mean()
    # ATR 需要 High/Low，只用在真實歷史資料上
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], close, window=CFG.atr_period)
    df["ATR"] = atr.average_true_range()
    return df

def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """模型特徵：只用 Close/Volume 產生，方便遞迴預測時更新"""
    df = df.copy()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    df["Ret1"] = np.log(close).diff()
    df["Ret5"] = np.log(close).diff(5)

    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()

    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    r1 = np.log(close).diff()
    df["Vol20"] = r1.rolling(20).std() * np.sqrt(252)

    df["VolChg"] = vol.pct_change().replace([np.inf, -np.inf], np.nan)

    return df.dropna().copy()


def compute_next_feature_row(close_hist: list[float], vol_hist: list[float]) -> np.ndarray:
    """遞迴預測用：根據目前 close_hist / vol_hist 生成下一步特徵列"""
    s = pd.Series(close_hist, dtype="float64")
    v = pd.Series(vol_hist, dtype="float64")

    # returns
    ret1 = float(np.log(s.iloc[-1]) - np.log(s.iloc[-2])) if len(s) >= 2 else 0.0
    ret5 = float(np.log(s.iloc[-1]) - np.log(s.iloc[-6])) if len(s) >= 6 else 0.0

    # moving avgs
    ma5 = float(s.iloc[-5:].mean()) if len(s) >= 5 else float(s.mean())
    ma10 = float(s.iloc[-10:].mean()) if len(s) >= 10 else float(s.mean())
    ma20 = float(s.iloc[-20:].mean()) if len(s) >= 20 else float(s.mean())
    ma60 = float(s.iloc[-60:].mean()) if len(s) >= 60 else float(s.mean())

    # RSI needs enough points; fallback 50
    if len(s) >= 15:
        rsi = float(ta.momentum.RSIIndicator(s, window=14).rsi().iloc[-1])
    else:
        rsi = 50.0

    # vol20 based on log returns
    r1 = np.log(s).diff().dropna()
    if len(r1) >= 20:
        vol20 = float(r1.iloc[-20:].std() * np.sqrt(252))
    elif len(r1) >= 2:
        vol20 = float(r1.std() * np.sqrt(252))
    else:
        vol20 = 0.0

    # volume change
    if len(v) >= 2 and v.iloc[-2] != 0:
        volchg = float(v.iloc[-1] / v.iloc[-2] - 1.0)
    else:
        volchg = 0.0

    row = np.array([[ret1, ret5, ma5, ma10, ma20, ma60, rsi, vol20, volchg]], dtype="float64")
    return row


# =========================
# Models: Ensemble + CV weighting
# =========================
def train_ensemble_with_cv(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """
    兩模型集成：HGB + RF
    權重：以 TimeSeriesSplit 的 CV MAE 反向加權（越準權重越高）
    """
    models = {
        "HGB": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=CFG.hgb_max_iter,
            random_state=seed
        ),
        "RF": RandomForestRegressor(
            n_estimators=CFG.rf_estimators,
            max_depth=CFG.rf_max_depth,
            min_samples_split=6,
            random_state=seed,
            n_jobs=-1
        )
    }

    tscv = TimeSeriesSplit(n_splits=5)
    cv_mae = {}

    for name, model in models.items():
        fold_mae = []
        for tr, te in tscv.split(X):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            fold_mae.append(mean_absolute_error(y[te], pred))
        cv_mae[name] = float(np.mean(fold_mae))

    # weights = inverse MAE
    inv = {k: 1.0 / max(v, 1e-9) for k, v in cv_mae.items()}
    s = sum(inv.values())
    weights = {k: inv[k] / s for k in inv}

    # fit full
    trained = {}
    for name, model in models.items():
        model.fit(X, y)
        trained[name] = model

    return trained, weights, cv_mae

def ensemble_predict(models: dict, weights: dict, X: np.ndarray) -> np.ndarray:
    pred = None
    for name, model in models.items():
        p = model.predict(X)
        w = weights.get(name, 0.0)
        pred = p * w if pred is None else pred + p * w
    return pred


def estimate_sigma(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """用近 80 筆殘差估 sigma，做 80% 區間"""
    resid = (y_true - y_pred)
    if resid.size >= 80:
        resid = resid[-80:]
    return float(np.std(resid))


# =========================
# Forecast: Recursive 10 business days
# =========================
@st.cache_data(ttl=3600)
def run_forecast(code: str, lookback_days: int):
    end = pd.Timestamp(datetime.today().date()) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days)

    df_raw = safe_download(code, start, end)
    if df_raw.empty:
        return None

    # indicators for guard rail (MA20/ATR) on REAL data only
    df_guard = add_base_indicators(df_raw).dropna().copy()

    # features for model
    df_feat = add_model_features(df_raw)
    if len(df_feat) < CFG.min_train_rows:
        return {
            "error": "資料不足，請調高訓練資料量（lookback_days）或換標的。",
            "df_raw": df_raw
        }

    X = df_feat[FEATURES].values
    y = df_feat["Close"].values

    models, weights, cv_mae = train_ensemble_with_cv(X, y)

    # in-sample pred to estimate sigma (for interval)
    y_pred = ensemble_predict(models, weights, X)
    sigma = estimate_sigma(y, y_pred)

    # recursive forecast 10 business days
    future_dates = pd.bdate_range(start=df_feat.index[-1], periods=CFG.forecast_days + 1)[1:]

    close_hist = df_feat["Close"].astype(float).tolist()
    vol_hist = df_raw.loc[df_feat.index, "Volume"].astype(float).tolist()

    # volume assumption: use last 20 avg
    if len(vol_hist) >= 20:
        future_vol = float(np.mean(vol_hist[-20:]))
    else:
        future_vol = float(np.mean(vol_hist))

    # guard rails from last real MA20/ATR
    # (如果 df_guard 比 df_feat 少一些，取最新可用的)
    last_ma20 = float(df_guard["MA20"].iloc[-1]) if "MA20" in df_guard.columns else float(df_feat["MA20"].iloc[-1])
    last_atr = float(df_guard["ATR"].iloc[-1]) if "ATR" in df_guard.columns else 0.0

    upper = last_ma20 + CFG.guard_atr_mult * last_atr if last_atr > 0 else np.inf
    lower = last_ma20 - CFG.guard_atr_mult * last_atr if last_atr > 0 else -np.inf

    preds = []
    hi = []
    lo = []

    for _ in range(CFG.forecast_days):
        x_next = compute_next_feature_row(close_hist, vol_hist)
        p = float(ensemble_predict(models, weights, x_next)[0])

        # guard rails
        p = min(max(p, lower), upper)

        preds.append(p)
        hi.append(p + CFG.interval_z * sigma)
        lo.append(p - CFG.interval_z * sigma)

        # update history for next step
        close_hist.append(p)
        vol_hist.append(future_vol)

    # build result
    last_close = float(df_feat["Close"].iloc[-1])
    result_df = pd.DataFrame({
        "日期": [d.date() for d in future_dates],
        "預測價": np.round(preds, 2),
        "漲跌幅": [f"{(p - last_close) / last_close * 100:+.2f}%" for p in preds],
        "區間下界(約80%)": np.round(lo, 2),
        "區間上界(約80%)": np.round(hi, 2)
    })

    out = {
        "df_raw": df_raw,
        "df_feat": df_feat,
        "last_close": last_close,
        "result_df": result_df,
        "models": models,
        "weights": weights,
        "cv_mae": cv_mae,
        "sigma": sigma,
        "future_dates": future_dates,
        "preds": preds
    }
    return out


# =========================
# Plot
# =========================
def plot_price(df_raw: pd.DataFrame, future_dates, preds):
    tail = df_raw.tail(120).copy()
    fut = pd.DataFrame({"Close": preds}, index=future_dates)
    merged = pd.concat([tail[["Close"]], fut], axis=0)
    return merged

def plot_candles_with_forecast(df_raw: pd.DataFrame, future_dates, preds, lo=None, hi=None):
    if not HAS_PLOTLY:
        return None

    dfp = df_raw.tail(140).copy()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28],
                        subplot_titles=("K線 + 10日預測", "成交量"))

    fig.add_trace(go.Candlestick(
        x=dfp.index, open=dfp["Open"], high=dfp["High"], low=dfp["Low"], close=dfp["Close"], name="K線"
    ), row=1, col=1)

    # forecast line
    connect_x = [dfp.index[-1]] + list(future_dates)
    connect_y = [float(dfp["Close"].iloc[-1])] + list(preds)
    fig.add_trace(go.Scatter(x=connect_x, y=connect_y, name="預測", line=dict(dash="dash", width=3)), row=1, col=1)

    # interval band if provided
    if lo is not None and hi is not None:
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=list(hi) + list(lo)[::-1],
            fill="toself", opacity=0.18,
            line=dict(width=0),
            name="約80%區間"
        ), row=1, col=1)

    fig.add_trace(go.Bar(x=dfp.index, y=dfp["Volume"], name="Volume"), row=2, col=1)
    fig.update_layout(height=680, xaxis_rangeslider_visible=False, hovermode="x unified")
    return fig


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="AI 股價預測（10日）", layout="wide", page_icon="🔮")

st.title("🔮 AI 股價預測（未來 10 個交易日）")
st.caption("說明：本工具用『遞迴預測』逐日更新特徵，因此不會出現 10 天都一樣的預測。")

with st.sidebar:
    st.header("⚙️ 設定")
    data_source = st.radio("資料來源", ["自動下載 (yfinance)", "手動貼上CSV資料"])

    if data_source == "自動下載 (yfinance)":
        code = st.text_input("股票代號（台股輸入 2330、美股 AAPL）", "2330")
        if code.strip().isdigit():
            code = code.strip() + ".TW"
        code = code.strip().upper()

        lookback_days = st.selectbox(
            "訓練資料量（越多越穩、越慢）",
            options=[600, 900, 1200, 1600, 2000],
            index=2
        )
        show_interval = st.checkbox("顯示預測區間（約80%）", value=True)
        use_plotly = st.checkbox("使用 Plotly K 線圖（需安裝 plotly）", value=True)

        st.divider()
        st.write(f"預測交易日數：**{CFG.forecast_days}**（固定）")
        st.write(f"台股交易日曆：{'已啟用(XTAI)' if HAS_TW_CAL else '未安裝套件（不影響預測）'}")

    else:
        st.info("手動 CSV：只做『遞迴預測10日』，不做交易日曆判斷。")
        show_interval = st.checkbox("顯示預測區間（約80%）", value=True)
        use_plotly = st.checkbox("使用 Plotly K 線圖（需安裝 plotly）", value=True)

run_btn = st.button("🚀 開始預測", type="primary", use_container_width=True)

if run_btn:
    if data_source == "手動貼上CSV資料":
        manual = st.text_area("貼上 CSV（需含 Date, Open, High, Low, Close, Volume 欄位）", height=240)
        if not manual.strip():
            st.error("請先貼上 CSV。")
            st.stop()

        try:
            df_raw = pd.read_csv(io.StringIO(manual))
            df_raw["Date"] = pd.to_datetime(df_raw["Date"])
            df_raw = df_raw.set_index("Date").sort_index()

            # 直接用 df_raw 當作資料來源，但仍走同一套 features/forecast 流程
            df_guard = add_base_indicators(df_raw).dropna().copy()
            df_feat = add_model_features(df_raw)

            if len(df_feat) < CFG.min_train_rows:
                st.error("資料不足，至少需要更長的歷史資料才能穩定預測。")
                st.stop()

            X = df_feat[FEATURES].values
            y = df_feat["Close"].values

            models, weights, cv_mae = train_ensemble_with_cv(X, y)
            y_pred = ensemble_predict(models, weights, X)
            sigma = estimate_sigma(y, y_pred)

            future_dates = pd.bdate_range(start=df_feat.index[-1], periods=CFG.forecast_days + 1)[1:]

            close_hist = df_feat["Close"].astype(float).tolist()
            vol_hist = df_raw.loc[df_feat.index, "Volume"].astype(float).tolist()
            future_vol = float(np.mean(vol_hist[-20:])) if len(vol_hist) >= 20 else float(np.mean(vol_hist))

            last_ma20 = float(df_guard["MA20"].iloc[-1]) if "MA20" in df_guard.columns else float(df_feat["MA20"].iloc[-1])
            last_atr = float(df_guard["ATR"].iloc[-1]) if "ATR" in df_guard.columns else 0.0

            upper = last_ma20 + CFG.guard_atr_mult * last_atr if last_atr > 0 else np.inf
            lower = last_ma20 - CFG.guard_atr_mult * last_atr if last_atr > 0 else -np.inf

            preds, hi, lo = [], [], []
            for _ in range(CFG.forecast_days):
                x_next = compute_next_feature_row(close_hist, vol_hist)
                p = float(ensemble_predict(models, weights, x_next)[0])
                p = min(max(p, lower), upper)

                preds.append(p)
                hi.append(p + CFG.interval_z * sigma)
                lo.append(p - CFG.interval_z * sigma)

                close_hist.append(p)
                vol_hist.append(future_vol)

            last_close = float(df_feat["Close"].iloc[-1])
            result_df = pd.DataFrame({
                "日期": [d.date() for d in future_dates],
                "預測價": np.round(preds, 2),
                "漲跌幅": [f"{(p - last_close) / last_close * 100:+.2f}%" for p in preds],
                "區間下界(約80%)": np.round(lo, 2),
                "區間上界(約80%)": np.round(hi, 2)
            })

            st.success("✅ CSV 讀取成功，已完成預測。")

            st.subheader("📌 模型摘要")
            st.write(f"最後收盤價：**{last_close:.2f}**")
            st.write(f"CV MAE：HGB {cv_mae.get('HGB', np.nan):.3f}｜RF {cv_mae.get('RF', np.nan):.3f}")
            st.write(f"權重：HGB {weights.get('HGB', 0):.2f}｜RF {weights.get('RF', 0):.2f}")
            st.write(f"sigma（近80日殘差標準差）：**{sigma:.3f}**")

            st.subheader("🔮 未來 10 個交易日預測")
            st.dataframe(result_df, use_container_width=True)

            st.subheader("📈 走勢（歷史 + 預測）")
            merged = plot_price(df_raw, future_dates, preds)
            st.line_chart(merged)

            if use_plotly and HAS_PLOTLY:
                fig = plot_candles_with_forecast(df_raw, future_dates, preds, lo if show_interval else None, hi if show_interval else None)
                st.plotly_chart(fig, use_container_width=True)
            elif use_plotly and not HAS_PLOTLY:
                st.warning(f"未安裝 plotly：{PLOTLY_ERROR}")

        except Exception as e:
            st.error(f"CSV 解析失敗：{e}")
            st.stop()

    else:
        # yfinance mode
        status = market_status(code)
        st.info(f"市場狀態：**{status}**（提示用；本預測基於日線資料）")

        with st.spinner("下載資料、訓練 Ensemble、進行 10 日遞迴預測中..."):
            out = run_forecast(code, lookback_days)

        if out is None:
            st.error("資料下載失敗，請檢查代號或網路。")
            st.stop()

        if "error" in out:
            st.error(out["error"])
            if "df_raw" in out and isinstance(out["df_raw"], pd.DataFrame) and not out["df_raw"].empty:
                st.write("已下載到部分資料，但有效樣本不足。")
            st.stop()

        df_raw = out["df_raw"]
        result_df = out["result_df"]
        last_close = out["last_close"]
        weights = out["weights"]
        cv_mae = out["cv_mae"]
        sigma = out["sigma"]
        future_dates = out["future_dates"]
        preds = out["preds"]

        st.subheader("📌 模型摘要")
        c1, c2, c3 = st.columns(3)
        c1.metric("最後收盤價", f"{last_close:.2f}")
        c2.metric("CV MAE (HGB)", f"{cv_mae.get('HGB', np.nan):.3f}")
        c3.metric("CV MAE (RF)", f"{cv_mae.get('RF', np.nan):.3f}")

        st.write(f"權重：HGB **{weights.get('HGB', 0):.2f}**｜RF **{weights.get('RF', 0):.2f}**")
        st.write(f"sigma（近80日殘差標準差）：**{sigma:.3f}**")

        st.subheader("🔮 未來 10 個交易日預測")
        if not show_interval:
            result_df2 = result_df.drop(columns=["區間下界(約80%)", "區間上界(約80%)"])
            st.dataframe(result_df2, use_container_width=True)
        else:
            st.dataframe(result_df, use_container_width=True)

        st.subheader("📈 走勢（歷史 + 預測）")
        merged = plot_price(df_raw, future_dates, preds)
        st.line_chart(merged)

        if use_plotly and HAS_PLOTLY:
            lo = result_df["區間下界(約80%)"].astype(float).tolist()
            hi = result_df["區間上界(約80%)"].astype(float).tolist()
            fig = plot_candles_with_forecast(df_raw, future_dates, preds, lo if show_interval else None, hi if show_interval else None)
            st.plotly_chart(fig, use_container_width=True)
        elif use_plotly and not HAS_PLOTLY:
            st.warning(f"未安裝 plotly：{PLOTLY_ERROR}")

st.caption("⚠️ 免責聲明：本工具僅供研究與學習，不構成任何投資建議。")
