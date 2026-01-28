import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
import warnings
from dataclasses import dataclass
from datetime import datetime, time

import ta
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

# ====== Plotly optional ======
PLOTLY_ERROR = ""
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception as e:
    HAS_PLOTLY = False
    PLOTLY_ERROR = str(e)

# ====== TW market calendar optional ======
HAS_TW_CAL = False
try:
    import pandas_market_calendars as mcal
    HAS_TW_CAL = True
except Exception:
    HAS_TW_CAL = False

# ====== 參數設定 ======
@dataclass
class Config:
    bottom_lookback: int = 20
    top_lookback: int = 20
    higher_high_lookback: int = 5
    lower_low_lookback: int = 5

    stoch_k: int = 9
    stoch_d: int = 3
    stoch_smooth: int = 3
    kd_threshold: float = 20.0
    kd_threshold_sell: float = 80.0

    ma_short: int = 20
    ma_long: int = 60
    volume_ma: int = 20
    atr_period: int = 14

    risk_per_trade: float = 0.01
    capital: float = 1_000_000

    fwd_days: int = 5
    # 回測與模型訓練控制（強化版會更吃算力，這裡做合理限制）
    train_min_rows: int = 140           # 最小訓練樣本
    backtest_max_rows: int = 420        # 近約 1.5~2 年交易日
    retrain_every: int = 5              # 回測時每 N 天重訓一次（大幅加速）

CFG = Config()

# ====== 股票代碼對照表（可自行擴充） ======
stock_name_dict = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
    "2303.TW": "聯電", "3711.TW": "日月光投控", "3034.TW": "聯詠", "2379.TW": "瑞昱",
    "3008.TW": "大立光", "2327.TW": "國巨", "2382.TW": "廣達", "3231.TW": "緯創",
    "2357.TW": "華碩", "2356.TW": "英業達", "2301.TW": "光寶科", "2412.TW": "中華電",
    "3045.TW": "台灣大", "4904.TW": "遠傳", "2345.TW": "智邦", "2368.TW": "金像電",
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2886.TW": "兆豐金",
    "2884.TW": "玉山金", "2892.TW": "第一金", "2885.TW": "元大金", "2880.TW": "華南金",
    "2883.TW": "開發金", "2890.TW": "永豐金",
    "2002.TW": "中鋼", "1301.TW": "台塑", "1303.TW": "南亞", "1326.TW": "台化",
    "6505.TW": "台塑化", "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海",
    "2618.TW": "長榮航", "2610.TW": "華航", "1101.TW": "台泥", "1102.TW": "亞泥",
    "1216.TW": "統一", "2912.TW": "統一超",
    "2376.TW": "技嘉", "2377.TW": "微星", "6669.TW": "緯穎", "3035.TW": "智原",
    "3443.TW": "創意", "3661.TW": "世芯-KY", "3017.TW": "奇鋐", "3324.TW": "雙鴻"
}

# =========================
# Utils
# =========================
def safe_download(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def pick_market_index(stock_code: str):
    code = stock_code.upper()
    if code.endswith(".TW"):
        return ["^TWII", "0050.TW"]  # 台股：加權指數優先，ETF fallback
    return ["^GSPC"]  # 美股：S&P 500

def market_status(code: str):
    """更準確的『是否交易日/盤中』顯示。台股優先用交易日曆；沒有套件就 fallback。"""
    is_tw = code.upper().endswith(".TW")

    # 台股用 XTAI 交易日曆（如果可用）
    if is_tw and HAS_TW_CAL:
        try:
            cal = mcal.get_calendar("XTAI")
            now = pd.Timestamp.now(tz="Asia/Taipei")
            sched = cal.schedule(start_date=now.date(), end_date=now.date())
            if sched.empty:
                return "非交易日", False
            open_t = sched.iloc[0]["market_open"].tz_convert("Asia/Taipei")
            close_t = sched.iloc[0]["market_close"].tz_convert("Asia/Taipei")
            if open_t <= now <= close_t:
                return "盤中", True
            return "已收盤", False
        except Exception:
            pass

    # fallback（簡化）：週末非交易；台股 9:00~13:30，美股不做盤中判斷（避免誤導）
    now_tw = datetime.now()
    if now_tw.weekday() >= 5:
        return "非交易日(推測)", False

    if is_tw:
        if time(9, 0) <= now_tw.time() <= time(13, 30):
            return "盤中(推測)", True
        return "已收盤(推測)", False

    return "日線資料(不判斷盤中)", False

# =========================
# Feature engineering
# =========================
def add_technical_indicators(df: pd.DataFrame, cfg: Config):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # MA
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA_S"] = df["MA20"]
    df["MA_L"] = df["MA60"]
    df["MA_S_SLOPE"] = df["MA_S"] - df["MA_S"].shift(5)

    # RSI / MACD / BB / ADX
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    df["ADX"] = ADXIndicator(high, low, close, window=14).adx()

    # KD（Stochastic）
    stoch = StochasticOscillator(high=high, low=low, close=close, window=cfg.stoch_k, smooth_window=cfg.stoch_smooth)
    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()

    # ATR
    atr_indicator = ta.volatility.AverageTrueRange(high, low, close, window=cfg.atr_period)
    df["ATR"] = atr_indicator.average_true_range()

    # 底/頂參考 + 量均
    df["RecentLow"] = close.rolling(cfg.bottom_lookback).min()
    df["PriorHigh"] = close.shift(1).rolling(cfg.higher_high_lookback).max()
    df["RecentHigh"] = close.rolling(cfg.top_lookback).max()
    df["PriorLow"] = close.shift(1).rolling(cfg.lower_low_lookback).min()
    df["VOL_MA"] = df["Volume"].rolling(cfg.volume_ma).mean()

    return df

def add_return_features(df: pd.DataFrame):
    df = df.copy()
    df["Ret1"] = np.log(df["Close"]).diff()
    df["Ret5"] = np.log(df["Close"]).diff(5)
    df["Vol10"] = df["Ret1"].rolling(10).std() * np.sqrt(252)
    df["Vol20"] = df["Ret1"].rolling(20).std() * np.sqrt(252)
    df["VolChg"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    return df

def build_dataset(stock_code: str, lookback_days: int):
    end = pd.Timestamp(datetime.today().date()) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days)

    df = safe_download(stock_code, start, end)
    if df.empty:
        return pd.DataFrame()

    # market index
    idx_df = pd.DataFrame()
    for idx in pick_market_index(stock_code):
        tmp = safe_download(idx, start, end)
        if not tmp.empty and "Close" in tmp.columns:
            idx_df = tmp
            break

    if idx_df.empty:
        df["Market_Close"] = np.nan
    else:
        df["Market_Close"] = idx_df["Close"].reindex(df.index).ffill()

    df = add_technical_indicators(df, CFG)
    df = add_return_features(df)

    # relative strength
    df["Mkt_Ret1"] = np.log(df["Market_Close"]).diff()
    df["RelStrength1"] = df["Ret1"] - df["Mkt_Ret1"]

    df = df.dropna().copy()
    return df

# =========================
# Signal logic (保留你原策略，但用於「技術面訊號」區塊)
# =========================
def generate_signal_row_buy(row_prior, row_now, cfg: Config):
    reasons = []
    bottom_built = (row_now["Close"] <= row_now["RecentLow"] * 1.08) and (row_now["Close"] > (row_now["PriorHigh"] * 0.8))
    if bottom_built: reasons.append("接近近期低點後回升")

    kd_cross_up = (row_prior["K"] < row_prior["D"]) and (row_now["K"] > row_now["D"])
    kd_above_threshold = row_now["K"] > cfg.kd_threshold
    kd_ok = kd_cross_up and kd_above_threshold
    if kd_ok: reasons.append(f"KD黃金交叉且K>{cfg.kd_threshold:.0f}")

    macd_up = (row_now["MACD"] > 0) and (row_now["MACD"] > row_prior["MACD"])
    if macd_up: reasons.append("MACD柱轉正且走揚")

    trend_ok = (row_now["MA_S"] > row_now["MA_L"]) and (row_now["MA_S_SLOPE"] > 0)
    if trend_ok: reasons.append("多頭趨勢濾網通過")

    volume_ok = row_now["Volume"] >= row_now["VOL_MA"]
    if volume_ok: reasons.append("量能不弱於均量")

    all_ok = bottom_built and kd_ok and macd_up and trend_ok and volume_ok
    return all_ok, reasons

def generate_signal_row_sell(row_prior, row_now, cfg: Config):
    reasons = []
    top_built = (row_now["Close"] >= row_now["RecentHigh"] * 0.92) and (row_now["Close"] < (row_now["PriorLow"] * 1.2))
    if top_built: reasons.append("接近近期高點後回落")

    kd_cross_down = (row_prior["K"] > row_prior["D"]) and (row_now["K"] < row_now["D"])
    kd_below_threshold = row_now["K"] < cfg.kd_threshold_sell
    kd_ok = kd_cross_down and kd_below_threshold
    if kd_ok: reasons.append(f"KD死亡交叉且K<{cfg.kd_threshold_sell:.0f}")

    macd_down = (row_now["MACD"] < 0) and (row_now["MACD"] < row_prior["MACD"])
    if macd_down: reasons.append("MACD柱轉負且走弱")

    trend_ok = (row_now["MA_S"] < row_now["MA_L"]) and (row_now["MA_S_SLOPE"] < 0)
    if trend_ok: reasons.append("空頭趨勢濾網通過")

    volume_ok = row_now["Volume"] >= row_now["VOL_MA"]
    if volume_ok: reasons.append("量能不弱於均量")

    all_ok = top_built and kd_ok and macd_down and trend_ok and volume_ok
    return all_ok, reasons

def evaluate_latest(df: pd.DataFrame, cfg: Config, strategy_type: str):
    need = max(cfg.ma_long, cfg.bottom_lookback, cfg.top_lookback, cfg.atr_period) + 5
    if len(df) < need:
        return {"是否符合訊號": False, "理由": "資料樣本太短", "動作": "無", "建議停損": 0, "估計ATR": 0, "建議股數": 0}

    df = df.dropna().copy()
    if len(df) < 2:
        return {"是否符合訊號": False, "理由": "有效樣本不足", "動作": "無", "建議停損": 0, "估計ATR": 0, "建議股數": 0}

    row_now = df.iloc[-1]
    row_prior = df.iloc[-2]
    atr = float(row_now["ATR"])

    if strategy_type == "buy":
        signal, reasons = generate_signal_row_buy(row_prior, row_now, cfg)
        stop_level = float(row_now["Close"] - 2.5 * atr)
        position_risk = float(row_now["Close"] - stop_level)
        action_text = "多方(買進)模式"
    else:
        signal, reasons = generate_signal_row_sell(row_prior, row_now, cfg)
        stop_level = float(row_now["Close"] + 2.5 * atr)
        position_risk = float(stop_level - row_now["Close"])
        action_text = "空方(放空)模式"

    position_size = 0
    if position_risk > 0:
        position_size = int((cfg.capital * cfg.risk_per_trade) // position_risk)

    return {
        "日期": df.index[-1].strftime("%Y-%m-%d"),
        "收盤": round(float(row_now["Close"]), 2),
        "是否符合訊號": bool(signal),
        "理由": "、".join(reasons) if reasons else "條件不足",
        "動作": action_text,
        "建議停損": round(float(stop_level), 2),
        "估計ATR": round(float(atr), 2),
        "建議股數": int(position_size)
    }

# =========================
# Strong ML: Ensemble + TimeSeries CV weights
# =========================
def cv_weighted_ensemble_train(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """
    回傳：trained_models(dict), weights(dict), cv_metrics(dict)
    權重 = 1 / CV_MAE（再正規化）
    """
    models = {
        "HGB": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            random_state=seed
        ),
        "RF": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=6,
            random_state=seed,
            n_jobs=-1
        )
    }

    tscv = TimeSeriesSplit(n_splits=5)
    maes = {}

    for name, model in models.items():
        fold_mae = []
        for tr, te in tscv.split(X):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            fold_mae.append(mean_absolute_error(y[te], pred))
        maes[name] = float(np.mean(fold_mae))

    # weights: inverse MAE
    inv = {k: (1.0 / max(v, 1e-9)) for k, v in maes.items()}
    s = sum(inv.values())
    weights = {k: (inv[k] / s) for k in inv}

    # train final on full
    trained = {}
    for name, model in models.items():
        model.fit(X, y)
        trained[name] = model

    cv_metrics = {"cv_mae": maes, "weights": weights}
    return trained, weights, cv_metrics

def ensemble_predict(models: dict, weights: dict, X: np.ndarray) -> np.ndarray:
    pred = None
    for name, m in models.items():
        p = m.predict(X)
        w = weights.get(name, 0.0)
        pred = p * w if pred is None else pred + p * w
    return pred

def estimate_interval_sigma(y_true: np.ndarray, y_pred: np.ndarray):
    resid = y_true - y_pred
    if resid.size >= 80:
        resid = resid[-80:]
    return float(np.std(resid))

# =========================
# Strong prediction (5 days) + interval
# =========================
@st.cache_data(ttl=3600)
def predict_next_5_strong(stock_code: str, lookback_days: int):
    df = build_dataset(stock_code, lookback_days)
    if df.empty or len(df) < CFG.train_min_rows:
        return None, None, None, pd.DataFrame(), {"error": "資料不足或下載失敗"}

    feats = [
        "Ret1", "Ret5", "Vol10", "Vol20", "VolChg",
        "MA5", "MA10", "MA20", "MA60",
        "RSI", "MACD", "ADX",
        "BB_High", "BB_Low",
        "RelStrength1"
    ]

    X = df[feats].values
    y = df["Close"].values

    models, weights, cvm = cv_weighted_ensemble_train(X, y)
    y_pred = ensemble_predict(models, weights, X)

    df = df.copy()
    df["AI_Pred"] = y_pred

    sigma = estimate_interval_sigma(y, y_pred)
    last_close = float(df["Close"].iloc[-1])

    # 5 business days
    future_dates = pd.bdate_range(start=df.index[-1], periods=6)[1:]

    # minimal extrapolation: use last feature row unchanged
    x_last = df[feats].iloc[-1:].values
    point_preds = ensemble_predict(models, weights, np.repeat(x_last, repeats=5, axis=0)).tolist()

    # guard rails by MA20 +/- 3*ATR
    last_ma20 = float(df["MA20"].iloc[-1])
    last_atr = float(df["ATR"].iloc[-1])
    upper = last_ma20 + 3 * last_atr
    lower = last_ma20 - 3 * last_atr

    point_preds = [min(max(float(p), lower), upper) for p in point_preds]

    # ~80% interval using 1.28*sigma plus small ATR cushion
    hi = [p + 1.28 * sigma + 0.25 * last_atr for p in point_preds]
    lo = [p - 1.28 * sigma - 0.25 * last_atr for p in point_preds]

    forecast = {d.date(): float(p) for d, p in zip(future_dates, point_preds)}
    preds_dict = {f"T+{i+1}": float(p) for i, p in enumerate(point_preds)}

    extra = {
        "cv_mae": cvm["cv_mae"],
        "weights": cvm["weights"],
        "sigma": sigma,
        "future_dates": list(future_dates),
        "pred_hi": hi,
        "pred_lo": lo
    }
    return last_close, forecast, preds_dict, df, extra

# =========================
# Strong realistic backtest (Close->Close + MFE/MAE)
# - speed optimized: limit rows + retrain every N steps
# =========================
@st.cache_data(ttl=3600)
def realistic_backtest_strong(df: pd.DataFrame, direction: str):
    """
    direction:
      - "buy": 看多策略回測
      - "sell": 看空策略回測（以反向報酬計算）
    回測觸發：使用你原本的技術面訊號（買/賣）作為 entry
    報酬計算：Close-to-Close（T+fwd_days 的收盤）
    MFE/MAE：使用未來 fwd_days 內的最高/最低（用來看承受回撤與潛在）
    """
    if df is None or df.empty:
        return {}

    # limit backtest window for speed
    df_bt = df.tail(CFG.backtest_max_rows).dropna().copy()
    if len(df_bt) < CFG.train_min_rows + CFG.fwd_days + 10:
        return {}

    feats = [
        "Ret1", "Ret5", "Vol10", "Vol20", "VolChg",
        "MA5", "MA10", "MA20", "MA60",
        "RSI", "MACD", "ADX",
        "BB_High", "BB_Low",
        "RelStrength1"
    ]

    # Precompute signal points
    signal_idx = []
    for i in range(2, len(df_bt) - CFG.fwd_days):
        row_prior = df_bt.iloc[i-1]
        row_now = df_bt.iloc[i]
        if direction == "buy":
            ok, _ = generate_signal_row_buy(row_prior, row_now, CFG)
        else:
            ok, _ = generate_signal_row_sell(row_prior, row_now, CFG)
        if ok:
            signal_idx.append(i)

    if not signal_idx:
        return {"樣本數": 0, "勝率(%)": 0.0, "平均報酬(%)": 0.0, "平均MFE(%)": 0.0, "平均MAE(%)": 0.0}

    results = []
    models = None
    weights = None
    last_train_end = None

    for j, i in enumerate(signal_idx):
        train_end = i  # exclusive
        if train_end < CFG.train_min_rows:
            continue

        # retrain every N signals OR if no model yet
        if (models is None) or (last_train_end is None) or ((train_end - last_train_end) >= CFG.retrain_every):
            train_df = df_bt.iloc[:train_end].copy()
            Xtr = train_df[feats].values
            ytr = train_df["Close"].values
            if len(train_df) < CFG.train_min_rows:
                continue
            models, weights, _ = cv_weighted_ensemble_train(Xtr, ytr)
            last_train_end = train_end

        entry = float(df_bt["Close"].iloc[i])
        future = df_bt.iloc[i+1:i+1+CFG.fwd_days]
        if future.empty or len(future) < CFG.fwd_days:
            continue

        exit_close = float(future["Close"].iloc[-1])
        future_high = float(future["High"].max())
        future_low = float(future["Low"].min())

        # Close->Close return
        if direction == "buy":
            ret = (exit_close - entry) / entry
            mfe = (future_high - entry) / entry
            mae = (future_low - entry) / entry
        else:
            # short: profit when price drops
            ret = (entry - exit_close) / entry
            mfe = (entry - future_low) / entry      # best favorable move (price down)
            mae = (entry - future_high) / entry     # adverse move (price up) => usually negative

        results.append({"ret": ret, "mfe": mfe, "mae": mae})

    if not results:
        return {"樣本數": 0, "勝率(%)": 0.0, "平均報酬(%)": 0.0, "平均MFE(%)": 0.0, "平均MAE(%)": 0.0}

    r = pd.DataFrame(results)
    return {
        "樣本數": int(len(r)),
        "勝率(%)": round(float((r["ret"] > 0).mean() * 100), 1),
        "平均報酬(%)": round(float(r["ret"].mean() * 100), 2),
        "中位數報酬(%)": round(float(r["ret"].median() * 100), 2),
        "平均MFE(%)": round(float(r["mfe"].mean() * 100), 2),
        "平均MAE(%)": round(float(r["mae"].mean() * 100), 2),
        "5%最差報酬(%)": round(float(np.percentile(r["ret"], 5) * 100), 2),
        "95%最好報酬(%)": round(float(np.percentile(r["ret"], 95) * 100), 2),
    }

# =========================
# Plot
# =========================
def plot_stock_data(df: pd.DataFrame, extra=None):
    if not HAS_PLOTLY:
        return None

    df = df.copy()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.7, 0.3],
        subplot_titles=("股價走勢（含AI軌跡/預測）", "成交量")
    )

    fig.add_trace(
        go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K線"),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA60"], name="MA60"), row=1, col=1)

    if "AI_Pred" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["AI_Pred"], name="AI 歷史預測(Ensemble)", line=dict(dash="dot")), row=1, col=1)

    if extra and extra.get("future_dates") is not None:
        fd = extra["future_dates"]
        hi = extra.get("pred_hi")
        lo = extra.get("pred_lo")

        # point use mid of band if provided, otherwise just skip band
        if hi is not None and lo is not None:
            mid = [(h + l) / 2 for h, l in zip(hi, lo)]
            connect_x = [df.index[-1]] + list(fd)
            connect_y = [float(df["Close"].iloc[-1])] + list(mid)
            fig.add_trace(go.Scatter(x=connect_x, y=connect_y, name="AI 未來預測", line=dict(dash="dash", width=3)), row=1, col=1)

            fig.add_trace(
                go.Scatter(
                    x=list(fd) + list(fd)[::-1],
                    y=hi + lo[::-1],
                    fill="toself",
                    name="預測區間(約80%)",
                    opacity=0.2,
                    line=dict(width=0)
                ),
                row=1, col=1
            )

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, hovermode="x unified")
    return fig

def plot_error_chart(df: pd.DataFrame):
    if not HAS_PLOTLY or "AI_Pred" not in df.columns:
        return None
    d = df.tail(80).copy()
    d["ErrPct"] = ((d["AI_Pred"] - d["Close"]) / d["Close"]) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["ErrPct"], mode="lines+markers", name="誤差(%)"))
    fig.add_shape(type="line", x0=d.index[0], y0=0, x1=d.index[-1], y1=0, line=dict(dash="dash"))
    fig.update_layout(height=320, hovermode="x unified", title="AI 歷史誤差趨勢（近80日）", yaxis_title="(AI_Pred - Close)/Close %")
    return fig

# =========================
# Advice
# =========================
def get_trade_advice(last, forecast):
    if not forecast:
        return "無法判斷"
    avg_pred = float(np.mean(list(forecast.values())))
    change_percent = ((avg_pred - last) / last) * 100
    if change_percent > 2.0:
        return f"強烈看漲 (預期 +{change_percent:.1f}%)"
    elif change_percent > 0.5:
        return f"看漲 (預期 +{change_percent:.1f}%)"
    elif change_percent < -2.0:
        return f"強烈看跌 (預期 {change_percent:.1f}%)"
    elif change_percent < -0.5:
        return f"看跌 (預期 {change_percent:.1f}%)"
    return f"盤整 (預期 {change_percent:.1f}%)"

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="AI 智能股價分析 Pro", layout="wide", page_icon="📈")

st.markdown("""
<style>
.metric-card {background-color:#f0f2f6;border-radius:10px;padding:15px;margin:10px 0;}
.suggestion-box {padding:18px;border-radius:12px;text-align:center;margin-bottom:14px;}
</style>
""", unsafe_allow_html=True)

st.title("📈 AI 智能股價分析 Pro（更強重寫版｜真實回測 + Ensemble + 台股交易日曆）")
st.caption("提醒：本工具僅供研究與學習，不構成投資建議。")

if "recent_stocks" not in st.session_state:
    st.session_state.recent_stocks = []

with st.sidebar:
    st.header("⚙️ 設定參數")
    data_source = st.radio("資料來源", ["自動下載 (yfinance)", "手動貼上CSV資料"])

    if data_source == "自動下載 (yfinance)":
        if st.session_state.recent_stocks:
            selected_history = st.selectbox("📜 最近瀏覽紀錄", ["請選擇..."] + st.session_state.recent_stocks)
            if selected_history != "請選擇...":
                default_code = selected_history.split(" ")[0].replace(".TW", "")
            else:
                default_code = "2330"
        else:
            default_code = "2330"

        code = st.text_input("股票代號（台股可輸入 2330）", value=default_code)

        strategy_type = st.radio("技術面訊號方向", ["買進策略", "賣出策略"])
        mode = st.selectbox("訓練資料量（越多越穩、越慢）", ["短 (約 1 年)", "中 (約 2 年)", "長 (約 4 年)"])
        mode_map = {"短 (約 1 年)": 420, "中 (約 2 年)": 840, "長 (約 4 年)": 1680}
        lookback_days = mode_map[mode]

        show_interval = st.checkbox("顯示預測區間（建議開）", value=True)
        show_backtest = st.checkbox("顯示真實回測（較耗時）", value=True)

        st.divider()
        if code.strip().upper().endswith(".TW") or code.strip().isdigit():
            st.caption(f"台股交易日曆：{'已啟用(XTAI)' if HAS_TW_CAL else '未安裝套件，使用推測模式'}")
        else:
            st.caption("美股：不做盤中判斷（避免時區/盤中誤導）")
    else:
        show_interval = False
        show_backtest = False
        st.info("手動模式：僅技術面訊號與指標，不跑 AI 預測 / 回測（避免錯誤）")

run_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)

if run_btn:
    df_result = pd.DataFrame()
    last_price = None
    forecast = None
    preds = None
    extra = {}

    if data_source == "自動下載 (yfinance)":
        full_code = code.strip().upper()
        if full_code.isdigit():
            full_code += ".TW"

        stock_name = stock_name_dict.get(full_code, "未知名稱")

        with st.spinner(f"下載 + 訓練 Ensemble + 預測中：{stock_name} ({full_code}) ..."):
            last_price, forecast, preds, df_result, extra = predict_next_5_strong(full_code, lookback_days)

        if df_result is None or df_result.empty or last_price is None:
            st.error("無法取得資料或有效樣本不足。請檢查代號或調高訓練資料量。")
            st.stop()

        history_item = f"{full_code} {stock_name}"
        if history_item not in st.session_state.recent_stocks:
            st.session_state.recent_stocks.insert(0, history_item)
            if len(st.session_state.recent_stocks) > 10:
                st.session_state.recent_stocks.pop()

        st.subheader(f"{stock_name} ({full_code}) - 分析報告（資料日期：{df_result.index[-1].strftime('%Y-%m-%d')}）")
        status_text, is_open = market_status(full_code)

    else:
        manual_data = st.text_area("貼上 CSV（需含 Date, Open, High, Low, Close, Volume）", height=220)
        if not manual_data:
            st.warning("請先貼上 CSV。")
            st.stop()

        try:
            df_result = pd.read_csv(io.StringIO(manual_data))
            df_result["Date"] = pd.to_datetime(df_result["Date"])
            df_result.set_index("Date", inplace=True)
            df_result = add_technical_indicators(df_result, CFG)
            df_result = add_return_features(df_result)
            df_result["Market_Close"] = np.nan
            df_result["Mkt_Ret1"] = np.nan
            df_result["RelStrength1"] = 0.0
            df_result = df_result.dropna().copy()
            last_price = float(df_result["Close"].iloc[-1])
            status_text, is_open = ("手動資料", False)
            st.success("CSV 讀取成功（手動模式不跑 AI）。")
        except Exception as e:
            st.error(f"CSV 格式錯誤：{e}")
            st.stop()

    # ===== 技術面訊號區 =====
    strat_key = "buy" if strategy_type == "買進策略" else "sell"
    summary = evaluate_latest(df_result, CFG, strat_key)

    # AI trend
    ai_trend_pct = 0.0
    if forecast:
        avg_pred = float(np.mean(list(forecast.values())))
        ai_trend_pct = ((avg_pred - last_price) / last_price) * 100

    # signal light
    if summary["是否符合訊號"]:
        if summary["動作"].startswith("多方"):
            signal_color, signal_emoji, signal_text = "#d4edda", "🟢", "買進訊號 (BUY)"
        else:
            signal_color, signal_emoji, signal_text = "#f8d7da", "🔴", "放空訊號 (SELL)"
        ai_hint = ""
    else:
        signal_color, signal_emoji = "#fff3cd", "🟡"
        trend_direction = "偏多" if ai_trend_pct > 0 else "偏空"
        signal_text = f"觀望 (WAIT) - 趨勢{trend_direction}"
        ai_hint = f" | <b>AI 趨勢:</b> {trend_direction} (預期 {ai_trend_pct:+.1f}%)，但技術面尚未確認" if forecast else ""

    st.markdown(f"""
    <div style="background-color:{signal_color};padding:18px;border-radius:14px;text-align:center;border:2px solid #ccc;color:#333;">
        <div style="color:#666;">市場狀態：{status_text} | 資料日期：{summary.get('日期','-')}</div>
        <div style="font-size:34px;margin:8px 0;">{signal_emoji} {signal_text}</div>
        <div style="font-size:16px;"><b>模式:</b> {summary.get('動作','-')} | <b>收盤:</b> {summary.get('收盤','-')}{ai_hint}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📌 訊號與風控")
        st.metric("基準收盤價 (Last Close)", f"{last_price:.2f}")
        st.write(f"**技術面訊號**：{summary.get('理由','-')}")
        st.write(f"**建議停損**：{summary.get('建議停損','-')}")
        st.write(f"**ATR**：{summary.get('估計ATR','-')}")
        st.write(f"**建議股數（依資金風險）**：{summary.get('建議股數','-')}")

        if forecast:
            st.markdown("### 🤖 AI（Ensemble）")
            st.info(f"AI 建議：**{get_trade_advice(last_price, forecast)}**")

            st.markdown("#### 🧩 模型權重（自動用 CV MAE 決定）")
            w = extra.get("weights", {})
            c = extra.get("cv_mae", {})
            if w and c:
                st.write(f"- 權重：HGB {w.get('HGB',0):.2f}｜RF {w.get('RF',0):.2f}")
                st.write(f"- CV MAE：HGB {c.get('HGB',np.nan):.3f}｜RF {c.get('RF',np.nan):.3f}")
            st.write(f"- 殘差波動 sigma（用於區間估計）：{extra.get('sigma', np.nan):.3f}")

    with col2:
        st.markdown("### 📈 圖表")
        plot_df = df_result.tail(180).copy()

        if HAS_PLOTLY:
            fig = plot_stock_data(plot_df, extra if (show_interval and forecast) else None)
            st.plotly_chart(fig, use_container_width=True)
            err_fig = plot_error_chart(df_result)
            if err_fig:
                st.plotly_chart(err_fig, use_container_width=True)
        else:
            st.warning("⚠️ 未安裝 plotly，改用簡易線圖。")
            st.line_chart(plot_df[["Close", "MA20", "MA60"]].dropna())

        if forecast:
            st.markdown("### 🔮 未來 5 日預測")
            f_dates = list(forecast.keys())
            f_vals = list(forecast.values())

            f_df = pd.DataFrame({
                "日期": [str(d) for d in f_dates],
                "預測價": [f"{v:.2f}" for v in f_vals],
                "漲跌幅": [f"{(v - last_price) / last_price * 100:+.2f}%" for v in f_vals],
            })

            if show_interval:
                hi = extra.get("pred_hi", None)
                lo = extra.get("pred_lo", None)
                if hi is not None and lo is not None:
                    f_df["區間下界(約80%)"] = [f"{v:.2f}" for v in lo]
                    f_df["區間上界(約80%)"] = [f"{v:.2f}" for v in hi]

            st.table(f_df)

    # ===== 真實回測（更強）=====
    if show_backtest and data_source == "自動下載 (yfinance)":
        st.markdown("---")
        st.subheader("📊 真實回測（Close→Close + MFE/MAE）")
        st.caption("說明：只在『技術面訊號成立』的日子進場，持有 fwd_days 天後以收盤出場；同時計算 MFE/MAE 觀察潛在與回撤。")

        with st.spinner("回測計算中（較耗時，但已做加速：只取近一段資料 + 每 N 天重訓一次）..."):
            bt = realistic_backtest_strong(df_result, "buy" if strat_key == "buy" else "sell")

        if not bt:
            st.info("回測資料不足或訊號太少，無法計算。你可改用『中/長』資料量或換標的。")
        else:
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("樣本數", bt.get("樣本數", 0))
            m2.metric("勝率", f"{bt.get('勝率(%)', 0):.1f}%")
            m3.metric("平均報酬", f"{bt.get('平均報酬(%)', 0):.2f}%")
            m4.metric("中位數報酬", f"{bt.get('中位數報酬(%)', 0):.2f}%")
            m5.metric("平均MFE", f"{bt.get('平均MFE(%)', 0):.2f}%")
            m6.metric("平均MAE", f"{bt.get('平均MAE(%)', 0):.2f}%")
            st.write(f"5%最差報酬：**{bt.get('5%最差報酬(%)', 0):.2f}%**｜95%最好報酬：**{bt.get('95%最好報酬(%)', 0):.2f}%**")

    st.markdown("---")
    st.caption("免責聲明：本工具僅供技術研究與學習，不構成投資建議。")
