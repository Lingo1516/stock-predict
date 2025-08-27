import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import ta
from datetime import datetime, timedelta
import time

@st.cache_data
def predict_next_5(stock, days, decay_factor):
    try:
        end = pd.Timestamp(datetime.today().date())
        start = end - pd.Timedelta(days=days)

        # 下載資料並添加錯誤處理
        max_retries = 3
        df, twii, sp = None, None, None
        
        for attempt in range(max_retries):
            try:
                df = yf.download(stock, start=start, end=end + pd.Timedelta(days=1), 
                               interval="1d", auto_adjust=True, progress=False)
                twii = yf.download("^TWII", start=start, end=end + pd.Timedelta(days=1), 
                                 interval="1d", auto_adjust=True, progress=False)
                sp = yf.download("^GSPC", start=start, end=end + pd.Timedelta(days=1), 
                               interval="1d", auto_adjust=True, progress=False)
                
                if not (df.empty or twii.empty or sp.empty):
                    break
                    
            except Exception as e:
                st.warning(f"嘗試 {attempt + 1}/{max_retries} 下載失敗: {e}")
                time.sleep(2)
                
            if attempt == max_retries - 1:
                st.error(f"無法下載資料：{stock}")
                return None, None, None

        # 檢查資料是否充足
        if df is None or len(df) < 50:
            st.error(f"資料不足，僅有 {len(df) if df is not None else 0} 行數據")
            return None, None, None

        # 處理多重索引
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = [col[0] for col in twii.columns]
        if isinstance(sp.columns, pd.MultiIndex):
            sp.columns = [col[0] for col in sp.columns]

        # 確保收盤價是一維序列
        close = df['Close'].squeeze() if 'Close' in df.columns else df.iloc[:, 3].squeeze()
        
        # 填充外部指數資料
        df['TWII_Close'] = twii['Close'].reindex(df.index, method='ffill').fillna(method='bfill')
        df['SP500_Close'] = sp['Close'].reindex(df.index, method='ffill').fillna(method='bfill')

        # 計算技術指標
        df['MA10'] = close.rolling(10, min_periods=1).mean()
        df['MA20'] = close.rolling(20, min_periods=1).mean()
        df['MA5'] = close.rolling(5, min_periods=1).mean()
        
        # 計算 RSI
        try:
            df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        except:
            # 簡單的 RSI 計算作為後備
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

        # 計算 MACD
        try:
            macd = ta.trend.MACD(close)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
        except:
            # 簡單的 MACD 計算作為後備
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        # 添加滯後特徵
        df['Prev_Close'] = close.shift(1)
        for i in range(1, 4):  # 減少滯後特徵數量
            df[f'Prev_Close_Lag{i}'] = close.shift(i)

        # 添加成交量特徵
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(10, min_periods=1).mean()
        else:
            df['Volume_MA'] = 0

        # 添加波動率指標
        df['Volatility'] = close.rolling(10, min_periods=1).std()
        
        # 選擇特徵
        feats = ['Prev_Close', 'MA5', 'MA10', 'MA20', 'Volume_MA', 'RSI', 'MACD', 
                'MACD_Signal', 'TWII_Close', 'SP500_Close', 'Volatility'] + \
                [f'Prev_Close_Lag{i}' for i in range(1, 4)]
        
        # 檢查缺失的特徵
        missing_feats = [f for f in feats if f not in df.columns]
        if missing_feats:
            st.error(f"缺少特徵: {missing_feats}")
            return None, None, None

        # 移除有缺失值的行
        df_clean = df[feats + ['Close']].dropna()
        
        if len(df_clean) < 30:
            st.error(f"清理後資料不足，僅有 {len(df_clean)} 行數據")
            return None, None, None

        # 準備訓練數據
        X = df_clean[feats].values
        y = df_clean['Close'].values
        
        # 標準化特徵
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # 避免除以零
        X_normalized = (X - X_mean) / X_std

        # 計算時間權重
        weights = np.exp(-decay_factor * np.arange(len(X))[::-1])
        weights = weights / np.sum(weights)

        # 分割訓練和驗證集
        split_idx = int(len(X_normalized) * 0.8)
        X_train, X_val = X_normalized[:split_idx], X_normalized[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        train_weights = weights[:split_idx]

        # 訓練多個模型來增加預測多樣性
        models = []
        
        # 主要隨機森林模型
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train, sample_weight=train_weights)
        models.append(('RF', rf_model))
        
        # 添加更多變化的模型
        rf_model2 = RandomForestRegressor(
            n_estimators=80,
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=123,
            n_jobs=-1
        )
        rf_model2.fit(X_train, y_train, sample_weight=train_weights)
        models.append(('RF2', rf_model2))
        
        rf_model3 = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_split=7,
            min_samples_leaf=3,
            random_state=456,
            n_jobs=-1
        )
        rf_model3.fit(X_train, y_train, sample_weight=train_weights)
        models.append(('RF3', rf_model3))

        # 預測未來 5 天 - 使用集成模型
        last_features = X_normalized[-1:].copy()
        last_close = float(y[-1])
        predictions = {}
        
        # 創建未來日期
        future_dates = []
        current_date = end
        for i in range(5):
            current_date = current_date + pd.offsets.BDay(1)
            future_dates.append(current_date.date())

        # 逐步預測 - 每次預測後更新特徵
        current_features = last_features.copy()
        predicted_prices = [last_close]  # 包含最後一天的實際價格
        
        for i, date in enumerate(future_dates):
            # 使用多個模型進行預測並取平均
            day_predictions = []
            for model_name, model in models:
                pred = model.predict(current_features)[0]
                # 添加小幅隨機變化來增加多樣性
                variation = np.random.normal(0, pred * 0.005)  # 0.5% 的隨機變化
                day_predictions.append(pred + variation)
            
            # 取加權平均（給主模型更高權重）
            weights = [0.5, 0.3, 0.2]  # 主模型50%權重
            ensemble_pred = np.average(day_predictions, weights=weights)
            
            # 添加基於歷史波動率的隨機變化
            historical_volatility = np.std(y[-30:]) / np.mean(y[-30:])  # 最近30天的波動率
            volatility_adjustment = np.random.normal(0, ensemble_pred * historical_volatility * 0.3)
            final_pred = ensemble_pred + volatility_adjustment
            
            predictions[date] = float(final_pred)
            predicted_prices.append(final_pred)
            
            # 為下一次預測更新特徵
            if i < 4:  # 不是最後一次預測
                new_features = current_features[0].copy()
                
                # 更新前一天收盤價相關特徵
                prev_close_idx = feats.index('Prev_Close')
                new_features[prev_close_idx] = (final_pred - X_mean[prev_close_idx]) / X_std[prev_close_idx]
                
                # 更新滯後特徵
                for j in range(1, min(4, len(predicted_prices))):
                    if f'Prev_Close_Lag{j}' in feats:
                        lag_idx = feats.index(f'Prev_Close_Lag{j}')
                        if len(predicted_prices) > j:
                            lag_price = predicted_prices[-(j+1)]
                            new_features[lag_idx] = (lag_price - X_mean[lag_idx]) / X_std[lag_idx]
                
                # 更新移動平均
                if 'MA5' in feats and len(predicted_prices) >= 2:
                    ma5_idx = feats.index('MA5')
                    recent_ma5 = np.mean(predicted_prices[-min(5, len(predicted_prices)):])
                    new_features[ma5_idx] = (recent_ma5 - X_mean[ma5_idx]) / X_std[ma5_idx]
                
                if 'MA10' in feats and len(predicted_prices) >= 2:
                    ma10_idx = feats.index('MA10')
                    recent_ma10 = np.mean(predicted_prices[-min(10, len(predicted_prices)):])
                    new_features[ma10_idx] = (recent_ma10 - X_mean[ma10_idx]) / X_std[ma10_idx]
                
                # 更新波動率特徵
                if 'Volatility' in feats and len(predicted_prices) >= 3:
                    volatility_idx = feats.index('Volatility')
                    recent_volatility = np.std(predicted_prices[-min(10, len(predicted_prices)):])
                    new_features[volatility_idx] = (recent_volatility - X_mean[volatility_idx]) / X_std[volatility_idx]
                
                current_features = new_features.reshape(1, -1)

        # 計算預測字典
        preds = {f'T+{i+1}': pred for i, pred in enumerate(predictions.values())}

        # 驗證模型（使用主模型）
        if len(X_val) > 0:
            y_pred_val = models[0][1].predict(X_val)  # 使用第一個模型進行驗證
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            st.info(f"模型驗證 - RMSE: {rmse:.2f} (約 {rmse/last_close*100:.1f}%)")
            
            # 顯示特徵重要性
            feature_importance = models[0][1].feature_importances_
            top_features = sorted(zip(feats, feature_importance), key=lambda x: x[1], reverse=True)[:5]
            st.info(f"重要特徵: {', '.join([f'{feat}({imp:.3f})' for feat, imp in top_features])}")
        
        return last_close, predictions, preds

    except Exception as e:
        st.error(f"預測過程發生錯誤: {str(e)}")
        return None, None, None
