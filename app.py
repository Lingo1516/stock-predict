import pandas as pd
import numpy as np
import random
from datetime import datetime
import streamlit as st

@st.cache_data
def predict_next_5(stock, days, decay_factor, models, last_features, last_close, y, feats, X_mean, X_std):
    try:
        # 定義預測期間的起止時間
        end = pd.Timestamp(datetime.today().date())  # 終止日期為今天
        start = end - pd.Timedelta(days=days)  # 起始日期基於days參數
        max_retries = 3
        df, twii, sp = None, None, None

        # 下載數據的邏輯 (此處略)
        # 在此下載股價及相關指數資料，並進行錯誤處理

        # 預測未來5個交易日的股價
        current_date = end  # 確保在這裡使用結束日期
        future_dates = []
        for i in range(5):
            current_date = current_date + pd.offsets.BDay(1)  # 每次移動1個交易日
            future_dates.append(current_date.date())  # 將日期加入預測日期列表

        # 初始化預測結果
        predictions = {}  # 用於存儲各日期的預測結果
        predicted_prices = [last_close]  # 預測價格從最後的收盤價開始
        max_deviation_pct = 0.10  # 設定最大偏差百分比

        # 開始預測
        for i, date in enumerate(future_dates):
            day_predictions = []
            for model in models:
                pred = model.predict(last_features)[0]  # 使用模型進行預測
                variation = np.random.normal(0, pred * 0.002)  # 隨機變異降至0.2%
                day_predictions.append(pred + variation)

            # 計算加權平均預測結果
            weights_ensemble = [0.5] * len(day_predictions)  # 所有模型權重相等
            ensemble_pred = np.average(day_predictions, weights=weights_ensemble)

            # 計算歷史波動率並調整預測
            historical_volatility = np.std(y[-30:]) / np.mean(y[-30:])
            volatility_adjustment = np.random.normal(0, ensemble_pred * historical_volatility * 0.05)

            # 最終預測
            final_pred = ensemble_pred + volatility_adjustment

            # 限制預測價格在合理範圍內
            upper_limit = last_close * (1 + max_deviation_pct)
            lower_limit = last_close * (1 - max_deviation_pct)
            final_pred = min(max(final_pred, lower_limit), upper_limit)

            # 儲存預測結果
            predictions[date] = float(final_pred)
            predicted_prices.append(final_pred)

            # 更新特徵資料以便預測下一個價格
            if i < 4:
                new_features = last_features.copy()

                prev_close_idx = feats.index('Prev_Close')
                new_features[prev_close_idx] = (final_pred - X_mean[prev_close_idx]) / X_std[prev_close_idx]

                # 更新滯後價格特徵
                for j in range(1, min(4, len(predicted_prices))):
                    if f'Prev_Close_Lag{j}' in feats:
                        lag_idx = feats.index(f'Prev_Close_Lag{j}')
                        if len(predicted_prices) > j:
                            lag_price = predicted_prices[-(j + 1)]
                            new_features[lag_idx] = (lag_price - X_mean[lag_idx]) / X_std[lag_idx]

                # 計算移動平均
                if 'MA5' in feats and len(predicted_prices) >= 2:
                    ma5_idx = feats.index('MA5')
                    recent_ma5 = np.mean(predicted_prices[-min(5, len(predicted_prices)):])
                    new_features[ma5_idx] = (recent_ma5 - X_mean[ma5_idx]) / X_std[ma5_idx]

                if 'MA10' in feats and len(predicted_prices) >= 2:
                    ma10_idx = feats.index('MA10')
                    recent_ma10 = np.mean(predicted_prices[-min(10, len(predicted_prices)):])
                    new_features[ma10_idx] = (recent_ma10 - X_mean[ma10_idx]) / X_std[ma10_idx]

                # 計算波動性
                if 'Volatility' in feats and len(predicted_prices) >= 3:
                    volatility_idx = feats.index('Volatility')
                    recent_volatility = np.std(predicted_prices[-min(10, len(predicted_prices)):])
                    new_features[volatility_idx] = (recent_volatility - X_mean[volatility_idx]) / X_std[volatility_idx]

                last_features = new_features.reshape(1, -1)  # 更新特徵資料以進行下一步預測

        # 最終返回預測結果
        preds = {f'T+{i + 1}': pred for i, pred in enumerate(predictions.values())}

        return preds, predictions, predicted_prices

    except Exception as e:
        st.error(f"預測過程發生錯誤: {str(e)}")
        return None, None, None
