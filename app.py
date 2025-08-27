# 在預測未來價格的部分
current_date = end
future_dates = []
for i in range(5):
    current_date = current_date + pd.offsets.BDay(1)
    future_dates.append(current_date.date())

current_features = last_features.copy()
predicted_prices = [last_close]

max_deviation_pct = 0.10  # 最大偏離限制 ±10%

for i, date in enumerate(future_dates):
    day_predictions = []
    for model_name, model in models:
        pred = model.predict(current_features)[0]
        variation = np.random.normal(0, pred * 0.002)  # 隨機變異降至0.2%
        day_predictions.append(pred + variation)

    # 確保 weights_ensemble 的長度與 day_predictions 一致
    weights_ensemble = [0.5] * len(day_predictions)  # 使用相同的權重
    ensemble_pred = np.average(day_predictions, weights=weights_ensemble)

    historical_volatility = np.std(y[-30:]) / np.mean(y[-30:])
    volatility_adjustment = np.random.normal(0, ensemble_pred * historical_volatility * 0.05)  # 調整到0.05

    final_pred = ensemble_pred + volatility_adjustment

    # 限制預測價格在合理範圍內
    upper_limit = last_close * (1 + max_deviation_pct)
    lower_limit = last_close * (1 - max_deviation_pct)
    final_pred = min(max(final_pred, lower_limit), upper_limit)

    predictions[date] = float(final_pred)
    predicted_prices.append(final_pred)

    if i < 4:
        new_features = current_features[0].copy()
        prev_close_idx = feats.index('Prev_Close')
        new_features[prev_close_idx] = (final_pred - X_mean[prev_close_idx]) / X_std[prev_close_idx]

        for j in range(1, min(4, len(predicted_prices))):
            if f'Prev_Close_Lag{j}' in feats:
                lag_idx = feats.index(f'Prev_Close_Lag{j}')
                if len(predicted_prices) > j:
                    lag_price = predicted_prices[-(j + 1)]
                    new_features[lag_idx] = (lag_price - X_mean[lag_idx]) / X_std[lag_idx]

        if 'MA5' in feats and len(predicted_prices) >= 2:
            ma5_idx = feats.index('MA5')
            recent_ma5 = np.mean(predicted_prices[-min(5, len(predicted_prices)):])
            new_features[ma5_idx] = (recent_ma5 - X_mean[ma5_idx]) / X_std[ma5_idx]

        if 'MA10' in feats and len(predicted_prices) >= 2:
            ma10_idx = feats.index('MA10')
            recent_ma10 = np.mean(predicted_prices[-min(10, len(predicted_prices)):])
            new_features[ma10_idx] = (recent_ma10 - X_mean[ma10_idx]) / X_std[ma10_idx]

        if 'Volatility' in feats and len(predicted_prices) >= 3:
            volatility_idx = feats.index('Volatility')
            recent_volatility = np.std(predicted_prices[-min(10, len(predicted_prices)):])
            new_features[volatility_idx] = (recent_volatility - X_mean[volatility_idx]) / X_std[volatility_idx]

        current_features = new_features.reshape(1, -1)
