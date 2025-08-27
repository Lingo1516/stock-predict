import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import ta
from datetime import datetime
import time

@st.cache_data
def predict_next_5(stock, days, decay_factor):
    # 省略函數內部實作，與原本相同
    # （這裡請貼入你原本提供的 predict_next_5 函數完整實現）
    pass

def get_trade_advice(last, preds):
    if not preds:
        return "無法判斷"
    price_changes = [preds[f'T+{d}'] - last for d in range(1, 6)]
    avg_change = np.mean(price_changes)
    change_percent = (avg_change / last) * 100
    if change_percent > 2:
        return f"強烈買入 (預期上漲 {change_percent:.1f}%)"
    elif change_percent > 0.5:
        return f"買入 (預期上漲 {change_percent:.1f}%)"
    elif change_percent < -2:
        return f"強烈賣出 (預期下跌 {abs(change_percent):.1f}%)"
    elif change_percent < -0.5:
        return f"賣出 (預期下跌 {abs(change_percent):.1f}%)"
    else:
        return f"持有 (預期變動 {change_percent:.1f}%)"

# Streamlit 介面設定
st.title("📈 5 日股價預測系統")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    code = st.text_input("請輸入股票代號（僅輸入數字部分即可）", "2330")
with col2:
    mode = st.selectbox("預測模式", ["中期模式", "短期模式", "長期模式"])

mode_info = {
    "短期模式": ("使用 100 天歷史資料，高敏感度", 100, 0.008),
    "中期模式": ("使用 200 天歷史資料，平衡敏感度", 200, 0.005),
    "長期模式": ("使用 400 天歷史資料，低敏感度", 400, 0.002)
}
st.info(f"**{mode}**: {mode_info[mode][0]}")
days, decay_factor = mode_info[mode][1], mode_info[mode][2]

if st.button("🔮 開始預測", type="primary"):
    full_code = code.strip()
    if not full_code.upper().endswith(".TW"):
        full_code = f"{full_code}.TW"
    with st.spinner("正在下載資料並進行預測..."):
        last, forecast, preds = predict_next_5(full_code, days, decay_factor)

    if last is None:
        st.error("❌ 預測失敗，請檢查股票代號或網路連線")
    else:
        st.success("✅ 預測完成！")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("當前股價", f"${last:.2f}")
            advice = get_trade_advice(last, preds)

            if "買入" in advice:
                st.success(f"📈 **交易建議**: {advice}")
            elif "賣出" in advice:
                st.error(f"📉 **交易建議**: {advice}")
            else:
                st.warning(f"📊 **交易建議**: {advice}")

        with col2:
            st.subheader("📅 未來 5 日預測")
            for date, price in forecast.items():
                change = price - last
                change_pct = (change / last) * 100
                if change > 0:
                    st.write(f"**{date}**: ${price:.2f} (+{change:.2f}, +{change_pct:.1f}%)")
                else:
                    st.write(f"**{date}**: ${price:.2f} ({change:.2f}, {change_pct:.1f}%)")

        st.subheader("📈 預測趨勢")
        chart_data = pd.DataFrame({
            '日期': ['今日'] + list(forecast.keys()),
            '股價': [last] + list(forecast.values())
        })
        st.line_chart(chart_data.set_index('日期'))

st.markdown("---")
st.caption("⚠️ 此預測僅供參考，投資有風險，請謹慎決策")
