import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from datetime import datetime, timedelta
import ta

# 內建常用股票清單
def get_taiwan_stocks():
    """獲取台灣股市股票清單（使用內建資料確保穩定性）"""
    # 擴展的台股清單
    stocks = {
        '2330': '台積電', '台積電': '2330',
        '2317': '鴻海', '鴻海': '2317',
        '2454': '聯發科', '聯發科': '2454',
        # 其他股票...
    }
    return stocks

def parse_stock_input(user_input, stock_dict):
    """
    解析用戶輸入，支援多種格式
    """
    user_input = user_input.strip()

    # 情況1: 直接輸入中文名稱（完整匹配）
    if user_input in stock_dict:
        code = stock_dict[user_input]
        return f"{code}.TW", user_input

    # 情況2: 純數字代號
    if user_input.isdigit():
        code = user_input
        name = stock_dict.get(code, f"股票{code}")
        return f"{code}.TW", name

    # 情況3: 已經包含 .TW/.TWO 的代號
    if '.TW' in user_input.upper() or '.TWO' in user_input.upper():
        parts = user_input.upper().split('.')
        code = parts[0]
        if code.isdigit():
            name = stock_dict.get(code, f"股票{code}")
            return user_input.upper(), name
        return user_input.upper(), user_input.upper()

    return None, None

@st.cache_data
def predict_next_5(stock_input, days, decay_factor):
    """股價預測主函數"""
    try:
        # 獲取股票清單
        stock_dict = get_taiwan_stocks()
        
        # 解析用戶輸入
        parsed_result = parse_stock_input(stock_input, stock_dict)
        if parsed_result[0] is None:
            st.error("無法解析輸入的股票")
            return None, None, None, None
            
        stock_code, stock_name = parsed_result
        
        # 設定時間範圍
        end = pd.Timestamp(datetime.today().date())
        start = end - pd.Timedelta(days=days)

        # 下載資料並添加錯誤處理
        max_retries = 3
        df, twii, sp = None, None, None
        
        for attempt in range(max_retries):
            try:
                st.write(f"嘗試下載 {stock_code} 的資料... (第 {attempt + 1} 次)")
                df = yf.download(stock_code, start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True)
                
                # 檢查是否有資料
                if df is not None and not df.empty:
                    st.write(f"✅ 成功下載 {stock_code} 資料，共 {len(df)} 筆")
                    break
                else:
                    st.write(f"❌ {stock_code} 無資料")
                    
            except Exception as e:
                st.warning(f"下載 {stock_code} 失敗: {str(e)}")
                time.sleep(1)
                
        if df is None or df.empty:
            st.error(f"無法下載 {stock_code} 的資料，請檢查股票代號")
            return None, None, None, None

        # 計算技術指標等
        close = df['Close'].squeeze() if 'Close' in df.columns else df.iloc[:, 3].squeeze()
        df['MA5'] = close.rolling(5, min_periods=1).mean()
        df['MA10'] = close.rolling(10, min_periods=1).mean()
        df['MA20'] = close.rolling(20, min_periods=1).mean()

        # 剩下的技術分析和模型訓練代碼...

        return df, stock_code, stock_name

    except Exception as e:
        st.error(f"預測過程發生錯誤: {str(e)}")
        return None, None, None, None

# Streamlit 介面
st.title("📈 股價預測系統")
st.markdown("---")

# 輸入區域
col1, col2 = st.columns([2, 1])
with col1:
    stock_input = st.text_input("🔍 輸入股票代號", "2330", help="例如：2330 (台積電)、AAPL (蘋果)")

with col2:
    mode = st.selectbox("預測模式", ["中期模式", "短期模式", "長期模式"])

# 模式說明
mode_info = {
    "短期模式": ("使用 100 天歷史資料，高敏感度", 100, 0.008),
    "中期模式": ("使用 200 天歷史資料，平衡敏感度", 200, 0.005),
    "長期模式": ("使用 400 天歷史資料，低敏感度", 400, 0.002)
}

st.info(f"**{mode}**: {mode_info[mode][0]}")
days, decay_factor = mode_info[mode][1], mode_info[mode][2]

if st.button("🔮 開始預測", type="primary"):
    with st.spinner("正在下載資料並進行預測..."):
        last, forecast, preds, stock_name = predict_next_5(stock_input.strip(), days, decay_factor)
    
    if last is None:
        st.error("❌ 預測失敗，請檢查股票代號或網路連線")
    else:
        # 顯示結果
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
        
        # 繪製趨勢圖
        st.subheader("📈 預測趨勢")
        chart_data = pd.DataFrame({
            '日期': ['今日'] + list(forecast.keys()),
            '股價': [last] + list(forecast.values())
        })
        st.line_chart(chart_data.set_index('日期'))

st.markdown("---")
st.caption("⚠️ 此預測僅供參考，投資有風險，請謹慎決策")
