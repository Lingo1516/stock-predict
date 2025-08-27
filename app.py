import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import ta

# 內建常用股票清單
def get_taiwan_stocks():
    """獲取台灣股市股票清單（使用內建資料確保穩定性）"""
    # 擴展的台股清單
    stocks = {
        # 熱門大型股
        '2330': '台積電', '台積電': '2330',
        '2317': '鴻海', '鴻海': '2317',
        '2454': '聯發科', '聯發科': '2454',
        '2881': '富邦金', '富邦金': '2881',
        '2412': '中華電', '中華電': '2412',
        '2303': '聯電', '聯電': '2303',
        '2002': '中鋼', '中鋼': '2002',
        '1301': '台塑', '台塑': '1301',
        '2882': '國泰金', '國泰金': '2882',
        '2886': '兆豐金', '兆豐金': '2886',
        '2891': '中信金', '中信金': '2891',
        '2884': '玉山金', '玉山金': '2884',
        '2885': '元大金', '元大金': '2885',
        '2892': '第一金', '第一金': '2892',
        '2883': '開發金', '開發金': '2883',
        
        # 科技股
        '2379': '瑞昱', '瑞昱': '2379',
        '2408': '南亞科', '南亞科': '2408',
        '3711': '日月光投控', '日月光投控': '3711',
        '2357': '華碩', '華碩': '2357',
        '2382': '廣達', '廣達': '2382',
        '2395': '研華', '研華': '2395',
        '6505': '台塑化', '台塑化': '6505',
        '2409': '友達', '友達': '2409',
        '2474': '可成', '可成': '2474',
        '3008': '大立光', '大立光': '3008',
        
        # 傳統產業
        '1303': '南亞', '南亞': '1303',
        '1326': '台化', '台化': '1326',
        '2207': '和泰車', '和泰車': '2207',
        '2301': '光寶科', '光寶科': '2301',
        '2308': '台達電', '台達電': '2308',
        '2105': '正新', '正新': '2105',
        '2912': '統一超', '統一超': '2912',
        '2801': '彰銀', '彰銀': '2801',
        '2880': '華南金', '華南金': '2880',
        '2890': '永豐金', '永豐金': '2890',
        
        # ETF
        '0050': '元大台灣50', '元大台灣50': '0050',
        '0056': '元大高股息', '元大高股息': '0056',
        '006208': '富邦台50', '富邦台50': '006208',
        '00878': '國泰永續高股息', '國泰永續高股息': '00878',
        '00692': '富邦公司治理', '富邦公司治理': '00692',
        
        # 簡化名稱映射
        '積電': '2330',
        '聯發': '2454',
        '富邦': '2881',
        '國泰': '2882',
        '中信': '2891',
        '玉山': '2884',
        '元大': '2885',
        '台塑': '1301',
        '南亞': '1303',
        '台化': '1326',
        '中鋼': '2002',
        '鴻海': '2317',
        '聯電': '2303',
        '中華電': '2412',
        '大立光': '3008',
        '台達電': '2308',
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

    # 情況4: 部分匹配中文名稱
    for name, code in stock_dict.items():
        if not name.isdigit() and len(name) > 1:
            if user_input in name or name in user_input:
                return f"{code}.TW" if code.isdigit() else f"{stock_dict.get(name, '0000')}.TW", name
    
    # 情況5: 美股或其他市場（直接使用）
    if user_input.isalpha() and len(user_input) <= 5:
        return user_input.upper(), user_input.upper()

    return None, None

@st.cache_data
def predict_next_5(stock_input, stock_list, days, decay_factor):
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

        # 下載資料
        df = yf.download(stock_code, start=start, end=end + pd.Timedelta(days=1), interval="1d", auto_adjust=True)
        
        if df is None or df.empty:
            st.error(f"無法下載 {stock_code} 的資料")
            return None, None, None, None

        # 計算技術指標等
        close = df['Close'].squeeze() if 'Close' in df.columns else df.iloc[:, 3].squeeze()
        df['MA5'] = close.rolling(5, min_periods=1).mean()
        df['MA10'] = close.rolling(10, min_periods=1).mean()
        df['MA20'] = close.rolling(20, min_periods=1).mean()

        # 剩下的技術分析和模型训练代码...
        
        return df, stock_code, stock_name

    except Exception as e:
        st.error(f"預測過程發生錯誤: {str(e)}")
        return None, None, None, None

# Streamlit 介面
st.title("📈 股價預測系統")
st.markdown("---")

# 輸入區域
col1, col2 = st.columns([3, 2])
with col1:
    stock_input = st.text_input("🔍 輸入股票", "台積電", help="支援格式：\n• 中文名稱：台積電、鴻海\n• 純數字：2330、2317\n• 完整代號：2330.TW\n• 美股：AAPL、TSLA")

with col2:
    mode = st.selectbox("📊 預測模式", ["中期模式", "短期模式", "長期模式"])

if st.button("🔮 開始預測", type="primary"):
    with st.spinner("📥 正在下載資料並進行預測..."):
        last, forecast, preds, stock_name = predict_next_5(stock_input.strip(), None, 100, 0.005)
    
    if last is None:
        st.error("❌ 預測失敗，請檢查股票輸入或網路連線")
    else:
        st.success("✅ 預測完成！")
        st.write(f"預測股價: {forecast}")

