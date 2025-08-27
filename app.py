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

# 从台湾证券交易所获取所有上市、上柜的公司中文名称和代号
def fetch_twse_stock_codes():
    url = 'https://www.twse.com.tw/zh/listed/listed_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 解析 HTML 获取股票代码和公司名称
    stock_data = []
    rows = soup.find_all('tr', {'class': 'tableRow'})
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 2:
            stock_name = cols[1].text.strip()
            stock_code = cols[0].text.strip()
            stock_data.append([stock_name, stock_code])

    # 将数据存储为 pandas DataFrame
    df = pd.DataFrame(stock_data, columns=['Stock Name', 'Stock Code'])
    return df

# 将股票信息存储到内存（例如城市环境）以进行查询
stock_list = fetch_twse_stock_codes()

# 通过股票名称获取对应的股票代号
def get_stock_code(stock_name):
    stock_name = stock_name.strip()
    # 查找股票代号
    code_row = stock_list[stock_list['Stock Name'] == stock_name]
    if not code_row.empty:
        return f"{code_row.iloc[0]['Stock Code']}.TW"
    else:
        return None

# 获取股票名称
def get_stock_name(stock_code):
    stock_code = stock_code.strip().upper()
    if not stock_code.endswith('.TW'):
        stock_code += '.TW'  # 如果未提供 .TW，则自动补全
    try:
        stock = yf.Ticker(stock_code)
        info = stock.info
        return info.get('longName', '未知股票名稱')
    except Exception as e:
        st.error(f"无法获取股票名称：{e}")
        return None

# 下载股票数据
def download_stock_data(stock_name, start_date, end_date):
    stock_code = get_stock_code(stock_name)
    if stock_code:
        df = yf.download(stock_code, start=start_date, end=end_date)
        return df
    else:
        return None

def predict_next_5(stock, stock_list, days, decay_factor):
    try:
        end = pd.Timestamp(datetime.today().date())
        start = end - pd.Timedelta(days=days)

        # 根据股票名称取得代号
        stock_code = get_stock_code(stock.strip())

        # 如果没有获取到股票代号，返回错误信息
        if stock_code is None:
            st.error("无法获取股票代号，请检查股票名称或代号。")
            return None, None, None

        # 下载数据并添加错误处理
        max_retries = 3
        df, twii, sp = None, None, None

        for attempt in range(max_retries):
            try:
                df = yf.download(stock_code, start=start, end=end + pd.Timedelta(days=1),
                                 interval="1d", auto_adjust=True, progress=False)
                twii = yf.download("^TWII", start=start, end=end + pd.Timedelta(days=1),
                                   interval="1d", auto_adjust=True, progress=False)
                sp = yf.download("^GSPC", start=start, end=end + pd.Timedelta(days=1),
                                 interval="1d", auto_adjust=True, progress=False)

                if not (df.empty or twii.empty or sp.empty):
                    break

            except Exception as e:
                st.warning(f"尝试 {attempt + 1}/{max_retries} 下载失败: {e}")
                time.sleep(2)

            if attempt == max_retries - 1:
                st.error(f"无法下载数据：{stock_code}")
                return None, None, None

        # 检查数据是否充足
        if df is None or len(df) < 50:
            st.error(f"数据不足，仅有 {len(df) if df is not None else 0} 行数据")
            return None, None, None

        # 处理多重索引
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = [col[0] for col in twii.columns]
        if isinstance(sp.columns, pd.MultiIndex):
            sp.columns = [col[0] for col in sp.columns]

        # 确保收盘价是一维序列
        close = df['Close'].squeeze() if 'Close' in df.columns else df.iloc[:, 3].squeeze()

        # 填充外部指数资料
        df['TWII_Close'] = twii['Close'].reindex(df.index, method='ffill').fillna(method='bfill')
        df['SP500_Close'] = sp['Close'].reindex(df.index, method='ffill').fillna(method='bfill')

        # 计算技术指标
        df['MA10'] = close.rolling(10, min_periods=1).mean()
        df['MA20'] = close.rolling(20, min_periods=1).mean()
        df['MA5'] = close.rolling(5, min_periods=1).mean()

        # 计算 RSI
        try:
            df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        except:
            # 简单的 RSI 计算作为后备
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

        # 计算 MACD
        try:
            macd = ta.trend.MACD(close)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
        except:
            # 简单的 MACD 计算作为后备
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        # 添加滞后特征
        df['Prev_Close'] = close.shift(1)
        for i in range(1, 4):  # 减少滞后特征数量
            df[f'Prev_Close_Lag{i}'] = close.shift(i)

        # 添加成交量特征
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(10, min_periods=1).mean()
        else:
            df['Volume_MA'] = 0

        # 添加波动率指标
        df['Volatility'] = close.rolling(10, min_periods=1).std()

        # 选择特征
        feats = ['Prev_Close', 'MA5', 'MA10', 'MA20', 'Volume_MA', 'RSI', 'MACD',
                 'MACD_Signal', 'TWII_Close', 'SP500_Close', 'Volatility'] + \
                [f'Prev_Close_Lag{i}' for i in range(1, 4)]

        # 检查缺失的特征
        missing_feats = [f for f in feats if f not in df.columns]
        if missing_feats:
            st.error(f"缺少特征: {missing_feats}")
            return None, None, None

        # 移除有缺失值的行
        df_clean = df[feats + ['Close']].dropna()

        if len(df_clean) < 30:
            st.error(f"数据不足，仅有 {len(df_clean)} 行数据")
            return None, None, None

        # 准备训练数据
        X = df_clean[feats].values
        y = df_clean['Close'].values

        # 标准化特征
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # 避免除以零
        X_normalized = (X - X_mean) / X_std

        # 计算时间权重
        weights = np.exp(-decay_factor * np.arange(len(X))[::-1])
        weights = weights / np.sum(weights)

        # 分割训练和验证集
        split_idx = int(len(X_normalized) * 0.8)
        X_train, X_val = X_normalized[:split_idx], X_normalized[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        train_weights = weights[:split_idx]

        # 训练多个模型来增加预测多样性
        models = []

        # 主要随机森林模型
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

        # 预测未来 5 天 - 使用集成模型
        last_features = X_normalized[-1:].copy()
        last_close = float(y[-1])
        predictions = {}

        # 创建未来日期
        future_dates = []
        current_date = end
        for i in range(5):
            current_date = current_date + pd.offsets.BDay(1)
            future_dates.append(current_date.date())

        # 逐步预测 - 每次预测后更新特征
        current_features = last_features.copy()
        predicted_prices = [last_close]  # 包含最后一天的实际价格

        for i, date in enumerate(future_dates):
            # 使用模型进行预测并加上随机变化
            pred = rf_model.predict(current_features)[0]
            variation = np.random.normal(0, pred * 0.005)  # 0.5% 随机变化
            final_pred = pred + variation
            predictions[date] = final_pred
            predicted_prices.append(final_pred)

            # 更新特征
            new_features = current_features[0].copy()
            prev_close_idx = feats.index('Prev_Close')
            new_features[prev_close_idx] = (final_pred - X_mean[prev_close_idx]) / X_std[prev_close_idx]

            # 更新滞后特征
            for j in range(1, min(4, len(predicted_prices))):
                if f'Prev_Close_Lag{j}' in feats:
                    lag_idx = feats.index(f'Prev_Close_Lag{j}')
                    lag_price = predicted_prices[-(j+1)]
                    new_features[lag_idx] = (lag_price - X_mean[lag_idx]) / X_std[lag_idx]

            current_features = new_features.reshape(1, -1)

        # 计算预测字典
        preds = {f'T+{i+1}': pred for i, pred in enumerate(predictions.values())}

        return last_close, predictions, preds

    except Exception as e:
        st.error(f"预测过程发生错误: {str(e)}")
        return None, None, None


# Streamlit 界面
st.title("📈 5 日股价预测系统")
st.markdown("---")

# 输入区域
col1, col2 = st.columns([2, 1])
with col1:
    stock_input = st.text_input("请输入股票代号或名称", "台积电", help="例如：2330 (台积电)、AAPL (苹果)")

with col2:
    mode = st.selectbox("预测模式", ["中期模式", "短期模式", "长期模式"])

# 获取台湾股市股票数据
stock_list = fetch_twse_stock_codes()

# 根据股票代号查询股票名称
stock_name = get_stock_name(stock_input.strip())

# 显示选择的股票名称
if stock_name:
    st.info(f"您选择的股票是: {stock_name}")
else:
    st.error("无法识别此股票代号，请检查代号或名称")

# 模式说明
mode_info = {
    "短期模式": ("使用 100 天历史资料，高敏感度", 100, 0.008),
    "中期模式": ("使用 200 天历史资料，平衡敏感度", 200, 0.005),
    "长期模式": ("使用 400 天历史资料，低敏感度", 400, 0.002)
}

st.info(f"**{mode}**: {mode_info[mode][0]}")
days, decay_factor = mode_info[mode][1], mode_info[mode][2]

if st.button("🔮 开始预测", type="primary"):
    with st.spinner("正在下载资料并进行预测..."):
        last, forecast, preds = predict_next_5(stock_input.strip(), stock_list, days, decay_factor)

    if last is None:
        st.error("❌ 预测失败，请检查股票代号或网络连接")
    else:
        # 显示结果
        st.success("✅ 预测完成！")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("当前股价", f"${last:.2f}")
            advice = get_trade_advice(last, preds)
            
            if "买入" in advice:
                st.success(f"📈 **交易建议**: {advice}")
            elif "卖出" in advice:
                st.error(f"📉 **交易建议**: {advice}")
            else:
                st.warning(f"📊 **交易建议**: {advice}")
        
        with col2:
            st.subheader("📅 未来 5 日预测")
            for date, price in forecast.items():
                change = price - last
                change_pct = (change / last) * 100
                
                if change > 0:
                    st.write(f"**{date}**: ${price:.2f} (+{change:.2f}, +{change_pct:.1f}%)")
                else:
                    st.write(f"**{date}**: ${price:.2f} ({change:.2f}, {change_pct:.1f}%)")
        
        # 绘制趋势图
        st.subheader("📈 预测趋势")
        chart_data = pd.DataFrame({
            '日期': ['今日'] + list(forecast.keys()),
            '股价': [last] + list(forecast.values())
        })
        st.line_chart(chart_data.set_index('日期'))

st.markdown("---")
st.caption("⚠️ 此预测仅供参考，投资有风险，请谨慎决策")
