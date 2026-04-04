import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 網頁設定 ---
st.set_page_config(page_title="AI 專家級股票分析系統", layout="wide")

# 自定義 CSS 讓介面更精美
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI 專家級股票技術分析系統")

# --- 側邊欄參數設定 ---
st.sidebar.header("⚙️ 技術指標設定")
sma_fast = st.sidebar.slider("短均線 (SMA Fast)", 5, 20, 5)
sma_slow = st.sidebar.slider("長均線 (SMA Slow)", 20, 60, 20)
rsi_period = st.sidebar.slider("RSI 週期", 7, 28, 14)

# --- 資料抓取模組 (DB1) ---

@st.cache_data(ttl=3600) # 快取 1 小時，避免重複抓取
def fetch_chip_data(code):
    """抓取台灣籌碼資料 (法人買賣、融資券)"""
    try:
        dl = DataLoader()
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        # 1. 法人買賣超
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=code, start_date=start_date)
        df_pivot = pd.DataFrame()
        if not df_inst.empty:
            df_pivot = df_inst.pivot_table(index='date', columns='name', values='buy_sell', aggfunc='sum').reset_index()
            df_pivot.rename(columns={'date': 'Date'}, inplace=True)
            
            # 轉換為張數 (通常原始資料為股數)
            cols = df_pivot.columns
            f_col = '外陸資買賣超股數(不含外資自營商)'
            t_col = '投信買賣超股數'
            df_pivot['Foreign_Buy'] = df_pivot[f_col] / 1000 if f_col in cols else 0
            df_pivot['Trust_Buy'] = df_pivot[t_col] / 1000 if t_col in cols else 0
            df_pivot = df_pivot[['Date', 'Foreign_Buy', 'Trust_Buy']]

        # 2. 融資融券
        df_margin = dl.taiwan_stock_margin_purchase_short_sale(stock_id=code, start_date=start_date)
        if not df_margin.empty:
            df_margin = df_margin[['date', 'MarginPurchaseTodayBalance', 'ShortSaleTodayBalance']]
            df_margin.rename(columns={
                'date': 'Date', 
                'MarginPurchaseTodayBalance': 'Margin_Bal',
                'ShortSaleTodayBalance': 'Short_Bal'
            }, inplace=True)
            
        if not df_pivot.empty and not df_margin.empty:
            return pd.merge(df_pivot, df_margin, on='Date', how='outer').fillna(0)
        elif not df_pivot.empty: return df_pivot
        elif not df_margin.empty: return df_margin
        return None
    except:
        return None

@st.cache_data(ttl=3600)
def fetch_stock_data(code):
    """抓取股價資料並結合籌碼"""
    df_price, symbol = None, None
    for suffix in [".TW", ".TWO"]:
        try:
            temp_sym = f"{code}{suffix}"
            # 使用 progress=False 避免 Streamlit 顯示雜亂進度條
            df = yf.download(temp_sym, period="1y", progress=False)
            if not df.empty:
                df.reset_index(inplace=True)
                # 解決新版 yfinance MultiIndex 欄位問題
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                df_price, symbol = df, temp_sym
                break
        except:
            continue
            
    if df_price is None:
        return None, None, False

    # 結合籌碼資料
    df_chip = fetch_chip_data(code)
    has_chip_data = False
    if df_chip is not None:
        df_price = pd.merge(df_price, df_chip, on='Date', how='left')
        # 籌碼資料若當天沒更新，以前一天為準，若全無則補 0
        df_price.fillna(method='ffill', inplace=True)
        df_price.fillna(0, inplace=True)
        has_chip_data = True
    else:
        for col in ['Foreign_Buy', 'Trust_Buy', 'Margin_Bal', 'Short_Bal']:
            df_price[col] = 0
        
    return df_price, symbol, has_chip_data

# --- 技術指標計算 (DB2) ---

def generate_db2(df):
    db2 = df.copy()
    
    # 均線
    db2['SMA_Fast'] = db2['Close'].rolling(window=sma_fast).mean()
    db2['SMA_Slow'] = db2['Close'].rolling(window=sma_slow).mean()
    
    # RSI
    delta = db2['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    db2['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = db2['Close'].ewm(span=12, adjust=False).mean()
    exp2 = db2['Close'].ewm(span=26, adjust=False).mean()
    db2['MACD'] = exp1 - exp2
    db2['MACD_Sig'] = db2['MACD'].ewm(span=9, adjust=False).mean()
    db2['MACD_Hist'] = db2['MACD'] - db2['MACD_Sig']
    
    # 布林通道
    db2['BB_Mid'] = db2['Close'].rolling(window=20).mean()
    std = db2['Close'].rolling(window=20).std()
    db2['BB_Up'] = db2['BB_Mid'] + (2 * std)
    db2['BB_Low'] = db2['BB_Mid'] - (2 * std)
    
    # KD
    low_9 = db2['Low'].rolling(window=9).min()
    high_9 = db2['High'].rolling(window=9).max()
    db2['RSV'] = (db2['Close'] - low_9) / (high_9 - low_9) * 100
    db2['K'] = db2['RSV'].ewm(com=2, adjust=False).mean()
    db2['D'] = db2['K'].ewm(com=2, adjust=False).mean()
    
    # 紀錄前一日數據計算型態
    db2['Prev_Close'] = db2['Close'].shift(1)
    db2['Prev_High'] = db2['High'].shift(1)
    db2['Prev_Low'] = db2['Low'].shift(1)
    db2['Margin_Diff'] = db2['Margin_Bal'].diff()
    
    return db2.dropna().round(2)

# --- 專家決策與報告 (DB3) ---

def generate_db3(db2, has_chip):
    latest = db2.iloc[-1]
    date = latest['Date']
    close_price = latest['Close']
    
    analysis_list = []
    bull_count, bear_count = 0, 0 # 正確的變數名稱定義
    
    # 1. 均線判定
    is_trend_up = latest['SMA_Fast'] > latest['SMA_Slow']
    if is_trend_up:
        analysis_list.append(["均線理論", "🟢 看多", "短均線在長均線之上，趨勢偏多"])
        bull_count += 1
    else:
        analysis_list.append(["均線理論", "🔴 看空", "短均線在長均線之下，趨勢偏弱"])
        bear_count += 1
        
    # 2. RSI 判定
    if latest['RSI'] > 75:
        analysis_list.append(["動能 (RSI)", "🔴 看空", f"RSI 為 {latest['RSI']} (超買區)"])
        bear_count += 1
    elif latest['RSI'] < 25:
        analysis_list.append(["動能 (RSI)", "🟢 看多", f"RSI 為 {latest['RSI']} (超賣區)"])
        bull_count += 1
    else:
        analysis_list.append(["動能 (RSI)", "⚪ 中立", "震盪區間"])
        
    # 3. MACD 判定
    if latest['MACD_Hist'] > 0:
        analysis_list.append(["波段 (MACD)", "🟢 看多", "MACD 紅柱增長中"])
        bull_count += 1
    else:
        analysis_list.append(["波段 (MACD)", "🔴 看空", "MACD 綠柱區間"])
        bear_count += 1
        
    # 4. K線型態
    body = abs(latest['Close'] - latest['Open'])
    lower_shadow = min(latest['Close'], latest['Open']) - latest['Low']
    if lower_shadow > body * 2 and body > 0:
        analysis_list.append(["K線型態", "🟢 看多", "出現長下影線，具支撐力道"])
        bull_count += 1
    else:
        analysis_list.append(["K線型態", "⚪ 中立", "無特殊型態"])

    # 5. 籌碼理論 (法人)
    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 500:
            analysis_list.append(["法人籌碼", "🟢 看多", f"三大法人單日大買 {int(inst_net)} 張"])
            bull_count += 1
        elif inst_net < -500:
            analysis_list.append(["法人籌碼", "🔴 看空", f"三大法人單日大賣 {int(inst_net)} 張"])
            bear_count += 1
        else:
            analysis_list.append(["法人籌碼", "⚪ 中立", "法人動作不明顯"])
    else:
        analysis_list.append(["法人籌碼", "⚪ 未知", "暫無籌碼資料"])

    # --- 專家報告生成 ---
    expert_report = "#### 🔍 專家深度解析與建議\n"
    
    # 1. 趨勢位階
    if is_trend_up and latest['MACD_Hist'] > 0:
        expert_report += "- **格局判定**：目前處於強勢多頭格局，均線與動能同步向上。\n"
    elif not is_trend_up and latest['MACD_Hist'] < 0:
        expert_report += "- **格局判定**：目前處於空頭調整格局，建議觀望保守。\n"
    else:
        expert_report += "- **格局判定**：盤勢進入震盪整理，指標出現分歧。\n"

    # 2. 矛盾檢查 (解決 NameError 後的核心邏輯)
    if is_trend_up and latest['RSI'] > 75:
        expert_report += "- **⚠️ 指標警訊**：雖然趨勢看多，但 RSI 已進入極度超買區，隨時有回檔壓力，不宜追高。\n"
    
    # 3. 具體策略總結 (修正了 bulls -> bull_count 的錯誤)
    expert_report += "\n#### 🎯 具體操作策略\n"
    if bull_count > bear_count * 2:
        expert_report += "> **👉 【偏多操作】** 多方指標佔有絕對優勢。建議可隨 5 日線佈局，並以 20 日線作為波段防守點。"
    elif bear_count > bull_count * 2:
        expert_report += "> **👉 【建議觀望】** 空方力量強勁。建議暫時避開，待股價止跌並站回均線後再行考慮。"
    else:
        expert_report += "> **👉 【區間操作】** 多空勢均力敵。適合低買高賣，或等待突破布林通道上/下軌道後的方向確認。"

    db3_df = pd.DataFrame(analysis_list, columns=["分析方法", "當前訊號", "狀態描述"])
    return db3_df, date, close_price, bull_count, bear_count, expert_report

# --- 繪圖功能 ---

def plot_advanced_chart(df):
    """繪製包含 K線、均線、布林通道與成交量的圖表"""
    # 建立子圖 (Row 1: K線, Row 2: 成交量)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=('K線與技術指標', '成交量'), 
                       row_width=[0.3, 0.7])

    # K線圖
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="K線"
    ), row=1, col=1)

    # 均線
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_Fast'], line=dict(color='blue', width=1), name=f"{sma_fast}MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_Slow'], line=dict(color='orange', width=1), name=f"{sma_slow}MA"), row=1, col=1)
    
    # 布林通道 (虛線)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Up'], line=dict(color='rgba(255,0,0,0.2)', dash='dot'), name="布林上軌"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Low'], line=dict(color='rgba(0,255,0,0.2)', dash='dot'), name="布林下軌"), row=1, col=1)

    # 成交量 (紅漲綠跌)
    colors = ['#EF5350' if row['Close'] >= row['Open'] else '#26A69A' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="成交量"), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(t=50, b=50, l=10, r=10))
    return fig

# --- 網頁主要 UI 邏輯 ---

col1, col2 = st.columns([3, 1])
with col1:
    stock_code = st.text_input("🔍 請輸入台股代號 (例如: 2330, 8069)", "2330").strip()
with col2:
    st.write("") 
    st.write("")
    search_btn = st.button("開始分析", use_container_width=True)

if stock_code:
    with st.spinner(f"正在分析 {stock_code}，請稍候..."):
        df_raw, actual_sym, has_chip = fetch_stock_data(stock_code)
        
        if df_raw is not None:
            # 1. 生成資料
            db2_ta = generate_db2(df_raw)
            db3_df, target_date, current_price, bulls, bears, expert_report = generate_db3(db2_ta, has_chip)
            
            # 2. 顯示關鍵數據指標
            st.markdown(f"### 🎯 分析對象：{actual_sym}")
            m1, m2, m3 = st.columns(3)
            price_change = round(current_price - db2_ta.iloc[-2]['Close'], 2)
            m1.metric("最新收盤價", f"{current_price} TWD", f"{price_change}")
            m2.metric("看多訊號數", f"{bulls} 🟢")
            m3.metric("看空訊號數", f"{bears} 🔴")
            
            # 3. 顯示圖表
            st.plotly_chart(plot_advanced_chart(db2_ta.tail(120)), use_container_width=True)
            
            # 4. 顯示報告與表格
            st.write("---")
            st.markdown("### 🧠 系統研判與專家報告")
            
            # 佈局：左邊表格，右邊文字報告
            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.write("**基礎指標統計表**")
                st.table(db3_df)
            with c_right:
                st.info(expert_report)
                
            # 5. 原始資料展開
            with st.expander("🗄️ 展開查看詳細數據紀錄 (最近 10 筆)"):
                st.dataframe(db2_ta.tail(10), use_container_width=True)
        else:
            st.error(f"❌ 無法獲取代號 {stock_code} 的資料，請確認代號是否正確或 API 是否連線正常。")
