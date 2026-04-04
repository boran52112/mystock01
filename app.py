import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader # <-- 引入台灣本土籌碼 API
from datetime import datetime, timedelta

st.set_page_config(page_title="個人股票技術分析系統", layout="wide")
st.title("📈 個人股票技術分析系統")

st.markdown("### ⚙️ 快速查詢")
col1, col2 = st.columns([3, 1])
with col1:
    stock_code = st.text_input("請輸入台股代號 (例如: 2330 或 8069)", "2330")
with col2:
    st.write("") 
    st.write("")
    update_button = st.button("🔄 查詢 / 更新資料", use_container_width=True)

st.write("---") 

# --- 新增：抓取台灣三大法人籌碼 ---
def fetch_chip_data(code):
    try:
        dl = DataLoader()
        # 抓取過去 60 天的資料就夠用了
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=code, start_date=start_date)
        
        if df_inst.empty:
            return None

        # 整理 FinMind 複雜的資料格式，轉換成以「日期」為基準的表格
        df_pivot = df_inst.pivot_table(index='date', columns='name', values='buy_sell', aggfunc='sum').reset_index()
        df_pivot.rename(columns={'date': 'Date'}, inplace=True)
        
        # 為了避免某些股票沒有投信或外資資料，做個安全防護
        if '外陸資買賣超股數(不含外資自營商)' in df_pivot.columns:
            df_pivot['Foreign_Buy'] = df_pivot['外陸資買賣超股數(不含外資自營商)'] / 1000 # 換算成「張」
        else:
            df_pivot['Foreign_Buy'] = 0
            
        if '投信買賣超股數' in df_pivot.columns:
            df_pivot['Trust_Buy'] = df_pivot['投信買賣超股數'] / 1000
        else:
            df_pivot['Trust_Buy'] = 0

        return df_pivot[['Date', 'Foreign_Buy', 'Trust_Buy']]
    except Exception as e:
        return None

# DB1：原始資料庫 (雙核心合併版)
def fetch_stock_data(code):
    df_price, symbol = None, None
    for suffix in [".TW", ".TWO"]:
        try:
            temp_sym = f"{code}{suffix}"
            ticker = yf.Ticker(temp_sym)
            df = ticker.history(period="1y")
            if not df.empty:
                df.reset_index(inplace=True)
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                df_price, symbol = df, temp_sym
                break
        except:
            continue
            
    if df_price is None:
        return None, None, False

    # 嘗試抓取籌碼資料並「合併 (Merge)」
    df_chip = fetch_chip_data(code)
    has_chip_data = False
    if df_chip is not None:
        # 將價格表和籌碼表，用「日期」像拉鍊一樣接起來！(左外部連接)
        df_price = pd.merge(df_price, df_chip, on='Date', how='left')
        df_price['Foreign_Buy'].fillna(0, inplace=True)
        df_price['Trust_Buy'].fillna(0, inplace=True)
        has_chip_data = True
    else:
        # 如果 API 異常，塞入 0 讓程式不會崩潰
        df_price['Foreign_Buy'] = 0
        df_price['Trust_Buy'] = 0
        
    return df_price, symbol, has_chip_data

# DB2：技術分析資料庫
def generate_db2(df):
    db2 = df.copy()
    db2['SMA_5'] = db2['Close'].rolling(window=5).mean()
    db2['SMA_20'] = db2['Close'].rolling(window=20).mean()
    
    delta = db2['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    db2['RSI_14'] = 100 - (100 / (1 + rs))
    
    exp1 = db2['Close'].ewm(span=12, adjust=False).mean()
    exp2 = db2['Close'].ewm(span=26, adjust=False).mean()
    db2['MACD'] = exp1 - exp2
    db2['MACD_Signal'] = db2['MACD'].ewm(span=9, adjust=False).mean()
    db2['MACD_Hist'] = db2['MACD'] - db2['MACD_Signal']
    
    db2['BB_Mid'] = db2['Close'].rolling(window=20).mean()
    db2['BB_Std'] = db2['Close'].rolling(window=20).std()
    db2['BB_Up'] = db2['BB_Mid'] + (2 * db2['BB_Std'])
    db2['BB_Low'] = db2['BB_Mid'] - (2 * db2['BB_Std'])
    
    low_min = db2['Low'].rolling(window=9).min()
    high_max = db2['High'].rolling(window=9).max()
    db2['RSV'] = (db2['Close'] - low_min) / (high_max - low_min) * 100
    db2['K'] = db2['RSV'].ewm(com=2, adjust=False).mean()
    db2['D'] = db2['K'].ewm(com=2, adjust=False).mean()
    
    db2['Prev_High'] = db2['High'].shift(1)
    db2['Prev_Low'] = db2['Low'].shift(1)
    
    db2.dropna(inplace=True) 
    return db2.round(2)

# DB3：結論與矛盾分析 
def generate_db3(db2, has_chip):
    latest = db2.iloc[-1]
    date = latest['Date']
    close_price = latest['Close']
    
    analysis_list = []
    bull_count, bear_count = 0, 0
    
    # [1~7: 技術指標區塊保持原樣]
    if latest['SMA_5'] > latest['SMA_20']:
        analysis_list.append(["1. 均線理論", "🟢 看多", "5日線大於20日線"])
        bull_count += 1
    else:
        analysis_list.append(["1. 均線理論", "🔴 看空", "5日線小於20日線"])
        bear_count += 1
        
    if latest['RSI_14'] > 75:
        analysis_list.append(["2. 動能理論 (RSI)", "🔴 看空", f"RSI為 {latest['RSI_14']} (超買)"])
        bear_count += 1
    elif latest['RSI_14'] < 25:
        analysis_list.append(["2. 動能理論 (RSI)", "🟢 看多", f"RSI為 {latest['RSI_14']} (超賣)"])
        bull_count += 1
    else:
        analysis_list.append(["2. 動能理論 (RSI)", "⚪ 中立", "正常震盪"])
        
    if latest['MACD_Hist'] > 0:
        analysis_list.append(["3. 波段理論 (MACD)", "🟢 看多", "柱狀圖為正"])
        bull_count += 1
    else:
        analysis_list.append(["3. 波段理論 (MACD)", "🔴 看空", "柱狀圖為負"])
        bear_count += 1
        
    if latest['Close'] > latest['BB_Up']:
        analysis_list.append(["4. 布林通道", "🟢 看多", "突破上軌"])
        bull_count += 1
    elif latest['Close'] < latest['BB_Low']:
        analysis_list.append(["4. 布林通道", "🔴 看空", "跌破下軌"])
        bear_count += 1
    else:
        analysis_list.append(["4. 布林通道", "⚪ 中立", "通道內游走"])
        
    if latest['K'] > 80 and latest['D'] > 80:
        analysis_list.append(["5. KD 指標", "🔴 看空", "高檔超買"])
        bear_count += 1
    elif latest['K'] < 20 and latest['D'] < 20:
        analysis_list.append(["5. KD 指標", "🟢 看多", "低檔超賣"])
        bull_count += 1
    else:
        analysis_list.append(["5. KD 指標", "⚪ 中立", "數值居中"])

    body = abs(latest['Close'] - latest['Open'])
    lower_shadow = min(latest['Close'], latest['Open']) - latest['Low']
    upper_shadow = latest['High'] - max(latest['Close'], latest['Open'])
    
    if lower_shadow > (body * 2) and body > 0:
        analysis_list.append(["6. K線型態", "🟢 看多", "長下影線支撐"])
        bull_count += 1
    elif upper_shadow > (body * 2) and body > 0:
        analysis_list.append(["6. K線型態", "🔴 看空", "長上影線賣壓"])
        bear_count += 1
    else:
        analysis_list.append(["6. K線型態", "⚪ 中立", "無特殊型態"])
        
    if latest['Low'] > latest['Prev_High']:
        analysis_list.append(["7. 缺口理論", "🟢 看多", "向上跳空缺口"])
        bull_count += 1
    elif latest['High'] < latest['Prev_Low']:
        analysis_list.append(["7. 缺口理論", "🔴 看空", "向下跳空缺口"])
        bear_count += 1
    else:
        analysis_list.append(["7. 缺口理論", "⚪ 中立", "無跳空缺口"])

    # --- 8. 新增籌碼理論 (三大法人) ---
    if has_chip:
        # 計算近3日外資與投信的總買賣超
        recent_3_days = db2.tail(3)
        foreign_3d_sum = recent_3_days['Foreign_Buy'].sum()
        trust_3d_sum = recent_3_days['Trust_Buy'].sum()
        
        desc_chip = f"近三日外資買賣: {int(foreign_3d_sum)}張, 投信買賣: {int(trust_3d_sum)}張"
        
        if trust_3d_sum > 500 or foreign_3d_sum > 2000:
            analysis_list.append(["8. 籌碼理論 (法人)", "🟢 看多", desc_chip + " (法人積極買超)"])
            bull_count += 1
        elif trust_3d_sum < -500 or foreign_3d_sum < -2000:
            analysis_list.append(["8. 籌碼理論 (法人)", "🔴 看空", desc_chip + " (法人積極賣超)"])
            bear_count += 1
        else:
            analysis_list.append(["8. 籌碼理論 (法人)", "⚪ 中立", desc_chip + " (法人動作不大)"])
    else:
        analysis_list.append(["8. 籌碼理論 (法人)", "⚪ 未知", "今日籌碼資料尚無或API異常"])

    db3_df = pd.DataFrame(analysis_list, columns=["分析方法", "當前訊號", "狀態描述"])
    
    # 專家矛盾引擎升級：加入籌碼判斷
    expert_comment = ""
    trend_is_up = (latest['SMA_5'] > latest['SMA_20'])
    trend_is_down = (latest['SMA_5'] < latest['SMA_20'])
    
    # 籌碼與技術面的矛盾分析
    if has_chip:
        if trend_is_down and (trust_3d_sum > 500 or foreign_3d_sum > 2000):
            expert_comment += "💡 **【專家視角：法人逢低偷接】** 目前技術均線呈現空頭，但發現「外資或投信連續買超」。這代表主力法人正在逢低承接，籌碼從散戶流向法人，隨時有醞釀大反彈的可能！不建議在此追空。\n\n"
        elif trend_is_up and (trust_3d_sum < -500 or foreign_3d_sum < -2000):
            expert_comment += "💡 **【專家視角：主力逢高出貨】** 目前技術面看似強勢，但「外資或投信卻在大量賣超」。請小心這是主力在拉高出貨的假突破，技術面指標即將反轉，建議多單嚴格設停損或減碼。\n\n"

    # 若無籌碼矛盾，執行常規判斷
    if expert_comment == "":
        if (bull_count > 0) and (bear_count > 0):
            expert_comment += "💡 **【專家視角：盤勢震盪】** 指標方向分歧，處於「盤整區間」或「趨勢轉換期」。建議空手觀望。\n\n"
        elif bull_count > 3:
            expert_comment += "💡 **【專家視角：順勢做多】** 多方共識強烈，順勢操作勝率極高。\n\n"
        elif bear_count > 3:
            expert_comment += "💡 **【專家視角：順勢偏空】** 空方共識強烈，嚴格執行多單停損。\n\n"

    return db3_df, date, close_price, bull_count, bear_count, expert_comment

def plot_candlestick(db2):
    plot_data = db2.tail(120) 
    fig = go.Figure(data=[go.Candlestick(x=plot_data['Date'],
                    open=plot_data['Open'], high=plot_data['High'],
                    low=plot_data['Low'], close=plot_data['Close'],
                    name='K線')])
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['SMA_5'], mode='lines', name='5日均線', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['SMA_20'], mode='lines', name='20日均線', line=dict(color='orange', width=1.5)))
    fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), height=400)
    return fig

# --- 網頁畫面呈現 ---
if update_button or stock_code:
    stock_code = stock_code.strip() 
    with st.spinner('正在搜尋資料與計算籌碼... (可能需要幾秒鐘)'):
        db1_price_data, actual_symbol, has_chip = fetch_stock_data(stock_code)
        
    if db1_price_data is not None:
        market_type = "上市" if ".TW" in actual_symbol else "上櫃"
        st.write(f"### 🎯 分析標的：{stock_code} ({market_type})")
        
        db2_ta_data = generate_db2(db1_price_data)
        chart_fig = plot_candlestick(db2_ta_data)
        st.plotly_chart(chart_fig, use_container_width=True)
        
        st.write("### 🧠 系統研判與專家視角 (DB3)")
        db3_df, target_date, current_price, bulls, bears, expert_comment = generate_db3(db2_ta_data, has_chip)
        st.write(f"**分析日期：** {target_date} ｜ **最新收盤價：** {current_price}")
        st.table(db3_df)
        
        st.write("#### ⚖️ 專家矛盾分析與權重判讀")
        st.info(expert_comment)
                
        with st.expander("🗄️ 展開查看原始資料 (含法人籌碼) 與 技術指標"):
            st.dataframe(db2_ta_data.tail(5), use_container_width=True)
    else:
        st.error(f"❌ 找不到代號為「{stock_code}」的股票。")
