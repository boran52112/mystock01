import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 設定網頁標題
st.set_page_config(page_title="個人股票技術分析系統", layout="wide")
st.title("📈 個人股票技術分析系統")

# 【優化 1】把輸入框移到主畫面最上方，手機版體驗大幅提升！
st.markdown("### ⚙️ 快速查詢")
col1, col2 = st.columns([3, 1]) # 切割排版比例
with col1:
    stock_code = st.text_input("請輸入台股代號 (例如: 2330 或 8069)", "2330")
with col2:
    st.write("") # 為了對齊輸入框留白
    st.write("")
    update_button = st.button("🔄 查詢 / 更新資料", use_container_width=True)

st.write("---") # 畫一條分隔線

# DB1：原始資料庫 (【優化 2】強化上市/上櫃的獨立判斷邏輯)
def fetch_stock_data(code):
    # 嘗試抓取【上市 .TW】
    symbol_tw = f"{code}.TW"
    try:
        ticker_tw = yf.Ticker(symbol_tw)
        df_tw = ticker_tw.history(period="1y")
        if not df_tw.empty:
            df_tw.reset_index(inplace=True)
            df_tw = df_tw[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df_tw['Date'] = df_tw['Date'].dt.strftime('%Y-%m-%d')
            return df_tw, symbol_tw
    except:
        pass # 如果報錯，什麼都不做，繼續往下找上櫃

    # 嘗試抓取【上櫃 .TWO】
    symbol_two = f"{code}.TWO"
    try:
        ticker_two = yf.Ticker(symbol_two)
        df_two = ticker_two.history(period="1y")
        if not df_two.empty:
            df_two.reset_index(inplace=True)
            df_two = df_two[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df_two['Date'] = df_two['Date'].dt.strftime('%Y-%m-%d')
            return df_two, symbol_two
    except:
        pass

    # 如果兩個都找不到，回傳空值
    return None, None

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
    db2.dropna(inplace=True) 
    return db2.round(2)

# DB3：結論與矛盾分析
def generate_db3(db2):
    latest = db2.iloc[-1]
    date = latest['Date']
    close_price = latest['Close']
    
    analysis_list = []
    bull_count, bear_count = 0, 0
    
    if latest['SMA_5'] > latest['SMA_20']:
        signal, desc = "🟢 看多", "5日均線 > 20日均線，短期趨勢偏多"
        bull_count += 1
    else:
        signal, desc = "🔴 看空", "5日均線 < 20日均線，短期趨勢偏空"
        bear_count += 1
    analysis_list.append(["均線理論 (Trend)", signal, desc])
    
    if latest['RSI_14'] > 70:
        signal, desc = "🔴 看空 (超買)", f"RSI高達 {latest['RSI_14']}，慎防高檔反轉回落"
        bear_count += 1
    elif latest['RSI_14'] < 30:
        signal, desc = "🟢 看多 (超賣)", f"RSI低至 {latest['RSI_14']}，隨時可能醞釀反彈"
        bull_count += 1
    else:
        signal, desc = "⚪ 中立", f"RSI為 {latest['RSI_14']}，處於正常整理區間"
    analysis_list.append(["動能理論 (RSI)", signal, desc])
    
    if latest['MACD_Hist'] > 0:
        signal, desc = "🟢 看多", "MACD 柱狀圖為正，多方掌握動能"
        bull_count += 1
    else:
        signal, desc = "🔴 看空", "MACD 柱狀圖為負，空方掌握動能"
        bear_count += 1
    analysis_list.append(["波段理論 (MACD)", signal, desc])
    
    db3_df = pd.DataFrame(analysis_list, columns=["分析方法", "當前訊號", "狀態描述"])
    has_conflict = (bull_count > 0) and (bear_count > 0)
    
    return db3_df, date, close_price, bull_count, bear_count, has_conflict

def plot_candlestick(db2):
    plot_data = db2.tail(120) 
    fig = go.Figure(data=[go.Candlestick(x=plot_data['Date'],
                    open=plot_data['Open'], high=plot_data['High'],
                    low=plot_data['Low'], close=plot_data['Close'],
                    name='K線')])
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['SMA_5'], mode='lines', name='5日均線', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['SMA_20'], mode='lines', name='20日均線', line=dict(color='orange', width=1.5)))
    
    # 針對手機版微調圖表邊界，讓圖表顯示更大
    fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), height=400)
    return fig

# --- 網頁畫面呈現邏輯 ---
if update_button or stock_code:
    stock_code = stock_code.strip() 
    
    with st.spinner('正在搜尋台股資料庫...'):
        db1_price_data, actual_symbol = fetch_stock_data(stock_code)
        
    if db1_price_data is not None:
        market_type = "上市" if ".TW" in actual_symbol else "上櫃"
        st.write(f"### 🎯 分析標的：{stock_code} ({market_type})")
        
        db2_ta_data = generate_db2(db1_price_data)
        
        # 把圖表放在最顯眼的地方
        chart_fig = plot_candlestick(db2_ta_data)
        st.plotly_chart(chart_fig, use_container_width=True)
        
        # DB3 決策直接顯示在圖表下方
        st.write("### 🧠 系統研判與矛盾分析 (DB3)")
        db3_df, target_date, current_price, bulls, bears, conflict = generate_db3(db2_ta_data)
        st.write(f"**分析日期：** {target_date} ｜ **最新收盤價：** {current_price}")
        st.table(db3_df)
        
        if conflict:
            st.warning(f"⚠️ **【發現矛盾訊號】** 目前共有 {bulls} 個指標看多，{bears} 個指標看空。建議保守觀望。")
        else:
            if bulls > 0:
                st.success(f"🚀 **【強烈多頭共識】** 目前 {bulls} 個指標全部看多！")
            elif bears > 0:
                st.error(f"📉 **【強烈空頭共識】** 目前 {bears} 個指標全部看空！切勿輕易摸底。")
            else:
                st.info("目前所有指標皆為中立，建議持續觀察。")
                
        # 原始資料收到最底下的折疊區塊，保持畫面乾淨
        with st.expander("🗄️ 展開查看原始資料 (DB1) 與 技術指標 (DB2)"):
            st.write("**DB1：原始股價資料**")
            st.dataframe(db1_price_data.tail(5), use_container_width=True)
            st.write("**DB2：技術分析結果**")
            st.dataframe(db2_ta_data.tail(5), use_container_width=True)

    else:
        st.error(f"❌ 找不到代號為「{stock_code}」的股票，請確認是否為有效的台股代號（可能為剛上市或無交易資料）。")
