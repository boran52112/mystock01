import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go  # <-- 新增：載入畫圖套件

st.set_page_config(page_title="個人股票技術分析系統", layout="wide")
st.title("📈 個人股票技術分析系統")

st.sidebar.header("設定")
stock_symbol = st.sidebar.text_input("請輸入股票代號 (台股請加 .TW，例如 2330.TW)", "2330.TW")
update_button = st.sidebar.button("🔄 每日更新資料")

# DB1：原始資料庫
def fetch_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y")
        df.reset_index(inplace=True)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        st.error(f"抓取資料失敗，請檢查股票代號。錯誤訊息: {e}")
        return None

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
    
    # 均線
    if latest['SMA_5'] > latest['SMA_20']:
        signal, desc = "🟢 看多", "5日均線 > 20日均線，短期趨勢偏多"
        bull_count += 1
    else:
        signal, desc = "🔴 看空", "5日均線 < 20日均線，短期趨勢偏空"
        bear_count += 1
    analysis_list.append(["均線理論 (Trend)", signal, desc])
    
    # RSI
    if latest['RSI_14'] > 70:
        signal, desc = "🔴 看空 (超買)", f"RSI高達 {latest['RSI_14']}，慎防高檔反轉回落"
        bear_count += 1
    elif latest['RSI_14'] < 30:
        signal, desc = "🟢 看多 (超賣)", f"RSI低至 {latest['RSI_14']}，隨時可能醞釀反彈"
        bull_count += 1
    else:
        signal, desc = "⚪ 中立", f"RSI為 {latest['RSI_14']}，處於正常整理區間"
    analysis_list.append(["動能理論 (RSI)", signal, desc])
    
    # MACD
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

# --- 繪製 K線圖函數 ---
def plot_candlestick(db2):
    # 只取最近半年(約 120 根 K 線)來畫圖，畫面比較好看
    plot_data = db2.tail(120) 
    
    fig = go.Figure(data=[go.Candlestick(x=plot_data['Date'],
                    open=plot_data['Open'], high=plot_data['High'],
                    low=plot_data['Low'], close=plot_data['Close'],
                    name='K線')])
    
    # 加入均線
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['SMA_5'], mode='lines', name='5日均線', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['SMA_20'], mode='lines', name='20日均線', line=dict(color='orange', width=1.5)))
    
    # 隱藏下方的滑動條，讓圖表更簡潔
    fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0), height=400)
    return fig

# --- 網頁畫面呈現邏輯 ---
if update_button or stock_symbol:
    st.write(f"### 🎯 目前分析標的：{stock_symbol}")
    
    db1_price_data = fetch_stock_data(stock_symbol)
    if db1_price_data is not None and not db1_price_data.empty:
        db2_ta_data = generate_db2(db1_price_data)
        
        # --- 插入圖表區 ---
        st.write("### 📊 近期股價走勢與均線")
        chart_fig = plot_candlestick(db2_ta_data)
        st.plotly_chart(chart_fig, use_container_width=True)
        # -------------------
        
        with st.expander("🗄️ 展開查看原始資料 (DB1) 與 技術指標 (DB2)"):
            st.write("**DB1：原始股價資料**")
            st.dataframe(db1_price_data.tail(5), use_container_width=True)
            st.write("**DB2：技術分析結果**")
            st.dataframe(db2_ta_data.tail(5), use_container_width=True)
            
        st.write("---")
        st.write("### 🧠 第三資料庫：結論與矛盾分析 (DB3)")
        
        db3_df, target_date, current_price, bulls, bears, conflict = generate_db3(db2_ta_data)
        st.write(f"**分析日期：** {target_date} ｜ **最新收盤價：** {current_price}")
        st.table(db3_df)
        
        st.write("#### ⚖️ 綜合研判與行動建議")
        if conflict:
            st.warning(f"⚠️ **【發現矛盾訊號】** 目前共有 {bulls} 個指標看多，{bears} 個指標看空。")
            st.write("> **解決方案提供：** 指標出現分歧，代表目前盤勢進入震盪或面臨轉折。")
            st.write("> 1. **保守操作者**：建議空手觀望，等待所有指標方向一致再行進場。")
            st.write("> 2. **積極操作者**：若欲買進，請將資金切分批試單，並嚴格設定前波低點為停損。")
        else:
            if bulls > 0:
                st.success(f"🚀 **【強烈多頭共識】** 目前 {bulls} 個指標全部看多！")
                st.write("> **行動參考：** 技術面互相支持，無矛盾。為偏多操作格局，可尋找拉回均線附近的買點伺機佈局。")
            elif bears > 0:
                st.error(f"📉 **【強烈空頭共識】** 目前 {bears} 個指標全部看空！")
                st.write("> **行動參考：** 技術面互相支持，無矛盾。趨勢偏弱，建議多單減碼或保持空手觀望，切勿輕易摸底接刀。")
            else:
                st.info("目前所有指標皆為中立，建議持續觀察。")