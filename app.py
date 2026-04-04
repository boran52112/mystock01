import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

# DB1：原始資料庫
def fetch_stock_data(code):
    for suffix in [".TW", ".TWO"]:
        try:
            symbol = f"{code}{suffix}"
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1y")
            if not df.empty:
                df.reset_index(inplace=True)
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                return df, symbol
        except:
            continue
    return None, None

# DB2：技術分析資料庫 (加入 7 大武器計算)
def generate_db2(df):
    db2 = df.copy()
    
    # 1. 均線
    db2['SMA_5'] = db2['Close'].rolling(window=5).mean()
    db2['SMA_20'] = db2['Close'].rolling(window=20).mean()
    
    # 2. RSI
    delta = db2['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    db2['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    exp1 = db2['Close'].ewm(span=12, adjust=False).mean()
    exp2 = db2['Close'].ewm(span=26, adjust=False).mean()
    db2['MACD'] = exp1 - exp2
    db2['MACD_Signal'] = db2['MACD'].ewm(span=9, adjust=False).mean()
    db2['MACD_Hist'] = db2['MACD'] - db2['MACD_Signal']
    
    # 4. 布林通道 (20日, 2個標準差)
    db2['BB_Mid'] = db2['Close'].rolling(window=20).mean()
    db2['BB_Std'] = db2['Close'].rolling(window=20).std()
    db2['BB_Up'] = db2['BB_Mid'] + (2 * db2['BB_Std'])
    db2['BB_Low'] = db2['BB_Mid'] - (2 * db2['BB_Std'])
    
    # 5. KD 指標 (9日)
    low_min = db2['Low'].rolling(window=9).min()
    high_max = db2['High'].rolling(window=9).max()
    db2['RSV'] = (db2['Close'] - low_min) / (high_max - low_min) * 100
    db2['K'] = db2['RSV'].ewm(com=2, adjust=False).mean() # 近似公式
    db2['D'] = db2['K'].ewm(com=2, adjust=False).mean()
    
    # 6. 缺口判定用 (昨日高低點)
    db2['Prev_High'] = db2['High'].shift(1)
    db2['Prev_Low'] = db2['Low'].shift(1)
    
    db2.dropna(inplace=True) 
    return db2.round(2)

# DB3：結論與矛盾分析 (專家大腦)
def generate_db3(db2):
    latest = db2.iloc[-1]
    date = latest['Date']
    close_price = latest['Close']
    
    analysis_list = []
    bull_count, bear_count = 0, 0
    
    # --- 單一指標判定 ---
    # 1. 均線
    if latest['SMA_5'] > latest['SMA_20']:
        analysis_list.append(["1. 均線理論 (趨勢)", "🟢 看多", "5日線大於20日線，短多格局"])
        bull_count += 1
    else:
        analysis_list.append(["1. 均線理論 (趨勢)", "🔴 看空", "5日線小於20日線，短空格局"])
        bear_count += 1
        
    # 2. RSI
    if latest['RSI_14'] > 75:
        analysis_list.append(["2. 動能理論 (RSI)", "🔴 看空", f"RSI高達 {latest['RSI_14']}，進入超買區"])
        bear_count += 1
    elif latest['RSI_14'] < 25:
        analysis_list.append(["2. 動能理論 (RSI)", "🟢 看多", f"RSI低達 {latest['RSI_14']}，進入超賣區"])
        bull_count += 1
    else:
        analysis_list.append(["2. 動能理論 (RSI)", "⚪ 中立", "RSI處於正常震盪區間"])
        
    # 3. MACD
    if latest['MACD_Hist'] > 0:
        analysis_list.append(["3. 波段理論 (MACD)", "🟢 看多", "柱狀圖為正，多方動能延續"])
        bull_count += 1
    else:
        analysis_list.append(["3. 波段理論 (MACD)", "🔴 看空", "柱狀圖為負，空方動能延續"])
        bear_count += 1
        
    # 4. 布林通道
    if latest['Close'] > latest['BB_Up']:
        analysis_list.append(["4. 布林通道 (BB)", "🟢 看多", "股價突破上軌，強勢表態 (或短線過熱)"])
        bull_count += 1
    elif latest['Close'] < latest['BB_Low']:
        analysis_list.append(["4. 布林通道 (BB)", "🔴 看空", "股價跌破下軌，弱勢探底"])
        bear_count += 1
    else:
        analysis_list.append(["4. 布林通道 (BB)", "⚪ 中立", "股價在通道內正常游走"])
        
    # 5. KD 指標
    if latest['K'] > 80 and latest['D'] > 80:
        analysis_list.append(["5. KD 隨機指標", "🔴 看空", f"K({latest['K']}) D({latest['D']}) 處於高檔超買區"])
        bear_count += 1
    elif latest['K'] < 20 and latest['D'] < 20:
        analysis_list.append(["5. KD 隨機指標", "🟢 看多", f"K({latest['K']}) D({latest['D']}) 處於低檔超賣區"])
        bull_count += 1
    else:
        analysis_list.append(["5. KD 隨機指標", "⚪ 中立", "數值居中，無極端超買賣現象"])

    # 6. K線型態學 (簡單判定：長下影線/實體紅黑K)
    body = abs(latest['Close'] - latest['Open'])
    lower_shadow = min(latest['Close'], latest['Open']) - latest['Low']
    upper_shadow = latest['High'] - max(latest['Close'], latest['Open'])
    
    if lower_shadow > (body * 2) and body > 0:
        analysis_list.append(["6. K線型態學", "🟢 看多", "出現長下影線，下方買盤支撐強勁"])
        bull_count += 1
    elif upper_shadow > (body * 2) and body > 0:
        analysis_list.append(["6. K線型態學", "🔴 看空", "出現長上影線，上方賣壓沉重"])
        bear_count += 1
    else:
        analysis_list.append(["6. K線型態學", "⚪ 中立", "今日K線型態無特殊強烈反轉暗示"])
        
    # 7. 缺口理論
    if latest['Low'] > latest['Prev_High']:
        analysis_list.append(["7. 缺口理論 (Gap)", "🟢 看多", "今日出現向上跳空缺口，多方極度強勢"])
        bull_count += 1
    elif latest['High'] < latest['Prev_Low']:
        analysis_list.append(["7. 缺口理論 (Gap)", "🔴 看空", "今日出現向下跳空缺口，空方極度強勢"])
        bear_count += 1
    else:
        analysis_list.append(["7. 缺口理論 (Gap)", "⚪ 中立", "今日無跳空缺口產生"])

    db3_df = pd.DataFrame(analysis_list, columns=["分析方法", "當前訊號", "狀態描述"])
    
    # --- 🧠 專家級綜合矛盾分析引擎 ---
    expert_comment = ""
    trend_is_up = (latest['SMA_5'] > latest['SMA_20']) and (latest['MACD_Hist'] > 0)
    trend_is_down = (latest['SMA_5'] < latest['SMA_20']) and (latest['MACD_Hist'] < 0)
    
    overbought_signals = (latest['RSI_14'] > 75) or (latest['K'] > 80)
    oversold_signals = (latest['RSI_14'] < 25) or (latest['K'] < 20)
    
    # 邏輯判定 1：趨勢強烈時的指標鈍化 (這就是專家的視角)
    if trend_is_up and overbought_signals:
        expert_comment += "💡 **【專家視角：高檔鈍化現象】** 目前均線與MACD呈現強烈多頭趨勢。雖然 RSI 或 KD 顯示「超買看空」，但這在強勢股中常是「高檔鈍化」現象（強者恆強）。**此時不應盲目採信超買看空訊號而提早下車或放空**，建議以均線是否跌破作為主要出場依據。\n\n"
    elif trend_is_down and oversold_signals:
        expert_comment += "💡 **【專家視角：低檔鈍化現象】** 目前大趨勢強烈偏空。雖然 RSI 或 KD 顯示「超賣看多」，但在弱勢股中這可能只是「低檔鈍化」（弱者恆弱）。**此時超賣的看多訊號參考價值極低，切勿輕易摸底接刀**。\n\n"
    
    # 邏輯判定 2：價格行為 (Price Action) 最優先
    if latest['Low'] > latest['Prev_High'] or (lower_shadow > (body * 2)):
        expert_comment += "💡 **【專家視角：價格行為表態】** 今日出現「跳空缺口」或「長下影線」，這是市場資金最真實、最即時的表態。在短線操作上，此類 K 線型態的看多權重應大於其他落後的均線指標。可以今日低點作為停損防守線。\n\n"

    # 邏輯判定 3：常規矛盾
    if expert_comment == "":
        if (bull_count > 0) and (bear_count > 0):
            expert_comment += "💡 **【專家視角：盤勢震盪】** 目前各指標缺乏共識，趨勢指標與震盪指標方向分歧，代表目前處於「盤整區間」或「趨勢轉換期」。建議空手觀望，或切換至布林通道的上下軌進行區間高出低進操作。\n\n"
        elif bull_count > 0:
            expert_comment += "💡 **【專家視角：順勢做多】** 7大指標呈現完美的偏多共識，無矛盾。順勢操作勝率極高。\n\n"
        elif bear_count > 0:
            expert_comment += "💡 **【專家視角：順勢偏空】** 7大指標呈現強烈的偏空共識，無矛盾。嚴格執行多單停損，空手觀望。\n\n"

    return db3_df, date, close_price, bull_count, bear_count, expert_comment

def plot_candlestick(db2):
    plot_data = db2.tail(120) 
    fig = go.Figure(data=[go.Candlestick(x=plot_data['Date'],
                    open=plot_data['Open'], high=plot_data['High'],
                    low=plot_data['Low'], close=plot_data['Close'],
                    name='K線')])
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['SMA_5'], mode='lines', name='5日均線', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['SMA_20'], mode='lines', name='20日均線', line=dict(color='orange', width=1.5)))
    # 圖表也加入布林通道
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['BB_Up'], mode='lines', name='布林上軌', line=dict(color='rgba(255, 0, 0, 0.3)', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['BB_Low'], mode='lines', name='布林下軌', line=dict(color='rgba(0, 255, 0, 0.3)', width=1, dash='dot')))
    
    fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), height=400)
    return fig

# --- 網頁畫面呈現 ---
if update_button or stock_code:
    stock_code = stock_code.strip() 
    with st.spinner('正在搜尋台股資料庫與計算 7 大指標...'):
        db1_price_data, actual_symbol = fetch_stock_data(stock_code)
        
    if db1_price_data is not None:
        market_type = "上市" if ".TW" in actual_symbol else "上櫃"
        st.write(f"### 🎯 分析標的：{stock_code} ({market_type})")
        
        db2_ta_data = generate_db2(db1_price_data)
        chart_fig = plot_candlestick(db2_ta_data)
        st.plotly_chart(chart_fig, use_container_width=True)
        
        st.write("### 🧠 系統研判與專家視角 (DB3)")
        db3_df, target_date, current_price, bulls, bears, expert_comment = generate_db3(db2_ta_data)
        st.write(f"**分析日期：** {target_date} ｜ **最新收盤價：** {current_price}")
        
        # 顯示 7 大武器清單
        st.table(db3_df)
        
        # 顯示專家級解讀
        st.write("#### ⚖️ 專家矛盾分析與權重判讀")
        st.info(expert_comment)
        
        st.write(f"*【基礎統計】看多指標：{bulls} 個 ｜ 看空指標：{bears} 個*")
                
        with st.expander("🗄️ 展開查看原始資料 (DB1) 與 技術指標 (DB2)"):
            st.dataframe(db2_ta_data.tail(5), use_container_width=True)
    else:
        st.error(f"❌ 找不到代號為「{stock_code}」的股票。")
