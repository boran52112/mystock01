import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# 設定網頁標題與排版
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

# --- 抓取台灣本土籌碼資料 (法人 + 融資券) ---
def fetch_chip_data(code):
    try:
        dl = DataLoader()
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=code, start_date=start_date)
        df_pivot = pd.DataFrame()
        if not df_inst.empty:
            df_pivot = df_inst.pivot_table(index='date', columns='name', values='buy_sell', aggfunc='sum').reset_index()
            df_pivot.rename(columns={'date': 'Date'}, inplace=True)
            
            if '外陸資買賣超股數(不含外資自營商)' in df_pivot.columns:
                df_pivot['Foreign_Buy'] = df_pivot['外陸資買賣超股數(不含外資自營商)'] / 1000
            else:
                df_pivot['Foreign_Buy'] = 0
                
            if '投信買賣超股數' in df_pivot.columns:
                df_pivot['Trust_Buy'] = df_pivot['投信買賣超股數'] / 1000
            else:
                df_pivot['Trust_Buy'] = 0
                
            df_pivot = df_pivot[['Date', 'Foreign_Buy', 'Trust_Buy']]

        df_margin = dl.taiwan_stock_margin_purchase_short_sale(stock_id=code, start_date=start_date)
        if not df_margin.empty:
            df_margin = df_margin[['date', 'MarginPurchaseTodayBalance', 'ShortSaleTodayBalance']]
            df_margin.rename(columns={
                'date': 'Date', 
                'MarginPurchaseTodayBalance': 'Margin_Bal',
                'ShortSaleTodayBalance': 'Short_Bal'
            }, inplace=True)
            
        if not df_pivot.empty and not df_margin.empty:
            df_final_chip = pd.merge(df_pivot, df_margin, on='Date', how='outer')
        elif not df_pivot.empty:
            df_final_chip = df_pivot
        elif not df_margin.empty:
            df_final_chip = df_margin
        else:
            return None

        return df_final_chip
    except Exception as e:
        return None

# DB1：原始資料庫 (價量 + 籌碼)
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

    df_chip = fetch_chip_data(code)
    has_chip_data = False
    if df_chip is not None:
        df_price = pd.merge(df_price, df_chip, on='Date', how='left')
        df_price.fillna(0, inplace=True) 
        has_chip_data = True
    else:
        df_price['Foreign_Buy'] = 0
        df_price['Trust_Buy'] = 0
        df_price['Margin_Bal'] = 0
        df_price['Short_Bal'] = 0
        
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
    db2['Prev_Close'] = db2['Close'].shift(1)
    
    if 'Margin_Bal' in db2.columns and 'Short_Bal' in db2.columns:
        db2['Margin_Diff'] = db2['Margin_Bal'].diff()
        db2['Short_Diff'] = db2['Short_Bal'].diff()
    else:
        db2['Margin_Diff'] = 0
        db2['Short_Diff'] = 0
    
    db2.dropna(inplace=True) 
    return db2.round(2)

# DB3：結論與矛盾分析 (升級版專家大腦)
def generate_db3(db2, has_chip):
    latest = db2.iloc[-1]
    date = latest['Date']
    close_price = latest['Close']
    
    analysis_list = []
    bull_count, bear_count = 0, 0
    
    # 紀錄各指標狀態供專家引擎使用
    is_trend_up = latest['SMA_5'] > latest['SMA_20']
    is_macd_up = latest['MACD_Hist'] > 0
    is_rsi_overbought = latest['RSI_14'] > 75
    is_rsi_oversold = latest['RSI_14'] < 25
    is_kd_overbought = latest['K'] > 80 and latest['D'] > 80
    is_kd_oversold = latest['K'] < 20 and latest['D'] < 20
    is_bb_up_break = latest['Close'] > latest['BB_Up']
    is_bb_low_break = latest['Close'] < latest['BB_Low']
    
    # 1. 均線
    if is_trend_up:
        analysis_list.append(["1. 均線理論", "🟢 看多", "5日線大於20日線"])
        bull_count += 1
    else:
        analysis_list.append(["1. 均線理論", "🔴 看空", "5日線小於20日線"])
        bear_count += 1
        
    # 2. RSI
    if is_rsi_overbought:
        analysis_list.append(["2. 動能理論 (RSI)", "🔴 看空", f"RSI為 {latest['RSI_14']} (超買)"])
        bear_count += 1
    elif is_rsi_oversold:
        analysis_list.append(["2. 動能理論 (RSI)", "🟢 看多", f"RSI為 {latest['RSI_14']} (超賣)"])
        bull_count += 1
    else:
        analysis_list.append(["2. 動能理論 (RSI)", "⚪ 中立", "正常震盪"])
        
    # 3. MACD
    if is_macd_up:
        analysis_list.append(["3. 波段理論 (MACD)", "🟢 看多", "MACD紅柱，多方動能"])
        bull_count += 1
    else:
        analysis_list.append(["3. 波段理論 (MACD)", "🔴 看空", "MACD綠柱，空方動能"])
        bear_count += 1
        
    # 4. 布林通道
    if is_bb_up_break:
        analysis_list.append(["4. 布林通道", "🟢 看多", "突破上軌"])
        bull_count += 1
    elif is_bb_low_break:
        analysis_list.append(["4. 布林通道", "🔴 看空", "跌破下軌"])
        bear_count += 1
    else:
        analysis_list.append(["4. 布林通道", "⚪ 中立", "通道內游走"])
        
    # 5. KD 指標
    if is_kd_overbought:
        analysis_list.append(["5. KD 指標", "🔴 看空", "高檔超買"])
        bear_count += 1
    elif is_kd_oversold:
        analysis_list.append(["5. KD 指標", "🟢 看多", "低檔超賣"])
        bull_count += 1
    else:
        analysis_list.append(["5. KD 指標", "⚪ 中立", "數值居中"])

    # 6. K線型態
    body = abs(latest['Close'] - latest['Open'])
    lower_shadow = min(latest['Close'], latest['Open']) - latest['Low']
    upper_shadow = latest['High'] - max(latest['Close'], latest['Open'])
    is_lower_shadow_long = lower_shadow > (body * 2) and body > 0
    is_upper_shadow_long = upper_shadow > (body * 2) and body > 0
    
    if is_lower_shadow_long:
        analysis_list.append(["6. K線型態", "🟢 看多", "長下影線支撐"])
        bull_count += 1
    elif is_upper_shadow_long:
        analysis_list.append(["6. K線型態", "🔴 看空", "長上影線賣壓"])
        bear_count += 1
    else:
        analysis_list.append(["6. K線型態", "⚪ 中立", "無特殊型態"])
        
    # 7. 缺口理論
    is_gap_up = latest['Low'] > latest['Prev_High']
    is_gap_down = latest['High'] < latest['Prev_Low']
    if is_gap_up:
        analysis_list.append(["7. 缺口理論", "🟢 看多", "向上跳空缺口"])
        bull_count += 1
    elif is_gap_down:
        analysis_list.append(["7. 缺口理論", "🔴 看空", "向下跳空缺口"])
        bear_count += 1
    else:
        analysis_list.append(["7. 缺口理論", "⚪ 中立", "無跳空缺口"])

    # --- 籌碼區塊 ---
    trust_3d_sum, foreign_3d_sum = 0, 0
    margin_diff, short_diff = 0, 0
    
    if has_chip:
        recent_3_days = db2.tail(3)
        foreign_3d_sum = recent_3_days['Foreign_Buy'].sum()
        trust_3d_sum = recent_3_days['Trust_Buy'].sum()
        
        desc_inst = f"近三日外資:{int(foreign_3d_sum)}張, 投信:{int(trust_3d_sum)}張"
        if trust_3d_sum > 500 or foreign_3d_sum > 2000:
            analysis_list.append(["8. 籌碼理論 (法人)", "🟢 看多", desc_inst + " (積極買超)"])
            bull_count += 1
        elif trust_3d_sum < -500 or foreign_3d_sum < -2000:
            analysis_list.append(["8. 籌碼理論 (法人)", "🔴 看空", desc_inst + " (積極賣超)"])
            bear_count += 1
        else:
            analysis_list.append(["8. 籌碼理論 (法人)", "⚪ 中立", desc_inst + " (動作不大)"])
            
        margin_diff = latest['Margin_Diff']
        short_diff = latest['Short_Diff']
        price_diff = latest['Close'] - latest['Prev_Close']
        desc_retail = f"本日融資增減: {int(margin_diff)}張, 融券增減: {int(short_diff)}張"
        
        if margin_diff < 0 and short_diff > 0:
            analysis_list.append(["9. 籌碼理論 (散戶)", "🟢 看多", desc_retail + " (資減券增，醞釀軋空)"])
            bull_count += 1
        elif margin_diff > 0 and price_diff < 0:
            analysis_list.append(["9. 籌碼理論 (散戶)", "🔴 看空", desc_retail + " (資增價跌，散戶接刀)"])
            bear_count += 1
        else:
            analysis_list.append(["9. 籌碼理論 (散戶)", "⚪ 中立", desc_retail + " (散戶籌碼無明顯背離)"])
    else:
        analysis_list.append(["8. 籌碼理論 (法人)", "⚪ 未知", "免費版API連線限制，暫無籌碼資料"])
        analysis_list.append(["9. 籌碼理論 (散戶)", "⚪ 未知", "免費版API連線限制，暫無籌碼資料"])

    db3_df = pd.DataFrame(analysis_list, columns=["分析方法", "當前訊號", "狀態描述"])
    
    # ==========================================
    # 🧠 全新架構：專家矛盾分析與權重判讀引擎
    # ==========================================
    expert_report = ""

    # 段落 1：主趨勢與位階判定
    expert_report += "#### 📌 1. 主趨勢與位階判定\n"
    if is_trend_up and is_macd_up:
        expert_report += "> 目前均線與 MACD 皆偏多，**屬於強勢多頭格局**。在這種格局下，我們應該「順勢做多」，任何拉回均線都是潛在買點。\n\n"
    elif not is_trend_up and not is_macd_up:
        expert_report += "> 目前均線與 MACD 皆偏空，**屬於弱勢空頭格局**。在這種格局下，壓力沉重，任何反彈都容易遇到解套賣壓，應避免輕易摸底。\n\n"
    else:
        expert_report += "> 目前均線與 MACD 方向不一致，**屬於震盪整理格局**。趨勢尚未明朗，建議縮小部位操作。\n\n"

    # 段落 2：矛盾與盲點解析 (核心靈魂)
    expert_report += "#### 🔍 2. 指標矛盾與盲點解析\n"
    conflict_found = False

    # 矛盾A：趨勢 vs 震盪指標鈍化
    if is_trend_up and (is_rsi_overbought or is_kd_overbought):
        expert_report += "- **【高檔鈍化忽略原則】**：雖然 RSI 或 KD 亮起「紅燈(超買看空)」，但在強多頭格局中，這往往是「強者恆強」的高檔鈍化現象。**專家建議：此時應忽略 KD/RSI 的看空訊號，不可因此提早放空或賣出，應以均線是否跌破為防守線。**\n"
        conflict_found = True
    elif not is_trend_up and (is_rsi_oversold or is_kd_oversold):
        expert_report += "- **【低檔鈍化忽略原則】**：雖然 RSI 或 KD 亮起「綠燈(超賣看多)」，但在空頭格局中，這通常是「弱者恆弱」的低檔鈍化。**專家建議：此時 KD/RSI 的買進訊號是無效的雜訊，切勿因為看到超賣就進場接刀。**\n"
        conflict_found = True

    # 矛盾B：價格表態優先
    if not is_trend_up and (is_lower_shadow_long or is_gap_up):
        expert_report += "- **【價格表態優於指標】**：雖然整體趨勢偏空，但今日出現了真實的資金買盤介入（長下影線或向上缺口）。**專家建議：K線型態反映了最即時的主力動作，此時型態學的看多權重應大於落後的均線，可視為短線搶反彈的契機。**\n"
        conflict_found = True

    # 矛盾C：籌碼背離
    if has_chip:
        if not is_trend_up and (trust_3d_sum > 500 or foreign_3d_sum > 2000):
            expert_report += "- **【籌碼背離 (法人偷接)】**：技術面偏空，但外資或投信卻連續大買。**專家建議：法人在逢低建倉，籌碼面看多權重大於技術面看空，隨時醞釀反轉。**\n"
            conflict_found = True
        elif is_trend_up and (trust_3d_sum < -500 or foreign_3d_sum < -2000):
            expert_report += "- **【籌碼背離 (拉高出貨)】**：技術面強勢，但法人卻在大量倒貨。**專家建議：請高度警戒，這可能是假突破，籌碼已鬆動，切勿追高。**\n"
            conflict_found = True

    if not conflict_found:
        expert_report += "- 目前各項指標方向大致同調，**未偵測到明顯的邏輯矛盾或指標陷阱**，可直接參考基礎統計的紅綠燈數量。\n"
    expert_report += "\n"

    # 段落 3：具體操作策略
    expert_report += "#### 🎯 3. 具體操作策略結論\n"
    if bulls > bears * 2:
        expert_report += "**👉 【偏多操作】** 綜合研判目前多方佔有絕對優勢，建議持股續抱，或尋找量縮拉回時佈局，並以 20 日均線作為波段停損點。"
    elif bears > bulls * 2:
        expert_report += "**👉 【偏空操作/觀望】** 綜合研判目前空方壓力沉重，多單應嚴格執行停損，空手者請耐心觀望，切勿急於進場。"
    else:
        expert_report += "**👉 【保守觀望】** 目前多空勢均力敵，盤勢陷入膠著。建議空手觀望，等待趨勢表態（例如帶量突破布林上軌或跌破下軌）後再行操作。"

    return db3_df, date, close_price, bull_count, bear_count, expert_report

def plot_candlestick(db2):
    plot_data = db2.tail(120) 
    fig = go.Figure(data=[go.Candlestick(x=plot_data['Date'],
                    open=plot_data['Open'], high=plot_data['High'],
                    low=plot_data['Low'], close=plot_data['Close'],
                    name='K線')])
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['SMA_5'], mode='lines', name='5日均線', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['SMA_20'], mode='lines', name='20日均線', line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['BB_Up'], mode='lines', name='布林上軌', line=dict(color='rgba(255, 0, 0, 0.3)', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['BB_Low'], mode='lines', name='布林下軌', line=dict(color='rgba(0, 255, 0, 0.3)', width=1, dash='dot')))
    
    fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), height=400)
    return fig

# --- 網頁畫面呈現 ---
if update_button or stock_code:
    stock_code = stock_code.strip() 
    with st.spinner('正在搜尋資料與計算籌碼... (免費版API若無回應屬正常限制)'):
        db1_price_data, actual_symbol, has_chip = fetch_stock_data(stock_code)
        
    if db1_price_data is not None:
        market_type = "上市" if ".TW" in actual_symbol else "上櫃"
        st.write(f"### 🎯 分析標的：{stock_code} ({market_type})")
        
        db2_ta_data = generate_db2(db1_price_data)
        chart_fig = plot_candlestick(db2_ta_data)
        st.plotly_chart(chart_fig, use_container_width=True)
        
        st.write("### 🧠 系統研判與專家報告 (DB3)")
        db3_df, target_date, current_price, bulls, bears, expert_report = generate_db3(db2_ta_data, has_chip)
        st.write(f"**分析日期：** {target_date} ｜ **最新收盤價：** {current_price}")
        
        # 基礎統計與表格
        st.write(f"📊 **基礎指標統計**：🟢 看多 {bulls} 個 ｜ 🔴 看空 {bears} 個")
        st.table(db3_df)
        
        # 專家報告 (保證絕對不留白)
        st.write("---")
        st.info(expert_report)
                
        with st.expander("🗄️ 展開查看原始資料 (含籌碼數據) 與 技術指標"):
            st.dataframe(db2_ta_data.tail(5), use_container_width=True)
    else:
        st.error(f"❌ 找不到代號為「{stock_code}」的股票。")
