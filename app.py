import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 網頁配置 ---
st.set_page_config(page_title="AI 專家級 9 大指標分析系統", layout="wide")

st.markdown("""
    <style>
    .report-card { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.05); min-height: 200px; }
    .strategy-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI 專家級 9 大指標技術分析系統")

# --- 資料抓取 (DB1) ---
@st.cache_data(ttl=3600)
def fetch_complete_data(code):
    df_price, actual_sym = None, None
    for suffix in [".TW", ".TWO"]:
        try:
            temp_sym = f"{code}{suffix}"
            df = yf.download(temp_sym, period="1y", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df.reset_index(inplace=True)
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                df_price, actual_sym = df, temp_sym
                break
        except: continue
    if df_price is None: return None, None, False

    try:
        dl = DataLoader()
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=code, start_date=start_date)
        df_margin = dl.taiwan_stock_margin_purchase_short_sale(stock_id=code, start_date=start_date)
        
        df_pivot = pd.DataFrame()
        if not df_inst.empty:
            df_pivot = df_inst.pivot_table(index='date', columns='name', values='buy_sell', aggfunc='sum').reset_index()
            df_pivot.rename(columns={'date': 'Date'}, inplace=True)
            f_col, t_col = '外陸資買賣超股數(不含外資自營商)', '投信買賣超股數'
            df_pivot['Foreign_Buy'] = df_pivot[f_col] / 1000 if f_col in df_pivot.columns else 0
            df_pivot['Trust_Buy'] = df_pivot[t_col] / 1000 if t_col in df_pivot.columns else 0
            df_pivot = df_pivot[['Date', 'Foreign_Buy', 'Trust_Buy']]

        if not df_margin.empty:
            df_margin = df_margin[['date', 'MarginPurchaseTodayBalance', 'ShortSaleTodayBalance']]
            df_margin.rename(columns={'date': 'Date', 'MarginPurchaseTodayBalance': 'Margin_Bal', 'ShortSaleTodayBalance': 'Short_Bal'}, inplace=True)
            
        if not df_pivot.empty: df_price = pd.merge(df_price, df_pivot, on='Date', how='left')
        if not df_margin.empty: df_price = pd.merge(df_price, df_margin, on='Date', how='left')
        df_price.fillna(method='ffill', inplace=True)
        df_price.fillna(0, inplace=True)
        return df_price, actual_sym, True
    except:
        return df_price, actual_sym, False

# --- 技術指標計算 (DB2) ---
def generate_ta_db2(df):
    db2 = df.copy()
    db2['SMA_5'] = db2['Close'].rolling(5).mean()
    db2['SMA_20'] = db2['Close'].rolling(20).mean()
    delta = db2['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    db2['RSI'] = 100 - (100 / (1 + gain/loss))
    exp1 = db2['Close'].ewm(span=12).mean()
    exp2 = db2['Close'].ewm(span=26).mean()
    db2['MACD'] = exp1 - exp2
    db2['MACD_S'] = db2['MACD'].ewm(span=9).mean()
    db2['MACD_H'] = db2['MACD'] - db2['MACD_S']
    db2['BB_Mid'] = db2['Close'].rolling(20).mean()
    std = db2['Close'].rolling(20).std()
    db2['BB_Up'] = db2['BB_Mid'] + 2*std
    db2['BB_Low'] = db2['BB_Mid'] - 2*std
    l9, h9 = db2['Low'].rolling(9).min(), db2['High'].rolling(9).max()
    rsv = (db2['Close'] - l9) / (h9 - l9) * 100
    db2['K'] = rsv.ewm(com=2).mean()
    db2['D'] = db2['K'].ewm(com=2).mean()
    db2['Prev_Close'] = db2['Close'].shift(1)
    db2['Prev_High'] = db2['High'].shift(1)
    db2['Prev_Low'] = db2['Low'].shift(1)
    if 'Margin_Bal' in db2.columns:
        db2['Margin_Diff'] = db2['Margin_Bal'].diff()
        db2['Short_Diff'] = db2['Short_Bal'].diff()
    return db2.dropna()

# --- 專家分析引擎 (DB3) ---
def generate_expert_db3(db2, has_chip):
    latest = db2.iloc[-1]
    prev = db2.iloc[-2]
    analysis = []
    bull_score, bear_score, neutral_score = 0, 0, 0
    
    # 1. 均線理論
    if latest['SMA_5'] > latest['SMA_20']:
        analysis.append(["1. 均線理論", "🟢 看多", "多頭排列"])
        bull_score += 1
    else:
        analysis.append(["1. 均線理論", "🔴 看空", "趨勢偏弱"])
        bear_score += 1

    # 2. RSI 動能
    if latest['RSI'] > 75:
        analysis.append(["2. 動能 (RSI)", "🔴 看空", "超買"])
        bear_score += 1
    elif latest['RSI'] < 25:
        analysis.append(["2. 動能 (RSI)", "🟢 看多", "超賣"])
        bull_score += 1
    else:
        analysis.append(["2. 動能 (RSI)", "⚪ 中立", "震盪區"])
        neutral_score += 1

    # 3. MACD 波段
    if latest['MACD_H'] > 0:
        analysis.append(["3. 波段 (MACD)", "🟢 看多", "紅柱向上"])
        bull_score += 1
    else:
        analysis.append(["3. 波段 (MACD)", "🔴 看空", "綠柱向下"])
        bear_score += 1

    # 4. 布林通道
    if latest['Close'] > latest['BB_Up']:
        analysis.append(["4. 布林通道", "🟢 看多", "強勢上攻"])
        bull_score += 1
    elif latest['Close'] < latest['BB_Low']:
        analysis.append(["4. 布林通道", "🔴 看空", "跌穿底線"])
        bear_score += 1
    else:
        analysis.append(["4. 布林通道", "⚪ 中立", "通道內盤整"])
        neutral_score += 1

    # 5. KD 指標
    if latest['K'] > 80:
        analysis.append(["5. KD 指標", "🔴 看空", "高檔區"])
        bear_score += 1
    elif latest['K'] < 20:
        analysis.append(["5. KD 指標", "🟢 看多", "低檔區"])
        bull_score += 1
    else:
        analysis.append(["5. KD 指標", "⚪ 中立", "無訊號"])
        neutral_score += 1

    # 6. K線型態
    body = abs(latest['Close'] - latest['Open'])
    lower_s = min(latest['Close'], latest['Open']) - latest['Low']
    if lower_s > body * 2 and body > 0:
        analysis.append(["6. K線型態", "🟢 看多", "下影線支撐"])
        bull_score += 1
    else:
        analysis.append(["6. K線型態", "⚪ 中立", "常規型態"])
        neutral_score += 1

    # 7. 缺口理論
    if latest['Low'] > prev['High']:
        analysis.append(["7. 缺口理論", "🟢 看多", "跳空上漲"])
        bull_score += 1
    elif latest['High'] < prev['Low']:
        analysis.append(["7. 缺口理論", "🔴 看空", "跳空下跌"])
        bear_score += 1
    else:
        analysis.append(["7. 缺口理論", "⚪ 中立", "無缺口"])
        neutral_score += 1

    # 8. 籌碼 (法人)
    if has_chip:
        inst_net = latest['Foreign_Buy'] + latest['Trust_Buy']
        if inst_net > 500:
            analysis.append(["8. 籌碼 (法人)", "🟢 看多", "大戶進場"])
            bull_score += 1
        elif inst_net < -500:
            analysis.append(["8. 籌碼 (法人)", "🔴 看空", "大戶撤退"])
            bear_score += 1
        else:
            analysis.append(["8. 籌碼 (法人)", "⚪ 中立", "觀望"])
            neutral_score += 1
    else:
        analysis.append(["8. 籌碼 (法人)", "⚪ 未知", "缺少資料"])

    # 9. 籌碼 (散戶)
    if has_chip:
        if latest['Margin_Diff'] > 500 and latest['Close'] < prev['Close']:
            analysis.append(["9. 籌碼 (散戶)", "🔴 看空", "散戶被套"])
            bear_score += 1
        elif latest['Margin_Diff'] < -500 and latest['Short_Diff'] > 200:
            analysis.append(["9. 籌碼 (散戶)", "🟢 看多", "軋空預備"])
            bull_score += 1
        else:
            analysis.append(["9. 籌碼 (散戶)", "⚪ 中立", "穩定"])
            neutral_score += 1
    else:
        analysis.append(["9. 籌碼 (散戶)", "⚪ 未知", "缺少資料"])

    # ==========================
    # 🧠 專家報告引擎 (保證不留白)
    # ==========================
    report = "#### 🔍 1. 核心矛盾與信度解析\n"
    
    # A. 概括總結 (保證第一段文字)
    summary_text = f"目前市場呈現 **{'偏空' if bear_score > bull_score else '偏多' if bull_score > bear_score else '中立震盪'}** 態勢。"
    summary_text += f"在 9 大指標中，看多 {bull_score} 項，看空 {bear_score} 項，其餘 {neutral_score + (0 if has_chip else 2)} 項為中立或資訊不足。"
    report += f"> {summary_text}\n\n"

    # B. 邏輯衝突檢查
    conflict_found = False
    if latest['SMA_5'] > latest['SMA_20'] and latest['RSI'] > 75:
        report += "- **指標衝突(強勢鈍化)**：均線雖然多頭排列，但動能指標已達超買警戒。這意味著股價進入「噴發期」，信度以均線為準，但切記不可追高。\n"
        conflict_found = True
    if latest['MACD_H'] < 0 and latest['Close'] > latest['SMA_5']:
        report += "- **指標衝突(背離訊號)**：價格雖然反彈站上均線，但 MACD 動能尚未轉正。專家認為這是「弱勢反彈」，信度以波段指標為準，需防範再次回落。\n"
        conflict_found = True
    if not has_chip:
        report += "- **資料限制提醒**：目前缺少本土籌碼數據(法人/融資)，僅能以技術型態判定，信度約為 60%。若股價出現異常放量，應補足籌碼面分析。\n"
        conflict_found = True
    
    # C. 若無衝突，則給予一致性評論
    if not conflict_found:
        if neutral_score > 4:
            report += "- **趨勢評論**：目前大多數指標處於「中立」狀態。這代表市場缺乏明確的方向感，成交量萎縮，屬於技術面上的「橫盤整理」，建議等待指標集體轉向再表態。\n"
        else:
            report += "- **趨勢評論**：目前各項指標方向大致相符，無明顯背離現象。建議順著目前的市場趨勢操作即可。\n"

    # ==========================
    # 🎯 2. 具體操作策略
    # ==========================
    strategy = "#### 🎯 2. 具體操作策略建議\n"
    entry_p = latest['SMA_5'].round(2)
    stop_p = (latest['Close'] * 0.95).round(2)
    
    if bull_score > bear_score + 2:
        strategy += f"- **操作方向**：【偏多介入】\n- **建議位階**：目前多方優勢明顯。若拉回至 **{entry_p} 元** 附近不破可視為買點。\n- **風險控制**：止損位設定在 **{stop_p} 元**。"
    elif bear_score > bull_score + 2:
        strategy += f"- **操作方向**：【偏空/觀望】\n- **建議位階**：目前壓力沉重。建議等待股價重回 **{latest['SMA_20'].round(2)} 元** 站穩後再進場。\n- **風險控制**：空手者切勿在此摸底。"
    else:
        strategy += f"- **操作方向**：【區間盤整】\n- **建議位階**：不建議大幅建倉。可在 **{latest['BB_Low'].round(2)}** 到 **{latest['BB_Up'].round(2)}** 之間進行極短線操作。\n- **觀察指標**：等待 MACD 紅柱再次增長作為訊號。"

    return pd.DataFrame(analysis, columns=["分析維度", "訊號", "專家描述"]), bull_score, bear_score, report, strategy

# --- 繪圖 ---
def plot_stock_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_5'], line=dict(color='blue', width=1), name="5MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Up'], line=dict(color='rgba(255,0,0,0.2)', dash='dot'), name="布林上軌"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Low'], line=dict(color='rgba(0,255,0,0.2)', dash='dot'), name="布林下軌"), row=1, col=1)
    colors = ['red' if c >= o else 'green' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="成交量"), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, l=10, r=10, b=10))
    return fig

# --- 主 UI ---
s_code = st.text_input("📈 請輸入台股代號", "2330").strip()
if s_code:
    with st.spinner("專家大腦運算中..."):
        raw_data, sym_name, has_c = fetch_complete_data(s_code)
        if raw_data is not None:
            db2 = generate_ta_db2(raw_data)
            db3_df, bulls, bears, rpt, stg = generate_expert_db3(db2, has_c)
            
            c_price = db2.iloc[-1]['Close']
            prev_price = db2.iloc[-2]['Close']
            m1, m2, m3 = st.columns(3)
            m1.metric("當前股價", f"{c_price} TWD", f"{round(c_price-prev_price, 2)}")
            m2.metric("看多指標", f"{bulls} / 9")
            m3.metric("看空指標", f"{bears} / 9")
            
            st.plotly_chart(plot_stock_chart(db2.tail(120)), use_container_width=True)
            
            st.markdown("### 🧠 專家系統研判 (DB3)")
            col_res_l, col_res_r = st.columns([4, 6])
            with col_res_l:
                st.dataframe(db3_df, hide_index=True, use_container_width=True)
            with col_res_r:
                st.markdown(f'<div class="report-card">{rpt}</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown(f'<div class="strategy-card">{stg}</div>', unsafe_allow_html=True)
        else:
            st.error("查無資料，請確認代號。")
